"""Stage 6 - generation: answer a question from retrieved chunks with a local Ollama model.

The model is pulled automatically on startup if Ollama doesn't have it yet;
`status` reports progress so the UI can show it while the download runs.
"""
import json
import logging
import threading
import time
from collections.abc import Iterator

import httpx

from .store import Hit

log = logging.getLogger("llm")

SYSTEM_PROMPT = """You answer questions using only the numbered sources the user provides.
You always cite sources with their number in square brackets."""

# Small local models follow rules far better when they come after the
# sources (closest to where generation starts) and include a worked example.
INSTRUCTIONS = """Answer the question using only the sources above.
- After every sentence, cite the source(s) it came from, like [1] or [2][3].
- If the sources answer only part of the question, answer that part and say what is missing.
- Only if the sources contain nothing relevant, reply exactly: I couldn't find that in the documents.
- Be concise: a few sentences or a short list. Do not mention "the sources" by name.

Example of the expected style:
Employees get 20 days of leave per year [1]. Requests need manager approval [2]."""


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        where = h.filename
        if h.page:
            where += f", page {h.page}"
        if h.heading:
            where += f", section \"{h.heading}\""
        blocks.append(f"[{i}] ({where})\n{h.text}")
    return ("Sources:\n\n" + "\n\n".join(blocks)
            + f"\n\n{INSTRUCTIONS}\n\nQuestion: {question}\nAnswer:")


class OllamaLLM:
    def __init__(self, base_url: str, model: str, num_ctx: int = 4096, temperature: float = 0.1):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.options = {"num_ctx": num_ctx, "temperature": temperature}
        self.status: dict = {"state": "starting", "model": model}
        self._client = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(10, read=300))

    @property
    def ready(self) -> bool:
        return self.status["state"] == "ready"

    def ensure_model_async(self) -> None:
        threading.Thread(target=self._ensure_model, name="ollama-pull", daemon=True).start()

    def _ensure_model(self, attempts: int = 5) -> None:
        # A pull can drop mid-download (network, Ollama restart). Ollama keeps
        # the partial blobs, so retrying resumes rather than starting over.
        for attempt in range(1, attempts + 1):
            if self._try_ensure_model():
                return
            if attempt < attempts:
                time.sleep(5 * attempt)
                log.warning("retrying model pull (%d/%d)", attempt + 1, attempts)

    def _try_ensure_model(self) -> bool:
        # Ollama may still be booting when we start; keep trying for a minute.
        for _ in range(30):
            try:
                tags = self._client.get("/api/tags").json()
                break
            except httpx.HTTPError:
                self.status = {"state": "unreachable", "model": self.model}
                time.sleep(2)
        else:
            log.error("Ollama not reachable at %s", self.base_url)
            return False

        names = {m["name"] for m in tags.get("models", [])}
        if self.model in names or f"{self.model}:latest" in names:
            self.status = {"state": "ready", "model": self.model}
            return True

        log.info("pulling %s", self.model)
        self.status = {"state": "pulling", "model": self.model, "progress": 0.0}
        try:
            with self._client.stream("POST", "/api/pull", json={"model": self.model},
                                     timeout=httpx.Timeout(10, read=None)) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    msg = json.loads(line)
                    if "error" in msg:
                        raise RuntimeError(msg["error"])
                    if msg.get("total"):
                        self.status["progress"] = round(msg.get("completed", 0) / msg["total"], 3)
            self.status = {"state": "ready", "model": self.model}
            log.info("%s ready", self.model)
            return True
        except Exception as e:  # noqa: BLE001
            log.exception("pull failed")
            self.status = {"state": "error", "model": self.model, "error": str(e)[:300]}
            return False

    def stream_answer(self, question: str, hits: list[Hit]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "stream": True,
            "options": self.options,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, hits)},
            ],
        }
        with self._client.stream("POST", "/api/chat", json=payload) as r:
            if r.status_code != 200:
                r.read()
                raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:300]}")
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                if piece := msg.get("message", {}).get("content"):
                    yield piece
                if msg.get("done"):
                    return
