"""HTTP API + static upload UI."""
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import SUPPORTED_EXTENSIONS, settings
from .embedder import make_embedder
from .llm import OllamaLLM
from .pipeline import Pipeline
from .store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC = Path(__file__).parent / "static"

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    embedder = make_embedder(settings.embedder, settings.model_name,
                             settings.model_cache, settings.embed_batch)
    store = Store(settings.db_path, embedder.dim)
    state.update(
        embedder=embedder,
        store=store,
        pipeline=Pipeline(settings, embedder, store),
        # One worker: documents are processed in upload order and the model
        # is never run from two threads at once.
        executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest"),
        llm=OllamaLLM(settings.ollama_url, settings.ollama_model, settings.ollama_num_ctx),
    )
    state["llm"].ensure_model_async()
    yield
    state["executor"].shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Local Embed", lifespan=lifespan)


def _doc_out(d: dict) -> dict:
    d = dict(d)
    d["timings"] = json.loads(d["timings"]) if d.get("timings") else None
    d.pop("sha256", None)
    return d


@app.get("/api/health")
def health():
    e = state["embedder"]
    return {"status": "ok", "model": e.name, "dim": e.dim, **state["store"].stats(),
            "chunk_tokens": settings.chunk_tokens, "chunk_overlap": settings.chunk_overlap,
            "llm": state["llm"].status}


@app.post("/api/documents")
async def upload(files: list[UploadFile] = File(...)):
    store: Store = state["store"]
    results = []
    for f in files:
        name = Path(f.filename or "upload").name
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results.append({"filename": name, "error": f"Unsupported type {ext or '(none)'}"})
            continue
        data = await f.read()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            results.append({"filename": name, "error": f"Larger than {settings.max_upload_mb} MB"})
            continue
        if not data:
            results.append({"filename": name, "error": "Empty file"})
            continue

        sha = hashlib.sha256(data).hexdigest()
        existing = store.find_by_hash(sha)
        if existing and existing["status"] != "failed":
            results.append({**_doc_out(existing), "duplicate": True})
            continue
        if existing:  # retrying a failed document: start clean
            store.delete_document(existing["id"])

        doc_id = store.create_document(name, sha, len(data))
        path = settings.upload_dir / f"{doc_id}{ext}"
        path.write_bytes(data)
        state["executor"].submit(state["pipeline"].process, doc_id, path)
        results.append({**_doc_out(store.get_document(doc_id)), "duplicate": False})
    return {"documents": results}


@app.get("/api/documents")
def list_documents():
    return {"documents": [_doc_out(d) for d in state["store"].list_documents()]}


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: int):
    store: Store = state["store"]
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return {**_doc_out(doc), "chunks": store.get_chunks(doc_id)}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int):
    store: Store = state["store"]
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    if doc["status"] not in ("indexed", "failed"):
        raise HTTPException(409, "Document is still being processed")
    store.delete_document(doc_id)
    for p in settings.upload_dir.glob(f"{doc_id}.*"):
        p.unlink(missing_ok=True)
    return {"deleted": doc_id}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=50)
    doc_ids: list[int] | None = None
    min_score: float = Field(0.0, ge=-1.0, le=1.0)


@app.post("/api/search")
def search(req: SearchRequest):
    t = time.perf_counter()
    qvec = state["embedder"].embed([req.query])[0]
    hits = state["store"].search(qvec, req.k, req.doc_ids, req.min_score)
    return {
        "query": req.query,
        "took_ms": round((time.perf_counter() - t) * 1000, 1),
        "results": [h.__dict__ for h in hits],
    }


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(None, ge=1, le=20)
    doc_ids: list[int] | None = None
    stream: bool = True


NOT_FOUND = "I couldn't find that in the documents."


@app.post("/api/ask")
def ask(req: AskRequest):
    """Retrieve the top chunks, then have the local LLM answer with [n] citations.

    Streaming responses are NDJSON: one `sources` event, then `token` events,
    then a final `done` (or `error`) event.
    """
    llm: OllamaLLM = state["llm"]
    t = time.perf_counter()
    qvec = state["embedder"].embed([req.question])[0]
    hits = state["store"].search(qvec, req.k or settings.ask_top_k, req.doc_ids, settings.ask_min_score)
    retrieval_ms = round((time.perf_counter() - t) * 1000, 1)
    sources = [{"n": i, **h.__dict__} for i, h in enumerate(hits, 1)]

    if hits and not llm.ready:
        raise HTTPException(503, {"message": "Language model is not ready yet", "llm": llm.status})

    def events():
        yield {"type": "sources", "sources": sources, "retrieval_ms": retrieval_ms}
        if not hits:
            yield {"type": "token", "text": NOT_FOUND}
        else:
            try:
                for piece in llm.stream_answer(req.question, hits):
                    yield {"type": "token", "text": piece}
            except Exception as e:  # noqa: BLE001
                yield {"type": "error", "message": str(e)[:300]}
                return
        yield {"type": "done", "took_ms": round((time.perf_counter() - t) * 1000, 1),
               "model": llm.model}

    if req.stream:
        return StreamingResponse((json.dumps(e) + "\n" for e in events()),
                                 media_type="application/x-ndjson")

    answer, out = [], {}
    for e in events():
        if e["type"] == "token":
            answer.append(e["text"])
        elif e["type"] == "error":
            raise HTTPException(502, e["message"])
        else:
            out.update({k: v for k, v in e.items() if k != "type"})
    return {"question": req.question, "answer": "".join(answer).strip(), **out}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
