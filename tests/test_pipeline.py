import importlib
import json
import io
import time

import docx
import pytest
from fastapi.testclient import TestClient

from app import cleaning, loaders
from app.chunking import chunk_sections, count_tokens
from app.loaders import Section


def make_pdf(pages: list[str]) -> bytes:
    """Minimal valid PDF with one Helvetica text line per page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        lines = text.split("\n")
        ops = "".join(f"BT /F1 12 Tf 72 {720 - 16 * i} Td ({ln}) Tj ET\n" for i, ln in enumerate(lines))
        objs.append(f"<< /Length {len(ops)} >>\nstream\n{ops}endstream")
        content = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content} 0 R >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for i, body in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n{body}\nendobj\n".encode())
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return out.getvalue()


# -- loaders ---------------------------------------------------------------
def test_markdown_heading_path_and_fences():
    md = "# Guide\nintro\n## Setup\nrun it\n```\n# not a heading\n```\n## Usage\nuse it"
    secs = loaders.load_markdown(md)
    assert [s.heading for s in secs] == ["Guide", "Guide > Setup", "Guide > Usage"]
    assert "# not a heading" in secs[1].text


def test_docx_headings_and_tables(tmp_path):
    d = docx.Document()
    d.add_heading("Policy", level=1)
    d.add_paragraph("Employees get 25 days of leave.")
    t = d.add_table(rows=1, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Plan", "Gold"
    path = tmp_path / "p.docx"
    d.save(path)
    secs = loaders.load(path)
    assert secs[0].heading == "Policy" and "25 days" in secs[0].text
    assert secs[-1].text == "Plan | Gold"


def test_pdf_pages_and_header_removal(tmp_path):
    pages = [f"ACME Confidential\nPage body number {w} about topic {w}\n{i}"
             for i, w in enumerate(["alpha", "beta", "gamma", "delta"], 1)]
    path = tmp_path / "doc.pdf"
    path.write_bytes(make_pdf(pages))
    secs = cleaning.clean(loaders.load(path))
    assert [s.page for s in secs] == [1, 2, 3, 4]
    assert "ACME Confidential" in secs[0].text
    assert all("ACME Confidential" not in s.text for s in secs[1:])
    assert "topic gamma" in secs[2].text


def test_unsupported_extension(tmp_path):
    p = tmp_path / "x.xlsx"
    p.write_bytes(b"x")
    with pytest.raises(loaders.UnsupportedFormat):
        loaders.load(p)


# -- cleaning --------------------------------------------------------------
def test_clean_text():
    raw = "Re-\nembedding   is\x00 great\r\n\n\n\nNext para"
    assert cleaning.clean_text(raw) == "Reembedding is great\n\nNext para"


# -- chunking --------------------------------------------------------------
def test_chunks_respect_budget_and_overlap():
    text = "\n\n".join(f"Sentence number {i} talks about vectors and retrieval." for i in range(200))
    chunks = chunk_sections([Section(text, page=3, heading="H")], max_tokens=60, overlap=15)
    assert len(chunks) > 5
    assert all(c.tokens <= 60 for c in chunks)
    assert all(c.page == 3 and c.heading == "H" for c in chunks)
    # neighbouring chunks share text
    tail = chunks[0].text.split()[-3:]
    assert " ".join(tail) in chunks[1].text
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_long_unbroken_text_still_splits():
    chunks = chunk_sections([Section("word " * 1000)], max_tokens=50, overlap=10)
    assert all(c.tokens <= 50 for c in chunks)
    assert count_tokens(" ".join(c.text for c in chunks)) >= 1000


# -- end to end through the API -----------------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDER", "hash")
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")
    import app.config, app.main
    importlib.reload(app.config)
    importlib.reload(app.main)
    with TestClient(app.main.app) as c:
        yield c


def wait_indexed(client, doc_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        d = client.get(f"/api/documents/{doc_id}").json()
        if d["status"] in ("indexed", "failed"):
            return d
        time.sleep(0.05)
    raise TimeoutError


def test_upload_search_dedupe_delete(client):
    files = [
        ("files", ("pets.md", b"# Pets\nCats purr and dogs bark loudly at the mailman.", "text/markdown")),
        ("files", ("space.txt", b"Rockets reach orbit using liquid fuel and oxygen.", "text/plain")),
        ("files", ("bad.exe", b"MZ", "application/octet-stream")),
    ]
    docs = client.post("/api/documents", files=files).json()["documents"]
    assert docs[2]["error"].startswith("Unsupported")
    pets, space = (wait_indexed(client, d["id"]) for d in docs[:2])
    assert pets["status"] == space["status"] == "indexed"
    assert pets["chunks"][0]["heading"] == "Pets"

    hits = client.post("/api/search", json={"query": "rockets orbit fuel", "k": 2}).json()["results"]
    assert hits[0]["filename"] == "space.txt"

    filtered = client.post("/api/search", json={"query": "rockets", "doc_ids": [pets["id"]]}).json()
    assert {h["filename"] for h in filtered["results"]} == {"pets.md"}

    again = client.post("/api/documents", files=[files[1]]).json()["documents"][0]
    assert again["duplicate"] is True and again["id"] == space["id"]

    assert client.delete(f"/api/documents/{space['id']}").status_code == 200
    hits = client.post("/api/search", json={"query": "rockets orbit"}).json()["results"]
    assert all(h["filename"] != "space.txt" for h in hits)
    assert client.get("/api/health").json()["indexed"] == 1


def test_failed_document_reports_error(client):
    d = client.post("/api/documents", files=[("files", ("broken.pdf", b"%PDF-garbage", "application/pdf"))])
    doc = wait_indexed(client, d.json()["documents"][0]["id"])
    assert doc["status"] == "failed" and doc["error"]


# -- ask (RAG answer) --------------------------------------------------------
class FakeLLM:
    model = "fake"

    def __init__(self, ready=True):
        self.ready = ready
        self.status = {"state": "ready" if ready else "pulling", "model": "fake", "progress": 0.5}
        self.calls = []

    def stream_answer(self, question, hits):
        self.calls.append((question, hits))
        yield "Rockets use liquid fuel "
        yield "[1]."


def ingest(client, name, body):
    d = client.post("/api/documents", files=[("files", (name, body, "text/plain"))]).json()
    return wait_indexed(client, d["documents"][0]["id"])


def ask_events(client, **body):
    r = client.post("/api/ask", json=body)
    assert r.status_code == 200, r.text
    return [json.loads(line) for line in r.text.splitlines() if line]


def test_ask_streams_sources_then_tokens(client):
    import app.main
    app.main.state["llm"] = fake = FakeLLM()
    ingest(client, "space.txt", b"Rockets reach orbit using liquid fuel and oxygen.")
    events = ask_events(client, question="how do rockets reach orbit")
    assert [e["type"] for e in events] == ["sources", "token", "token", "done"]
    assert events[0]["sources"][0]["n"] == 1 and events[0]["sources"][0]["filename"] == "space.txt"
    assert len(fake.calls) == 1

    full = client.post("/api/ask", json={"question": "rockets orbit fuel", "stream": False}).json()
    assert full["answer"] == "Rockets use liquid fuel [1]." and full["model"] == "fake"


def test_ask_without_relevant_chunks_skips_llm(client):
    import app.main
    app.main.state["llm"] = fake = FakeLLM(ready=False)
    events = ask_events(client, question="anything at all")
    assert events[0]["sources"] == []
    assert events[1]["text"] == app.main.NOT_FOUND
    assert fake.calls == []


def test_ask_503_while_model_downloads(client):
    import app.main
    app.main.state["llm"] = FakeLLM(ready=False)
    ingest(client, "space.txt", b"Rockets reach orbit using liquid fuel and oxygen.")
    r = client.post("/api/ask", json={"question": "rockets orbit liquid fuel"})
    assert r.status_code == 503 and r.json()["detail"]["llm"]["state"] == "pulling"


def test_prompt_numbers_sources():
    from app.llm import build_prompt
    from app.store import Hit
    hits = [Hit(1, 1, "a.pdf", 0, "alpha", 2, "Intro", 0.9), Hit(2, 2, "b.md", 0, "beta", None, None, 0.5)]
    p = build_prompt("q?", hits)
    assert '[1] (a.pdf, page 2, section "Intro")\nalpha' in p and "[2] (b.md)\nbeta" in p
    assert p.endswith("Question: q?\nAnswer:") and "cite the source" in p
