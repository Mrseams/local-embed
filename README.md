# Local Embed

A small, fully local embedding engine for RAG. Upload TXT / Markdown / PDF / DOCX,
and it runs them through a document pipeline, stores the vectors, and serves semantic search.
The model (`all-MiniLM-L6-v2`, 384-d, ONNX via fastembed, no torch) is baked into the
Docker image, so the container works offline.

```bash
docker compose up -d --build
```

Open http://localhost:8077, drop files in, then **Ask** for a cited answer or **Search** for raw matches.
On first start the app pulls the answer model (`llama3.2:3b`, about 2 GB) into Ollama. The header shows
download progress, and Ask returns 503 until the model is ready.

## Pipeline

```
upload ─► dedupe (sha256) ─► extract ─► clean ─► chunk ─► embed ─► index ─► search
                                                                          └─► ask: top-k ─► Ollama ─► cited answer
```

| Stage    | Module            | What it does |
|----------|-------------------|--------------|
| extract  | `app/loaders.py`  | TXT (encoding sniffing), Markdown (heading path, skips headings inside code fences), PDF (per page), DOCX (heading styles + tables) |
| clean    | `app/cleaning.py` | NFKC, control chars, de-hyphenation across line breaks, whitespace; removes running PDF headers/footers and page numbers |
| chunk    | `app/chunking.py` | Recursive split (paragraph → line → sentence → word), packed to 200 tokens with a 40-token overlap; never crosses page/section boundaries |
| embed    | `app/embedder.py` | Batched, L2-normalized; heading is prefixed to the chunk for context |
| index    | `app/store.py`    | SQLite (docs, chunks, float32 BLOBs) + an in-memory matrix for exact cosine top-k |
| ask      | `app/llm.py`      | Chunks scoring ≥ `ASK_MIN_SCORE` become numbered sources; the local LLM answers with `[n]` citations. If no chunk qualifies, it replies "not found" without calling the LLM |

Each document's status moves through `queued → extracting → cleaning → chunking → embedding → indexed`
(or `failed` with the error), and per-stage timings are recorded. Re-uploading identical content
is skipped; re-uploading a failed document retries it.

## API

| Method | Path | |
|--------|------|-|
| `POST` | `/api/documents` | multipart `files` (one or many) |
| `GET`  | `/api/documents` | list with status |
| `GET`  | `/api/documents/{id}` | detail + chunks |
| `DELETE` | `/api/documents/{id}` | removes doc, chunks and file |
| `POST` | `/api/search` | `{"query": "...", "k": 5, "doc_ids": [1,2], "min_score": 0.2}` |
| `POST` | `/api/ask` | `{"question": "...", "k": 5, "doc_ids": [..], "stream": true}` |
| `GET`  | `/api/health` | model, dim, counts, LLM status |

`/api/ask` streams NDJSON by default:
`{"type":"sources",...}` → `{"type":"token","text":...}`… → `{"type":"done","took_ms":...}` (or `{"type":"error"}`).
With `"stream": false` it returns `{answer, sources, retrieval_ms, took_ms, model}` instead.

Interactive docs: http://localhost:8077/docs

## Config (env)

`CHUNK_TOKENS` (200), `CHUNK_OVERLAP` (40), `MIN_CHUNK_TOKENS` (8), `EMBED_BATCH` (32),
`MAX_UPLOAD_MB` (25), `EMBEDDER` (`fastembed` or `hash` for tests),
`OLLAMA_MODEL` (`llama3.2:3b`; set it in `.env` or the shell, e.g. `qwen2.5:7b` for better answers),
`OLLAMA_NUM_CTX` (4096), `ASK_TOP_K` (5), `ASK_MIN_SCORE` (0.2).
Use another model with `docker compose build --build-arg EMBED_MODEL=BAAI/bge-small-en-v1.5`.
After switching models, clear the volume, because vectors from different models can't be compared.

## Tests

```bash
docker compose run --rm -e EMBEDDER=hash embed pytest -q
```

## Limits

- No OCR. Scanned PDFs fail with a clear error.
- Token counts are a regex approximation of WordPiece, and the 200-token budget keeps chunks under MiniLM's 256 limit.
- Brute-force search is fine up to roughly 100k chunks. Past that, swap `Store.search` for FAISS or HNSW.
- Ollama runs CPU-only in Docker here. An answer takes about 20–60 s, and the first one after idle is slower while the model loads.
  For GPU, add an NVIDIA `deploy.resources` block to the `ollama` service.
- The 3B model sometimes adds unnecessary "couldn't find X" caveats. A 7B+ model follows the citation rules more strictly.
