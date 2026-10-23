"""Orchestrates the full flow: extract -> clean -> chunk -> embed -> index.

Every stage writes its status to the documents table so the UI can follow
along, and its duration to `timings`.
"""
import json
import logging
import time
from pathlib import Path

from . import cleaning, loaders
from .chunking import chunk_sections
from .config import Settings
from .embedder import Embedder
from .store import Store

log = logging.getLogger("pipeline")


class Pipeline:
    def __init__(self, settings: Settings, embedder: Embedder, store: Store):
        self.settings = settings
        self.embedder = embedder
        self.store = store

    def process(self, doc_id: int, path: Path) -> None:
        timings: dict[str, float] = {}

        def stage(name: str):
            self.store.update_document(doc_id, status=name)
            return time.perf_counter()

        try:
            t = stage("extracting")
            sections = loaders.load(path)
            timings["extract"] = time.perf_counter() - t

            t = stage("cleaning")
            sections = cleaning.clean(sections)
            if not sections:
                raise ValueError("Document has no text after cleaning")
            timings["clean"] = time.perf_counter() - t

            t = stage("chunking")
            s = self.settings
            chunks = chunk_sections(sections, s.chunk_tokens, s.chunk_overlap, s.min_chunk_tokens)
            timings["chunk"] = time.perf_counter() - t

            t = stage("embedding")
            vectors = self.embedder.embed([c.embed_text for c in chunks])
            timings["embed"] = time.perf_counter() - t

            t = time.perf_counter()
            self.store.replace_chunks(doc_id, chunks, vectors)
            timings["index"] = time.perf_counter() - t

            self.store.update_document(
                doc_id,
                status="indexed",
                error=None,
                n_sections=len(sections),
                n_chunks=len(chunks),
                n_chars=sum(len(sec.text) for sec in sections),
                model=self.embedder.name,
                timings=json.dumps({k: round(v, 3) for k, v in timings.items()}),
            )
            self.store.invalidate()
            log.info("doc %s indexed: %d chunks in %.2fs", doc_id, len(chunks), sum(timings.values()))
        except Exception as e:  # noqa: BLE001 - any failure is reported on the document
            log.exception("doc %s failed", doc_id)
            self.store.update_document(doc_id, status="failed", error=str(e)[:500],
                                       timings=json.dumps(timings))
