"""Stage 5 - storage: SQLite for documents/chunks, an in-memory matrix for search.

Vectors live as float32 BLOBs next to their chunk. Search is exact cosine
similarity (dot product of normalized vectors) over a matrix that is loaded
lazily and invalidated on every write. That is plenty for tens of thousands
of chunks; past that, swap this class for a FAISS/HNSW-backed one.
"""
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    sha256      TEXT NOT NULL UNIQUE,
    size_bytes  INTEGER NOT NULL,
    status      TEXT NOT NULL,          -- queued|extracting|cleaning|chunking|embedding|indexed|failed
    error       TEXT,
    n_sections  INTEGER DEFAULT 0,
    n_chunks    INTEGER DEFAULT 0,
    n_chars     INTEGER DEFAULT 0,
    model       TEXT,
    timings     TEXT,                   -- JSON: seconds per stage
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id    INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx       INTEGER NOT NULL,
    text      TEXT NOT NULL,
    tokens    INTEGER NOT NULL,
    page      INTEGER,
    heading   TEXT,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
"""


@dataclass
class Hit:
    chunk_id: int
    doc_id: int
    filename: str
    idx: int
    text: str
    page: int | None
    heading: str | None
    score: float


class Store:
    def __init__(self, path: Path, dim: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(SCHEMA)
        self._matrix: np.ndarray | None = None
        self._meta: list[tuple[int, int]] = []  # (chunk_id, doc_id) per matrix row
        # Anything left mid-pipeline by a crash/restart is not coming back.
        self._db.execute(
            "UPDATE documents SET status='failed', error='Interrupted by restart' "
            "WHERE status NOT IN ('indexed','failed')"
        )
        self._db.commit()

    # -- documents ---------------------------------------------------------
    def find_by_hash(self, sha256: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM documents WHERE sha256=?", (sha256,)).fetchone()
            return dict(row) if row else None

    def create_document(self, filename: str, sha256: str, size: int) -> int:
        now = time.time()
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO documents (filename, sha256, size_bytes, status, created_at, updated_at) "
                "VALUES (?, ?, ?, 'queued', ?, ?)",
                (filename, sha256, size, now, now),
            )
            self._db.commit()
            return cur.lastrowid

    def update_document(self, doc_id: int, **fields) -> None:
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE documents SET {cols} WHERE id=?", (*fields.values(), doc_id))
            self._db.commit()

    def get_document(self, doc_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            return dict(row) if row else None

    def list_documents(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM documents ORDER BY id DESC").fetchall()
            return [dict(r) for r in rows]

    def delete_document(self, doc_id: int) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            self._db.commit()
            self._matrix = None
            return cur.rowcount > 0

    # -- chunks ------------------------------------------------------------
    def replace_chunks(self, doc_id: int, chunks: list[Chunk], vectors: np.ndarray) -> None:
        assert vectors.shape == (len(chunks), self.dim)
        with self._lock:
            self._db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self._db.executemany(
                "INSERT INTO chunks (doc_id, idx, text, tokens, page, heading, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (doc_id, c.index, c.text, c.tokens, c.page, c.heading,
                     vectors[i].astype(np.float32).tobytes())
                    for i, c in enumerate(chunks)
                ],
            )
            self._db.commit()
            self._matrix = None

    def get_chunks(self, doc_id: int) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, idx, text, tokens, page, heading FROM chunks WHERE doc_id=? ORDER BY idx",
                (doc_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # -- search ------------------------------------------------------------
    def _load_matrix(self) -> None:
        rows = self._db.execute(
            "SELECT c.id, c.doc_id, c.embedding FROM chunks c "
            "JOIN documents d ON d.id = c.doc_id WHERE d.status='indexed' ORDER BY c.id"
        ).fetchall()
        self._meta = [(r["id"], r["doc_id"]) for r in rows]
        if rows:
            self._matrix = np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32)
            self._matrix = self._matrix.reshape(len(rows), self.dim)
        else:
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)

    def invalidate(self) -> None:
        with self._lock:
            self._matrix = None

    def search(self, query_vec: np.ndarray, k: int = 5, doc_ids: list[int] | None = None,
               min_score: float = 0.0) -> list[Hit]:
        with self._lock:
            if self._matrix is None:
                self._load_matrix()
            matrix, meta = self._matrix, self._meta
            if not len(meta):
                return []
            scores = matrix @ query_vec.astype(np.float32)
            if doc_ids:
                allowed = np.array([d in set(doc_ids) for _, d in meta])
                scores = np.where(allowed, scores, -np.inf)
            k = min(k, len(scores))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            picked = [(meta[i][0], float(scores[i])) for i in top if scores[i] >= min_score]
            if not picked:
                return []
            ids = [cid for cid, _ in picked]
            rows = self._db.execute(
                f"SELECT c.id, c.doc_id, c.idx, c.text, c.page, c.heading, d.filename "
                f"FROM chunks c JOIN documents d ON d.id=c.doc_id "
                f"WHERE c.id IN ({','.join('?' * len(ids))})",
                ids,
            ).fetchall()
            by_id = {r["id"]: r for r in rows}
            return [
                Hit(cid, by_id[cid]["doc_id"], by_id[cid]["filename"], by_id[cid]["idx"],
                    by_id[cid]["text"], by_id[cid]["page"], by_id[cid]["heading"], score)
                for cid, score in picked if cid in by_id
            ]

    def stats(self) -> dict:
        with self._lock:
            row = self._db.execute(
                "SELECT (SELECT COUNT(*) FROM documents) AS documents, "
                "(SELECT COUNT(*) FROM documents WHERE status='indexed') AS indexed, "
                "(SELECT COUNT(*) FROM chunks) AS chunks"
            ).fetchone()
            return dict(row)
