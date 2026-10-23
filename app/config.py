"""Runtime settings, read once from environment variables."""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    model_name: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model_cache: str | None = os.getenv("MODEL_CACHE") or None
    # "fastembed" (real model) or "hash" (dependency-free, for tests)
    embedder: str = os.getenv("EMBEDDER", "fastembed")
    # MiniLM truncates at 256 word pieces; stay safely under it.
    chunk_tokens: int = int(os.getenv("CHUNK_TOKENS", "200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "40"))
    min_chunk_tokens: int = int(os.getenv("MIN_CHUNK_TOKENS", "8"))
    embed_batch: int = int(os.getenv("EMBED_BATCH", "32"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    # Chunks below this cosine score are not shown to the LLM.
    ask_min_score: float = float(os.getenv("ASK_MIN_SCORE", "0.2"))
    ask_top_k: int = int(os.getenv("ASK_TOP_K", "5"))

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "index.db"


settings = Settings()

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx"}
