import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SYSTEM_PROMPT = (
    "You are my local AI assistant. Give accurate, useful and concise answers. "
    "If you are uncertain, say so instead of inventing information. "
    "Use provided document context when available."
    "You are a helpful general-purpose AI assistant running locally. "
    "Answer the user's questions directly and naturally. "
    "Help with general knowledge, explanations, coding, mathematics, writing, "
    "reasoning, and everyday questions. "
    "Do not refuse harmless questions. "
    "Do not claim that you lack access to information unless that limitation "
    "is actually relevant. "
    "If you do not know an answer, say that you are unsure rather than making "
    "up facts. "
    "Keep answers clear and useful. "
    "Use local document context when it is provided and relevant."

)


def _load_env() -> None:
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_env()
_e = os.getenv


@dataclass
class Settings:
    ollama_url: str = _e("OLLAMA_URL", "http://127.0.0.1:11434")
    model: str = _e("OLLAMA_MODEL", "qwen2.5:3b")
    embed_model: str = _e("EMBED_MODEL", "nomic-embed-text")
    data_dir: Path = Path(_e("DATA_DIR", str(ROOT / "data")))
    num_ctx: int = int(_e("NUM_CTX", 4096))
    temperature: float = float(_e("TEMPERATURE", 0.7))
    keep_alive: str = _e("KEEP_ALIVE", "10m")
    max_history_messages: int = int(_e("MAX_HISTORY_MESSAGES", 12))
    max_history_chars: int = int(_e("MAX_HISTORY_CHARS", 12000))
    max_prompt_chars: int = int(_e("MAX_PROMPT_CHARS", 8000))
    max_upload_bytes: int = int(_e("MAX_UPLOAD_MB", 15)) * 1024 * 1024
    request_timeout: float = float(_e("REQUEST_TIMEOUT", 120))
    rag_top_k: int = int(_e("RAG_TOP_K", 4))
    rag_min_score: float = float(_e("RAG_MIN_SCORE", 0.30))
    chunk_size: int = 800
    chunk_overlap: int = 100
    max_chunks_per_doc: int = 2000

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "documents"


settings = Settings()
