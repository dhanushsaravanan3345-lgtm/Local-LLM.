import numpy as np

from ..config import settings
from ..database.database import db
from ..errors import AppError
from . import llm_service


def chunk_text(text: str) -> list[str]:
    size, overlap = settings.chunk_size, settings.chunk_overlap
    text = "\n".join(l.rstrip() for l in text.replace("\r", "").split("\n"))
    chunks, i, n = [], 0, len(text)
    while i < n:
        end = min(i + size, n)
        if end < n:  # prefer breaking on paragraph / line / sentence / space
            for sep in ("\n\n", "\n", ". ", " "):
                p = text.rfind(sep, i + size // 2, end)
                if p != -1:
                    end = p + len(sep)
                    break
        piece = text[i:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    if len(chunks) > settings.max_chunks_per_doc:
        raise AppError("document_too_large", "This document is too large to index. Try a smaller file.", 413)
    return chunks


def _is_nomic() -> bool:
    return "nomic" in settings.embed_model.lower()


def _norm(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(a)
    return a / n if n else a


async def index_document(name: str, stored: str, ext: str, size: int, chunks: list[str]) -> int:
    pre = "search_document: " if _is_nomic() else ""
    vecs = await llm_service.embed([pre + c for c in chunks])
    with db() as c:
        doc_id = c.execute("INSERT INTO documents(name,stored_name,ext,size,chunk_count) VALUES(?,?,?,?,?)",
                           (name, stored, ext, size, len(chunks))).lastrowid
        c.executemany("INSERT INTO chunks(document_id,idx,text,embedding) VALUES(?,?,?,?)",
                      [(doc_id, i, t, _norm(v).tobytes()) for i, (t, v) in enumerate(zip(chunks, vecs))])
    return doc_id


def has_documents() -> bool:
    with db() as c:
        return c.execute("SELECT 1 FROM chunks LIMIT 1").fetchone() is not None


async def retrieve(query: str) -> list[dict]:
    pre = "search_query: " if _is_nomic() else ""
    q = _norm((await llm_service.embed([pre + query]))[0])
    with db() as c:
        rows = c.execute("SELECT c.text,c.embedding,d.name FROM chunks c JOIN documents d ON d.id=c.document_id").fetchall()
    if not rows:
        return []
    m = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    if m.shape[1] != q.shape[0]:
        return []  # embedding model changed; old vectors are incompatible
    scores = m @ q
    out = []
    for i in np.argsort(-scores)[: settings.rag_top_k]:
        if scores[i] >= settings.rag_min_score:
            out.append({"name": rows[i]["name"], "text": rows[i]["text"], "score": round(float(scores[i]), 3)})
    return out
