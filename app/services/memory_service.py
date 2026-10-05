import json

from ..config import DEFAULT_SYSTEM_PROMPT, settings
from ..database.database import db
from ..errors import AppError


def get_setting(key, default=None):
    with db() as c:
        r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return r["value"] if r else default


def set_setting(key: str, value: str) -> None:
    with db() as c:
        c.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  (key, value))


def system_prompt() -> str:
    return get_setting("system_prompt", DEFAULT_SYSTEM_PROMPT)


def create_conversation(title: str = "New chat") -> int:
    with db() as c:
        return c.execute("INSERT INTO conversations(title) VALUES(?)", (title[:60],)).lastrowid


def conversation_exists(cid: int) -> bool:
    with db() as c:
        return c.execute("SELECT 1 FROM conversations WHERE id=?", (cid,)).fetchone() is not None


def list_conversations(limit: int = 50) -> list[dict]:
    with db() as c:
        rows = c.execute("SELECT id,title,updated_at FROM conversations ORDER BY updated_at DESC, id DESC LIMIT ?",
                         (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_messages(cid: int) -> list[dict]:
    with db() as c:
        rows = c.execute("SELECT id,role,content,sources FROM messages WHERE conversation_id=? ORDER BY id",
                         (cid,)).fetchall()
    return [{"id": r["id"], "role": r["role"], "content": r["content"],
             "sources": json.loads(r["sources"]) if r["sources"] else []} for r in rows]


def update_message(mid: int, content: str) -> None:
    with db() as c:
        c.execute("UPDATE messages SET content=? WHERE id=?", (content, mid))


def add_message(cid: int, role: str, content: str, sources=None) -> int:
    with db() as c:
        mid = c.execute("INSERT INTO messages(conversation_id,role,content,sources) VALUES(?,?,?,?)",
                        (cid, role, content, json.dumps(sources) if sources else None)).lastrowid
        c.execute("UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (cid,))
        if role == "user":
            c.execute("UPDATE conversations SET title=? WHERE id=? AND title='New chat'", (content[:50], cid))
    return mid


def pop_last_for_regen(cid: int) -> str:
    """Remove the trailing assistant message (if any) and return the last user text."""
    with db() as c:
        last = c.execute("SELECT id,role FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 1",
                         (cid,)).fetchone()
        if last and last["role"] == "assistant":
            c.execute("DELETE FROM messages WHERE id=?", (last["id"],))
        u = c.execute("SELECT content FROM messages WHERE conversation_id=? AND role='user' ORDER BY id DESC LIMIT 1",
                      (cid,)).fetchone()
    if not u:
        raise AppError("nothing_to_regenerate", "There is no previous message to regenerate.", 400)
    return u["content"]


def history_for_model(cid: int) -> list[dict]:
    """Last N messages, trimmed (oldest first) to the character budget. Newest is always kept."""
    with db() as c:
        rows = c.execute("SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                         (cid, settings.max_history_messages)).fetchall()
    kept, total = [], 0
    for r in rows:
        if kept and total + len(r["content"]) > settings.max_history_chars:
            break
        kept.append({"role": r["role"], "content": r["content"]})
        total += len(r["content"])
    return kept[::-1]


def delete_conversation(cid: int) -> None:
    with db() as c:
        c.execute("DELETE FROM conversations WHERE id=?", (cid,))


def clear_all() -> None:
    with db() as c:
        c.execute("DELETE FROM conversations")
