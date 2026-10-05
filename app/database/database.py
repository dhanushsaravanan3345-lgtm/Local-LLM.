import sqlite3
from contextlib import contextmanager

from ..config import settings
from ..errors import AppError

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations(
  id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL DEFAULT 'New chat',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL, content TEXT NOT NULL, sources TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, stored_name TEXT NOT NULL,
  ext TEXT NOT NULL, size INTEGER NOT NULL, chunk_count INTEGER NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  idx INTEGER NOT NULL, text TEXT NOT NULL, embedding BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunks(document_id);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@contextmanager
def db():
    try:
        conn = sqlite3.connect(settings.db_path, timeout=10)
    except sqlite3.Error as e:
        raise AppError("database_error", "The local database is unavailable.", 500) from e
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        conn.rollback()
        raise AppError("database_error", "A local database error occurred. Please try again.", 500) from e
    finally:
        conn.close()


def init_db() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    with db() as c:
        c.executescript(SCHEMA)
