import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_tmp = tempfile.mkdtemp(prefix="localai_test_")
os.environ["DATA_DIR"] = _tmp
os.environ["OLLAMA_MODEL"] = "fake-model:1b"
os.environ["EMBED_MODEL"] = "nomic-embed-text"
os.environ["MAX_PROMPT_CHARS"] = "500"

import pytest
import uvicorn


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session", autouse=True)
def fake_ollama():
    import fake_ollama as fo
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(fo.app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    from app.config import settings
    settings.ollama_url = f"http://127.0.0.1:{port}"
    yield settings.ollama_url
    server.should_exit = True


@pytest.fixture(scope="session")
def client(fake_ollama):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c
