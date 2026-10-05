import io
import json

import pytest

from app.config import settings


def events(resp):
    return [json.loads(l[6:]) for l in resp.text.split("\n\n") if l.startswith("data: ")]


def test_health_and_frontend(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["runtime_up"] and r.json()["model_installed"]
    assert "My Local AI" in client.get("/").text


def test_models_info(client):
    d = client.get("/api/models").json()
    assert d["status"] == "ready" and d["runtime"] == "Ollama" and d["local"] is True
    assert "ram_percent" in d["system"]


def test_model_switch(client):
    assert client.post("/api/models/select", json={"model": "other-model:1b"}).status_code == 200
    assert client.get("/api/models").json()["current"] == "other-model:1b"
    assert client.post("/api/models/select", json={"model": "nope:1b"}).json()["error"]["code"] == "model_missing"
    client.post("/api/models/select", json={"model": "fake-model:1b"})


def test_chat_and_memory(client):
    r = client.post("/api/chat", json={"message": "alpha question"}).json()
    cid = r["conversation_id"]
    assert "alpha question" in r["answer"]
    r2 = client.post("/api/chat", json={"message": "follow up", "conversation_id": cid}).json()
    assert "[2]" in r2["answer"] and "alpha question" in r2["answer"]  # earlier turn reached the model
    msgs = client.get(f"/api/conversations/{cid}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert any(c["id"] == cid for c in client.get("/api/conversations").json()["conversations"])


def test_history_limit(client, monkeypatch):
    monkeypatch.setattr(settings, "max_history_messages", 2)
    cid = client.post("/api/chat", json={"message": "m1"}).json()["conversation_id"]
    for i in (2, 3):
        client.post("/api/chat", json={"message": f"m{i}", "conversation_id": cid})
    a = client.post("/api/chat", json={"message": "m4", "conversation_id": cid}).json()["answer"]
    assert "m1" not in a and "m4" in a


def test_streaming(client):
    r = client.post("/api/chat/stream", json={"message": "stream me please"})
    ev = events(r)
    assert ev[0]["type"] == "meta" and ev[-1]["type"] == "done"
    toks = [e["text"] for e in ev if e["type"] == "token"]
    assert len(toks) > 3 and "stream me please" in "".join(toks)
    assert ev[-1]["stats"]["tokens_per_s"] and ev[-1]["stats"]["first_token_s"] is not None
    cid = ev[0]["conversation_id"]
    assert client.get(f"/api/conversations/{cid}").json()["messages"][-1]["role"] == "assistant"


def test_regenerate(client):
    cid = client.post("/api/chat", json={"message": "regen me"}).json()["conversation_id"]
    ev = events(client.post("/api/chat/stream", json={"conversation_id": cid, "regenerate": True}))
    assert ev[-1]["type"] == "done"
    roles = [m["role"] for m in client.get(f"/api/conversations/{cid}").json()["messages"]]
    assert roles == ["user", "assistant"]


def test_empty_and_large_prompt(client):
    for body in ({"message": ""}, {"message": "   \n "}):
        r = client.post("/api/chat", json=body)
        assert r.status_code == 400 and r.json()["error"]["code"] == "empty_prompt"
    r = client.post("/api/chat/stream", json={"message": "x" * 501})
    assert r.status_code == 413 and r.json()["error"]["code"] == "prompt_too_large"
    assert client.post("/api/chat", json={"message": "x" * 2_000_000}).status_code == 413  # body-size limit


def test_oom_error_is_friendly(client):
    r = client.post("/api/chat", json={"message": "OOM please"})
    assert r.status_code == 507 and r.json()["error"]["code"] == "out_of_memory"
    assert "Traceback" not in r.text
    ev = events(client.post("/api/chat/stream", json={"message": "OOM again"}))
    assert ev[-1]["type"] == "error" and ev[-1]["code"] == "out_of_memory"


def test_model_not_installed(client, monkeypatch):
    monkeypatch.setattr(settings, "model", "ghost:7b")
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 404 and "ollama pull ghost:7b" in r.json()["error"]["message"]
    assert client.get("/api/models").json()["status"] == "model_missing"


def test_runtime_unavailable(client, monkeypatch):
    monkeypatch.setattr(settings, "ollama_url", "http://127.0.0.1:1")
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "runtime_unavailable"
    h = client.get("/api/health").json()
    assert h["runtime_up"] is False and h["status"] == "ok"
    ev = events(client.post("/api/chat/stream", json={"message": "hi"}))
    assert ev[-1]["code"] == "runtime_unavailable"


def test_timeout(client, monkeypatch):
    monkeypatch.setattr(settings, "request_timeout", 0.3)
    r = client.post("/api/chat", json={"message": "SLOW please"})
    assert r.status_code == 504 and r.json()["error"]["code"] == "timeout" and "Traceback" not in r.text


def test_system_prompt_setting(client):
    assert client.put("/api/settings", json={"system_prompt": "Be a pirate."}).status_code == 200
    assert client.get("/api/settings").json()["system_prompt"] == "Be a pirate."
    assert client.put("/api/settings", json={"system_prompt": ""}).status_code == 422


def test_delete_history(client):
    client.post("/api/chat", json={"message": "to be deleted"})
    assert client.delete("/api/chat/history").status_code == 200
    assert client.get("/api/conversations").json()["conversations"] == []
    assert client.get("/api/conversations/99999").status_code == 404


def _pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<</Type/Catalog/Pages 2 0 R>>", b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
            b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
            b"<</Length %d>>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    x = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offs:
        out += b"%010d 00000 n \n" % o
    return out + b"trailer<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, x)


def _docx(text: str) -> bytes:
    import docx
    d = docx.Document()
    d.add_paragraph(text)
    b = io.BytesIO()
    d.save(b)
    return b.getvalue()


def upload(client, name, data):
    return client.post("/api/documents/upload", files={"file": (name, data)})


def test_document_upload_all_types_and_delete(client):
    for name, data in [("a.txt", b"Plain text about gardening and tomatoes."), ("b.md", b"# Title\nMarkdown about bicycles"),
                       ("c.docx", _docx("Word file about submarines")), ("d.pdf", _pdf("PDF file about volcanoes"))]:
        r = upload(client, name, data)
        assert r.status_code == 200, (name, r.text)
    docs = client.get("/api/documents").json()["documents"]
    assert {d["name"] for d in docs} >= {"a.txt", "b.md", "c.docx", "d.pdf"}
    for d in docs:
        assert client.delete(f"/api/documents/{d['id']}").status_code == 200
    assert client.get("/api/documents").json()["documents"] == []
    assert client.delete("/api/documents/9999").status_code == 404
    assert list(settings.docs_dir.glob("*")) == []  # files removed from disk too


def test_document_validation(client):
    assert upload(client, "evil.exe", b"MZ").status_code == 415
    assert upload(client, "empty.txt", b"").status_code == 400
    r = upload(client, "broken.pdf", b"this is not a pdf")
    assert r.status_code == 422 and "Traceback" not in r.text
    r = upload(client, "blank.txt", b"   \n  ")
    assert r.json()["error"]["code"] == "empty_document"
    r = upload(client, "../../../etc/passwd.txt", b"harmless words here")
    assert r.status_code == 200 and r.json()["name"] == "passwd.txt"
    assert not any("etc" in p.name for p in settings.docs_dir.iterdir())
    client.delete(f"/api/documents/{r.json()['id']}")


def test_rag_retrieval_and_sources(client):
    upload(client, "zorbia.txt", b"The capital of Zorbia is Quillon. Zorbia exports blue copper and silk.")
    upload(client, "cooking.md", b"# Pasta\nBoil water, add salt, cook spaghetti for nine minutes, drain.")
    ev = events(client.post("/api/chat/stream", json={"message": "What is the capital of Zorbia?"}))
    assert ev[0]["sources"] == ["zorbia.txt"]
    assert "CTX" in "".join(e.get("text", "") for e in ev)  # context actually reached the model
    r = client.post("/api/chat", json={"message": "What is the capital of Zorbia?", "use_documents": False}).json()
    assert r["sources"] == [] and "CTX" not in r["answer"]
    r = client.post("/api/chat", json={"message": "qwerty zxcvb unrelated"}).json()
    assert r["sources"] == []  # below relevance threshold -> no sources shown


def test_upload_when_embed_model_missing(client, monkeypatch):
    monkeypatch.setattr(settings, "embed_model", "no-such-embed")
    before = len(list(settings.docs_dir.iterdir()))
    r = upload(client, "x.txt", b"some text")
    assert r.status_code == 404 and "ollama pull no-such-embed" in r.json()["error"]["message"]
    assert len(list(settings.docs_dir.iterdir())) == before  # failed import leaves no orphan file
    # chat still works with documents present but embeddings unavailable
    assert client.post("/api/chat", json={"message": "still works"}).status_code == 200


def test_database_error_is_friendly(client, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", settings.data_dir / "does" / "not" / "exist")
    r = client.get("/api/conversations")
    assert r.status_code == 500 and r.json()["error"]["code"] == "database_error"
