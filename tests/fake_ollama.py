"""Minimal fake of the Ollama HTTP API used ONLY by tests (no real model is needed)."""
import asyncio
import json
import re
import zlib

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
INSTALLED = ["fake-model:1b", "other-model:1b", "nomic-embed-text:latest"]


def vec(text: str, dim: int = 4096):
    v = [0.0] * dim
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        v[zlib.crc32(w.encode()) % dim] += 1.0
    return v


@app.get("/api/version")
async def version():
    return {"version": "0.0-fake"}


@app.get("/api/tags")
async def tags():
    return {"models": [{"name": n, "size": 1000} for n in INSTALLED]}


@app.get("/api/ps")
async def ps():
    return {"models": [{"name": "fake-model:1b", "size": 2**30, "size_vram": 0}]}


@app.post("/api/embed")
async def embed(request: Request):
    b = await request.json()
    if b["model"] not in INSTALLED and f"{b['model']}:latest" not in INSTALLED:
        return JSONResponse({"error": f"model '{b['model']}' not found"}, status_code=404)
    return {"embeddings": [vec(t) for t in b["input"]]}


@app.post("/api/chat")
async def chat(request: Request):
    b = await request.json()
    if b["model"] not in INSTALLED:
        return JSONResponse({"error": f"model '{b['model']}' not found"}, status_code=404)
    msgs = b["messages"]
    users = [m["content"] for m in msgs if m["role"] == "user"]
    if users[-1].startswith("OOM"):
        return JSONResponse({"error": "model requires more system memory (9.0 GiB) than is available"}, status_code=500)
    if users[-1].startswith("SLOW"):
        await asyncio.sleep(1.5)
    ctx = "CTX " if "[Source:" in msgs[0]["content"] else ""
    reply = f"{ctx}[{len(users)}] " + " | ".join(users)
    words = reply.split(" ")

    async def gen():
        for i, w in enumerate(words):
            if users[-1].startswith("word"):
                await asyncio.sleep(0.15)
            yield json.dumps({"message": {"role": "assistant", "content": w + (" " if i < len(words) - 1 else "")}, "done": False}) + "\n"
        yield json.dumps({"message": {"role": "assistant", "content": ""}, "done": True, "eval_count": len(words),
                          "eval_duration": 10**9, "prompt_eval_count": 5, "load_duration": 5 * 10**8}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
