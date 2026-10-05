"""Thin async client for the local Ollama runtime."""
import json
import subprocess

import httpx

from ..config import settings
from ..errors import AppError


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=settings.ollama_url,
        timeout=httpx.Timeout(settings.request_timeout, connect=3.0),
    )


def _err_text(body) -> str:
    if isinstance(body, bytes):
        body = body.decode("utf-8", "replace")
    try:
        return str(json.loads(body).get("error", body))
    except Exception:
        return body


def _map_error(status: int, text: str, model: str) -> AppError:
    t = (text or "").lower()
    if status == 404 or "not found" in t:
        return AppError("model_missing",
                        f"Model '{model}' is not installed. Run:  ollama pull {model}", 404)
    if "memory" in t:
        return AppError("out_of_memory",
                        "Not enough memory to run this model. Close other apps or pick a smaller model in Settings.", 507)
    return AppError("llm_error", "The local model returned an error. Please try again.", 502)


def _down() -> AppError:
    return AppError("runtime_unavailable",
                    "The local model runtime (Ollama) is not running. Start it with:  ollama serve", 503)


def _timeout() -> AppError:
    return AppError("timeout", "The model took too long to respond. Try a shorter prompt or a smaller model.", 504)


async def _get(path: str):
    try:
        async with _client() as c:
            return await c.get(path)
    except httpx.TimeoutException:
        raise _timeout()
    except httpx.HTTPError:
        raise _down()


async def list_models() -> list[dict]:
    r = await _get("/api/tags")
    if r.status_code != 200:
        raise _down()
    return [{"name": m["name"], "size": m.get("size", 0)} for m in r.json().get("models", [])]


async def runtime_info() -> dict:
    """Never raises: used by health/status endpoints."""
    info = {"runtime": "Ollama", "runtime_up": False, "version": None, "installed": [], "loaded": []}
    try:
        r = await _get("/api/version")
        info["runtime_up"], info["version"] = True, r.json().get("version")
        info["installed"] = await list_models()
        ps = await _get("/api/ps")
        info["loaded"] = [{"name": m["name"], "size": m.get("size", 0), "size_vram": m.get("size_vram", 0)}
                          for m in ps.json().get("models", [])]
    except Exception:
        pass
    return info


async def stream_chat(messages: list[dict], model: str):
    payload = {
        "model": model, "messages": messages, "stream": True, "keep_alive": settings.keep_alive,
        "options": {"num_ctx": settings.num_ctx, "temperature": settings.temperature},
    }
    try:
        async with _client() as c:
            async with c.stream("POST", "/api/chat", json=payload) as r:
                if r.status_code != 200:
                    raise _map_error(r.status_code, _err_text(await r.aread()), model)
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "error" in obj:
                        raise _map_error(500, str(obj["error"]), model)
                    tok = obj.get("message", {}).get("content", "")
                    if tok:
                        yield {"token": tok}
                    if obj.get("done"):
                        ev = obj.get("eval_count") or 0
                        ed = obj.get("eval_duration") or 0
                        yield {"done": True, "stats": {
                            "tokens": ev,
                            "tokens_per_s": round(ev / (ed / 1e9), 1) if ed else None,
                            "prompt_tokens": obj.get("prompt_eval_count"),
                            "model_load_s": round((obj.get("load_duration") or 0) / 1e9, 2),
                        }}
    except httpx.TimeoutException:
        raise _timeout()
    except httpx.HTTPError:
        raise _down()


async def chat_once(messages: list[dict], model: str):
    parts, stats = [], {}
    async for ev in stream_chat(messages, model):
        if "token" in ev:
            parts.append(ev["token"])
        elif ev.get("done"):
            stats = ev["stats"]
    return "".join(parts), stats


async def embed(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    model = settings.embed_model
    try:
        async with _client() as c:
            for i in range(0, len(texts), 16):
                r = await c.post("/api/embed", json={"model": model, "input": texts[i:i + 16]})
                if r.status_code != 200:
                    e = _map_error(r.status_code, _err_text(r.content), model)
                    if e.code == "model_missing":
                        e.message = f"Embedding model '{model}' is not installed. Run:  ollama pull {model}"
                    raise e
                out.extend(r.json()["embeddings"])
    except httpx.TimeoutException:
        raise _timeout()
    except httpx.HTTPError:
        raise _down()
    return out


def gpu_stats():
    try:
        o = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
        if o.returncode != 0 or not o.stdout.strip():
            return None
        n, u, mu, mt = [x.strip() for x in o.stdout.splitlines()[0].split(",")]
        return {"name": n, "util_percent": float(u), "vram_used_mb": float(mu), "vram_total_mb": float(mt)}
    except Exception:
        return None
