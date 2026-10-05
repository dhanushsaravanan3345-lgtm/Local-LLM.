# 🤖 My Local AI

A private chat assistant that runs entirely on your laptop: **Ollama** (local model) → **FastAPI** → plain HTML/CSS/JS UI, with **SQLite** memory and local **RAG** over your PDF / TXT / DOCX / Markdown files. No cloud AI APIs, no API keys, no telemetry.

```
Browser UI → FastAPI → Ollama → local quantized model
Documents → text extraction → chunks → local embeddings (Ollama) → SQLite → top-k context → local LLM
```

## 1. Setup (one time)

```bash
# from the project folder
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

python scripts/setup_model.py --dry-run          # inspect hardware + show the recommended model
python scripts/setup_model.py --install-ollama   # install Ollama if missing, pull models, write .env, verify
```

`setup_model.py` inspects OS / CPU / RAM / GPU+VRAM / free disk / Python and picks the smallest practical 4-bit model:

| Your machine | Chat model |
|---|---|
| NVIDIA ≥ 6 GB VRAM (and ≥ 12 GB RAM) | `qwen2.5:7b` |
| NVIDIA 3–6 GB VRAM | `qwen2.5:3b` |
| Apple Silicon ≥ 16 GB | `qwen2.5:7b` |
| CPU only, ≥ 6 GB RAM / Apple Silicon 8 GB | `qwen2.5:3b` |
| < 6 GB RAM | `qwen2.5:1.5b` (+ `all-minilm` embeddings) |

Embeddings use `nomic-embed-text` (`all-minilm` on low-RAM machines). Force a model with `--model NAME`, or edit `.env`.
Internet is needed only for the one-time install/pull; after that everything works offline.

## 2. Run

```bash
ollama serve                                   # skip if the Ollama app/service is already running
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open **http://127.0.0.1:8000**. The server binds to localhost only.

## 3. Test and measure

```bash
python -m pytest -q tests          # 19 tests; use a fake Ollama, so no model needed
python scripts/smoke_live.py       # real server over HTTP incl. "Stop" mid-stream
python scripts/benchmark.py        # REAL numbers on your hardware -> benchmark_result.json
```
`benchmark.py` reports cold model-load time, first-token latency, tokens/s, Ollama RAM, and GPU/VRAM usage. The chat UI also shows tokens/s and first-token time under each answer.

## Features
Streaming chat · Stop · Regenerate · Copy · follow-up memory (SQLite, configurable limits) · document chat with source names · model switcher · editable system prompt · dark mode · mobile layout · friendly errors (runtime down, model missing, out of memory, timeout, empty/oversized prompt, bad documents, DB errors).

## API
`GET /api/health` · `GET /api/models` · `POST /api/models/select` · `POST /api/chat` · `POST /api/chat/stream` (SSE) · `GET|DELETE /api/conversations[/id]` · `DELETE /api/chat/history[?conversation_id=]` · `GET|PUT /api/settings` · `POST /api/documents/upload` · `GET /api/documents` · `DELETE /api/documents/{id}`

## Configuration (`.env`, see `.env.example`)
`OLLAMA_MODEL`, `EMBED_MODEL`, `NUM_CTX`, `MAX_HISTORY_MESSAGES`, `MAX_HISTORY_CHARS`, `MAX_PROMPT_CHARS`, `MAX_UPLOAD_MB`, `REQUEST_TIMEOUT`, `RAG_TOP_K`, `RAG_MIN_SCORE`. The system prompt is edited in the UI (Settings) and stored in SQLite.

## Privacy and security
Prompts, chats, documents, embeddings and inference all stay on this machine. Uploads: allow-listed extensions, size-limited, stored under server-generated names (no path tricks), never executed. Request bodies are size-limited; the UI renders model output as plain text (no HTML injection); internal errors are logged server-side only. No secrets exist; `.env` is git-ignored.

## Known limitations
Scanned/image-only PDFs need OCR (not included). Retrieval loads all chunk vectors per question (fine for hundreds of documents, not millions). No authentication (localhost only, single user). Changing `EMBED_MODEL` requires re-uploading documents.

## Extending
LoRA/fine-tuning: train outside the app, `ollama create` the result, pick it in the model dropdown. Cloud fallback / voice / multilingual: add a new function in `services/llm_service.py`; routes don't need to change.
