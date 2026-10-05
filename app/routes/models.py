import psutil
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..config import settings
from ..errors import AppError
from ..services import llm_service as llm
from ..services import memory_service as mem

router = APIRouter(prefix="/api")
psutil.cpu_percent(None)  # prime the counter


@router.get("/health")
async def health():
    info = await llm.runtime_info()
    names = [m["name"] for m in info["installed"]]
    return {"status": "ok", "runtime_up": info["runtime_up"],
            "model": settings.model, "model_installed": settings.model in names}


@router.get("/models")
async def models():
    info = await llm.runtime_info()
    names = [m["name"] for m in info["installed"]]
    installed = settings.model in names
    status = "runtime_down" if not info["runtime_up"] else "ready" if installed else "model_missing"
    vm = psutil.virtual_memory()
    loaded = next((m for m in info["loaded"] if m["name"] == settings.model), None)
    return {
        "current": settings.model, "embed_model": settings.embed_model,
        "runtime": info["runtime"], "runtime_version": info["version"], "status": status,
        "model_loaded_in_memory": loaded is not None,
        "installed": info["installed"], "local": True,
        "system": {"cpu_percent": psutil.cpu_percent(None), "ram_percent": vm.percent,
                   "ram_used_gb": round(vm.used / 2**30, 1), "ram_total_gb": round(vm.total / 2**30, 1),
                   "model_vram_gb": round(loaded["size_vram"] / 2**30, 2) if loaded else None,
                   "gpu": llm.gpu_stats()},
    }


class ModelSelect(BaseModel):
    model: str = Field(min_length=1, max_length=100)


@router.post("/models/select")
async def select_model(body: ModelSelect):
    names = [m["name"] for m in await llm.list_models()]
    if body.model not in names:
        raise AppError("model_missing", f"Model '{body.model}' is not installed. Run:  ollama pull {body.model}", 404)
    settings.model = body.model
    mem.set_setting("model", body.model)
    return {"current": body.model}


class SettingsBody(BaseModel):
    system_prompt: str = Field(min_length=1, max_length=4000)


@router.get("/settings")
async def get_settings():
    return {"system_prompt": mem.system_prompt(), "max_history_messages": settings.max_history_messages,
            "max_prompt_chars": settings.max_prompt_chars, "num_ctx": settings.num_ctx}


@router.put("/settings")
async def put_settings(body: SettingsBody):
    mem.set_setting("system_prompt", body.system_prompt.strip())
    return {"system_prompt": body.system_prompt.strip()}
