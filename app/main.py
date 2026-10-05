import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import ROOT, settings
from .database.database import init_db
from .errors import AppError
from .routes import chat, documents, models
from .services import memory_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    saved = memory_service.get_setting("model")
    if saved:
        settings.model = saved
    yield


app = FastAPI(title="My Local AI", lifespan=lifespan, docs_url=None, redoc_url=None)


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    if request.method in ("POST", "PUT"):
        limit = settings.max_upload_bytes + 1_000_000 if request.url.path.endswith("/documents/upload") else 1_000_000
        try:
            if int(request.headers.get("content-length", 0)) > limit:
                return _err(413, "request_too_large", "That request is too large.")
        except ValueError:
            return _err(400, "bad_request", "Invalid request.")
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api") else resp.headers.get("Cache-Control", "no-cache")
    return resp


@app.exception_handler(AppError)
async def app_error(_: Request, e: AppError):
    return _err(e.status, e.code, e.message)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, e: RequestValidationError):
    return _err(422, "invalid_request", "Some of the information sent was invalid.")


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, e: StarletteHTTPException):
    return _err(e.status_code, "http_error", "Not found." if e.status_code == 404 else "Request failed.")


@app.exception_handler(Exception)
async def unhandled(_: Request, e: Exception):
    log.exception("unhandled error")  # details go to the server log only
    return _err(500, "internal", "Something went wrong. Please try again.")


app.include_router(models.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
