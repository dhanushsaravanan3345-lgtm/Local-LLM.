from fastapi import APIRouter, File, UploadFile

from ..config import settings
from ..errors import AppError
from ..services import document_service as docs
from ..services import rag_service as rag

router = APIRouter(prefix="/api/documents")


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    name, ext = docs.safe_name(file.filename)
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise AppError("file_too_large", f"File is too large (limit {settings.max_upload_bytes // 2**20} MB).", 413)
    if not data:
        raise AppError("empty_file", "The uploaded file is empty.", 400)
    path = docs.save_file(data, ext)
    try:
        text = docs.extract_text(path, ext)
        chunks = rag.chunk_text(text)
        doc_id = await rag.index_document(name, path.name, ext, len(data), chunks)
    except BaseException:
        path.unlink(missing_ok=True)  # don't leave orphan files after a failed import
        raise
    return {"id": doc_id, "name": name, "chunks": len(chunks)}


@router.get("")
async def list_documents():
    return {"documents": docs.list_documents()}


@router.delete("/{doc_id}")
async def delete_document(doc_id: int):
    if not docs.delete_document(doc_id):
        raise AppError("not_found", "Document not found.", 404)
    return {"deleted": True}
