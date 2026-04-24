from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException
from api.schemas import UploadResponse
from api.document_processor import DocumentProcessor
from utils.logger import log_step

router    = APIRouter()
processor = DocumentProcessor()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".md"}
    ext     = "." + file.filename.split(".")[-1].lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext} not supported. Use PDF, TXT, or MD."
        )

    try:
        file_bytes   = await file.read()
        log_step("UPLOAD", f"Processing {file.filename} ({len(file_bytes)} bytes)")

        chunks_added, total = processor.process_file(file_bytes, file.filename)

        return UploadResponse(
            filename=file.filename,
            chunks_added=chunks_added,
            total_docs=total,
            message=f"Successfully indexed {chunks_added} chunks from {file.filename}",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents():
    total = processor.get_total_docs()
    return {"total_chunks": total}