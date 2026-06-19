"""Document ingestion endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from ..config import get_settings
from ..rag import IngestError, ingest_pdf
from ..rate_limit import rate_limiter
from ..schemas import IngestResponse

logger = logging.getLogger("twyp.documents")

router = APIRouter()


@router.post(
    "/documents",
    response_model=IngestResponse,
    dependencies=[Depends(rate_limiter)],
)
async def create_document(file: UploadFile = File(...)) -> IngestResponse:
    settings = get_settings()

    filename = file.filename or "document.pdf"
    content_type = (file.content_type or "").lower()
    if content_type and content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File is too large. Maximum size is "
                f"{settings.max_upload_bytes // 1_000_000}MB."
            ),
        )

    try:
        document_id, chunk_count = ingest_pdf(data, filename)
    except IngestError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return IngestResponse(
        document_id=document_id,
        filename=filename,
        chunk_count=chunk_count,
    )
