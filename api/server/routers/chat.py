"""Chat / question-answering endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from ..config import get_settings
from ..rag import answer_question
from ..rate_limit import rate_limiter
from ..schemas import ChatRequest, ChatResponse

logger = logging.getLogger("twyp.chat")

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(rate_limiter)],
)
async def chat(payload: ChatRequest) -> ChatResponse:
    settings = get_settings()

    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must not be empty.",
        )
    if len(question) > settings.max_question_len:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Question is too long (max {settings.max_question_len} characters).",
        )

    try:
        answer, sources = answer_question(payload.document_id, question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to answer question")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The assistant could not generate an answer. Please try again.",
        ) from exc

    return ChatResponse(answer=answer, sources=sources)
