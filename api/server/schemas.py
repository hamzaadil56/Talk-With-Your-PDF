"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


class ChatRequest(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=128)
    question: str = Field(..., min_length=1)


class Source(BaseModel):
    text: str
    score: float
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    model: str
    rate_limiting: bool


class ErrorResponse(BaseModel):
    detail: str
