"""RAG pipeline: ingest PDFs into Upstash Vector and answer questions with Groq.

Serverless-native port of the original ``KnowledgeRetriever`` prototype:
- embeddings happen **server-side in Upstash** (no local torch/transformers),
- vectors persist in Upstash (no in-memory FAISS),
- each document lives in its own Upstash namespace (``document_id``).

Clients are created lazily and cached so warm invocations reuse them.
"""

from __future__ import annotations

import io
import logging
import uuid
from functools import lru_cache

from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from upstash_vector import Index

from .config import get_settings
from .schemas import Source

logger = logging.getLogger("twyp.rag")


class IngestError(Exception):
    """Raised when a PDF cannot be parsed or contains no extractable text."""


@lru_cache
def _vector_index() -> Index:
    settings = get_settings()
    return Index(
        url=settings.upstash_vector_rest_url,
        token=settings.upstash_vector_rest_token,
    )


@lru_cache
def _groq_client() -> Groq:
    return Groq(api_key=get_settings().groq_api_key)


def _extract_pages(data: bytes) -> list[tuple[int, str]]:
    """Return a list of (page_number, text) for pages with extractable text."""
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 — surface a clean error to the caller
        raise IngestError("The file could not be read as a PDF.") from exc

    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def ingest_pdf(data: bytes, filename: str) -> tuple[str, int]:
    """Parse, chunk, and upsert a PDF. Returns (document_id, chunk_count)."""
    settings = get_settings()
    pages = _extract_pages(data)
    if not pages:
        raise IngestError(
            "No extractable text found. Scanned/image-only PDFs are not supported."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    document_id = uuid.uuid4().hex
    vectors: list[dict] = []
    for page_number, text in pages:
        for chunk in splitter.split_text(text):
            vectors.append(
                {
                    "id": uuid.uuid4().hex,
                    "data": chunk,  # Upstash embeds this server-side
                    "metadata": {"page": page_number, "text": chunk},
                }
            )
            if len(vectors) >= settings.max_chunks_per_doc:
                break
        if len(vectors) >= settings.max_chunks_per_doc:
            logger.warning("Document %s truncated at %d chunks", document_id, len(vectors))
            break

    index = _vector_index()
    # Upsert in batches to stay within request limits.
    batch_size = 100
    for start in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[start : start + batch_size], namespace=document_id)

    logger.info("Ingested %s (%s): %d chunks", filename, document_id, len(vectors))
    return document_id, len(vectors)


def _retrieve(document_id: str, question: str) -> list[Source]:
    settings = get_settings()
    results = _vector_index().query(
        data=question,
        top_k=settings.top_k,
        include_metadata=True,
        namespace=document_id,
    )
    sources: list[Source] = []
    for r in results or []:
        metadata = r.metadata or {}
        sources.append(
            Source(
                text=metadata.get("text", ""),
                score=float(r.score),
                page=metadata.get("page"),
            )
        )
    return sources


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a PDF document. "
    "Use the provided context to answer the question. If the context does not "
    "contain the answer, you may use your own knowledge, but say so clearly. "
    "Be concise and accurate."
)


def answer_question(document_id: str, question: str) -> tuple[str, list[Source]]:
    """Retrieve relevant chunks and generate a grounded answer with Groq."""
    settings = get_settings()
    sources = _retrieve(document_id, question)
    context = "\n\n---\n\n".join(s.text for s in sources) or "(no relevant context found)"

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    completion = _groq_client().chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    answer = (completion.choices[0].message.content or "").strip()
    return answer, sources
