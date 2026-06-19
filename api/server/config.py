"""Application configuration with fail-fast validation.

Loaded once at import time. Missing required secrets raise immediately so a
misconfigured deployment fails on cold start rather than mid-request.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Required secrets ---
    groq_api_key: str
    upstash_vector_rest_url: str
    upstash_vector_rest_token: str

    # --- Optional: distributed rate limiting (Upstash Redis) ---
    upstash_redis_rest_url: str | None = None
    upstash_redis_rest_token: str | None = None

    # --- Tunables ---
    groq_model: str = "llama3-70b-8192"
    max_upload_bytes: int = 4_000_000  # ~4MB — Vercel function request-body limit
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    max_question_len: int = 2000
    max_chunks_per_doc: int = 600  # guardrail so ingestion stays within maxDuration

    # Requests allowed per window, per client IP (when Redis is configured)
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60

    @property
    def rate_limiting_enabled(self) -> bool:
        return bool(self.upstash_redis_rest_url and self.upstash_redis_rest_token)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
