"""Distributed rate limiting via Upstash Redis.

In-memory limiting cannot work across stateless serverless instances, so this
uses Upstash Redis when configured. When the Redis env vars are absent the
dependency is a no-op, so local dev and minimal deployments work unchanged.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request, status

from .config import get_settings

logger = logging.getLogger("twyp.ratelimit")

_ratelimit = None


def _get_ratelimit():
    global _ratelimit
    if _ratelimit is not None:
        return _ratelimit

    settings = get_settings()
    if not settings.rate_limiting_enabled:
        return None

    from upstash_ratelimit import FixedWindow, Ratelimit
    from upstash_redis import Redis

    _ratelimit = Ratelimit(
        redis=Redis(
            url=settings.upstash_redis_rest_url,
            token=settings.upstash_redis_rest_token,
        ),
        limiter=FixedWindow(
            max_requests=settings.rate_limit_requests,
            window=settings.rate_limit_window_seconds,
        ),
        prefix="twyp-ratelimit",
    )
    return _ratelimit


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "anonymous"


async def rate_limiter(request: Request) -> None:
    """FastAPI dependency. Raises 429 when the caller exceeds the limit."""
    limiter = _get_ratelimit()
    if limiter is None:
        return

    try:
        result = limiter.limit(_client_ip(request))
    except Exception:  # noqa: BLE001 — never let limiter outages break the API
        logger.exception("Rate limiter check failed; allowing request")
        return

    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please slow down and try again shortly.",
        )
