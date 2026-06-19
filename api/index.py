"""Vercel Python serverless entrypoint.

This is the only top-level module in ``api/`` so Vercel builds a single
function from it. The whole backend lives in the ``server`` subpackage, which
Vercel does not turn into separate functions. Vercel serves the exported ASGI
``app`` directly.
"""

import os
import sys

# Ensure the `server` package is importable whether this module is loaded as the
# top-level `index` (Vercel) or as `api.index` (local `uvicorn api.index:app`).
sys.path.insert(0, os.path.dirname(__file__))

from server.main import app  # noqa: E402,F401

__all__ = ["app"]
