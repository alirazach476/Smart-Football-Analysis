"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request

from .worker import JobQueue


def get_queue(request: Request) -> JobQueue:
    """Return the app-wide background job queue."""
    return request.app.state.queue
