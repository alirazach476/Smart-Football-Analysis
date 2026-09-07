"""
FastAPI application factory for the Smart Football Match Analysis System.

Wires configuration, database, background job queue, routers, CORS, and error
handling into a single ASGI app. Run with:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .config import Settings, get_settings
from .database import init_db
from .routers import analytics, downloads, health, matches, visuals
from .worker import JobQueue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sfa.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the database and background queue on startup."""
    settings: Settings = get_settings()
    settings.ensure_dirs()
    init_db()
    app.state.queue = JobQueue(settings)
    logger.info("%s v%s started (data dir: %s)", settings.app_name, __version__, settings.data_dir)
    try:
        yield
    finally:
        app.state.queue.shutdown(wait=False)
        logger.info("shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "REST API for uploading football match videos, processing them "
            "asynchronously with a YOLOv8 bird's-eye pipeline, and serving "
            "analytics, processed videos, and reports."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = settings.api_prefix
    app.include_router(health.router, prefix=prefix)
    app.include_router(matches.router, prefix=prefix)
    app.include_router(analytics.router, prefix=prefix)
    app.include_router(downloads.router, prefix=prefix)
    app.include_router(visuals.router, prefix=prefix)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:  # noqa: ANN001
        logger.exception("unhandled error on %s", request.url.path)
        detail = str(exc) if settings.debug else "Internal server error."
        return JSONResponse(status_code=500, content={"detail": detail})

    @app.get("/", tags=["health"])
    async def root() -> dict:
        return {
            "app": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "api_prefix": prefix,
        }

    return app


app = create_app()
