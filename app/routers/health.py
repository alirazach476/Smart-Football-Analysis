"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from .. import __version__
from ..config import get_settings
from ..schemas import HealthResponse
from ..services.analysis_service import models_available

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        app=settings.app_name,
        version=__version__,
        models_available=models_available(settings),
    )
