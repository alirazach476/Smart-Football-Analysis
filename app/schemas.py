"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .enums import JobStatus


class ProcessingOptions(BaseModel):
    """Optional processing knobs supplied at upload time."""

    enable_teams: bool = True
    enable_speed: bool = True
    enable_possession: bool = True
    enable_passes: bool = True
    enable_shots: bool = True
    enable_goals: bool = True
    enable_highlights: bool = True
    enable_heatmaps: bool = False
    device: str | None = Field(default=None, description="cpu, cuda, or GPU index")
    imgsz: int | None = Field(default=None, ge=320, le=2560)
    conf_threshold: float | None = Field(default=None, ge=0.01, le=0.95)


class JobSummary(BaseModel):
    """Compact job view used in list responses."""

    id: str
    original_filename: str
    status: JobStatus
    progress: float
    created_at: str | None = None
    updated_at: str | None = None


class JobDetail(BaseModel):
    """Full job view."""

    id: str
    original_filename: str
    status: JobStatus
    progress: float
    options: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    artifacts: list[str] = []
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class JobListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[JobSummary]


class UploadResponse(BaseModel):
    id: str
    status: JobStatus
    message: str = "Match uploaded and queued for processing."


class AnalyticsResponse(BaseModel):
    id: str
    type: str
    data: dict[str, Any]


class MessageResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    version: str
    models_available: bool
