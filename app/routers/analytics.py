"""Analytics endpoints: full summary and per-type JSON."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import repository as repo
from ..database import get_session
from ..enums import ANALYTICS_ARTIFACT_KEY, AnalyticsType, JobStatus
from ..schemas import AnalyticsResponse

router = APIRouter(prefix="/matches/{job_id}/analytics", tags=["analytics"])


def _require_completed(session: Session, job_id: str):
    job = repo.get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Match not found.")
    if job.status != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=409,
            detail=f"Analytics not ready (status: {job.status}).",
        )
    return job


def _load_artifact_json(job, artifact_key: str) -> dict:
    path = (job.artifacts or {}).get(artifact_key)
    if not path:
        raise HTTPException(status_code=404, detail="Requested analytics were not produced.")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read analytics: {exc}") from exc


@router.get("", summary="Full analytics summary for a match")
def get_analytics_summary(
    job_id: str,
    session: Session = Depends(get_session),
) -> dict:
    job = _require_completed(session, job_id)
    return {
        "id": job.id,
        "match": job.original_filename,
        "summary": job.summary or {},
        "available": sorted((job.artifacts or {}).keys()),
    }


@router.get("/{analytics_type}", response_model=AnalyticsResponse, summary="Typed analytics JSON")
def get_analytics(
    job_id: str,
    analytics_type: AnalyticsType,
    session: Session = Depends(get_session),
) -> AnalyticsResponse:
    job = _require_completed(session, job_id)
    artifact_key = ANALYTICS_ARTIFACT_KEY[analytics_type]
    data = _load_artifact_json(job, artifact_key)
    return AnalyticsResponse(id=job.id, type=analytics_type.value, data=data)
