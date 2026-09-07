"""Download endpoints: processed videos and reports."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .. import repository as repo
from ..config import Settings, get_settings
from ..database import get_session
from ..enums import VIDEO_ARTIFACT_KEY, JobStatus, VideoKind
from ..services.report_service import write_report

router = APIRouter(prefix="/matches/{job_id}", tags=["downloads"])


def _require_completed(session: Session, job_id: str):
    job = repo.get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Match not found.")
    if job.status != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=409, detail=f"Outputs not ready (status: {job.status})."
        )
    return job


@router.get("/video", summary="Download a processed video")
def download_video(
    job_id: str,
    kind: VideoKind = Query(default=VideoKind.BIRDSEYE),
    session: Session = Depends(get_session),
) -> FileResponse:
    job = _require_completed(session, job_id)
    artifact_key = VIDEO_ARTIFACT_KEY[kind]
    path = (job.artifacts or {}).get(artifact_key)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"No '{kind.value}' video available.")
    filename = f"{Path(job.original_filename).stem}_{kind.value}.mp4"
    return FileResponse(path, media_type="video/mp4", filename=filename)


@router.get("/report", summary="Download a match report (html or json)")
def download_report(
    job_id: str,
    format: str = Query(default="html", pattern="^(html|json)$"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    job = _require_completed(session, job_id)
    path = write_report(settings, job, fmt=format)
    media_type = "text/html" if format == "html" else "application/json"
    filename = f"{Path(job.original_filename).stem}_report.{format}"
    return FileResponse(path, media_type=media_type, filename=filename)
