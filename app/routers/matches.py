"""Match lifecycle endpoints: upload, list, detail, delete."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from .. import repository as repo
from ..config import Settings, get_settings
from ..database import get_session
from ..dependencies import get_queue
from ..schemas import (
    JobDetail,
    JobListResponse,
    JobSummary,
    MessageResponse,
    ProcessingOptions,
    UploadResponse,
)
from ..worker import JobQueue

router = APIRouter(prefix="/matches", tags=["matches"])

_CHUNK = 1024 * 1024


def _validate_extension(filename: str, settings: Settings) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_video_extensions:
        allowed = ", ".join(settings.allowed_video_extensions)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )
    return ext


def _save_upload(upload: UploadFile, dest: Path, settings: Settings) -> int:
    """Stream the upload to disk with a size guard. Returns bytes written."""
    written = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as out:
        while True:
            chunk = upload.file.read(_CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if written > settings.max_upload_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds max size of {settings.max_upload_bytes} bytes.",
                )
            out.write(chunk)
    return written


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a match video and queue it for analysis",
)
def upload_match(
    file: UploadFile = File(..., description="Match video file"),
    options: str | None = Form(
        default=None,
        description="JSON-encoded ProcessingOptions (optional)",
    ),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    queue: JobQueue = Depends(get_queue),
) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    ext = _validate_extension(file.filename, settings)

    # Parse optional processing options.
    if options:
        try:
            opts = ProcessingOptions(**json.loads(options))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid options: {exc}") from exc
    else:
        opts = ProcessingOptions()

    job_id = uuid.uuid4().hex
    source_path = settings.jobs_dir / job_id / f"source{ext}"

    written = _save_upload(file, source_path, settings)
    if written == 0:
        source_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    repo.create_job(
        session,
        job_id=job_id,
        original_filename=file.filename,
        source_path=str(source_path),
        options=opts.model_dump(),
    )
    queue.submit(job_id)

    return UploadResponse(id=job_id, status="queued")


@router.get("", response_model=JobListResponse, summary="List match jobs")
def list_matches(
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
) -> JobListResponse:
    jobs = repo.list_jobs(session, status=status_filter, limit=limit, offset=offset)
    total = repo.count_jobs(session, status=status_filter)
    return JobListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[JobSummary(**j.to_dict()) for j in jobs],
    )


@router.get("/{job_id}", response_model=JobDetail, summary="Get a match job")
def get_match(
    job_id: str,
    session: Session = Depends(get_session),
) -> JobDetail:
    job = repo.get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Match not found.")
    return JobDetail(**job.to_dict())


@router.delete("/{job_id}", response_model=MessageResponse, summary="Delete a match job")
def delete_match(
    job_id: str,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> MessageResponse:
    job = repo.get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Match not found.")

    repo.delete_job(session, job)
    job_dir = settings.jobs_dir / job_id
    shutil.rmtree(job_dir, ignore_errors=True)
    return MessageResponse(message=f"Match {job_id} deleted.")
