"""Data-access helpers for match jobs (thin repository over SQLAlchemy)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from .enums import JobStatus
from .models import MatchJob


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_job(
    session: Session,
    job_id: str,
    original_filename: str,
    source_path: str,
    options: dict,
) -> MatchJob:
    job = MatchJob(
        id=job_id,
        original_filename=original_filename,
        source_path=source_path,
        status=JobStatus.QUEUED.value,
        options=options,
        artifacts={},
        summary={},
        progress=0.0,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def get_job(session: Session, job_id: str) -> MatchJob | None:
    return session.get(MatchJob, job_id)


def list_jobs(
    session: Session,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[MatchJob]:
    stmt = select(MatchJob).order_by(MatchJob.created_at.desc())
    if status:
        stmt = stmt.where(MatchJob.status == status)
    stmt = stmt.limit(limit).offset(offset)
    return list(session.scalars(stmt))


def count_jobs(session: Session, status: str | None = None) -> int:
    stmt = select(MatchJob)
    if status:
        stmt = stmt.where(MatchJob.status == status)
    return len(list(session.scalars(stmt)))


def mark_processing(session: Session, job: MatchJob) -> MatchJob:
    job.status = JobStatus.PROCESSING.value
    job.started_at = _utcnow()
    job.progress = 0.05
    session.commit()
    session.refresh(job)
    return job


def mark_completed(
    session: Session, job: MatchJob, artifacts: dict, summary: dict
) -> MatchJob:
    job.status = JobStatus.COMPLETED.value
    job.artifacts = artifacts
    job.summary = summary
    job.progress = 1.0
    job.finished_at = _utcnow()
    session.commit()
    session.refresh(job)
    return job


def mark_failed(session: Session, job: MatchJob, error: str) -> MatchJob:
    job.status = JobStatus.FAILED.value
    job.error = error
    job.finished_at = _utcnow()
    session.commit()
    session.refresh(job)
    return job


def update_progress(session: Session, job: MatchJob, progress: float) -> None:
    job.progress = max(0.0, min(1.0, progress))
    session.commit()


def delete_job(session: Session, job: MatchJob) -> None:
    session.delete(job)
    session.commit()
