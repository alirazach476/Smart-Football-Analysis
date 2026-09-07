"""
Asynchronous job processing via a thread pool.

Video analysis is long-running and CPU/GPU bound, so it runs off the request
thread in a bounded `ThreadPoolExecutor`. Each task opens its own database
session (SQLAlchemy sessions are not shared across threads), transitions the job
through its lifecycle, runs the analysis, and records artifacts + summary.

The heavy analysis callable is injectable (`run_fn`) so the queue can be tested
with a lightweight stand-in that does not require the model weights.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from .config import Settings
from .database import SessionLocal
from . import repository as repo
from .services import analysis_service

logger = logging.getLogger("sfa.worker")

RunFn = Callable[..., None]


class JobQueue:
    """Bounded background queue that processes match-analysis jobs."""

    def __init__(
        self,
        settings: Settings,
        run_fn: RunFn | None = None,
    ) -> None:
        self._settings = settings
        self._run_fn: RunFn = run_fn or analysis_service.run_analysis
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, settings.max_workers),
            thread_name_prefix="sfa-worker",
        )

    def submit(self, job_id: str) -> None:
        """Enqueue a job for background processing."""
        self._executor.submit(self._process, job_id)

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    # ------------------------------------------------------------------ #
    def _process(self, job_id: str) -> None:
        session = SessionLocal()
        try:
            job = repo.get_job(session, job_id)
            if job is None:
                logger.warning("job %s vanished before processing", job_id)
                return

            repo.mark_processing(session, job)
            source_path = job.source_path
            options = dict(job.options or {})

            def _on_progress(frac: float) -> None:
                try:
                    repo.update_progress(session, job, frac)
                except Exception:  # noqa: BLE001
                    logger.debug("progress update failed for job %s", job_id)

            logger.info("processing job %s (%s)", job_id, job.original_filename)
            self._run_fn(self._settings, job_id, source_path, options, progress_cb=_on_progress)

            artifacts = analysis_service.collect_artifacts(self._settings, job_id)
            summary = analysis_service.build_summary(artifacts)
            repo.mark_completed(session, job, artifacts, summary)
            logger.info("job %s completed (%d artifacts)", job_id, len(artifacts))

        except Exception as exc:  # noqa: BLE001 - record any failure on the job
            logger.exception("job %s failed", job_id)
            try:
                job = repo.get_job(session, job_id)
                if job is not None:
                    detail = f"{type(exc).__name__}: {exc}"
                    if self._settings.debug:
                        detail += "\n" + traceback.format_exc()
                    repo.mark_failed(session, job, detail)
            except Exception:  # noqa: BLE001
                logger.exception("failed to record failure for job %s", job_id)
        finally:
            session.close()
