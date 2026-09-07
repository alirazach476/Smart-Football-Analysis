"""Serve generated PNG visualizations (heatmaps, pass networks, shot maps)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .. import repository as repo
from ..database import get_session
from ..enums import JobStatus

router = APIRouter(prefix="/matches/{job_id}/visuals", tags=["visuals"])

_VISUAL_DIRS = {
    "heatmap": "heatmap_dir",
    "pass_network": "pass_network_dir",
    "shot_map": "shot_map_dir",
}


def _require_completed(session: Session, job_id: str):
    job = repo.get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Match not found.")
    if job.status != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=409, detail=f"Visuals not ready (status: {job.status}).")
    return job


@router.get("", summary="List available visualization images")
def list_visuals(
    job_id: str,
    session: Session = Depends(get_session),
) -> dict:
    job = _require_completed(session, job_id)
    artifacts = job.artifacts or {}
    items: list[dict] = []

    for category, key in _VISUAL_DIRS.items():
        dir_path = artifacts.get(key)
        if not dir_path:
            continue
        folder = Path(dir_path)
        if not folder.is_dir():
            continue
        for png in sorted(folder.glob("*.png")):
            items.append({
                "id": f"{category}:{png.stem}",
                "category": category,
                "name": png.stem,
                "filename": png.name,
            })

    return {"job_id": job_id, "items": items}


@router.get("/file", summary="Serve a visualization PNG")
def get_visual(
    job_id: str,
    category: str = Query(..., pattern="^(heatmap|pass_network|shot_map)$"),
    name: str = Query(..., min_length=1),
    session: Session = Depends(get_session),
) -> FileResponse:
    job = _require_completed(session, job_id)
    key = _VISUAL_DIRS.get(category)
    if not key:
        raise HTTPException(status_code=400, detail="Invalid category.")
    dir_path = (job.artifacts or {}).get(key)
    if not dir_path:
        raise HTTPException(status_code=404, detail="Visualization directory not found.")

    # Prevent path traversal — only serve files directly in the artifact dir.
    safe_name = Path(name).name
    if not safe_name.endswith(".png"):
        safe_name = f"{safe_name}.png"
    file_path = Path(dir_path) / safe_name
    if not file_path.is_file() or file_path.parent.resolve() != Path(dir_path).resolve():
        raise HTTPException(status_code=404, detail="Visualization not found.")

    return FileResponse(file_path, media_type="image/png")
