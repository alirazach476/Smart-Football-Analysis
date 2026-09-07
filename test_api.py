"""
End-to-end smoke test for the FastAPI backend.

Uses FastAPI's TestClient with an injected fake analysis function so the full
API flow (upload -> async processing -> analytics -> downloads -> report) is
exercised without the YOLOv8 models or heavy compute.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import numpy as np
import cv2


def _make_video(path: str, frames: int = 10, fps: float = 25.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (320, 180))
    for i in range(frames):
        frame = np.full((180, 320, 3), (30, 60, 30), dtype=np.uint8)
        cv2.putText(frame, f"f{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def _fake_run(settings, job_id, source_path, options, progress_cb=None):
    """Stand-in for the pipeline: writes representative output artifacts."""
    from app.services.analysis_service import job_output_dir

    out = job_output_dir(settings, job_id)
    out.mkdir(parents=True, exist_ok=True)

    # A processed video (copy of the source is fine for the test).
    _make_video(str(out / "birdseye.mp4"), frames=8)
    _make_video(str(out / "highlights.mp4"), frames=6)

    (out / "possession.json").write_text(json.dumps({
        "summary": {"team_a_pct": 55.0, "team_b_pct": 45.0, "loose_pct": 0.0}
    }), encoding="utf-8")
    (out / "passes.json").write_text(json.dumps({
        "summary": {"total": 120, "successful": 90, "success_rate_pct": 75.0}
    }), encoding="utf-8")
    (out / "shots.json").write_text(json.dumps({
        "summary": {"total": 8, "on_target": 5, "accuracy_pct": 62.5}
    }), encoding="utf-8")
    (out / "goals.json").write_text(json.dumps({
        "summary": {"total": 2},
        "final_score": {"Team A": 1, "Team B": 1},
        "goals": [
            {"scoring_team": "Team A", "scorer_track_id": 7, "timestamp_sec": 12.3, "scoreline": "1-0"},
            {"scoring_team": "Team B", "scorer_track_id": 21, "timestamp_sec": 45.1, "scoreline": "1-1"},
        ],
    }), encoding="utf-8")
    (out / "highlights.json").write_text(json.dumps({
        "event_count": 6, "clip_count": 3
    }), encoding="utf-8")


def _build_client():
    # Isolate storage in a temp dir before importing app modules.
    tmp = tempfile.mkdtemp(prefix="sfa-api-test-")
    os.environ["SFA_DATA_DIR"] = tmp
    os.environ["SFA_DEBUG"] = "true"

    from fastapi.testclient import TestClient

    from app.config import get_settings
    from app.database import init_db
    from app.main import create_app
    from app.worker import JobQueue

    get_settings.cache_clear()  # pick up the temp data dir
    settings = get_settings()
    init_db()

    app = create_app()
    # Replace the default queue with one using the fake pipeline runner.
    app.state.queue = JobQueue(settings, run_fn=_fake_run)
    return TestClient(app), tmp


def _wait_completed(client, job_id, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/v1/matches/{job_id}")
        assert r.status_code == 200, r.text
        status = r.json()["status"]
        if status == "completed":
            return r.json()
        if status == "failed":
            raise AssertionError(f"job failed: {r.json().get('error')}")
        time.sleep(0.2)
    raise AssertionError("job did not complete in time")


def main() -> None:
    client, tmp = _build_client()

    # Health.
    r = client.get("/api/v1/health")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ok"

    # Upload.
    video = os.path.join(tmp, "match.mp4")
    _make_video(video, frames=10)
    options = json.dumps({"enable_heatmaps": False, "enable_speed": True})
    with open(video, "rb") as f:
        r = client.post(
            "/api/v1/matches",
            files={"file": ("match.mp4", f, "video/mp4")},
            data={"options": options},
        )
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]
    assert r.json()["status"] == "queued"

    # Wait for async processing.
    detail = _wait_completed(client, job_id)
    assert detail["status"] == "completed"
    assert "possession_json" in detail["artifacts"] or "possession" in detail["summary"]

    # List.
    r = client.get("/api/v1/matches")
    assert r.status_code == 200
    assert r.json()["total"] >= 1

    # Analytics summary.
    r = client.get(f"/api/v1/matches/{job_id}/analytics")
    assert r.status_code == 200, r.text
    assert "possession" in r.json()["summary"]

    # Typed analytics.
    for atype in ("possession", "passes", "shots", "goals", "highlights"):
        r = client.get(f"/api/v1/matches/{job_id}/analytics/{atype}")
        assert r.status_code == 200, f"{atype}: {r.text}"
        assert r.json()["type"] == atype
        assert r.json()["data"]

    # Download videos.
    for kind in ("birdseye", "highlights"):
        r = client.get(f"/api/v1/matches/{job_id}/video", params={"kind": kind})
        assert r.status_code == 200, f"{kind}: {r.text}"
        assert r.headers["content-type"] == "video/mp4"
        assert len(r.content) > 0

    # Reports.
    r = client.get(f"/api/v1/matches/{job_id}/report", params={"format": "html"})
    assert r.status_code == 200
    assert b"Match Analysis Report" in r.content
    r = client.get(f"/api/v1/matches/{job_id}/report", params={"format": "json"})
    assert r.status_code == 200
    assert r.json()["job_id"] == job_id

    # Delete.
    r = client.delete(f"/api/v1/matches/{job_id}")
    assert r.status_code == 200
    r = client.get(f"/api/v1/matches/{job_id}")
    assert r.status_code == 404

    print("[ok] FastAPI backend smoke test passed (upload -> process -> analytics -> download -> report -> delete)")


if __name__ == "__main__":
    main()
