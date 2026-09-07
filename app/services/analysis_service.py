"""
Analysis service: runs the bird's-eye pipeline for a job and collects outputs.

This module is the single bridge between the API and the heavy YOLOv8 pipeline
(`src.birdseye_pipeline.BirdseyePipeline`). The pipeline import is deferred to
call time so the rest of the API (and its tests) can be imported without the
torch/ultralytics stack present.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from ..config import Settings

logger = logging.getLogger("sfa.analysis")


# Artifact key -> path relative to the job's output directory.
_ARTIFACT_LAYOUT: dict[str, str] = {
    "birdseye_video": "birdseye.mp4",
    "highlights_video": "highlights.mp4",
    "coordinates_json": "birdseye_coordinates.json",
    "possession_json": "possession.json",
    "possession_csv": "possession.csv",
    "possession_timeline_csv": "possession_timeline.csv",
    "passes_json": "passes.json",
    "passes_csv": "passes.csv",
    "pass_statistics_csv": "pass_statistics.csv",
    "pass_network_dir": "pass_networks",
    "shots_json": "shots.json",
    "shots_csv": "shots.csv",
    "shot_statistics_csv": "shot_statistics.csv",
    "shot_map_dir": "shot_maps",
    "goals_json": "goals.json",
    "goals_csv": "goals.csv",
    "goal_statistics_csv": "goal_statistics.csv",
    "goal_notifications_txt": "goal_notifications.txt",
    "highlights_json": "highlights.json",
    "speed_json": "player_speed.json",
    "speed_csv": "player_speed.csv",
    "heatmap_json": "heatmap_data.json",
    "heatmap_html": "heatmaps.html",
    "heatmap_dir": "heatmaps",
}


class ModelsUnavailableError(RuntimeError):
    """Raised when the trained model weights are missing."""


def models_available(settings: Settings) -> bool:
    return Path(settings.field_model_path).exists() and Path(settings.object_model_path).exists()


def job_output_dir(settings: Settings, job_id: str) -> Path:
    return settings.jobs_dir / job_id / "output"


def build_pipeline(settings: Settings, job_id: str, source_path: str, options: dict, progress_cb=None):
    """Construct a configured `BirdseyePipeline` for a job (deferred import)."""
    from src.birdseye_pipeline import BirdseyePipeline

    out = job_output_dir(settings, job_id)
    out.mkdir(parents=True, exist_ok=True)

    def p(name: str) -> str:
        return str(out / _ARTIFACT_LAYOUT[name])

    device = options.get("device") or settings.device
    imgsz = options.get("imgsz") or settings.imgsz
    conf = options.get("conf_threshold") or settings.conf_threshold
    ball_conf = options.get("ball_conf_threshold") or settings.ball_conf_threshold
    possession_radius = options.get("possession_radius_m") or settings.possession_radius_m

    return BirdseyePipeline(
        field_model=str(settings.field_model_path),
        object_model=str(settings.object_model_path),
        source=source_path,
        output_path=p("birdseye_video"),
        coordinates_json=p("coordinates_json"),
        conf_threshold=conf,
        ball_conf_threshold=ball_conf,
        device=device,
        imgsz=imgsz,
        enable_teams=options.get("enable_teams", True),
        enable_speed=options.get("enable_speed", True),
        speed_json=p("speed_json"),
        speed_csv=p("speed_csv"),
        enable_heatmaps=options.get("enable_heatmaps", settings.enable_heatmaps_default),
        heatmap_dir=p("heatmap_dir"),
        heatmap_json=p("heatmap_json"),
        heatmap_html=p("heatmap_html"),
        enable_possession=options.get("enable_possession", True),
        possession_json=p("possession_json"),
        possession_csv=p("possession_csv"),
        possession_timeline_csv=p("possession_timeline_csv"),
        possession_radius_m=possession_radius,
        enable_passes=options.get("enable_passes", True),
        passes_json=p("passes_json"),
        passes_csv=p("passes_csv"),
        pass_statistics_csv=p("pass_statistics_csv"),
        pass_network_dir=p("pass_network_dir"),
        enable_shots=options.get("enable_shots", True),
        shots_json=p("shots_json"),
        shots_csv=p("shots_csv"),
        shot_statistics_csv=p("shot_statistics_csv"),
        shot_map_dir=p("shot_map_dir"),
        enable_goals=options.get("enable_goals", True),
        goals_json=p("goals_json"),
        goals_csv=p("goals_csv"),
        goal_statistics_csv=p("goal_statistics_csv"),
        goal_notifications_path=p("goal_notifications_txt"),
        enable_highlights=options.get("enable_highlights", True),
        highlights_path=p("highlights_video"),
        highlights_json=p("highlights_json"),
        enable_ball_model=options.get("enable_ball_model", settings.enable_ball_model),
        ball_model=options.get("ball_model") or settings.ball_model,
        position_smoothing=options.get("position_smoothing", settings.position_smoothing),
        progress_callback=progress_cb,
    )


def run_analysis(
    settings: Settings, job_id: str, source_path: str, options: dict, progress_cb=None
) -> None:
    """Execute the full pipeline for a job (blocking; call from a worker)."""
    if not models_available(settings):
        raise ModelsUnavailableError(
            "Trained model weights not found. Expected:\n"
            f"  field:  {settings.field_model_path}\n"
            f"  object: {settings.object_model_path}"
        )
    pipeline = build_pipeline(settings, job_id, source_path, options, progress_cb=progress_cb)
    pipeline.run()

    # OpenCV writes MPEG-4 Part 2 (mp4v), which HTML5 <video> cannot decode.
    # Transcode the browser-facing outputs to H.264 so they play in the dashboard.
    out = job_output_dir(settings, job_id)
    for name in ("birdseye.mp4", "highlights.mp4"):
        _transcode_h264(out / name)


def _transcode_h264(path: Path) -> None:
    """Re-encode a video to browser-friendly H.264 (yuv420p) in place, if ffmpeg exists."""
    if not path.exists():
        return
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found; %s may not play in the browser (mp4v codec)", path.name)
        return
    tmp = path.with_name(path.stem + "_h264.mp4")
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an", str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(path)
        logger.info("transcoded %s to H.264", path.name)
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.warning("H.264 transcode failed for %s: %s", path.name, exc)
        tmp.unlink(missing_ok=True)


def collect_artifacts(settings: Settings, job_id: str) -> dict[str, str]:
    """Return {artifact_key: absolute_path} for every produced output."""
    out = job_output_dir(settings, job_id)
    artifacts: dict[str, str] = {}
    for key, rel in _ARTIFACT_LAYOUT.items():
        path = out / rel
        if path.exists():
            artifacts[key] = str(path)
    return artifacts


def _read_json(path: str | None) -> dict:
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def build_summary(artifacts: dict[str, str]) -> dict:
    """Distil a compact, API-friendly summary from the JSON artifacts."""
    summary: dict = {}

    possession = _read_json(artifacts.get("possession_json"))
    if possession.get("summary"):
        summary["possession"] = possession["summary"]

    passes = _read_json(artifacts.get("passes_json"))
    if passes.get("summary"):
        summary["passes"] = passes["summary"]

    shots = _read_json(artifacts.get("shots_json"))
    if shots.get("summary"):
        summary["shots"] = shots["summary"]

    goals = _read_json(artifacts.get("goals_json"))
    if goals:
        summary["goals"] = {
            "total": goals.get("summary", {}).get("total", 0),
            "final_score": goals.get("final_score", {}),
        }

    highlights = _read_json(artifacts.get("highlights_json"))
    if highlights:
        summary["highlights"] = {
            "event_count": highlights.get("event_count", 0),
            "clip_count": highlights.get("clip_count", 0),
        }

    speed = _read_json(artifacts.get("speed_json"))
    if speed.get("summary"):
        summary["speed"] = speed["summary"]

    return summary
