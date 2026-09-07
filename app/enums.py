"""Shared enumerations for job status and analytics artifact kinds."""

from __future__ import annotations

from enum import Enum


class JobStatus(str, Enum):
    """Lifecycle states for a match-analysis job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalyticsType(str, Enum):
    """Analytics artifacts that can be queried as JSON."""

    POSSESSION = "possession"
    PASSES = "passes"
    SHOTS = "shots"
    GOALS = "goals"
    HIGHLIGHTS = "highlights"
    SPEED = "speed"
    HEATMAPS = "heatmaps"
    COORDINATES = "coordinates"


class VideoKind(str, Enum):
    """Downloadable processed videos."""

    BIRDSEYE = "birdseye"
    HIGHLIGHTS = "highlights"


# Maps an analytics type to the artifact key stored in the job manifest.
ANALYTICS_ARTIFACT_KEY: dict[AnalyticsType, str] = {
    AnalyticsType.POSSESSION: "possession_json",
    AnalyticsType.PASSES: "passes_json",
    AnalyticsType.SHOTS: "shots_json",
    AnalyticsType.GOALS: "goals_json",
    AnalyticsType.HIGHLIGHTS: "highlights_json",
    AnalyticsType.SPEED: "speed_json",
    AnalyticsType.HEATMAPS: "heatmap_json",
    AnalyticsType.COORDINATES: "coordinates_json",
}

VIDEO_ARTIFACT_KEY: dict[VideoKind, str] = {
    VideoKind.BIRDSEYE: "birdseye_video",
    VideoKind.HIGHLIGHTS: "highlights_video",
}
