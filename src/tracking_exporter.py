"""
Tracking exporter.

Accumulates tracked detections during a run and produces two artifacts:

* `tracking.json` - rich, structured output with BOTH a per-frame view and a
  per-track "history" (each track's trajectory over time, first/last frame,
  how long it lived, etc.).
* `tracking.csv`  - flat, one row per tracked detection, ideal for pandas /
  spreadsheets.

Only detections that carry a `track_id` are exported here; untracked boxes
(rare, on the first couple of frames before ByteTrack confirms a track) are
skipped so the tracking outputs stay clean.
"""

from __future__ import annotations

import csv
import json
import os

from .detector import Detection


class TrackHistory:
    """Per-track running summary and trajectory."""

    def __init__(self, track_id: int, class_name: str, class_id: int) -> None:
        self.track_id = track_id
        self.class_name = class_name
        self.class_id = class_id
        self.first_frame: int | None = None
        self.last_frame: int | None = None
        # Trajectory of box centres over time: [{frame, timestamp_sec, cx, cy}]
        self.trajectory: list[dict] = []
        self._confidences: list[float] = []

    def update(self, frame_index: int, timestamp: float | None, det: Detection) -> None:
        if self.first_frame is None:
            self.first_frame = frame_index
        self.last_frame = frame_index
        cx, cy = det.center
        self.trajectory.append(
            {"frame": frame_index, "timestamp_sec": timestamp, "cx": cx, "cy": cy}
        )
        self._confidences.append(det.confidence)

    def to_dict(self) -> dict:
        frames_seen = len(self.trajectory)
        span = (
            (self.last_frame - self.first_frame + 1)
            if self.first_frame is not None
            else 0
        )
        avg_conf = (
            sum(self._confidences) / len(self._confidences)
            if self._confidences
            else 0.0
        )
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "frames_seen": frames_seen,
            # Total lifespan in frames from first to last sighting; larger than
            # frames_seen means the track was occluded/lost for some frames but
            # kept the same ID on reappearance.
            "frame_span": span,
            "avg_confidence": round(avg_conf, 4),
            "trajectory": self.trajectory,
        }


class TrackingExporter:
    """Accumulates tracked detections and serialises tracking artifacts."""

    CSV_FIELDS = [
        "frame",
        "timestamp_sec",
        "track_id",
        "class_id",
        "class_name",
        "team",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "cx",
        "cy",
    ]

    def __init__(self, fps: float = 0.0) -> None:
        self.fps = fps if fps > 0 else 0.0
        # Per-frame records for the CSV / per-frame JSON view.
        self._frames: list[dict] = []
        # track_id -> TrackHistory
        self._tracks: dict[int, TrackHistory] = {}

    def add_frame(self, frame_index: int, detections: list[Detection]) -> None:
        """Record all *tracked* detections for a single frame."""
        timestamp = round(frame_index / self.fps, 3) if self.fps else None

        frame_tracks: list[dict] = []
        for det in detections:
            if det.track_id is None:
                continue  # skip not-yet-confirmed tracks

            frame_tracks.append(det.to_dict())

            # Update the per-track history / trajectory.
            history = self._tracks.get(det.track_id)
            if history is None:
                history = TrackHistory(det.track_id, det.class_name, det.class_id)
                self._tracks[det.track_id] = history
            history.update(frame_index, timestamp, det)

        self._frames.append(
            {
                "frame": frame_index,
                "timestamp_sec": timestamp,
                "tracks": frame_tracks,
            }
        )

    def save_json(self, path: str) -> None:
        """Write structured tracking JSON (per-frame + per-track history)."""
        _ensure_parent_dir(path)

        # Sort tracks by ID for a stable, readable file.
        tracks = [self._tracks[tid].to_dict() for tid in sorted(self._tracks)]

        # Class breakdown, e.g. how many unique players vs referees.
        class_counts: dict[str, int] = {}
        for t in self._tracks.values():
            class_counts[t.class_name] = class_counts.get(t.class_name, 0) + 1

        payload = {
            "fps": self.fps,
            "total_frames": len(self._frames),
            "unique_tracks": len(self._tracks),
            "unique_tracks_per_class": class_counts,
            "tracks": tracks,
            "frames": self._frames,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def save_csv(self, path: str) -> None:
        """Write a flat CSV with one row per tracked detection."""
        _ensure_parent_dir(path)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            for frame in self._frames:
                for det in frame["tracks"]:
                    writer.writerow(
                        {
                            "frame": frame["frame"],
                            "timestamp_sec": frame["timestamp_sec"],
                            "track_id": det["track_id"],
                            "class_id": det["class_id"],
                            "class_name": det["class_name"],
                            "team": det.get("team"),
                            "confidence": det["confidence"],
                            "x1": det["x1"],
                            "y1": det["y1"],
                            "x2": det["x2"],
                            "y2": det["y2"],
                            "width": det["width"],
                            "height": det["height"],
                            "cx": det["cx"],
                            "cy": det["cy"],
                        }
                    )

    @property
    def unique_tracks(self) -> int:
        return len(self._tracks)


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory for `path` if it does not exist."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
