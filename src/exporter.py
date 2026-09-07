"""
Detection exporters.

Collects per-frame detections during processing and writes them out to
`detections.json` (structured, grouped by frame) and `detections.csv`
(flat, one row per detection) at the end of the run.
"""

from __future__ import annotations

import csv
import json
import os

from .detector import Detection


class DetectionExporter:
    """Accumulates detections per frame and serialises them to disk."""

    # Column order used for the flat CSV export. Kept in sync with
    # Detection.to_dict(); `track_id` is null in pure detection mode.
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
        # Source video FPS, used to compute a timestamp for each frame.
        self.fps = fps if fps > 0 else 0.0
        # frames[frame_index] = list of detection dicts
        self._frames: list[dict] = []

    def add_frame(self, frame_index: int, detections: list[Detection]) -> None:
        """Record all detections for a single frame."""
        timestamp = round(frame_index / self.fps, 3) if self.fps else None
        self._frames.append(
            {
                "frame": frame_index,
                "timestamp_sec": timestamp,
                "detections": [d.to_dict() for d in detections],
            }
        )

    def save_json(self, path: str) -> None:
        """Write a structured JSON file grouped by frame."""
        _ensure_parent_dir(path)
        payload = {
            "fps": self.fps,
            "total_frames": len(self._frames),
            "total_detections": self.total_detections,
            "frames": self._frames,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def save_csv(self, path: str) -> None:
        """Write a flat CSV file with one row per detection."""
        _ensure_parent_dir(path)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            for frame in self._frames:
                for det in frame["detections"]:
                    writer.writerow(
                        {
                            "frame": frame["frame"],
                            "timestamp_sec": frame["timestamp_sec"],
                            **det,
                        }
                    )

    @property
    def total_detections(self) -> int:
        return sum(len(f["detections"]) for f in self._frames)


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory for `path` if it does not exist."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
