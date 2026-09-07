"""
Player speed statistics exporter and live video overlay.

Writes per-player aggregate stats and per-frame history to JSON and a flat CSV
summary. Also provides the drawing helper used to show live speed on video.
"""

from __future__ import annotations

import csv
import json
import os

import cv2

from .speed_estimator import PlayerSpeedStats, SpeedEstimator, SpeedSample


def draw_speed_overlay(
    frame: np.ndarray,
    detections,
    speed_samples: dict[int, SpeedSample],
) -> np.ndarray:
    """
  Draw live speed (km/h) below each tracked player's bounding box.

    Sprinting players get a red speed label; others get cyan. Only players with
    a current speed sample (i.e. homography was available that frame) are drawn.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for det in detections:
        if det.track_id is None:
            continue
        if det.class_name not in SpeedEstimator.TRACKABLE_CLASSES:
            continue
        sample = speed_samples.get(det.track_id)
        if sample is None or sample.speed_kmh < 0.5:
            continue

        text = f"{sample.speed_kmh:.1f} km/h"
        color = (0, 0, 255) if sample.is_sprinting else (255, 255, 0)  # red / cyan
        x = det.x1
        y = det.y2 + 18

        # Black outline for readability on any background.
        cv2.putText(frame, text, (x, y), font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, 0.55, color, 1, cv2.LINE_AA)

        if sample.is_sprinting:
            cv2.putText(
                frame, "SPRINT", (x, y + 16), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA
            )

    return frame


class SpeedExporter:
    """Serialises player speed statistics to JSON and CSV."""

    CSV_FIELDS = [
        "track_id",
        "class_name",
        "team",
        "total_distance_m",
        "avg_speed_m_s",
        "avg_speed_kmh",
        "max_speed_m_s",
        "max_speed_kmh",
        "sprint_count",
        "total_sprint_distance_m",
        "total_sprint_time_sec",
    ]

    def __init__(
        self,
        json_path: str = "output/player_speed.json",
        csv_path: str = "output/player_speed.csv",
        sprint_threshold_m_s: float = 7.0,
    ) -> None:
        self.json_path = json_path
        self.csv_path = csv_path
        self.sprint_threshold_m_s = sprint_threshold_m_s

    def save(self, stats: list[PlayerSpeedStats], fps: float, source: str) -> None:
        """Write JSON (full history + sprints) and CSV (aggregate summary)."""
        self._save_json(stats, fps, source)
        self._save_csv(stats)

    def _save_json(
        self, stats: list[PlayerSpeedStats], fps: float, source: str
    ) -> None:
        _ensure_parent(self.json_path)
        payload = {
            "source": source,
            "fps": fps,
            "sprint_threshold_m_s": self.sprint_threshold_m_s,
            "sprint_threshold_kmh": round(self.sprint_threshold_m_s * 3.6, 1),
            "players_tracked": len(stats),
            "players": [s.to_dict() for s in stats],
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _save_csv(self, stats: list[PlayerSpeedStats]) -> None:
        _ensure_parent(self.csv_path)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            for s in stats:
                writer.writerow(
                    {
                        "track_id": s.track_id,
                        "class_name": s.class_name,
                        "team": s.team or "",
                        "total_distance_m": round(s.total_distance_m, 2),
                        "avg_speed_m_s": round(s.avg_speed_m_s, 2),
                        "avg_speed_kmh": round(s.avg_speed_m_s * 3.6, 1),
                        "max_speed_m_s": round(s.max_speed_m_s, 2),
                        "max_speed_kmh": round(s.max_speed_kmh, 1),
                        "sprint_count": s.sprint_count,
                        "total_sprint_distance_m": round(
                            s.total_sprint_distance_m, 2
                        ),
                        "total_sprint_time_sec": round(s.total_sprint_time_sec, 2),
                    }
                )


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
