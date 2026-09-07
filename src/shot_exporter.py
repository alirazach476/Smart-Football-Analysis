"""
Shot detection statistics exporter.

Writes shot events JSON, per-shot CSV, summary CSV, and shot map PNGs.
"""

from __future__ import annotations

import csv
import json
import os

import cv2

from .shot_detector import ShotDetector
from .shot_map import render_shot_map


class ShotExporter:
    """Export shot statistics and shot map visualizations."""

    SHOT_CSV_FIELDS = [
        "shot_id", "team", "shooter_track_id", "outcome", "target_goal",
        "speed_ms", "speed_kmh", "location_x_cm", "location_y_cm",
        "distance_to_goal_m", "angle_deg", "projected_goal_y_cm",
        "start_frame", "end_frame", "start_time_sec", "end_time_sec",
    ]
    SUMMARY_CSV_FIELDS = [
        "team", "total", "on_target", "off_target", "blocked",
        "accuracy_pct", "max_speed_kmh", "avg_speed_kmh", "avg_distance_m",
    ]

    def __init__(
        self,
        json_path: str = "output/shots.json",
        csv_path: str = "output/shots.csv",
        summary_csv_path: str = "output/shot_statistics.csv",
        shot_map_dir: str = "output/shot_maps",
        pitch_scale: float = 0.06,
    ) -> None:
        self.json_path = json_path
        self.csv_path = csv_path
        self.summary_csv_path = summary_csv_path
        self.shot_map_dir = shot_map_dir
        self.pitch_scale = pitch_scale

    def save(
        self,
        detector: ShotDetector,
        source: str,
        fps: float,
        team_names: tuple[str, str] = ("Team A", "Team B"),
    ) -> dict:
        """Write JSON, CSVs, and shot map PNGs. Returns export summary."""
        _ensure_parent(self.json_path)
        os.makedirs(self.shot_map_dir, exist_ok=True)

        shots = detector.shots
        overall = detector.stats()
        team_stats = {t: detector.stats(t).to_dict() for t in team_names}

        payload = {
            "source": source,
            "fps": fps,
            "summary": overall.to_dict(),
            "by_team": team_stats,
            "shots": [s.to_dict() for s in shots],
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SHOT_CSV_FIELDS)
            writer.writeheader()
            for s in shots:
                writer.writerow(s.to_dict())

        with open(self.summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SUMMARY_CSV_FIELDS)
            writer.writeheader()

            def _row(label: str, stats: dict) -> dict:
                return {k: stats.get(k, "") for k in self.SUMMARY_CSV_FIELDS if k != "team"} | {"team": label}

            writer.writerow(_row("Overall", overall.to_dict()))
            for team in team_names:
                writer.writerow(_row(team, team_stats[team]))

        map_paths: list[str] = []
        combined = render_shot_map(
            shots, team=None, pitch_scale=self.pitch_scale, title="All shots"
        )
        combined_path = os.path.join(self.shot_map_dir, "shot_map_all.png")
        cv2.imwrite(combined_path, combined)
        map_paths.append(combined_path)

        for team in team_names:
            if not any(s.team == team for s in shots):
                continue
            safe = team.replace(" ", "_")
            path = os.path.join(self.shot_map_dir, f"shot_map_{safe}.png")
            img = render_shot_map(
                shots, team=team, pitch_scale=self.pitch_scale,
                title=f"{team} shots",
            )
            cv2.imwrite(path, img)
            map_paths.append(path)

        return {
            "json_path": self.json_path,
            "csv_path": self.csv_path,
            "summary_csv_path": self.summary_csv_path,
            "shot_map_dir": self.shot_map_dir,
            "shot_map_paths": map_paths,
            "total_shots": overall.total,
            "on_target": overall.on_target,
            "accuracy_pct": overall.to_dict()["accuracy_pct"],
        }


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
