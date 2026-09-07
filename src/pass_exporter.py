"""
Pass detection statistics exporter.

Writes pass events JSON, summary CSV, per-pass CSV, and pass network PNGs.
"""

from __future__ import annotations

import csv
import json
import os

import cv2

from .pass_detector import PassDetector, PassEvent
from .pass_network import render_pass_map, render_pass_network


class PassExporter:
    """Export pass statistics and network visualizations."""

    PASS_CSV_FIELDS = [
        "pass_id", "team", "from_track_id", "to_track_id", "successful",
        "primary_type", "types", "distance_m", "forward_m",
        "start_frame", "end_frame", "start_time_sec", "end_time_sec",
        "start_x_cm", "start_y_cm", "end_x_cm", "end_y_cm", "failure_reason",
    ]
    SUMMARY_CSV_FIELDS = [
        "team", "total", "successful", "unsuccessful", "success_rate_pct",
        "short", "long", "through_ball", "cross", "back",
    ]

    def __init__(
        self,
        json_path: str = "output/passes.json",
        csv_path: str = "output/passes.csv",
        summary_csv_path: str = "output/pass_statistics.csv",
        network_dir: str = "output/pass_networks",
        pitch_scale: float = 0.06,
    ) -> None:
        self.json_path = json_path
        self.csv_path = csv_path
        self.summary_csv_path = summary_csv_path
        self.network_dir = network_dir
        self.pitch_scale = pitch_scale

    def save(
        self,
        detector: PassDetector,
        source: str,
        fps: float,
        team_names: tuple[str, str] = ("Team A", "Team B"),
    ) -> dict:
        """Write JSON, CSVs, and network PNGs. Returns export summary."""
        _ensure_parent(self.json_path)
        os.makedirs(self.network_dir, exist_ok=True)

        passes = detector.passes
        overall = detector.stats()
        team_stats = {t: detector.stats(t).to_dict() for t in team_names}
        positions = detector.average_player_positions()

        payload = {
            "source": source,
            "fps": fps,
            "summary": overall.to_dict(),
            "by_team": team_stats,
            "passes": [p.to_dict() for p in passes],
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.PASS_CSV_FIELDS)
            writer.writeheader()
            for p in passes:
                row = p.to_dict()
                row["types"] = "|".join(p.types)
                writer.writerow(row)

        with open(self.summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SUMMARY_CSV_FIELDS)
            writer.writeheader()
            def _summary_row(team_label: str, stats: dict) -> dict:
                return {
                    "team": team_label,
                    "total": stats["total"],
                    "successful": stats["successful"],
                    "unsuccessful": stats["unsuccessful"],
                    "success_rate_pct": stats["success_rate_pct"],
                    "short": stats["short"],
                    "long": stats["long"],
                    "through_ball": stats["through_ball"],
                    "cross": stats["cross"],
                    "back": stats["back"],
                }

            writer.writerow(_summary_row("Overall", overall.to_dict()))
            for team in team_names:
                writer.writerow(_summary_row(team, team_stats[team]))

        network_paths: list[str] = []
        for team in team_names:
            team_passes = [p for p in passes if p.team == team]
            if not team_passes:
                continue
            safe = team.replace(" ", "_")
            net_path = os.path.join(self.network_dir, f"network_{safe}.png")
            map_path = os.path.join(self.network_dir, f"pass_map_{safe}.png")

            net_img = render_pass_network(
                passes, positions, team=team,
                pitch_scale=self.pitch_scale,
                title=f"{team} pass network",
            )
            cv2.imwrite(net_path, net_img)
            network_paths.append(net_path)

            map_img = render_pass_map(
                passes, team=team,
                pitch_scale=self.pitch_scale,
                title=f"{team} passes",
            )
            cv2.imwrite(map_path, map_img)
            network_paths.append(map_path)

        # Combined network if both teams have passes.
        if len(passes) > 0:
            combined_path = os.path.join(self.network_dir, "network_all.png")
            combined = render_pass_network(
                passes, positions, team=None,
                pitch_scale=self.pitch_scale,
                title="All passes",
            )
            cv2.imwrite(combined_path, combined)
            network_paths.append(combined_path)

        return {
            "json_path": self.json_path,
            "csv_path": self.csv_path,
            "summary_csv_path": self.summary_csv_path,
            "network_dir": self.network_dir,
            "network_paths": network_paths,
            "total_passes": overall.total,
            "success_rate_pct": overall.to_dict()["success_rate_pct"],
        }


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
