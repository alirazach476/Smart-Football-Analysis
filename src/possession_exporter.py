"""
Possession statistics exporter and live video overlay.

Writes possession JSON (timeline + team/player stats) and summary CSV.
Draws a live possession bar and current-holder label on video frames.
"""

from __future__ import annotations

import csv
import json
import os

import cv2

from .possession_analyzer import PossessionAnalyzer, PossessionState


def draw_possession_overlay(
    frame: np.ndarray,
    analyzer: PossessionAnalyzer,
    current_team_names: tuple[str, str] = ("Team A", "Team B"),
) -> np.ndarray:
    """
    Draw live possession bar and current holder on a video frame.

    Top bar shows Team A vs Team B split; label shows current possessor or
    LOOSE BALL / CONTESTED.
    """
    h, w = frame.shape[:2]
    stats = analyzer.team_stats()
    summary = analyzer.summary()
    current = analyzer.current

    ta_pct = summary.get("team_a_pct", 0.0)
    tb_pct = summary.get("team_b_pct", 0.0)

    # --- Possession bar (top, 24px tall) ---------------------------------
    bar_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)

    split = int(w * ta_pct / 100.0) if (ta_pct + tb_pct) > 0 else w // 2
    cv2.rectangle(overlay, (0, 0), (split, bar_h), (200, 80, 40), -1)   # Team A blue-ish
    cv2.rectangle(overlay, (split, 0), (w, bar_h), (40, 60, 200), -1)   # Team B red-ish

    label_a = f"{current_team_names[0]} {ta_pct:.0f}%"
    label_b = f"{tb_pct:.0f}% {current_team_names[1]}"
    cv2.putText(overlay, label_a, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, label_b, (w - 160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)

    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # --- Current possession label ----------------------------------------
    if current is None:
        return frame

    if current.state == PossessionState.LOOSE.value:
        status = "LOOSE BALL"
        color = (0, 255, 255)
    elif current.state == PossessionState.CONTESTED.value:
        status = "CONTESTED"
        color = (0, 165, 255)
    elif current.track_id is not None:
        team_str = f" ({current.team})" if current.team else ""
        dist = f" {current.nearest_distance_m:.1f}m" if current.nearest_distance_m else ""
        status = f"#{current.track_id}{team_str}{dist}"
        color = (0, 255, 0)
    else:
        status = current.state
        color = (200, 200, 200)

    cv2.putText(
        frame, status, (12, bar_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 0, 0), 4, cv2.LINE_AA,
    )
    cv2.putText(
        frame, status, (12, bar_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        color, 2, cv2.LINE_AA,
    )
    return frame


class PossessionExporter:
    """Export possession statistics to JSON and CSV."""

    TEAM_CSV_FIELDS = ["team", "possession_time_sec", "possession_pct"]
    PLAYER_CSV_FIELDS = [
        "track_id", "class_name", "team",
        "possession_time_sec", "possession_pct", "touches",
    ]
    TIMELINE_CSV_FIELDS = [
        "start_frame", "end_frame", "start_time_sec", "end_time_sec",
        "duration_sec", "state", "track_id", "team",
    ]

    def __init__(
        self,
        json_path: str = "output/possession.json",
        csv_path: str = "output/possession.csv",
        timeline_csv_path: str = "output/possession_timeline.csv",
    ) -> None:
        self.json_path = json_path
        self.csv_path = csv_path
        self.timeline_csv_path = timeline_csv_path

    def save(self, analyzer: PossessionAnalyzer, source: str, fps: float) -> None:
        """Write JSON + team/player CSV + timeline CSV."""
        _ensure_parent(self.json_path)
        teams = analyzer.team_stats()
        players = analyzer.player_stats()
        timeline = analyzer.timeline()
        summary = analyzer.summary()

        payload = {
            "source": source,
            "fps": fps,
            "summary": summary,
            "teams": [t.to_dict() for t in teams],
            "players": [p.to_dict() for p in players],
            "timeline": [s.to_dict() for s in timeline],
            "frames": [f.to_dict() for f in analyzer.history],
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.TEAM_CSV_FIELDS)
            writer.writeheader()
            for t in teams:
                writer.writerow(t.to_dict())
            writer.writerow({})
            writer = csv.DictWriter(f, fieldnames=self.PLAYER_CSV_FIELDS)
            writer.writeheader()
            for p in players:
                writer.writerow(p.to_dict())

        with open(self.timeline_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.TIMELINE_CSV_FIELDS)
            writer.writeheader()
            for seg in timeline:
                writer.writerow(seg.to_dict())


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
