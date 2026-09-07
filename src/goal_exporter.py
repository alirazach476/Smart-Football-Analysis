"""
Goal statistics exporter, live scoreboard overlay, and goal notifications.

Writes goal events JSON, per-goal CSV, summary CSV, and a notifications log.
Provides video overlays for the live scoreboard and goal alerts.
"""

from __future__ import annotations

import csv
import json
import os

import cv2

from .goal_detector import GoalDetector, GoalEvent


def draw_goal_overlay(
    frame: np.ndarray,
    detector: GoalDetector,
    team_names: tuple[str, str] = ("Team A", "Team B"),
) -> np.ndarray:
    """
    Draw live scoreboard (top-right) and a goal notification banner when active.
    """
    h, w = frame.shape[:2]
    score = detector.score
    ta, tb = team_names

    # --- Scoreboard (top-right) ------------------------------------------
    label = f"{ta}  {score.get(ta, 0)}  -  {score.get(tb, 0)}  {tb}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    pad = 12
    bx1 = w - tw - pad * 2 - 8
    by1 = 8
    bx2 = w - 8
    by2 = by1 + th + pad * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
    cv2.putText(
        frame, label, (bx1 + pad, by2 - pad),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )

    # --- Goal notification banner ----------------------------------------
    notif = detector.notification
    if notif is None:
        return frame

    banner_h = 56
    cy = h // 2
    overlay = frame.copy()
    cv2.rectangle(
        overlay, (0, cy - banner_h // 2), (w, cy + banner_h // 2),
        (0, 180, 0), -1,
    )
    frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)

    cv2.putText(
        frame, "GOAL!", (w // 2 - 80, cy - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA,
    )
    cv2.putText(
        frame, notif.message, (w // 2 - min(len(notif.message) * 7, w // 2 - 20), cy + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return frame


class GoalExporter:
    """Export goal statistics and notifications."""

    GOAL_CSV_FIELDS = [
        "goal_id", "scoring_team", "conceded_team", "scorer_track_id",
        "goal_line", "cross_y_cm", "frame", "timestamp_sec", "speed_ms",
        "score_team_a", "score_team_b", "scoreline",
    ]
    SUMMARY_CSV_FIELDS = [
        "team", "goals", "first_goal_sec", "last_goal_sec",
    ]

    def __init__(
        self,
        json_path: str = "output/goals.json",
        csv_path: str = "output/goals.csv",
        summary_csv_path: str = "output/goal_statistics.csv",
        notifications_path: str = "output/goal_notifications.txt",
    ) -> None:
        self.json_path = json_path
        self.csv_path = csv_path
        self.summary_csv_path = summary_csv_path
        self.notifications_path = notifications_path

    def save(
        self,
        detector: GoalDetector,
        source: str,
        fps: float,
        team_names: tuple[str, str] = ("Team A", "Team B"),
    ) -> dict:
        """Write JSON, CSVs, and notifications log."""
        _ensure_parent(self.json_path)

        goals = detector.goals
        stats = detector.stats()
        score = detector.score

        notifications = [
            {
                "goal_id": g.goal_id,
                "timestamp_sec": g.timestamp_sec,
                "message": (
                    f"GOAL! {g.scoring_team} — "
                    f"{'#' + str(g.scorer_track_id) if g.scorer_track_id else 'Unknown'} "
                    f"({g.score_team_a}-{g.score_team_b}) @ {g.timestamp_sec:.1f}s"
                ),
            }
            for g in goals
        ]

        payload = {
            "source": source,
            "fps": fps,
            "final_score": {
                team_names[0]: score.get(team_names[0], 0),
                team_names[1]: score.get(team_names[1], 0),
            },
            "summary": stats.to_dict(),
            "notifications": notifications,
            "goals": [g.to_dict() for g in goals],
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.GOAL_CSV_FIELDS)
            writer.writeheader()
            for g in goals:
                writer.writerow(g.to_dict())

        with open(self.summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SUMMARY_CSV_FIELDS)
            writer.writeheader()
            for team in team_names:
                team_goals = [g for g in goals if g.scoring_team == team]
                first = min((g.timestamp_sec for g in team_goals), default=None)
                last = max((g.timestamp_sec for g in team_goals), default=None)
                writer.writerow({
                    "team": team,
                    "goals": score.get(team, 0),
                    "first_goal_sec": round(first, 3) if first is not None else "",
                    "last_goal_sec": round(last, 3) if last is not None else "",
                })

        with open(self.notifications_path, "w", encoding="utf-8") as f:
            for n in notifications:
                f.write(f"[{n['timestamp_sec']:.1f}s] {n['message']}\n")

        return {
            "json_path": self.json_path,
            "csv_path": self.csv_path,
            "summary_csv_path": self.summary_csv_path,
            "notifications_path": self.notifications_path,
            "total_goals": stats.total,
            "final_score": payload["final_score"],
        }


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
