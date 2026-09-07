"""Smoke test for goal detection with synthetic bird's-eye coordinates."""

from __future__ import annotations

import json
import os
import tempfile

import src.field_config as fc
from src.goal_detector import GoalDetector
from src.goal_exporter import GoalExporter
from src.possession_analyzer import PossessionAnalyzer


def _frame(frame: int, players: list[dict], ball: dict, fps: float = 25.0) -> dict:
    return {
        "frame": frame,
        "timestamp_sec": round(frame / fps, 3),
        "homography_available": True,
        "players": players,
        "ball": ball,
    }


def build_synthetic_goals(fps: float = 25.0) -> list[dict]:
    """
    Team A attacks right (x=L). Simulate:
      1. Team A goal — ball crosses right goal line at centre
      2. Team B goal — ball crosses left goal line at centre
    """
    L = fc.PITCH_LENGTH
    W = fc.PITCH_WIDTH
    frames: list[dict] = []

    def teamA(x, y, tid=1):
        return {"track_id": tid, "class_name": "player", "team": "Team A",
                "x_cm": x, "y_cm": y}

    def teamB(x, y, tid=99):
        return {"track_id": tid, "class_name": "player", "team": "Team B",
                "x_cm": x, "y_cm": y}

    f = 0
    step = 140  # ~35 m/s at 25 fps

    # Warm-up: Team A left half, Team B right half.
    for _ in range(8):
        frames.append(_frame(f, [teamA(3000, W / 2), teamB(9000, W / 2)],
                             {"x_cm": 3030, "y_cm": W / 2}, fps))
        f += 1

    # Team A possession before goal.
    for _ in range(6):
        frames.append(_frame(f, [teamA(10000, W / 2), teamB(11000, 2000)],
                             {"x_cm": 10020, "y_cm": W / 2}, fps))
        f += 1

    # Ball crosses right goal line (Team A scores).
    bx = L - 800
    for i in range(8):
        bx = min(bx + step, L + 50)
        frames.append(_frame(f, [teamA(10000, W / 2), teamB(11000, 2000)],
                             {"x_cm": bx, "y_cm": W / 2}, fps))
        f += 1

    # Gap between goals.
    for _ in range(160):
        frames.append(_frame(f, [teamA(3000, W / 2), teamB(9000, W / 2)],
                             {"x_cm": 3000, "y_cm": W / 2}, fps))
        f += 1

    # Team B possession before goal.
    for _ in range(6):
        frames.append(_frame(f, [teamA(2000, W / 2), teamB(1500, W / 2)],
                             {"x_cm": 1480, "y_cm": W / 2}, fps))
        f += 1

    # Ball crosses left goal line (Team B scores).
    bx = 800
    for i in range(8):
        bx = max(bx - step, -50)
        frames.append(_frame(f, [teamA(2000, W / 2), teamB(1500, W / 2)],
                             {"x_cm": bx, "y_cm": W / 2}, fps))
        f += 1

    return frames


def main() -> None:
    fps = 25.0
    frames = build_synthetic_goals(fps)

    possession = PossessionAnalyzer(fps=fps, possession_radius_m=2.5)
    for rec in frames:
        possession.update(rec)
    possession.finalize()

    detector = GoalDetector(fps=fps, min_goal_gap_frames=100)
    goals = detector.analyze(frames, possession)

    assert len(goals) >= 2, f"expected >=2 goals, got {len(goals)}"
    assert detector.score["Team A"] >= 1
    assert detector.score["Team B"] >= 1
    assert goals[0].scorer_track_id is not None or goals[0].scoring_team

    with tempfile.TemporaryDirectory() as tmp:
        exporter = GoalExporter(
            json_path=os.path.join(tmp, "goals.json"),
            csv_path=os.path.join(tmp, "goals.csv"),
            summary_csv_path=os.path.join(tmp, "goal_statistics.csv"),
            notifications_path=os.path.join(tmp, "goal_notifications.txt"),
        )
        summary = exporter.save(detector, "synthetic", fps)
        assert summary["total_goals"] == len(goals)
        assert os.path.exists(summary["notifications_path"])
        with open(summary["json_path"], encoding="utf-8") as fp:
            payload = json.load(fp)
        assert "goals" in payload and "notifications" in payload

    print(
        f"[ok] goal detection smoke test passed "
        f"({len(goals)} goals, score {detector.score['Team A']}-{detector.score['Team B']})"
    )


if __name__ == "__main__":
    main()
