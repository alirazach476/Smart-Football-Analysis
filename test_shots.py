"""Smoke test for shot detection with synthetic bird's-eye coordinates."""

from __future__ import annotations

import json
import os
import tempfile

from src.possession_analyzer import PossessionAnalyzer
from src.shot_detector import ShotDetector, ShotOutcome
from src.shot_exporter import ShotExporter
import src.field_config as fc


def _frame(frame: int, players: list[dict], ball: dict, fps: float = 25.0) -> dict:
    return {
        "frame": frame,
        "timestamp_sec": round(frame / fps, 3),
        "homography_available": True,
        "players": players,
        "ball": ball,
    }


def build_synthetic_shots(fps: float = 25.0) -> list[dict]:
    """
    Team A attacks the right goal (x=L). Build three attempts:
      1. on target  - straight at the goal mouth centre
      2. off target - aimed wide of the post
      3. blocked    - defender sits on the ball path short of goal
    """
    L = fc.PITCH_LENGTH
    W = fc.PITCH_WIDTH
    frames: list[dict] = []

    # Establish attack direction: Team A players live in the left half.
    def teamA(x, y, tid=1):
        return {"track_id": tid, "class_name": "player", "team": "Team A",
                "x_cm": x, "y_cm": y}

    def teamB(x, y, tid=99):
        return {"track_id": tid, "class_name": "player", "team": "Team B",
                "x_cm": x, "y_cm": y}

    # Realistic per-frame ball displacement: ~140 cm/frame at 25 fps ~= 35 m/s.
    step = 140
    f = 0

    # Warm-up possession frames so attack directions resolve.
    for _ in range(6):
        frames.append(_frame(f, [teamA(3000, W / 2), teamB(9000, W / 2)],
                             {"x_cm": 3030, "y_cm": W / 2}, fps))
        f += 1

    # --- Shot 1: on target, straight to goal centre -----------------------
    for i in range(9):
        bx = min(9000 + i * step, L)
        frames.append(_frame(f, [teamA(9000, W / 2), teamB(11500, 2000)],
                             {"x_cm": bx, "y_cm": W / 2}, fps))
        f += 1
    for _ in range(10):
        frames.append(_frame(f, [teamA(3000, W / 2), teamB(9000, W / 2)],
                             {"x_cm": 3000, "y_cm": W / 2}, fps))
        f += 1

    # --- Shot 2: off target, drifting well wide of the post ----------------
    for i in range(9):
        bx = min(9000 + i * step, L)
        by = max(1600 - i * 60, 0)  # heading toward the corner/out
        frames.append(_frame(f, [teamA(9000, 1600), teamB(11500, 5000)],
                             {"x_cm": bx, "y_cm": by}, fps))
        f += 1
    for _ in range(10):
        frames.append(_frame(f, [teamA(3000, W / 2), teamB(9000, W / 2)],
                             {"x_cm": 3000, "y_cm": W / 2}, fps))
        f += 1

    # --- Shot 3: blocked - defender on the ball path short of goal --------
    for i in range(5):
        bx = 9000 + i * step  # blocked around x=9560 by defender
        frames.append(_frame(f, [teamA(9000, W / 2), teamB(9560, W / 2)],
                             {"x_cm": min(bx, 9560), "y_cm": W / 2}, fps))
        f += 1

    return frames


def main() -> None:
    fps = 25.0
    frames = build_synthetic_shots(fps)

    possession = PossessionAnalyzer(fps=fps, possession_radius_m=2.5)
    for rec in frames:
        possession.update(rec)
    possession.finalize()

    detector = ShotDetector(fps=fps)
    shots = detector.analyze(frames, possession)

    assert len(shots) >= 3, f"expected >=3 shots, got {len(shots)}"
    outcomes = [s.outcome for s in shots]
    assert ShotOutcome.ON_TARGET.value in outcomes, f"no on-target shot: {outcomes}"
    assert ShotOutcome.OFF_TARGET.value in outcomes, f"no off-target shot: {outcomes}"
    assert ShotOutcome.BLOCKED.value in outcomes, f"no blocked shot: {outcomes}"

    for s in shots:
        assert s.speed_kmh > 0
        assert 0 <= s.distance_to_goal_m <= 60
        assert s.target_goal in ("left", "right")

    with tempfile.TemporaryDirectory() as tmp:
        exporter = ShotExporter(
            json_path=os.path.join(tmp, "shots.json"),
            csv_path=os.path.join(tmp, "shots.csv"),
            summary_csv_path=os.path.join(tmp, "shot_statistics.csv"),
            shot_map_dir=os.path.join(tmp, "maps"),
        )
        summary = exporter.save(detector, "synthetic", fps)
        assert summary["total_shots"] == len(shots)
        assert os.path.exists(summary["json_path"])
        with open(summary["json_path"], encoding="utf-8") as fp:
            payload = json.load(fp)
        assert "shots" in payload and "summary" in payload
        for p in summary["shot_map_paths"]:
            assert os.path.exists(p)

    detail = ", ".join(f"{s.outcome}@{s.speed_kmh:.0f}km/h" for s in shots)
    print(f"[ok] shot detection smoke test passed ({len(shots)} shots: {detail})")


if __name__ == "__main__":
    main()
