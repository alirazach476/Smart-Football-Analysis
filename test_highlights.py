"""Smoke test for highlight generation with a synthetic video + coordinates."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import cv2

import src.field_config as fc
from src.event_detector import EventDetector, EventType
from src.goal_detector import GoalDetector
from src.highlight_generator import HighlightGenerator
from src.possession_analyzer import PossessionAnalyzer
from src.shot_detector import ShotDetector


W = fc.PITCH_WIDTH
L = fc.PITCH_LENGTH


def _frame(frame: int, players: list[dict], ball: dict, fps: float = 25.0) -> dict:
    return {
        "frame": frame,
        "timestamp_sec": round(frame / fps, 3),
        "homography_available": True,
        "players": players,
        "ball": ball,
    }


def build_frames(fps: float = 25.0) -> list[dict]:
    """Synthetic sequence with a corner, a penalty spot placement, and a goal."""
    frames: list[dict] = []
    f = 0

    def teamA(x, y, tid=1):
        return {"track_id": tid, "class_name": "player", "team": "Team A", "x_cm": x, "y_cm": y}

    def teamB(x, y, tid=99):
        return {"track_id": tid, "class_name": "player", "team": "Team B", "x_cm": x, "y_cm": y}

    def ref(x, y):
        return {"track_id": 200, "class_name": "referee", "team": None, "x_cm": x, "y_cm": y}

    # Corner: ball sits in the top-right corner arc, stationary.
    for _ in range(12):
        frames.append(_frame(f, [teamA(11500, 500), teamB(11000, 800)],
                             {"x_cm": L - 150, "y_cm": 150}, fps))
        f += 1

    # Gap.
    for _ in range(30):
        frames.append(_frame(f, [teamA(6000, W / 2), teamB(6500, W / 2)],
                             {"x_cm": 6000, "y_cm": W / 2}, fps))
        f += 1

    # Penalty: ball on the right penalty spot, stationary.
    spot_x = L - fc.PENALTY_SPOT_DISTANCE
    for _ in range(12):
        frames.append(_frame(f, [teamA(spot_x - 100, W / 2), teamB(200, W / 2)],
                             {"x_cm": spot_x, "y_cm": W / 2}, fps))
        f += 1

    # Gap.
    for _ in range(30):
        frames.append(_frame(f, [teamA(6000, W / 2), teamB(6500, W / 2)],
                             {"x_cm": 6000, "y_cm": W / 2}, fps))
        f += 1

    # Team A possession then goal into right goal.
    for _ in range(6):
        frames.append(_frame(f, [teamA(10000, W / 2), teamB(11000, 2000)],
                             {"x_cm": 10020, "y_cm": W / 2}, fps))
        f += 1
    bx = L - 800
    for _ in range(8):
        bx = min(bx + 140, L + 50)
        frames.append(_frame(f, [teamA(10000, W / 2), teamB(11000, 2000)],
                             {"x_cm": bx, "y_cm": W / 2}, fps))
        f += 1

    return frames


def write_dummy_video(path: str, num_frames: int, fps: float = 25.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = 320, 180
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(num_frames):
        frame = np.full((h, w, 3), (30, 60, 30), dtype=np.uint8)
        cv2.putText(frame, f"f{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def main() -> None:
    fps = 25.0
    frames = build_frames(fps)
    total_frames = frames[-1]["frame"] + 30

    possession = PossessionAnalyzer(fps=fps, possession_radius_m=2.5)
    for rec in frames:
        possession.update(rec)
    possession.finalize()

    goals = GoalDetector(fps=fps).analyze(frames, possession)
    shots = ShotDetector(fps=fps).analyze(frames, possession)
    events = EventDetector(fps=fps).analyze(frames)

    event_types = {e.type for e in events}
    assert EventType.CORNER.value in event_types, f"no corner: {event_types}"
    assert EventType.PENALTY.value in event_types, f"no penalty: {event_types}"
    assert len(goals) >= 1, "no goal detected"

    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, "match.mp4")
        write_dummy_video(video, total_frames, fps)

        out = os.path.join(tmp, "highlights.mp4")
        timeline = os.path.join(tmp, "highlights.json")
        gen = HighlightGenerator(
            source=video, output_path=out, timeline_json=timeline,
            pre_seconds=1.0, post_seconds=1.0,
        )
        gen.add_goals(goals)
        gen.add_shots(shots)
        gen.add_events(events)
        result = gen.generate()

        assert os.path.exists(out), "highlights.mp4 not created"
        assert os.path.getsize(out) > 0, "highlights.mp4 is empty"
        assert result["clips"] >= 1
        assert os.path.exists(timeline)
        with open(timeline, encoding="utf-8") as fp:
            payload = json.load(fp)
        assert payload["event_count"] >= 3

    counts = {}
    for e in events:
        counts[e.type] = counts.get(e.type, 0) + 1
    detail = ", ".join(f"{k}:{v}" for k, v in counts.items())
    print(
        f"[ok] highlight generation smoke test passed "
        f"({len(goals)} goals, {len(shots)} shots, events: {detail}, "
        f"{result['clips']} clips)"
    )


if __name__ == "__main__":
    main()
