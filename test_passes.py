"""Smoke test for pass detection with synthetic bird's-eye coordinates."""

from __future__ import annotations

import json
import os
import tempfile

from src.pass_detector import PassDetector, PassType
from src.pass_exporter import PassExporter
from src.possession_analyzer import PossessionAnalyzer


def _frame(
    frame: int,
    players: list[dict],
    ball: dict,
    fps: float = 25.0,
) -> dict:
    return {
        "frame": frame,
        "timestamp_sec": round(frame / fps, 3),
        "homography_available": True,
        "players": players,
        "ball": ball,
    }


def build_synthetic_match(fps: float = 25.0) -> list[dict]:
    """Simulate a short sequence with successful and failed passes."""
    frames: list[dict] = []

    def add_segment(
        start: int,
        length: int,
        holder: int,
        team: str,
        px: float,
        py: float,
    ) -> None:
        for i in range(length):
            f = start + i
            frames.append(
                _frame(
                    f,
                    [
                        {"track_id": holder, "class_name": "player", "team": team,
                         "x_cm": px, "y_cm": py},
                        {"track_id": 99, "class_name": "player", "team": "Team B",
                         "x_cm": 9000, "y_cm": 3500},
                    ],
                    {"x_cm": px + 30, "y_cm": py + 10},
                    fps,
                )
            )

    # Team A player 1 holds ball, passes to player 2 (short successful).
    add_segment(0, 10, 1, "Team A", 4000, 3500)
    for i in range(5):
        bx = 4200 + i * 80
        frames.append(
            _frame(
                10 + i,
                [
                    {"track_id": 1, "class_name": "player", "team": "Team A",
                     "x_cm": 4100, "y_cm": 3500},
                    {"track_id": 2, "class_name": "player", "team": "Team A",
                     "x_cm": bx, "y_cm": 3600},
                ],
                {"x_cm": bx, "y_cm": 3600},
                fps,
            )
        )
    add_segment(15, 10, 2, "Team A", 4600, 3600)

    # Long forward pass from player 2 to player 3 in final third (through ball).
    add_segment(25, 8, 2, "Team A", 5000, 2000)
    for i in range(6):
        bx = 5500 + i * 200
        frames.append(
            _frame(
                33 + i,
                [
                    {"track_id": 2, "class_name": "player", "team": "Team A",
                     "x_cm": 5200, "y_cm": 2000},
                    {"track_id": 3, "class_name": "player", "team": "Team A",
                     "x_cm": bx, "y_cm": 2500},
                ],
                {"x_cm": bx, "y_cm": 2500},
                fps,
            )
        )
    add_segment(39, 10, 3, "Team A", 9000, 3000)

    # Back pass player 3 -> player 1.
    add_segment(49, 8, 3, "Team A", 8500, 3500)
    for i in range(5):
        bx = 8000 - i * 100
        frames.append(
            _frame(
                57 + i,
                [
                    {"track_id": 3, "class_name": "player", "team": "Team A",
                     "x_cm": 8400, "y_cm": 3500},
                    {"track_id": 1, "class_name": "player", "team": "Team A",
                     "x_cm": bx, "y_cm": 3400},
                ],
                {"x_cm": bx, "y_cm": 3400},
                fps,
            )
        )
    add_segment(62, 8, 1, "Team A", 7500, 3400)

    # Failed pass: Team A player 1 -> loose -> Team B intercept.
    add_segment(70, 8, 1, "Team A", 6000, 1000)
    for i in range(20):
        frames.append(
            _frame(
                78 + i,
                [
                    {"track_id": 1, "class_name": "player", "team": "Team A",
                     "x_cm": 5800, "y_cm": 1000},
                    {"track_id": 99, "class_name": "player", "team": "Team B",
                     "x_cm": 7000, "y_cm": 1200},
                ],
                {"x_cm": 6500 + i * 30, "y_cm": 1100},
                fps,
            )
        )
    add_segment(98, 10, 99, "Team B", 7200, 1200)

    return frames


def main() -> None:
    fps = 25.0
    frames = build_synthetic_match(fps)

    possession = PossessionAnalyzer(fps=fps, possession_radius_m=2.5)
    for rec in frames:
        possession.update(rec)
    possession.finalize()

    detector = PassDetector(fps=fps)
    passes = detector.analyze(frames, possession)

    assert len(passes) >= 3, f"expected >=3 passes, got {len(passes)}"
    types_found = {p.primary_type for p in passes}
    assert PassType.SHORT.value in types_found or PassType.LONG.value in types_found
    assert any(p.successful for p in passes)
    assert any(not p.successful for p in passes)

    with tempfile.TemporaryDirectory() as tmp:
        exporter = PassExporter(
            json_path=os.path.join(tmp, "passes.json"),
            csv_path=os.path.join(tmp, "passes.csv"),
            summary_csv_path=os.path.join(tmp, "pass_statistics.csv"),
            network_dir=os.path.join(tmp, "networks"),
        )
        summary = exporter.save(detector, "synthetic", fps)
        assert summary["total_passes"] == len(passes)
        assert os.path.exists(summary["json_path"])
        with open(summary["json_path"], encoding="utf-8") as f:
            payload = json.load(f)
        assert "passes" in payload
        assert "summary" in payload

    print(f"[ok] pass detection smoke test passed ({len(passes)} passes)")


if __name__ == "__main__":
    main()
