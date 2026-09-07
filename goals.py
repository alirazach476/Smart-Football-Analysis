"""
Command-line entry point for automatic goal detection.

Detects ball crossing the goal line, attributes scorers, records timestamps,
and exports goal statistics and notifications.

Examples
--------
    python goals.py --coordinates output/birdseye_coordinates.json

Run `python goals.py --help` for all options.
"""

from __future__ import annotations

import argparse
import json

from src.goal_detector import GoalDetector
from src.goal_exporter import GoalExporter
from src.possession_analyzer import PossessionAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic football goal detection from bird's-eye coordinates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--coordinates",
        required=True,
        help="Path to birdseye_coordinates.json.",
    )
    parser.add_argument(
        "--json",
        dest="goals_json",
        default="output/goals.json",
        help="Goal events JSON output.",
    )
    parser.add_argument(
        "--csv",
        dest="goals_csv",
        default="output/goals.csv",
        help="Per-goal CSV output.",
    )
    parser.add_argument(
        "--summary",
        dest="summary_csv",
        default="output/goal_statistics.csv",
        help="Goal summary CSV.",
    )
    parser.add_argument(
        "--notifications",
        default="output/goal_notifications.txt",
        help="Goal notifications log.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Possession radius in metres (for scorer attribution).",
    )
    return parser.parse_args()


def analyze(
    coordinates_path: str,
    goals_json: str = "output/goals.json",
    goals_csv: str = "output/goals.csv",
    summary_csv: str = "output/goal_statistics.csv",
    notifications_path: str = "output/goal_notifications.txt",
    radius_m: float = 2.0,
) -> dict:
    """Load coordinates, detect goals, export results."""
    with open(coordinates_path, encoding="utf-8") as f:
        meta = json.load(f)
    fps = meta.get("fps", 25.0)
    source = meta.get("source", coordinates_path)
    frames = meta.get("frames", [])

    possession = PossessionAnalyzer(fps=fps, possession_radius_m=radius_m)
    for rec in frames:
        possession.update(rec)
    possession.finalize()

    detector = GoalDetector(fps=fps)
    detector.analyze(frames, possession)

    exporter = GoalExporter(
        json_path=goals_json,
        csv_path=goals_csv,
        summary_csv_path=summary_csv,
        notifications_path=notifications_path,
    )
    return exporter.save(detector, source, fps)


def main() -> None:
    args = parse_args()
    summary = analyze(
        coordinates_path=args.coordinates,
        goals_json=args.goals_json,
        goals_csv=args.goals_csv,
        summary_csv=args.summary,
        notifications_path=args.notifications,
        radius_m=args.radius,
    )

    score = summary["final_score"]
    teams = list(score.keys())
    print("\n[done] goal detection complete")
    print(f"       total goals     : {summary['total_goals']}")
    if teams:
        print(f"       final score     : {teams[0]} {score[teams[0]]} - {score[teams[1]]} {teams[1]}")
    print(f"       json            : {summary['json_path']}")
    print(f"       goals csv       : {summary['csv_path']}")
    print(f"       statistics csv  : {summary['summary_csv_path']}")
    print(f"       notifications   : {summary['notifications_path']}")


if __name__ == "__main__":
    main()
