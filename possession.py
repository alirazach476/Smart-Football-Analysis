"""
Command-line entry point for possession analysis from bird's-eye coordinates.

Analyzes ball possession from transformed pitch coordinates: nearest player
detection, team percentages, player possession time, and timeline export.

Examples
--------
    python possession.py --coordinates output/birdseye_coordinates.json

Run `python possession.py --help` for all options.
"""

from __future__ import annotations

import argparse
import json

from src.possession_analyzer import PossessionAnalyzer
from src.possession_exporter import PossessionExporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Football possession analysis from bird's-eye coordinates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--coordinates",
        required=True,
        help="Path to birdseye_coordinates.json.",
    )
    parser.add_argument(
        "--json",
        dest="possession_json",
        default="output/possession.json",
        help="Possession statistics JSON output.",
    )
    parser.add_argument(
        "--csv",
        dest="possession_csv",
        default="output/possession.csv",
        help="Team/player summary CSV.",
    )
    parser.add_argument(
        "--timeline",
        dest="timeline_csv",
        default="output/possession_timeline.csv",
        help="Possession timeline CSV.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Possession radius in metres.",
    )
    return parser.parse_args()


def analyze(
    coordinates_path: str,
    possession_json: str = "output/possession.json",
    possession_csv: str = "output/possession.csv",
    timeline_csv: str = "output/possession_timeline.csv",
    radius_m: float = 2.0,
) -> dict:
    """Load coordinates, run possession analysis, export results."""
    with open(coordinates_path, encoding="utf-8") as f:
        meta = json.load(f)
    fps = meta.get("fps", 25.0)
    source = meta.get("source", coordinates_path)

    analyzer = PossessionAnalyzer(fps=fps, possession_radius_m=radius_m)
    analyzer.load_from_json(coordinates_path)

    exporter = PossessionExporter(
        json_path=possession_json,
        csv_path=possession_csv,
        timeline_csv_path=timeline_csv,
    )
    exporter.save(analyzer, source, fps)
    return analyzer.summary()


def main() -> None:
    args = parse_args()
    summary = analyze(
        coordinates_path=args.coordinates,
        possession_json=args.possession_json,
        possession_csv=args.possession_csv,
        timeline_csv=args.timeline_csv,
        radius_m=args.radius,
    )

    print("\n[done] possession analysis complete")
    print(
        f"       Team A: {summary['team_a_pct']:.1f}% | "
        f"Team B: {summary['team_b_pct']:.1f}%"
    )
    print(f"       loose ball: {summary['loose_pct']:.1f}%")
    print(f"       contested:  {summary.get('contested_pct', 0):.1f}%")
    print(f"       json: {args.possession_json}")
    print(f"       csv:  {args.possession_csv}")
    print(f"       timeline: {args.timeline_csv}")


if __name__ == "__main__":
    main()
