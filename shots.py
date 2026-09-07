"""
Command-line entry point for automatic shot detection.

Detects shots and classifies them as on target, off target, or blocked from
bird's-eye coordinates, estimates shot speed and location, then exports
statistics and a shot map visualization.

Examples
--------
    python shots.py --coordinates output/birdseye_coordinates.json

Run `python shots.py --help` for all options.
"""

from __future__ import annotations

import argparse
import json

from src.possession_analyzer import PossessionAnalyzer
from src.shot_detector import ShotDetector
from src.shot_exporter import ShotExporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic football shot detection from bird's-eye coordinates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--coordinates",
        required=True,
        help="Path to birdseye_coordinates.json.",
    )
    parser.add_argument(
        "--json",
        dest="shots_json",
        default="output/shots.json",
        help="Shot events JSON output.",
    )
    parser.add_argument(
        "--csv",
        dest="shots_csv",
        default="output/shots.csv",
        help="Per-shot CSV output.",
    )
    parser.add_argument(
        "--summary",
        dest="summary_csv",
        default="output/shot_statistics.csv",
        help="Team shot summary CSV.",
    )
    parser.add_argument(
        "--map-dir",
        default="output/shot_maps",
        help="Directory for shot map PNG images.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.06,
        help="Pitch scale for shot map images (pixels per cm).",
    )
    parser.add_argument(
        "--min-speed",
        type=float,
        default=6.0,
        help="Minimum ball speed (m/s) toward goal to count as a shot.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=35.0,
        help="Maximum shot distance from goal (m).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Possession radius in metres (for shooter attribution).",
    )
    return parser.parse_args()


def analyze(
    coordinates_path: str,
    shots_json: str = "output/shots.json",
    shots_csv: str = "output/shots.csv",
    summary_csv: str = "output/shot_statistics.csv",
    map_dir: str = "output/shot_maps",
    pitch_scale: float = 0.06,
    min_speed_ms: float = 6.0,
    max_distance_m: float = 35.0,
    radius_m: float = 2.0,
) -> dict:
    """Load coordinates, detect shots, export results."""
    with open(coordinates_path, encoding="utf-8") as f:
        meta = json.load(f)
    fps = meta.get("fps", 25.0)
    source = meta.get("source", coordinates_path)
    frames = meta.get("frames", [])

    possession = PossessionAnalyzer(fps=fps, possession_radius_m=radius_m)
    for rec in frames:
        possession.update(rec)
    possession.finalize()

    detector = ShotDetector(
        fps=fps,
        min_shot_speed_ms=min_speed_ms,
        max_shot_distance_m=max_distance_m,
    )
    detector.analyze(frames, possession)

    exporter = ShotExporter(
        json_path=shots_json,
        csv_path=shots_csv,
        summary_csv_path=summary_csv,
        shot_map_dir=map_dir,
        pitch_scale=pitch_scale,
    )
    return exporter.save(detector, source, fps)


def main() -> None:
    args = parse_args()
    summary = analyze(
        coordinates_path=args.coordinates,
        shots_json=args.shots_json,
        shots_csv=args.shots_csv,
        summary_csv=args.summary,
        map_dir=args.map_dir,
        pitch_scale=args.scale,
        min_speed_ms=args.min_speed,
        max_distance_m=args.max_distance,
        radius_m=args.radius,
    )

    print("\n[done] shot detection complete")
    print(f"       total shots     : {summary['total_shots']}")
    print(f"       on target       : {summary['on_target']}")
    print(f"       accuracy        : {summary['accuracy_pct']:.1f}%")
    print(f"       json            : {summary['json_path']}")
    print(f"       shots csv       : {summary['csv_path']}")
    print(f"       statistics csv  : {summary['summary_csv_path']}")
    print(f"       shot maps       : {summary['shot_map_dir']}")
    for p in summary.get("shot_map_paths", []):
        print(f"         - {p}")


if __name__ == "__main__":
    main()
