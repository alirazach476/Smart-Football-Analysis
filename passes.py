"""
Command-line entry point for automatic pass detection.

Detects successful/unsuccessful passes and classifies them (short, long,
through ball, cross, back) from bird's-eye coordinates, then exports statistics
and pass network visualizations.

Examples
--------
    python passes.py --coordinates output/birdseye_coordinates.json

Run `python passes.py --help` for all options.
"""

from __future__ import annotations

import argparse
import json

from src.pass_detector import PassDetector
from src.pass_exporter import PassExporter
from src.possession_analyzer import PossessionAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic football pass detection from bird's-eye coordinates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--coordinates",
        required=True,
        help="Path to birdseye_coordinates.json.",
    )
    parser.add_argument(
        "--json",
        dest="passes_json",
        default="output/passes.json",
        help="Pass events JSON output.",
    )
    parser.add_argument(
        "--csv",
        dest="passes_csv",
        default="output/passes.csv",
        help="Per-pass CSV output.",
    )
    parser.add_argument(
        "--summary",
        dest="summary_csv",
        default="output/pass_statistics.csv",
        help="Team pass summary CSV.",
    )
    parser.add_argument(
        "--network-dir",
        default="output/pass_networks",
        help="Directory for pass network PNG images.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.06,
        help="Pitch scale for network images (pixels per cm).",
    )
    parser.add_argument(
        "--possession-json",
        default=None,
        help="Optional possession.json to reuse (skips re-computing possession).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Possession radius in metres (when possession is computed).",
    )
    return parser.parse_args()


def analyze(
    coordinates_path: str,
    passes_json: str = "output/passes.json",
    passes_csv: str = "output/passes.csv",
    summary_csv: str = "output/pass_statistics.csv",
    network_dir: str = "output/pass_networks",
    pitch_scale: float = 0.06,
    possession_json: str | None = None,
    radius_m: float = 2.0,
) -> dict:
    """Load coordinates, detect passes, export results."""
    with open(coordinates_path, encoding="utf-8") as f:
        meta = json.load(f)
    fps = meta.get("fps", 25.0)
    source = meta.get("source", coordinates_path)
    frames = meta.get("frames", [])

    possession = PossessionAnalyzer(fps=fps, possession_radius_m=radius_m)
    if possession_json:
        possession.load_from_json(coordinates_path)
    else:
        for rec in frames:
            possession.update(rec)
        possession.finalize()

    detector = PassDetector(fps=fps)
    detector.analyze(frames, possession)

    exporter = PassExporter(
        json_path=passes_json,
        csv_path=passes_csv,
        summary_csv_path=summary_csv,
        network_dir=network_dir,
        pitch_scale=pitch_scale,
    )
    return exporter.save(detector, source, fps)


def main() -> None:
    args = parse_args()
    summary = analyze(
        coordinates_path=args.coordinates,
        passes_json=args.passes_json,
        passes_csv=args.passes_csv,
        summary_csv=args.summary,
        network_dir=args.network_dir,
        pitch_scale=args.scale,
        possession_json=args.possession_json,
        radius_m=args.radius,
    )

    print("\n[done] pass detection complete")
    print(f"       total passes    : {summary['total_passes']}")
    print(f"       success rate    : {summary['success_rate_pct']:.1f}%")
    print(f"       json            : {summary['json_path']}")
    print(f"       passes csv      : {summary['csv_path']}")
    print(f"       statistics csv  : {summary['summary_csv_path']}")
    print(f"       network images  : {summary['network_dir']}")
    for p in summary.get("network_paths", []):
        print(f"         - {p}")


if __name__ == "__main__":
    main()
