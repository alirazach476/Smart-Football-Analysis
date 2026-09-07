"""
Command-line entry point for football heatmap generation.

Builds player, team, ball, zone-control and pressure heatmaps from
bird's-eye transformed coordinates. Can run standalone on a saved
coordinates JSON or is auto-invoked by `birdseye.py --heatmaps`.

Examples
--------
    # From saved bird's-eye coordinates
    python heatmaps.py --coordinates output/birdseye_coordinates.json

    # Custom output paths
    python heatmaps.py --coordinates output/birdseye_coordinates.json \\
        --output-dir output/heatmaps --html output/heatmaps.html

Run `python heatmaps.py --help` for all options.
"""

from __future__ import annotations

import argparse
import json

from src.heatmap_collector import HeatmapCollector
from src.heatmap_exporter import HeatmapExporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate football heatmaps from bird's-eye coordinates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--coordinates",
        required=True,
        help="Path to birdseye_coordinates.json (transformed pitch coords).",
    )
    parser.add_argument(
        "--output-dir",
        default="output/heatmaps",
        help="Directory for PNG heatmap images.",
    )
    parser.add_argument(
        "--json",
        dest="heatmap_json",
        default="output/heatmap_data.json",
        help="Path for exported heatmap grid data JSON.",
    )
    parser.add_argument(
        "--html",
        dest="heatmap_html",
        default="output/heatmaps.html",
        help="Path for interactive HTML dashboard.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.06,
        help="Pitch render scale (pixels per cm) for PNG output.",
    )
    return parser.parse_args()


def generate(
    coordinates_path: str,
    output_dir: str = "output/heatmaps",
    heatmap_json: str = "output/heatmap_data.json",
    heatmap_html: str = "output/heatmaps.html",
    pitch_scale: float = 0.06,
) -> dict:
    """Programmatic entry: load coordinates, build and export all heatmaps."""
    with open(coordinates_path, encoding="utf-8") as f:
        meta = json.load(f)
    source = meta.get("source", coordinates_path)

    collector = HeatmapCollector()
    data = collector.load_from_json(coordinates_path)

    exporter = HeatmapExporter(
        output_dir=output_dir,
        json_path=heatmap_json,
        html_path=heatmap_html,
        pitch_scale=pitch_scale,
    )
    return exporter.export(data, source=source)


def main() -> None:
    args = parse_args()
    summary = generate(
        coordinates_path=args.coordinates,
        output_dir=args.output_dir,
        heatmap_json=args.heatmap_json,
        heatmap_html=args.heatmap_html,
        pitch_scale=args.scale,
    )

    print("\n[done] heatmap generation complete")
    print(f"       frames used     : {summary['frames_processed']}")
    print(f"       players         : {summary['players']}")
    print(f"       png images      : {summary['png_count']} in {summary['output_dir']}")
    print(f"       heatmap json    : {summary['json_path']}")
    print(f"       interactive html: {summary['html_path']}")


if __name__ == "__main__":
    main()
