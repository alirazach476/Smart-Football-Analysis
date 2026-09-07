"""
Command-line entry point for automatic highlight generation.

Detects goals, shots, fouls, cards, penalties and corners from bird's-eye
coordinates, extracts a clip around each event from the source video, and
merges them into a single `highlights.mp4`.

Examples
--------
    python highlights.py --source match.mp4 \
        --coordinates output/birdseye_coordinates.json

Run `python highlights.py --help` for all options.
"""

from __future__ import annotations

import argparse
import json

from src.event_detector import EventDetector
from src.goal_detector import GoalDetector
from src.highlight_generator import HighlightGenerator
from src.possession_analyzer import PossessionAnalyzer
from src.shot_detector import ShotDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic football highlight generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source match video (clips are extracted from this).",
    )
    parser.add_argument(
        "--coordinates",
        required=True,
        help="Path to birdseye_coordinates.json.",
    )
    parser.add_argument(
        "--output",
        default="output/highlights.mp4",
        help="Merged highlight video output.",
    )
    parser.add_argument(
        "--timeline",
        default="output/highlights.json",
        help="Highlight timeline JSON output.",
    )
    parser.add_argument(
        "--pre", type=float, default=4.0,
        help="Seconds of footage before each event.",
    )
    parser.add_argument(
        "--post", type=float, default=3.0,
        help="Seconds of footage after each event.",
    )
    parser.add_argument(
        "--no-fouls", action="store_true", help="Skip foul detection.",
    )
    parser.add_argument(
        "--no-cards", action="store_true", help="Skip card detection.",
    )
    parser.add_argument(
        "--no-shots", action="store_true", help="Exclude shots from highlights.",
    )
    parser.add_argument(
        "--radius", type=float, default=2.0,
        help="Possession radius in metres (for goals/shots attribution).",
    )
    return parser.parse_args()


def generate(
    source: str,
    coordinates_path: str,
    output: str = "output/highlights.mp4",
    timeline_json: str = "output/highlights.json",
    pre_seconds: float = 4.0,
    post_seconds: float = 3.0,
    detect_fouls: bool = True,
    detect_cards: bool = True,
    include_shots: bool = True,
    radius_m: float = 2.0,
) -> dict:
    """Detect events and build the highlight reel."""
    with open(coordinates_path, encoding="utf-8") as f:
        meta = json.load(f)
    fps = meta.get("fps", 25.0)
    frames = meta.get("frames", [])

    possession = PossessionAnalyzer(fps=fps, possession_radius_m=radius_m)
    for rec in frames:
        possession.update(rec)
    possession.finalize()

    goal_detector = GoalDetector(fps=fps)
    goals = goal_detector.analyze(frames, possession)

    shots = []
    if include_shots:
        shot_detector = ShotDetector(fps=fps)
        shots = shot_detector.analyze(frames, possession)

    event_detector = EventDetector(
        fps=fps, detect_fouls=detect_fouls, detect_cards=detect_cards
    )
    events = event_detector.analyze(frames)

    generator = HighlightGenerator(
        source=source,
        output_path=output,
        timeline_json=timeline_json,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
    )
    generator.add_goals(goals)
    if include_shots:
        generator.add_shots(shots)
    generator.add_events(events)

    result = generator.generate()
    result["goals"] = len(goals)
    result["shots"] = len(shots)
    result["event_counts"] = event_detector.counts()
    return result


def main() -> None:
    args = parse_args()
    result = generate(
        source=args.source,
        coordinates_path=args.coordinates,
        output=args.output,
        timeline_json=args.timeline,
        pre_seconds=args.pre,
        post_seconds=args.post,
        detect_fouls=not args.no_fouls,
        detect_cards=not args.no_cards,
        include_shots=not args.no_shots,
        radius_m=args.radius,
    )

    print("\n[done] highlight generation complete")
    print(f"       goals detected  : {result['goals']}")
    print(f"       shots detected  : {result['shots']}")
    for etype, count in result.get("event_counts", {}).items():
        print(f"       {etype:15s} : {count}")
    print(f"       total events    : {result['events']}")
    print(f"       clips merged    : {result['clips']}")
    print(f"       duration        : {result['duration_sec']}s")
    print(f"       highlights      : {result['output_path']}")
    print(f"       timeline json   : {result['timeline_json']}")


if __name__ == "__main__":
    main()
