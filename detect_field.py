"""
Command-line entry point for the football field (pitch) detection pipeline.

Detects the 32 pitch keypoints with the trained YOLOv8 pose model, draws the
sidelines, goal lines, penalty boxes, goal boxes, halfway line and centre
circle, saves the visualization, and exports the field coordinates to JSON.

Examples
--------
    # Single image
    python detect_field.py --source pitch.jpg

    # Match video
    python detect_field.py --source match.mp4 --output output/field.mp4

Run `python detect_field.py --help` for all options.
"""

from __future__ import annotations

import argparse

from src.field_pipeline import FieldPipeline

DEFAULT_MODEL = "football-field-detection/football-field-detection.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Football field detection pipeline (YOLOv8 pose + OpenCV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Path to the trained pose .pt model."
    )
    parser.add_argument(
        "--source", required=True, help="Input image or video path."
    )
    parser.add_argument(
        "--output",
        default="output/field_annotated.mp4",
        help="Annotated output path (image inputs auto-switch to .png).",
    )
    parser.add_argument(
        "--json",
        dest="coordinates_json",
        default="output/field_coordinates.json",
        help="Path for the exported field coordinates JSON.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Pitch detection confidence."
    )
    parser.add_argument(
        "--kpt-conf",
        type=float,
        default=0.5,
        help="Per-keypoint confidence to treat a keypoint as visible.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280, help="Inference image size."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device: 'cpu', 'cuda', or a GPU index like '0'.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not draw keypoint index labels on the visualization.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show a live preview window (video inputs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = FieldPipeline(
        model_path=args.model,
        source=args.source,
        output_path=args.output,
        coordinates_json=args.coordinates_json,
        conf_threshold=args.conf,
        kpt_conf_threshold=args.kpt_conf,
        device=args.device,
        imgsz=args.imgsz,
        draw_labels=not args.no_labels,
        show_preview=args.show,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
