"""
Command-line entry point for the football object detection pipeline.

Examples
--------
    # Plain detection
    python detect.py --source match.mp4

    # Detection + ByteTrack tracking (persistent IDs, tracking.json/csv)
    python detect.py --source match.mp4 --track

    # Detection + tracking + automatic Team A / Team B classification
    python detect.py --source match.mp4 --teams

    # Detection + Kalman ball tracking (trajectory + speed + ball_trajectory.json)
    python detect.py --source match.mp4 --ball-track

Run `python detect.py --help` for all options.
"""

from __future__ import annotations

import argparse

from src.config import PipelineConfig
from src.pipeline import FootballPipeline

# Sensible default path to the trained model shipped with this project.
DEFAULT_MODEL = "football-players-detection/football-players-detection.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Football object detection pipeline (YOLOv8 + OpenCV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Path to the trained .pt model."
    )
    parser.add_argument(
        "--source", required=True, help="Path to the input football match video."
    )
    parser.add_argument(
        "--output",
        default="output/annotated.mp4",
        help="Path for the saved annotated video.",
    )
    parser.add_argument(
        "--json",
        dest="detections_json",
        default="output/detections.json",
        help="Path for the exported detections JSON.",
    )
    parser.add_argument(
        "--csv",
        dest="detections_csv",
        default="output/detections.csv",
        help="Path for the exported detections CSV.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold."
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="NMS IoU threshold."
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
        "--show",
        action="store_true",
        help="Show a live preview window while processing.",
    )

    # --- Tracking (ByteTrack) options --------------------------------------
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable ByteTrack multi-object tracking (persistent IDs).",
    )
    parser.add_argument(
        "--tracker",
        dest="tracker_config",
        default="bytetrack.yaml",
        help="Path to the ByteTrack YAML config.",
    )
    parser.add_argument(
        "--tracking-json",
        dest="tracking_json",
        default="output/tracking.json",
        help="Path for the exported tracking JSON (tracking mode only).",
    )
    parser.add_argument(
        "--tracking-csv",
        dest="tracking_csv",
        default="output/tracking.csv",
        help="Path for the exported tracking CSV (tracking mode only).",
    )
    parser.add_argument(
        "--class-colors",
        action="store_true",
        help="Colour boxes per class instead of per track ID (tracking mode).",
    )

    # --- Team classification options ---------------------------------------
    parser.add_argument(
        "--teams",
        action="store_true",
        help="Automatically split players into Team A / Team B by jersey "
        "colour (implies --track). No manual colour selection needed.",
    )
    parser.add_argument(
        "--team-warmup",
        dest="team_warmup",
        type=int,
        default=30,
        help="Frames of colour evidence to gather before the first clustering.",
    )

    # --- Ball tracking (Kalman filter) options -----------------------------
    parser.add_argument(
        "--ball-track",
        action="store_true",
        help="Enable Kalman-filter ball tracking with trajectory and speed.",
    )
    parser.add_argument(
        "--ball-json",
        dest="ball_trajectory_json",
        default="output/ball_trajectory.json",
        help="Path for the exported ball trajectory JSON.",
    )
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=0.10,
        help="Ball detection confidence floor (lower than players for blur).",
    )
    parser.add_argument(
        "--ball-coast",
        type=int,
        default=20,
        help="Max frames to predict through ball disappearance.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = PipelineConfig(
        model_path=args.model,
        source_video=args.source,
        output_video=args.output,
        detections_json=args.detections_json,
        detections_csv=args.detections_csv,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        show_preview=args.show,
        enable_tracking=args.track,
        tracker_config=args.tracker_config,
        tracking_json=args.tracking_json,
        tracking_csv=args.tracking_csv,
        # By default colour by track ID; --class-colors reverts to class colours.
        id_colors=not args.class_colors,
        enable_teams=args.teams,
        team_warmup_frames=args.team_warmup,
        enable_ball_tracking=args.ball_track,
        ball_trajectory_json=args.ball_trajectory_json,
        ball_conf_threshold=args.ball_conf,
        ball_max_coast_frames=args.ball_coast,
    )

    pipeline = FootballPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
