"""
Command-line entry point for the bird's-eye (homography) view pipeline.

Combines the field pose model and the object-detection model to project
players and the ball onto a top-down pitch, rendered side by side with the
source video and exported to JSON.

Examples
--------
    # Default: both bundled models, teams on
    python birdseye.py --source match.mp4

    # Custom output + no team colouring
    python birdseye.py --source match.mp4 --output output/tactical.mp4 --no-teams

Run `python birdseye.py --help` for all options.
"""

from __future__ import annotations

import argparse

from src.birdseye_pipeline import BirdseyePipeline

DEFAULT_FIELD_MODEL = "football-field-detection/football-field-detection.pt"
DEFAULT_OBJECT_MODEL = "football-players-detection/football-players-detection.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bird's-eye tactical view via homography (YOLOv8).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--field-model",
        default=DEFAULT_FIELD_MODEL,
        help="Path to the trained field (pose) .pt model.",
    )
    parser.add_argument(
        "--object-model",
        default=DEFAULT_OBJECT_MODEL,
        help="Path to the trained players/ball .pt model.",
    )
    parser.add_argument(
        "--source", required=True, help="Input football match video."
    )
    parser.add_argument(
        "--output",
        default="output/birdseye.mp4",
        help="Path for the synchronized side-by-side output video.",
    )
    parser.add_argument(
        "--json",
        dest="coordinates_json",
        default="output/birdseye_coordinates.json",
        help="Path for the exported transformed-coordinates JSON.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Detection confidence threshold."
    )
    parser.add_argument(
        "--kpt-conf",
        type=float,
        default=0.5,
        help="Field keypoint visibility threshold.",
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
        "--scale",
        type=float,
        default=0.06,
        help="Minimap scale in pixels per centimetre.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.5,
        help="Homography temporal smoothing in [0,1) (higher = smoother).",
    )
    parser.add_argument(
        "--no-teams",
        action="store_true",
        help="Disable jersey-colour team classification on the minimap.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show a live preview window while processing.",
    )

    # --- Speed estimation options ------------------------------------------
    parser.add_argument(
        "--no-speed",
        action="store_true",
        help="Disable player speed estimation and live speed overlay.",
    )
    parser.add_argument(
        "--speed-json",
        default="output/player_speed.json",
        help="Path for exported player speed statistics JSON.",
    )
    parser.add_argument(
        "--speed-csv",
        default="output/player_speed.csv",
        help="Path for exported player speed summary CSV.",
    )
    parser.add_argument(
        "--sprint-threshold",
        type=float,
        default=7.0,
        help="Sprint speed threshold in m/s (default 7.0 = 25.2 km/h).",
    )
    parser.add_argument(
        "--min-sprint",
        type=float,
        default=1.0,
        help="Minimum sprint duration in seconds.",
    )

    # --- Heatmap options ---------------------------------------------------
    parser.add_argument(
        "--heatmaps",
        action="store_true",
        help="Generate player/team/ball/zone/pressure heatmaps after processing.",
    )
    parser.add_argument(
        "--heatmap-dir",
        default="output/heatmaps",
        help="Directory for heatmap PNG images.",
    )
    parser.add_argument(
        "--heatmap-json",
        default="output/heatmap_data.json",
        help="Path for heatmap grid data JSON.",
    )
    parser.add_argument(
        "--heatmap-html",
        default="output/heatmaps.html",
        help="Path for interactive heatmap HTML dashboard.",
    )

    # --- Possession analysis options ---------------------------------------
    parser.add_argument(
        "--no-possession",
        action="store_true",
        help="Disable automatic possession analysis and live overlay.",
    )
    parser.add_argument(
        "--possession-json",
        default="output/possession.json",
        help="Path for possession statistics JSON.",
    )
    parser.add_argument(
        "--possession-csv",
        default="output/possession.csv",
        help="Path for possession summary CSV.",
    )
    parser.add_argument(
        "--possession-timeline",
        default="output/possession_timeline.csv",
        help="Path for possession timeline CSV.",
    )
    parser.add_argument(
        "--possession-radius",
        type=float,
        default=5.5,
        help="Max distance (m) from ball to player for possession.",
    )

    # --- Pass detection options --------------------------------------------
    parser.add_argument(
        "--no-passes",
        action="store_true",
        help="Disable automatic pass detection and network export.",
    )
    parser.add_argument(
        "--passes-json",
        default="output/passes.json",
        help="Path for pass events JSON.",
    )
    parser.add_argument(
        "--passes-csv",
        default="output/passes.csv",
        help="Path for per-pass CSV.",
    )
    parser.add_argument(
        "--pass-statistics",
        default="output/pass_statistics.csv",
        help="Path for pass summary statistics CSV.",
    )
    parser.add_argument(
        "--pass-network-dir",
        default="output/pass_networks",
        help="Directory for pass network PNG images.",
    )

    # --- Shot detection options --------------------------------------------
    parser.add_argument(
        "--no-shots",
        action="store_true",
        help="Disable automatic shot detection and shot map export.",
    )
    parser.add_argument(
        "--shots-json",
        default="output/shots.json",
        help="Path for shot events JSON.",
    )
    parser.add_argument(
        "--shots-csv",
        default="output/shots.csv",
        help="Path for per-shot CSV.",
    )
    parser.add_argument(
        "--shot-statistics",
        default="output/shot_statistics.csv",
        help="Path for shot summary statistics CSV.",
    )
    parser.add_argument(
        "--shot-map-dir",
        default="output/shot_maps",
        help="Directory for shot map PNG images.",
    )

    # --- Goal detection options --------------------------------------------
    parser.add_argument(
        "--no-goals",
        action="store_true",
        help="Disable automatic goal detection, scoreboard, and goal export.",
    )
    parser.add_argument(
        "--goals-json",
        default="output/goals.json",
        help="Path for goal events JSON.",
    )
    parser.add_argument(
        "--goals-csv",
        default="output/goals.csv",
        help="Path for per-goal CSV.",
    )
    parser.add_argument(
        "--goal-statistics",
        default="output/goal_statistics.csv",
        help="Path for goal summary statistics CSV.",
    )
    parser.add_argument(
        "--goal-notifications",
        default="output/goal_notifications.txt",
        help="Path for goal notifications log.",
    )

    # --- Highlight generation options --------------------------------------
    parser.add_argument(
        "--no-highlights",
        action="store_true",
        help="Disable automatic highlight reel generation.",
    )
    parser.add_argument(
        "--highlights",
        dest="highlights_path",
        default="output/highlights.mp4",
        help="Path for the merged highlight video.",
    )
    parser.add_argument(
        "--highlights-json",
        default="output/highlights.json",
        help="Path for the highlight timeline JSON.",
    )
    parser.add_argument(
        "--highlight-pre",
        type=float,
        default=4.0,
        help="Seconds of footage before each highlight event.",
    )
    parser.add_argument(
        "--highlight-post",
        type=float,
        default=3.0,
        help="Seconds of footage after each highlight event.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = BirdseyePipeline(
        field_model=args.field_model,
        object_model=args.object_model,
        source=args.source,
        output_path=args.output,
        coordinates_json=args.coordinates_json,
        conf_threshold=args.conf,
        kpt_conf_threshold=args.kpt_conf,
        device=args.device,
        imgsz=args.imgsz,
        enable_teams=not args.no_teams,
        pitch_scale=args.scale,
        smoothing=args.smoothing,
        show_preview=args.show,
        enable_speed=not args.no_speed,
        speed_json=args.speed_json,
        speed_csv=args.speed_csv,
        sprint_threshold_m_s=args.sprint_threshold,
        min_sprint_sec=args.min_sprint,
        enable_heatmaps=args.heatmaps,
        heatmap_dir=args.heatmap_dir,
        heatmap_json=args.heatmap_json,
        heatmap_html=args.heatmap_html,
        enable_possession=not args.no_possession,
        possession_json=args.possession_json,
        possession_csv=args.possession_csv,
        possession_timeline_csv=args.possession_timeline,
        possession_radius_m=args.possession_radius,
        enable_passes=not args.no_passes,
        passes_json=args.passes_json,
        passes_csv=args.passes_csv,
        pass_statistics_csv=args.pass_statistics,
        pass_network_dir=args.pass_network_dir,
        enable_shots=not args.no_shots,
        shots_json=args.shots_json,
        shots_csv=args.shots_csv,
        shot_statistics_csv=args.shot_statistics,
        shot_map_dir=args.shot_map_dir,
        enable_goals=not args.no_goals,
        goals_json=args.goals_json,
        goals_csv=args.goals_csv,
        goal_statistics_csv=args.goal_statistics,
        goal_notifications_path=args.goal_notifications,
        enable_highlights=not args.no_highlights,
        highlights_path=args.highlights_path,
        highlights_json=args.highlights_json,
        highlight_pre_sec=args.highlight_pre,
        highlight_post_sec=args.highlight_post,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
