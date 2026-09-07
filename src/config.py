"""
Central configuration for the football detection pipeline.

Keeping all tunable constants (class names, colours, drawing style,
inference defaults) in one place makes the rest of the code easier to
read and maintain.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------
# These match the classes the model was trained on (see data.yaml):
#   0: ball, 1: goalkeeper, 2: player, 3: referee
#
# We map each class *name* to a BGR colour (OpenCV uses BGR, not RGB) so that
# every object type is drawn in a distinct, easy-to-read colour.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "ball": (0, 255, 255),        # yellow
    "goalkeeper": (0, 128, 255),  # orange
    "player": (0, 255, 0),        # green
    "referee": (0, 0, 255),       # red
}

# Fallback colour for any class that is not explicitly listed above.
DEFAULT_COLOR: tuple[int, int, int] = (255, 255, 255)  # white


def color_for_track_id(track_id: int) -> tuple[int, int, int]:
    """
    Deterministically map a track ID to a distinct, bright BGR colour.

    Using the ID as the seed means the same track keeps the same colour for
    the whole video, which makes it easy to follow an individual visually.
    """
    # Simple hash spread across the hue circle; multiply by a large prime so
    # consecutive IDs get visually different colours.
    import colorsys

    hue = (track_id * 0.61803398875) % 1.0  # golden-ratio spacing
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR for OpenCV


@dataclass
class PipelineConfig:
    """Runtime configuration for a single pipeline run."""

    # Paths
    model_path: str
    source_video: str
    output_video: str = "output/annotated.mp4"
    detections_json: str = "output/detections.json"
    detections_csv: str = "output/detections.csv"
    tracking_json: str = "output/tracking.json"
    tracking_csv: str = "output/tracking.csv"

    # Inference settings
    conf_threshold: float = 0.25      # minimum confidence to keep a detection
    iou_threshold: float = 0.45       # NMS IoU threshold
    device: str = "cpu"               # "cpu", "cuda", "0", etc. (auto-resolved)
    imgsz: int = 1280                 # inference image size

    # Tracking settings
    enable_tracking: bool = False     # run ByteTrack instead of plain detection
    tracker_config: str = "bytetrack.yaml"  # path to the ByteTrack YAML
    id_colors: bool = True            # colour boxes per track ID (vs per class)

    # Team classification settings
    enable_teams: bool = False        # cluster players into Team A / Team B
    team_warmup_frames: int = 30      # frames to gather colour before first fit
    team_refit_interval: int = 30     # re-cluster every N frames
    team_min_samples: int = 3         # min sightings before a track is clustered

    # Ball tracking settings (Kalman filter)
    enable_ball_tracking: bool = False
    ball_trajectory_json: str = "output/ball_trajectory.json"
    ball_conf_threshold: float = 0.10   # lower than players (motion blur)
    ball_gate_radius: float = 120.0     # association gate in pixels
    ball_max_coast_frames: int = 20     # predict through brief disappearances
    ball_trajectory_length: int = 60    # trail length drawn on video

    # Drawing style
    box_thickness: int = 2
    font_scale: float = 0.5
    font_thickness: int = 1

    # Display / preview window
    show_preview: bool = False        # set True to see live annotated frames

    # Class colours (copied so callers can override per-run if desired)
    class_colors: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: dict(CLASS_COLORS)
    )

    def color_for(self, class_name: str) -> tuple[int, int, int]:
        """Return the BGR colour for a class name, falling back to white."""
        return self.class_colors.get(class_name, DEFAULT_COLOR)
