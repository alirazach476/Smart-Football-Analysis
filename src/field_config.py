"""
Football pitch geometry and semantics for the 32-keypoint field model.

The trained `football-field-detection.pt` is a YOLOv8 **pose** model with a
single class (`pitch`) and 32 keypoints. Those 32 keypoints follow Roboflow's
well-known `SoccerPitchConfiguration` layout, where each index corresponds to a
fixed landmark on a standard 105 x 68 m pitch.

This module gives those bare indices *meaning*: a human-readable label and a
semantic group (sideline / goal line / penalty box / goal box / halfway line /
centre circle) for every keypoint, plus the edge topology that connects them
into the actual pitch lines. Everything downstream (drawing, exporting) reads
from here so there is a single source of truth for the pitch layout.

All indices in this module are 0-based (0..31). The reference source numbers
vertices 1..32, so index = vertex_number - 1.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Real-world pitch template (centimetres), from SoccerPitchConfiguration.
# Useful as the target plane for homography / minimap work.
# ---------------------------------------------------------------------------
PITCH_LENGTH = 12000  # cm (touchline, x axis)
PITCH_WIDTH = 7000    # cm (goal line, y axis)
PENALTY_BOX_WIDTH = 4100
PENALTY_BOX_LENGTH = 2015
GOAL_BOX_WIDTH = 1832
GOAL_BOX_LENGTH = 550
CENTRE_CIRCLE_RADIUS = 915
PENALTY_SPOT_DISTANCE = 1100
GOAL_WIDTH = 732  # cm (standard 7.32 m goal mouth, centred on the goal line)


def _template_vertices() -> list[tuple[float, float]]:
    """The 32 landmark coordinates on the real pitch, in centimetres."""
    W, L = PITCH_WIDTH, PITCH_LENGTH
    pb, pbl = PENALTY_BOX_WIDTH, PENALTY_BOX_LENGTH
    gb, gbl = GOAL_BOX_WIDTH, GOAL_BOX_LENGTH
    r, psd = CENTRE_CIRCLE_RADIUS, PENALTY_SPOT_DISTANCE
    return [
        (0, 0),                              # 0
        (0, (W - pb) / 2),                   # 1
        (0, (W - gb) / 2),                   # 2
        (0, (W + gb) / 2),                   # 3
        (0, (W + pb) / 2),                   # 4
        (0, W),                              # 5
        (gbl, (W - gb) / 2),                 # 6
        (gbl, (W + gb) / 2),                 # 7
        (psd, W / 2),                        # 8
        (pbl, (W - pb) / 2),                 # 9
        (pbl, (W - gb) / 2),                 # 10
        (pbl, (W + gb) / 2),                 # 11
        (pbl, (W + pb) / 2),                 # 12
        (L / 2, 0),                          # 13
        (L / 2, W / 2 - r),                  # 14
        (L / 2, W / 2 + r),                  # 15
        (L / 2, W),                          # 16
        (L - pbl, (W - pb) / 2),             # 17
        (L - pbl, (W - gb) / 2),             # 18
        (L - pbl, (W + gb) / 2),             # 19
        (L - pbl, (W + pb) / 2),             # 20
        (L - psd, W / 2),                    # 21
        (L - gbl, (W - gb) / 2),             # 22
        (L - gbl, (W + gb) / 2),             # 23
        (L, 0),                              # 24
        (L, (W - pb) / 2),                   # 25
        (L, (W - gb) / 2),                   # 26
        (L, (W + gb) / 2),                   # 27
        (L, (W + pb) / 2),                   # 28
        (L, W),                              # 29
        (L / 2 - r, W / 2),                  # 30
        (L / 2 + r, W / 2),                  # 31
    ]


TEMPLATE_VERTICES = _template_vertices()

# ---------------------------------------------------------------------------
# Human-readable label for every keypoint index.
# ---------------------------------------------------------------------------
KEYPOINT_LABELS: list[str] = [
    "corner_top_left",              # 0
    "left_penalty_box_top_gl",      # 1
    "left_goal_box_top_gl",         # 2
    "left_goal_box_bottom_gl",      # 3
    "left_penalty_box_bottom_gl",   # 4
    "corner_bottom_left",           # 5
    "left_goal_box_top_front",      # 6
    "left_goal_box_bottom_front",   # 7
    "left_penalty_spot",            # 8
    "left_penalty_box_top_front",   # 9
    "left_penalty_box_top_inner",   # 10
    "left_penalty_box_bottom_inner",# 11
    "left_penalty_box_bottom_front",# 12
    "halfway_line_top",             # 13
    "centre_circle_top",            # 14
    "centre_circle_bottom",         # 15
    "halfway_line_bottom",          # 16
    "right_penalty_box_top_front",  # 17
    "right_penalty_box_top_inner",  # 18
    "right_penalty_box_bottom_inner",# 19
    "right_penalty_box_bottom_front",# 20
    "right_penalty_spot",           # 21
    "right_goal_box_top_front",     # 22
    "right_goal_box_bottom_front",  # 23
    "corner_top_right",             # 24
    "right_penalty_box_top_gl",     # 25
    "right_goal_box_top_gl",        # 26
    "right_goal_box_bottom_gl",     # 27
    "right_penalty_box_bottom_gl",  # 28
    "corner_bottom_right",          # 29
    "centre_circle_left",           # 30
    "centre_circle_right",          # 31
]

# ---------------------------------------------------------------------------
# Semantic groups. Each keypoint belongs to one primary group, used for
# colouring and for the grouped coordinate export.
# ---------------------------------------------------------------------------
GROUP_SIDELINE = "sideline"
GROUP_GOAL_LINE = "goal_line"
GROUP_PENALTY_BOX = "penalty_box"
GROUP_GOAL_BOX = "goal_box"
GROUP_HALFWAY = "halfway_line"
GROUP_CENTRE_CIRCLE = "centre_circle"
GROUP_PENALTY_SPOT = "penalty_spot"

# Primary group per keypoint index. Corners count as goal-line endpoints (they
# are also sideline endpoints; goal line is chosen as primary).
KEYPOINT_GROUPS: list[str] = [
    GROUP_GOAL_LINE,     # 0 corner TL
    GROUP_PENALTY_BOX,   # 1
    GROUP_GOAL_BOX,      # 2
    GROUP_GOAL_BOX,      # 3
    GROUP_PENALTY_BOX,   # 4
    GROUP_GOAL_LINE,     # 5 corner BL
    GROUP_GOAL_BOX,      # 6
    GROUP_GOAL_BOX,      # 7
    GROUP_PENALTY_SPOT,  # 8
    GROUP_PENALTY_BOX,   # 9
    GROUP_PENALTY_BOX,   # 10
    GROUP_PENALTY_BOX,   # 11
    GROUP_PENALTY_BOX,   # 12
    GROUP_HALFWAY,       # 13
    GROUP_CENTRE_CIRCLE, # 14
    GROUP_CENTRE_CIRCLE, # 15
    GROUP_HALFWAY,       # 16
    GROUP_PENALTY_BOX,   # 17
    GROUP_PENALTY_BOX,   # 18
    GROUP_PENALTY_BOX,   # 19
    GROUP_PENALTY_BOX,   # 20
    GROUP_PENALTY_SPOT,  # 21
    GROUP_GOAL_BOX,      # 22
    GROUP_GOAL_BOX,      # 23
    GROUP_GOAL_LINE,     # 24 corner TR
    GROUP_PENALTY_BOX,   # 25
    GROUP_GOAL_BOX,      # 26
    GROUP_GOAL_BOX,      # 27
    GROUP_PENALTY_BOX,   # 28
    GROUP_GOAL_LINE,     # 29 corner BR
    GROUP_CENTRE_CIRCLE, # 30
    GROUP_CENTRE_CIRCLE, # 31
]

# BGR colours per group (OpenCV order).
GROUP_COLORS: dict[str, tuple[int, int, int]] = {
    GROUP_SIDELINE: (255, 255, 255),      # white
    GROUP_GOAL_LINE: (0, 255, 0),         # green
    GROUP_PENALTY_BOX: (0, 165, 255),     # orange
    GROUP_GOAL_BOX: (255, 0, 255),        # magenta
    GROUP_HALFWAY: (0, 255, 255),         # yellow
    GROUP_CENTRE_CIRCLE: (0, 0, 255),     # red
    GROUP_PENALTY_SPOT: (255, 128, 0),    # azure
}


@dataclass(frozen=True)
class Edge:
    """A pitch line segment connecting two keypoint indices, tagged by group."""

    a: int
    b: int
    group: str


# Edge topology (0-indexed) grouped by the pitch feature each line belongs to.
# Derived directly from SoccerPitchConfiguration.edges.
EDGES: list[Edge] = [
    # Left goal line (x = 0)
    Edge(0, 1, GROUP_GOAL_LINE),
    Edge(1, 2, GROUP_GOAL_LINE),
    Edge(2, 3, GROUP_GOAL_LINE),
    Edge(3, 4, GROUP_GOAL_LINE),
    Edge(4, 5, GROUP_GOAL_LINE),
    # Right goal line (x = L)
    Edge(24, 25, GROUP_GOAL_LINE),
    Edge(25, 26, GROUP_GOAL_LINE),
    Edge(26, 27, GROUP_GOAL_LINE),
    Edge(27, 28, GROUP_GOAL_LINE),
    Edge(28, 29, GROUP_GOAL_LINE),
    # Top sideline (y = 0)
    Edge(0, 13, GROUP_SIDELINE),
    Edge(13, 24, GROUP_SIDELINE),
    # Bottom sideline (y = W)
    Edge(5, 16, GROUP_SIDELINE),
    Edge(16, 29, GROUP_SIDELINE),
    # Halfway line
    Edge(13, 14, GROUP_HALFWAY),
    Edge(14, 15, GROUP_HALFWAY),
    Edge(15, 16, GROUP_HALFWAY),
    # Left penalty box
    Edge(1, 9, GROUP_PENALTY_BOX),
    Edge(9, 10, GROUP_PENALTY_BOX),
    Edge(10, 11, GROUP_PENALTY_BOX),
    Edge(11, 12, GROUP_PENALTY_BOX),
    Edge(4, 12, GROUP_PENALTY_BOX),
    # Right penalty box
    Edge(17, 18, GROUP_PENALTY_BOX),
    Edge(18, 19, GROUP_PENALTY_BOX),
    Edge(19, 20, GROUP_PENALTY_BOX),
    Edge(17, 25, GROUP_PENALTY_BOX),
    Edge(20, 28, GROUP_PENALTY_BOX),
    # Left goal box
    Edge(2, 6, GROUP_GOAL_BOX),
    Edge(6, 7, GROUP_GOAL_BOX),
    Edge(3, 7, GROUP_GOAL_BOX),
    # Right goal box
    Edge(22, 23, GROUP_GOAL_BOX),
    Edge(22, 26, GROUP_GOAL_BOX),
    Edge(23, 27, GROUP_GOAL_BOX),
]

# Keypoint indices that describe the centre circle (its top/bottom/left/right
# extremes). Used to fit/draw the circle ellipse.
CENTRE_CIRCLE_INDICES: list[int] = [14, 15, 30, 31]

NUM_KEYPOINTS = 32


def label_for(index: int) -> str:
    """Human-readable label for a keypoint index."""
    if 0 <= index < len(KEYPOINT_LABELS):
        return KEYPOINT_LABELS[index]
    return f"kpt_{index}"


def group_for(index: int) -> str:
    """Semantic group for a keypoint index."""
    if 0 <= index < len(KEYPOINT_GROUPS):
        return KEYPOINT_GROUPS[index]
    return "unknown"


def color_for_group(group: str) -> tuple[int, int, int]:
    """BGR colour for a semantic group (falls back to grey)."""
    return GROUP_COLORS.get(group, (200, 200, 200))
