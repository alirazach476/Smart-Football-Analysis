"""
Heatmap data collector.

Accumulates bird's-eye pitch coordinates (centimetres) from each processed
frame and builds the density grids used for player, team, ball, zone-control
and pressure heatmaps.

All coordinates must already be transformed onto the pitch plane (from the
homography / bird's-eye pipeline). Frames without homography are skipped.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from . import field_config as fc

# Fine grid for density heatmaps (100 cm per cell on a 120x70 m pitch).
GRID_COLS = 120
GRID_ROWS = 70

# Coarse grid for zone-control / tactical zones (10 m per cell).
ZONE_COLS = 12
ZONE_ROWS = 7


def _cm_to_grid(x_cm: float, y_cm: float) -> tuple[int, int] | None:
    """Map pitch cm to fine-grid indices, or None if off-pitch."""
    if x_cm < 0 or x_cm > fc.PITCH_LENGTH or y_cm < 0 or y_cm > fc.PITCH_WIDTH:
        return None
    col = min(int(x_cm / fc.PITCH_LENGTH * GRID_COLS), GRID_COLS - 1)
    row = min(int(y_cm / fc.PITCH_WIDTH * GRID_ROWS), GRID_ROWS - 1)
    return col, row


def _cm_to_zone(x_cm: float, y_cm: float) -> tuple[int, int] | None:
    if x_cm < 0 or x_cm > fc.PITCH_LENGTH or y_cm < 0 or y_cm > fc.PITCH_WIDTH:
        return None
    col = min(int(x_cm / fc.PITCH_LENGTH * ZONE_COLS), ZONE_COLS - 1)
    row = min(int(y_cm / fc.PITCH_WIDTH * ZONE_ROWS), ZONE_ROWS - 1)
    return col, row


@dataclass
class HeatmapData:
    """All accumulated heatmap grids ready for rendering / export."""

    # Fine grids (GRID_COLS x GRID_ROWS), row-major [row, col].
    players: dict[int, np.ndarray] = field(default_factory=dict)
    player_meta: dict[int, dict] = field(default_factory=dict)
    teams: dict[str, np.ndarray] = field(default_factory=dict)
    ball: np.ndarray = field(default_factory=lambda: np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32))
    pressure: np.ndarray = field(default_factory=lambda: np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32))
    # Zone control: values in [-1, 1] (negative = Team B, positive = Team A).
    zone_control: np.ndarray = field(
        default_factory=lambda: np.zeros((ZONE_ROWS, ZONE_COLS), dtype=np.float32)
    )
    zone_team_a: np.ndarray = field(
        default_factory=lambda: np.zeros((ZONE_ROWS, ZONE_COLS), dtype=np.float32)
    )
    zone_team_b: np.ndarray = field(
        default_factory=lambda: np.zeros((ZONE_ROWS, ZONE_COLS), dtype=np.float32)
    )
    frames_processed: int = 0
    team_names: tuple[str, str] = ("Team A", "Team B")

    def to_dict(self) -> dict:
        """Serialise grids to nested lists for JSON export."""
        return {
            "grid": {"cols": GRID_COLS, "rows": GRID_ROWS},
            "zones": {"cols": ZONE_COLS, "rows": ZONE_ROWS},
            "pitch_length_cm": fc.PITCH_LENGTH,
            "pitch_width_cm": fc.PITCH_WIDTH,
            "frames_processed": self.frames_processed,
            "team_names": list(self.team_names),
            "players": {
                str(tid): {
                    **self.player_meta.get(tid, {}),
                    "grid": self.players[tid].tolist(),
                }
                for tid in self.players
            },
            "teams": {name: g.tolist() for name, g in self.teams.items()},
            "ball": self.ball.tolist(),
            "pressure": self.pressure.tolist(),
            "zone_control": self.zone_control.tolist(),
            "zone_team_a": self.zone_team_a.tolist(),
            "zone_team_b": self.zone_team_b.tolist(),
        }


class HeatmapCollector:
    """Accumulates positions from bird's-eye frame records."""

    TEAM_CLASSES = {"player", "goalkeeper"}

    def __init__(self, team_names: tuple[str, str] = ("Team A", "Team B")) -> None:
        self.team_names = team_names
        self._data = HeatmapData(team_names=team_names)
        # Initialise team grids.
        for name in team_names:
            self._data.teams[name] = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

    @property
    def data(self) -> HeatmapData:
        return self._data

    def add_frame(self, record: dict) -> None:
        """
        Ingest one frame from the bird's-eye coordinates export.

        Expected keys: homography_available, players (list), ball (dict|None).
        """
        if not record.get("homography_available"):
            return

        self._data.frames_processed += 1
        players = record.get("players", [])
        ball = record.get("ball")
        ball_xy: tuple[float, float] | None = None

        if ball and "x_cm" in ball:
            ball_xy = (ball["x_cm"], ball["y_cm"])
            self._accumulate_point(self._data.ball, ball_xy)

        # Per-frame zone tallies for zone-control update.
        zone_a = np.zeros((ZONE_ROWS, ZONE_COLS), dtype=np.float32)
        zone_b = np.zeros((ZONE_ROWS, ZONE_COLS), dtype=np.float32)

        for p in players:
            x, y = p.get("x_cm"), p.get("y_cm")
            if x is None or y is None:
                continue
            cls = p.get("class_name", "player")
            tid = p.get("track_id")
            team = p.get("team")

            if cls in self.TEAM_CLASSES and tid is not None:
                self._add_player(tid, x, y, cls, team)
                if team == self.team_names[0]:
                    self._accumulate_point(self._data.teams[self.team_names[0]], (x, y))
                    z = _cm_to_zone(x, y)
                    if z:
                        zone_a[z[1], z[0]] += 1
                elif team == self.team_names[1]:
                    self._accumulate_point(self._data.teams[self.team_names[1]], (x, y))
                    z = _cm_to_zone(x, y)
                    if z:
                        zone_b[z[1], z[0]] += 1
                elif team is None:
                    # Unclassified player: still counts for pressure, not zone split.
                    pass

                self._add_pressure(x, y, ball_xy)

        self._data.zone_team_a += zone_a
        self._data.zone_team_b += zone_b

    def finalize(self) -> HeatmapData:
        """Compute zone-control dominance ratios and return the finished data."""
        a = self._data.zone_team_a
        b = self._data.zone_team_b
        total = a + b
        # +1 = Team A dominated, -1 = Team B, 0 = contested / empty.
        with np.errstate(divide="ignore", invalid="ignore"):
            self._data.zone_control = np.where(
                total > 0, (a - b) / total, 0.0
            ).astype(np.float32)
        return self._data

    def load_from_json(self, path: str) -> HeatmapData:
        """Rebuild heatmaps from a saved birdseye_coordinates.json file."""
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        for frame in payload.get("frames", []):
            self.add_frame(frame)
        return self.finalize()

    # ------------------------------------------------------------------ #
    def _add_player(
        self, track_id: int, x: float, y: float, cls: str, team: str | None
    ) -> None:
        if track_id not in self._data.players:
            self._data.players[track_id] = np.zeros(
                (GRID_ROWS, GRID_COLS), dtype=np.float32
            )
            self._data.player_meta[track_id] = {
                "track_id": track_id,
                "class_name": cls,
                "team": team,
            }
        elif team and not self._data.player_meta[track_id].get("team"):
            self._data.player_meta[track_id]["team"] = team
        self._accumulate_point(self._data.players[track_id], (x, y))

    @staticmethod
    def _accumulate_point(grid: np.ndarray, xy: tuple[float, float]) -> None:
        cell = _cm_to_grid(xy[0], xy[1])
        if cell:
            grid[cell[1], cell[0]] += 1.0

    def _add_pressure(
        self, x: float, y: float, ball_xy: tuple[float, float] | None
    ) -> None:
        """
        Pressure = local player density weighted by proximity to the ball.

        When the ball is known, weight falls off with distance (15 m radius).
        Without ball, plain density is accumulated.
        """
        cell = _cm_to_grid(x, y)
        if not cell:
            return
        weight = 1.0
        if ball_xy is not None:
            dist_m = math.hypot(x - ball_xy[0], y - ball_xy[1]) / 100.0
            weight = max(0.0, 1.0 - dist_m / 15.0)  # 15 m influence radius
        col, row = cell
        # Spread a small Gaussian footprint for smoother pressure maps.
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = row + dr, col + dc
                if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                    self._data.pressure[r, c] += weight * (0.5 if dr or dc else 1.0)
