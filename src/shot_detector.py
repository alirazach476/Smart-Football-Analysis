"""
Automatic football shot detection from bird's-eye coordinates + possession.

A shot is inferred when the ball leaves an attacking player and travels fast
toward the opponent's goal from within shooting range. Each shot is:

  * located    -> pitch position, distance to goal, angle
  * speed-rated -> peak ball speed (m/s and km/h)
  * classified -> on_target | off_target | blocked

On target : the ball's trajectory projects inside the goal mouth and is not
            interrupted before reaching the goal line.
Off target: the trajectory projects wide/over the goal mouth.
Blocked   : a defender (or goalkeeper) is on the ball path before the goal and
            the ball's fast motion stops short of the goal line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from . import field_config as fc
from .possession_analyzer import PossessionAnalyzer, PossessionState


class ShotOutcome(str, Enum):
    ON_TARGET = "on_target"
    OFF_TARGET = "off_target"
    BLOCKED = "blocked"


CM_PER_M = 100.0

# Pitch geometry (cm).
L = fc.PITCH_LENGTH
W = fc.PITCH_WIDTH
GOAL_HALF = fc.GOAL_WIDTH / 2.0
GOAL_Y_LO = W / 2.0 - GOAL_HALF
GOAL_Y_HI = W / 2.0 + GOAL_HALF

# Detection thresholds.
MIN_SHOT_SPEED_MS = 6.0          # ball must exceed this toward goal
MAX_SHOT_DISTANCE_M = 35.0       # only count attempts within range
MAX_PLAUSIBLE_BALL_SPEED_MS = 45.0  # reject tracking glitches (~162 km/h)
GOAL_LINE_MARGIN_CM = 300.0      # "reached the goal line" tolerance
BLOCK_RADIUS_CM = 180.0          # defender proximity to ball path = block
MIN_SHOT_GAP_FRAMES = 8          # dead frames before a new shot can start


@dataclass
class ShotEvent:
    """A single detected shot attempt."""

    shot_id: int
    team: str
    shooter_track_id: int | None
    outcome: str
    target_goal: str  # "left" | "right"
    speed_ms: float
    speed_kmh: float
    location_x_cm: float
    location_y_cm: float
    distance_to_goal_m: float
    angle_deg: float
    projected_goal_y_cm: float | None
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float

    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "team": self.team,
            "shooter_track_id": self.shooter_track_id,
            "outcome": self.outcome,
            "target_goal": self.target_goal,
            "speed_ms": round(self.speed_ms, 2),
            "speed_kmh": round(self.speed_kmh, 1),
            "location_x_cm": round(self.location_x_cm, 1),
            "location_y_cm": round(self.location_y_cm, 1),
            "distance_to_goal_m": round(self.distance_to_goal_m, 2),
            "angle_deg": round(self.angle_deg, 1),
            "projected_goal_y_cm": (
                round(self.projected_goal_y_cm, 1)
                if self.projected_goal_y_cm is not None
                else None
            ),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_sec": round(self.start_time_sec, 3),
            "end_time_sec": round(self.end_time_sec, 3),
        }


@dataclass
class ShotStats:
    """Aggregate shot statistics."""

    total: int = 0
    on_target: int = 0
    off_target: int = 0
    blocked: int = 0
    max_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0
    avg_distance_m: float = 0.0
    by_player: dict[int, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        acc = (self.on_target / self.total * 100.0) if self.total else 0.0
        return {
            "total": self.total,
            "on_target": self.on_target,
            "off_target": self.off_target,
            "blocked": self.blocked,
            "accuracy_pct": round(acc, 1),
            "max_speed_kmh": round(self.max_speed_kmh, 1),
            "avg_speed_kmh": round(self.avg_speed_kmh, 1),
            "avg_distance_m": round(self.avg_distance_m, 2),
            "by_player": {str(k): v for k, v in self.by_player.items()},
        }


class ShotDetector:
    """Detects and classifies shots from bird's-eye frame records."""

    TEAM_STATES = {PossessionState.TEAM_A.value, PossessionState.TEAM_B.value}
    TRACKABLE = {"player", "goalkeeper"}

    def __init__(
        self,
        fps: float = 25.0,
        min_shot_speed_ms: float = MIN_SHOT_SPEED_MS,
        max_shot_distance_m: float = MAX_SHOT_DISTANCE_M,
        team_names: tuple[str, str] = ("Team A", "Team B"),
    ) -> None:
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 25.0
        self.min_shot_speed_ms = min_shot_speed_ms
        self.max_shot_distance_cm = max_shot_distance_m * CM_PER_M
        self.team_names = team_names

        self._shots: list[ShotEvent] = []
        self._ball_series: list[dict] = []
        self._frame_players: dict[int, list[dict]] = {}
        self._team_attack_dir: dict[str, int] = {}

    def analyze(
        self,
        frames: list[dict],
        possession: PossessionAnalyzer | None = None,
    ) -> list[ShotEvent]:
        """Detect shots from bird's-eye frame records."""
        if possession is None:
            possession = PossessionAnalyzer(fps=self.fps)
            for rec in frames:
                possession.update(rec)
            possession.finalize()

        self._index_frames(frames)
        self._compute_attack_directions(possession)

        self._shots = []
        self._scan_shots(possession)
        return self._shots

    def stats(self, team: str | None = None) -> ShotStats:
        """Aggregate statistics, optionally filtered by team."""
        shots = [s for s in self._shots if team is None or s.team == team]
        st = ShotStats()
        st.total = len(shots)
        if not shots:
            return st

        speeds = [s.speed_kmh for s in shots]
        distances = [s.distance_to_goal_m for s in shots]
        st.max_speed_kmh = max(speeds)
        st.avg_speed_kmh = sum(speeds) / len(speeds)
        st.avg_distance_m = sum(distances) / len(distances)

        for s in shots:
            if s.outcome == ShotOutcome.ON_TARGET.value:
                st.on_target += 1
            elif s.outcome == ShotOutcome.OFF_TARGET.value:
                st.off_target += 1
            elif s.outcome == ShotOutcome.BLOCKED.value:
                st.blocked += 1

            if s.shooter_track_id is not None:
                pp = st.by_player.setdefault(
                    s.shooter_track_id,
                    {"shots": 0, "on_target": 0, "blocked": 0, "max_speed_kmh": 0.0},
                )
                pp["shots"] += 1
                if s.outcome == ShotOutcome.ON_TARGET.value:
                    pp["on_target"] += 1
                elif s.outcome == ShotOutcome.BLOCKED.value:
                    pp["blocked"] += 1
                pp["max_speed_kmh"] = round(max(pp["max_speed_kmh"], s.speed_kmh), 1)
        return st

    @property
    def shots(self) -> list[ShotEvent]:
        return list(self._shots)

    # ------------------------------------------------------------------ #
    def _index_frames(self, frames: list[dict]) -> None:
        """Build an ordered ball series and per-frame player lists."""
        self._ball_series = []
        self._frame_players = {}

        for rec in sorted(frames, key=lambda r: r["frame"]):
            if not rec.get("homography_available"):
                continue
            frame = rec["frame"]
            ts = rec.get("timestamp_sec")
            if ts is None:
                ts = frame * self.dt

            players = []
            for p in rec.get("players", []):
                if p.get("class_name") not in self.TRACKABLE:
                    continue
                x, y = p.get("x_cm"), p.get("y_cm")
                if x is None or y is None:
                    continue
                players.append(
                    {
                        "track_id": p.get("track_id"),
                        "team": p.get("team"),
                        "class_name": p.get("class_name", "player"),
                        "x_cm": float(x),
                        "y_cm": float(y),
                    }
                )
            self._frame_players[frame] = players

            ball = rec.get("ball")
            if ball and ball.get("x_cm") is not None and ball.get("y_cm") is not None:
                self._ball_series.append(
                    {
                        "frame": frame,
                        "ts": ts,
                        "x": float(ball["x_cm"]),
                        "y": float(ball["y_cm"]),
                    }
                )

    def _compute_attack_directions(self, possession: PossessionAnalyzer) -> None:
        """Estimate each team's attacking direction (+1 toward x=L, -1 toward x=0)."""
        team_x: dict[str, list[float]] = {t: [] for t in self.team_names}
        for pf in possession.history:
            if pf.track_id is None or pf.team not in team_x:
                continue
            pos = self._player_pos_at(pf.track_id, pf.frame)
            if pos:
                team_x[pf.team].append(pos[0])

        # Teams attack opposite directions; assign them relative to each other
        # so both are never labelled the same way (which happens if each is
        # tested against the halfway line independently on noisy homography).
        a, b = self.team_names
        med: dict[str, float | None] = {}
        for team in self.team_names:
            xs = team_x[team]
            med[team] = sorted(xs)[len(xs) // 2] if xs else None

        if med[a] is None or med[b] is None:
            self._team_attack_dir = {a: 1, b: -1}
        elif med[a] <= med[b]:
            # Team a sits further left -> attacks toward x=L (right).
            self._team_attack_dir = {a: 1, b: -1}
        else:
            self._team_attack_dir = {a: -1, b: 1}

    def _scan_shots(self, possession: PossessionAnalyzer) -> None:
        """Find fast ball movements toward a goal and classify them."""
        series = self._ball_series
        if len(series) < 2:
            return

        shot_id = 1
        i = 0
        n = len(series)
        last_shot_end_frame = -MIN_SHOT_GAP_FRAMES - 1

        while i < n - 1:
            a = series[i]
            b = series[i + 1]

            # The two samples forming the shot must be temporally adjacent,
            # otherwise a tracking gap makes the velocity meaningless.
            if b["frame"] - a["frame"] > 2:
                i += 1
                continue

            # A shot must not START right after the ball was lost/teleported:
            # the position at re-acquisition is unreliable (bad homography can
            # fling the ball to the wrong end). Require a continuous preceding
            # sample so we only trust well-tracked ball motion.
            if i == 0 or (a["frame"] - series[i - 1]["frame"]) > 2:
                i += 1
                continue

            speed, vx, vy = self._segment_velocity(a, b)

            if speed < self.min_shot_speed_ms or speed > MAX_PLAUSIBLE_BALL_SPEED_MS:
                i += 1
                continue

            target = self._goal_target(a["x"], vx)
            if target is None:
                i += 1
                continue

            if a["frame"] - last_shot_end_frame < MIN_SHOT_GAP_FRAMES:
                i += 1
                continue

            goal_x = 0.0 if target == "left" else float(L)
            dist_cm = self._distance_to_goal(a["x"], a["y"], goal_x)
            if dist_cm > self.max_shot_distance_cm:
                i += 1
                continue

            shooter, team = self._shooter_at(a["frame"], target)
            if team is None:
                i += 1
                continue

            # Extend the shot window while the ball keeps moving toward goal.
            j = i + 1
            peak_speed = speed
            while j < n - 1:
                s2, vx2, _ = self._segment_velocity(series[j], series[j + 1])
                if self._goal_target(series[j]["x"], vx2) != target:
                    break
                if s2 > MAX_PLAUSIBLE_BALL_SPEED_MS:
                    break
                if s2 < self.min_shot_speed_ms * 0.5:
                    break
                peak_speed = max(peak_speed, s2)
                j += 1

            end = series[j]
            outcome, proj_y = self._classify(
                a, end, vx, vy, target, goal_x
            )

            angle = self._shot_angle(a["x"], a["y"], goal_x)
            self._shots.append(
                ShotEvent(
                    shot_id=shot_id,
                    team=team,
                    shooter_track_id=shooter,
                    outcome=outcome,
                    target_goal=target,
                    speed_ms=peak_speed,
                    speed_kmh=peak_speed * 3.6,
                    location_x_cm=a["x"],
                    location_y_cm=a["y"],
                    distance_to_goal_m=dist_cm / CM_PER_M,
                    angle_deg=angle,
                    projected_goal_y_cm=proj_y,
                    start_frame=a["frame"],
                    end_frame=end["frame"],
                    start_time_sec=a["ts"],
                    end_time_sec=end["ts"],
                )
            )
            shot_id += 1
            last_shot_end_frame = end["frame"]
            i = j + 1

    def _classify(
        self,
        start: dict,
        end: dict,
        vx: float,
        vy: float,
        target: str,
        goal_x: float,
    ) -> tuple[str, float | None]:
        """Determine shot outcome and projected goal-line crossing."""
        proj_y = self._project_to_goal_line(start["x"], start["y"], vx, vy, goal_x)

        # Blocked: a defender/keeper sits on the ball path before the goal and
        # the ball did not reach the goal line.
        reached_goal = abs(end["x"] - goal_x) <= GOAL_LINE_MARGIN_CM
        if not reached_goal and self._path_blocked(start, end, target):
            return ShotOutcome.BLOCKED.value, proj_y

        if proj_y is not None and GOAL_Y_LO <= proj_y <= GOAL_Y_HI:
            return ShotOutcome.ON_TARGET.value, proj_y
        return ShotOutcome.OFF_TARGET.value, proj_y

    def _path_blocked(self, start: dict, end: dict, target: str) -> bool:
        """True if an opposing player is close to the ball path before the goal."""
        shooter_team_dir = 1 if target == "right" else -1
        for frame in range(start["frame"], end["frame"] + 1):
            players = self._frame_players.get(frame)
            ball = self._ball_at_frame(frame)
            if not players or ball is None:
                continue
            for p in players:
                # Defender = player attacking the opposite goal.
                team = p.get("team")
                if team is None:
                    continue
                dist = math.hypot(p["x_cm"] - ball[0], p["y_cm"] - ball[1])
                if dist <= BLOCK_RADIUS_CM:
                    p_dir = self._team_attack_dir.get(team, 1)
                    if p_dir == -shooter_team_dir:
                        return True
        return False

    # ------------------------------------------------------------------ #
    def _segment_velocity(self, a: dict, b: dict) -> tuple[float, float, float]:
        """Speed (m/s) and velocity components between two ball samples."""
        dt = (b["ts"] - a["ts"]) or self.dt
        if dt <= 0:
            dt = self.dt
        dx = b["x"] - a["x"]
        dy = b["y"] - a["y"]
        speed = math.hypot(dx, dy) / CM_PER_M / dt
        return speed, dx / dt, dy / dt

    def _goal_target(self, x: float, vx: float) -> str | None:
        """Which goal the ball is heading toward, if clearly moving in x."""
        if vx < -50.0:  # moving toward x=0
            return "left"
        if vx > 50.0:   # moving toward x=L
            return "right"
        return None

    @staticmethod
    def _distance_to_goal(x: float, y: float, goal_x: float) -> float:
        gy = min(max(y, GOAL_Y_LO), GOAL_Y_HI)
        return math.hypot(x - goal_x, y - gy)

    @staticmethod
    def _shot_angle(x: float, y: float, goal_x: float) -> float:
        """Angle (deg) between shot location and the goal centre line."""
        dx = abs(x - goal_x)
        dy = abs(y - W / 2.0)
        return math.degrees(math.atan2(dy, max(dx, 1e-6)))

    @staticmethod
    def _project_to_goal_line(
        x: float, y: float, vx: float, vy: float, goal_x: float
    ) -> float | None:
        """Extrapolate the straight ball path to the goal-line x, return y."""
        if abs(vx) < 1e-6:
            return None
        t = (goal_x - x) / vx
        if t < 0:
            return None
        return y + vy * t

    def _shooter_at(
        self, frame: int, target: str
    ) -> tuple[int | None, str | None]:
        """Nearest attacking player to the ball at (or near) the shot start."""
        attack_dir = 1 if target == "right" else -1
        ball = self._ball_at_frame(frame)
        if ball is None:
            return None, None

        best: dict | None = None
        best_dist = float("inf")
        for delta in range(0, 4):
            players = self._frame_players.get(frame - delta)
            if not players:
                continue
            for p in players:
                team = p.get("team")
                if team is None:
                    continue
                if self._team_attack_dir.get(team, 1) != attack_dir:
                    continue
                dist = math.hypot(p["x_cm"] - ball[0], p["y_cm"] - ball[1])
                if dist < best_dist:
                    best_dist = dist
                    best = p
            if best is not None:
                break

        if best is None:
            return None, None
        return best.get("track_id"), best.get("team")

    def _ball_at_frame(self, frame: int) -> tuple[float, float] | None:
        for s in self._ball_series:
            if s["frame"] == frame:
                return (s["x"], s["y"])
        return None

    def _player_pos_at(self, track_id: int, frame: int) -> tuple[float, float] | None:
        for delta in range(0, 4):
            for f in (frame - delta, frame + delta):
                players = self._frame_players.get(f)
                if not players:
                    continue
                for p in players:
                    if p.get("track_id") == track_id:
                        return (p["x_cm"], p["y_cm"])
        return None
