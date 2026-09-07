"""
Automatic football goal detection from bird's-eye ball trajectory.

A goal is recorded when the ball crosses a goal line (x = 0 or x = pitch length)
with its y coordinate inside the goal mouth and sufficient velocity toward the
goal. The scorer is attributed from the last attacking-team possession touch
before the crossing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import field_config as fc
from .possession_analyzer import PossessionAnalyzer, PossessionState


CM_PER_M = 100.0

L = fc.PITCH_LENGTH
W = fc.PITCH_WIDTH
GOAL_HALF = fc.GOAL_WIDTH / 2.0
GOAL_Y_LO = W / 2.0 - GOAL_HALF
GOAL_Y_HI = W / 2.0 + GOAL_HALF
# Extra tolerance for noisy homography near the goal mouth. Kept tight so a
# glitchy projection off to the side of the goal (e.g. near a corner) is not
# counted as a goal.
GOAL_MOUTH_MARGIN_CM = 200.0
# How close (cm) to the goal line the ball must be to count as "in the goal".
GOAL_BOX_DEPTH_CM = 450.0
# Max per-frame ball movement (cm) while dwelling in the goal. A ball resting
# in the net is nearly still; a homography glitch near the goal jitters widely.
GOAL_STABLE_CM = 300.0

MIN_CROSS_SPEED_MS = 2.0
MIN_GOAL_GAP_FRAMES = 150
SCORER_LOOKBACK_FRAMES = 75
MAX_PLAUSIBLE_BALL_SPEED_MS = 80.0
GOAL_ZONE_DEPTH_CM = 600.0


@dataclass
class GoalEvent:
    """A single detected goal."""

    goal_id: int
    scoring_team: str
    conceded_team: str
    scorer_track_id: int | None
    goal_line: str  # "left" | "right"
    cross_y_cm: float
    frame: int
    timestamp_sec: float
    speed_ms: float | None
    score_team_a: int
    score_team_b: int

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "scoring_team": self.scoring_team,
            "conceded_team": self.conceded_team,
            "scorer_track_id": self.scorer_track_id,
            "goal_line": self.goal_line,
            "cross_y_cm": round(self.cross_y_cm, 1),
            "frame": self.frame,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "speed_ms": round(self.speed_ms, 2) if self.speed_ms is not None else None,
            "score_team_a": self.score_team_a,
            "score_team_b": self.score_team_b,
            "scoreline": f"{self.score_team_a}-{self.score_team_b}",
        }


@dataclass
class GoalNotification:
    """Transient on-screen goal alert."""

    goal: GoalEvent
    expires_frame: int
    message: str


@dataclass
class GoalStats:
    """Aggregate goal statistics."""

    total: int = 0
    team_a: int = 0
    team_b: int = 0
    by_player: dict[int, int] = field(default_factory=dict)
    first_goal_sec: float | None = None
    last_goal_sec: float | None = None

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "team_a": self.team_a,
            "team_b": self.team_b,
            "by_player": {str(k): v for k, v in self.by_player.items()},
            "first_goal_sec": self.first_goal_sec,
            "last_goal_sec": self.last_goal_sec,
        }


class GoalDetector:
    """Detects goals from ball trajectory and possession history."""

    TEAM_STATES = {PossessionState.TEAM_A.value, PossessionState.TEAM_B.value}

    def __init__(
        self,
        fps: float = 25.0,
        min_cross_speed_ms: float = MIN_CROSS_SPEED_MS,
        min_goal_gap_frames: int = MIN_GOAL_GAP_FRAMES,
        scorer_lookback_frames: int = SCORER_LOOKBACK_FRAMES,
        notification_sec: float = 4.0,
        team_names: tuple[str, str] = ("Team A", "Team B"),
    ) -> None:
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 25.0
        self.min_cross_speed_ms = min_cross_speed_ms
        self.min_goal_gap = min_goal_gap_frames
        self.scorer_lookback = scorer_lookback_frames
        self.notification_frames = int(notification_sec * fps)

        self.team_names = team_names
        self._team_a, self._team_b = team_names

        self._goals: list[GoalEvent] = []
        self._score: dict[str, int] = {self._team_a: 0, self._team_b: 0}
        self._prev_ball: dict | None = None
        self._last_goal_frame = -min_goal_gap_frames
        self._team_attack_dir: dict[str, int] = {}
        self._possession_history: list = []
        self._notification: GoalNotification | None = None

        # A real goal keeps the ball inside the goal box for several frames; a
        # homography glitch that briefly projects the ball near a goal does not.
        # We require this dwell before registering, which removes spurious goals
        # on camera cuts / re-acquisition.
        self._zone_line: str | None = None
        self._zone_run: int = 0
        self._zone_last_pos: tuple[float, float] | None = None
        self._goal_dwell_frames = 4

    # ------------------------------------------------------------------ #
    def update(
        self,
        record: dict,
        possession_frame=None,
    ) -> list[GoalEvent]:
        """
        Process one frame. Returns newly detected goals (usually 0 or 1).
        """
        if possession_frame is not None:
            self._possession_history.append(possession_frame)

        frame = record["frame"]
        self._expire_notification(frame)

        if not record.get("homography_available"):
            return []

        ball = record.get("ball_raw") or record.get("ball")
        if not ball or ball.get("x_cm") is None or ball.get("y_cm") is None:
            return []

        cur = {
            "frame": frame,
            "ts": record.get("timestamp_sec", frame * self.dt),
            "x": float(ball["x_cm"]),
            "y": float(ball["y_cm"]),
        }

        new_goals: list[GoalEvent] = []

        # Robust rule: a goal is when the ball settles inside the goal box
        # (tight mouth, close to the line) for several frames. This handles
        # both clean crossings and the common broadcast case where the ball
        # "teleports" into the net across a homography gap / camera cut.
        line = self._ball_in_goal(cur)
        if line is not None:
            pos = (cur["x"], cur["y"])
            stable = (
                self._zone_last_pos is not None
                and math.hypot(
                    pos[0] - self._zone_last_pos[0], pos[1] - self._zone_last_pos[1]
                )
                <= GOAL_STABLE_CM
            )
            if line == self._zone_line and stable:
                self._zone_run += 1
            else:
                self._zone_line = line
                self._zone_run = 1
            self._zone_last_pos = pos
            if (
                self._zone_run >= self._goal_dwell_frames
                and frame - self._last_goal_frame >= self.min_goal_gap
            ):
                speed = self._recent_ball_speed(cur)
                goal = self._register_goal(line, cur["y"], frame, cur["ts"], speed)
                if goal is not None:
                    new_goals.append(goal)
        else:
            self._zone_line = None
            self._zone_run = 0
            self._zone_last_pos = None

        self._prev_ball = cur
        return new_goals

    def _ball_in_goal(self, ball: dict) -> str | None:
        """
        Return the goal line ("left"/"right") if the ball is inside that goal
        box (within the mouth and close to/behind the line), else None.
        """
        if not (
            GOAL_Y_LO - GOAL_MOUTH_MARGIN_CM
            <= ball["y"]
            <= GOAL_Y_HI + GOAL_MOUTH_MARGIN_CM
        ):
            return None
        if ball["x"] <= GOAL_BOX_DEPTH_CM:
            return "left"
        if ball["x"] >= L - GOAL_BOX_DEPTH_CM:
            return "right"
        return None

    def _recent_ball_speed(self, cur: dict) -> float:
        """Rough ball speed (m/s) from the last accepted sample, clamped."""
        if self._prev_ball is None:
            return self.min_cross_speed_ms
        dt = (cur["ts"] - self._prev_ball["ts"]) or self.dt
        if dt <= 0:
            dt = self.dt
        dist = math.hypot(cur["x"] - self._prev_ball["x"], cur["y"] - self._prev_ball["y"])
        speed = dist / CM_PER_M / dt
        return float(min(max(speed, self.min_cross_speed_ms), MAX_PLAUSIBLE_BALL_SPEED_MS))

    def analyze(
        self,
        frames: list[dict],
        possession: PossessionAnalyzer | None = None,
    ) -> list[GoalEvent]:
        """Batch-detect goals from saved frame records."""
        self.reset()

        if possession is None:
            possession = PossessionAnalyzer(fps=self.fps)
            for rec in frames:
                possession.update(rec)
            possession.finalize()

        self._compute_attack_directions(frames, possession)
        self._possession_history = list(possession.history)

        for rec in sorted(frames, key=lambda r: r["frame"]):
            frame = rec["frame"]
            pf = next(
                (p for p in possession.history if p.frame == frame), None
            )
            self.update(rec, pf)

        return self._goals

    def reset(self) -> None:
        """Clear all goals, score, and notification state."""
        self._goals = []
        self._score = {self._team_a: 0, self._team_b: 0}
        self._prev_ball = None
        self._last_goal_frame = -self.min_goal_gap
        self._notification = None
        self._possession_history = []
        self._zone_line = None
        self._zone_run = 0
        self._zone_last_pos = None

    def stats(self) -> GoalStats:
        """Aggregate goal statistics."""
        st = GoalStats()
        st.total = len(self._goals)
        st.team_a = self._score[self._team_a]
        st.team_b = self._score[self._team_b]

        for g in self._goals:
            if g.scorer_track_id is not None:
                st.by_player[g.scorer_track_id] = (
                    st.by_player.get(g.scorer_track_id, 0) + 1
                )
            if st.first_goal_sec is None or g.timestamp_sec < st.first_goal_sec:
                st.first_goal_sec = g.timestamp_sec
            if st.last_goal_sec is None or g.timestamp_sec > st.last_goal_sec:
                st.last_goal_sec = g.timestamp_sec

        if st.first_goal_sec is not None:
            st.first_goal_sec = round(st.first_goal_sec, 3)
        if st.last_goal_sec is not None:
            st.last_goal_sec = round(st.last_goal_sec, 3)
        return st

    @property
    def goals(self) -> list[GoalEvent]:
        return list(self._goals)

    @property
    def score(self) -> dict[str, int]:
        return dict(self._score)

    @property
    def notification(self) -> GoalNotification | None:
        return self._notification

    def set_attack_directions(self, directions: dict[str, int]) -> None:
        """Inject pre-computed attack directions (for live pipeline use)."""
        self._team_attack_dir = dict(directions)

    @staticmethod
    def infer_attack_directions(
        frames: list[dict],
        team_names: tuple[str, str] = ("Team A", "Team B"),
    ) -> dict[str, int]:
        """
        Infer attack direction per team from player positions.

        Teams always attack opposite directions, so we assign them relative to
        each other: the team whose players sit on the *left* half (smaller
        median x) attacks toward the right goal (+1) and vice versa. This is
        far more robust than testing each team against the halfway line
        independently (which can label both teams the same way).
        """
        team_x: dict[str, list[float]] = {t: [] for t in team_names}
        for rec in frames:
            if not rec.get("homography_available"):
                continue
            for p in rec.get("players", []):
                team = p.get("team")
                x = p.get("x_cm")
                if team in team_x and x is not None:
                    team_x[team].append(float(x))

        a, b = team_names
        med: dict[str, float | None] = {}
        for team in team_names:
            xs = team_x[team]
            med[team] = sorted(xs)[len(xs) // 2] if xs else None

        if med[a] is None or med[b] is None:
            return {a: 1, b: -1}

        # Team positioned further left defends the left goal, attacks right.
        if med[a] <= med[b]:
            return {a: 1, b: -1}
        return {a: -1, b: 1}

    # ------------------------------------------------------------------ #
    def _compute_attack_directions(
        self,
        frames: list[dict],
        possession: PossessionAnalyzer,
    ) -> None:
        """Estimate each team's attacking direction from player positions."""
        self._team_attack_dir = self.infer_attack_directions(frames, self.team_names)

    def _detect_crossing(
        self, a: dict, b: dict
    ) -> tuple[str, float, float] | None:
        """Return (goal_line, cross_y_cm, speed_ms) if ball crossed a goal line."""
        dt = (b["ts"] - a["ts"]) or self.dt
        if dt <= 0:
            dt = self.dt

        for goal_x, goal_line in ((0.0, "left"), (float(L), "right")):
            da = a["x"] - goal_x
            db = b["x"] - goal_x
            if da * db >= 0:
                continue

            t = (goal_x - a["x"]) / (b["x"] - a["x"])
            if not (0.0 <= t <= 1.0):
                continue

            cross_y = a["y"] + t * (b["y"] - a["y"])
            if not (
                GOAL_Y_LO - GOAL_MOUTH_MARGIN_CM
                <= cross_y
                <= GOAL_Y_HI + GOAL_MOUTH_MARGIN_CM
            ):
                continue

            vx = (b["x"] - a["x"]) / dt / CM_PER_M
            if goal_line == "left" and vx > -self.min_cross_speed_ms:
                continue
            if goal_line == "right" and vx < self.min_cross_speed_ms:
                continue

            speed = math.hypot(b["x"] - a["x"], b["y"] - a["y"]) / CM_PER_M / dt
            if speed > MAX_PLAUSIBLE_BALL_SPEED_MS:
                continue

            return goal_line, cross_y, speed

        return None

    def _detect_goal_zone(
        self, a: dict, b: dict
    ) -> tuple[str, float, float] | None:
        """
        Fallback: ball enters the goal mouth zone with attacking velocity.

        Handles homography noise where the projected ball overshoots the line
        without a clean segment crossing.
        """
        dt = (b["ts"] - a["ts"]) or self.dt
        if dt <= 0:
            dt = self.dt

        vx = (b["x"] - a["x"]) / dt / CM_PER_M
        vy = (b["y"] - a["y"]) / dt / CM_PER_M
        speed = math.hypot(b["x"] - a["x"], b["y"] - a["y"]) / CM_PER_M / dt
        if speed > MAX_PLAUSIBLE_BALL_SPEED_MS:
            return None

        for goal_line, lo, hi, sign in (
            ("left", -GOAL_ZONE_DEPTH_CM, GOAL_ZONE_DEPTH_CM, -1),
            ("right", L - GOAL_ZONE_DEPTH_CM, L + GOAL_ZONE_DEPTH_CM, 1),
        ):
            in_zone = lo <= b["x"] <= hi and (
                GOAL_Y_LO - GOAL_MOUTH_MARGIN_CM
                <= b["y"]
                <= GOAL_Y_HI + GOAL_MOUTH_MARGIN_CM
            )
            was_outside = (a["x"] - lo) * sign < 0
            moving_in = vx * sign > self.min_cross_speed_ms * 0.5
            if in_zone and (was_outside or moving_in):
                return goal_line, b["y"], max(speed, self.min_cross_speed_ms)

        return None

    def _register_goal(
        self,
        goal_line: str,
        cross_y: float,
        frame: int,
        timestamp_sec: float,
        speed_ms: float,
    ) -> GoalEvent | None:
        """Create a goal event, update score, and fire notification."""
        # Prefer the team that actually last possessed the ball before it
        # entered the goal — far more reliable than attack-direction geometry
        # on noisy broadcast homography. Fall back to goal-line geometry.
        scoring_team, scorer = self._scorer_from_possession(frame)
        if scoring_team is None:
            scoring_team = self._team_for_goal_line(goal_line)
            if scoring_team is None:
                return None
            scorer = self._find_scorer(scoring_team, frame)

        conceded = self._team_b if scoring_team == self._team_a else self._team_a

        self._score[scoring_team] += 1
        self._last_goal_frame = frame

        goal = GoalEvent(
            goal_id=len(self._goals) + 1,
            scoring_team=scoring_team,
            conceded_team=conceded,
            scorer_track_id=scorer,
            goal_line=goal_line,
            cross_y_cm=cross_y,
            frame=frame,
            timestamp_sec=timestamp_sec,
            speed_ms=speed_ms,
            score_team_a=self._score[self._team_a],
            score_team_b=self._score[self._team_b],
        )
        self._goals.append(goal)

        scorer_str = f"#{scorer}" if scorer is not None else "Unknown"
        msg = (
            f"GOAL! {scoring_team} — {scorer_str} "
            f"({self._score[self._team_a]}-{self._score[self._team_b]}) "
            f"@ {timestamp_sec:.1f}s"
        )
        self._notification = GoalNotification(
            goal=goal,
            expires_frame=frame + self.notification_frames,
            message=msg,
        )
        return goal

    def _team_for_goal_line(self, goal_line: str) -> str | None:
        """Map a goal line to the team that scored."""
        if goal_line == "right":
            for team, direction in self._team_attack_dir.items():
                if direction > 0:
                    return team
            return self._team_a
        if goal_line == "left":
            for team, direction in self._team_attack_dir.items():
                if direction < 0:
                    return team
            return self._team_b
        return None

    def _scorer_from_possession(
        self, frame: int
    ) -> tuple[str | None, int | None]:
        """
        Attribute the goal to the last team that held the ball before it went
        in. Returns (team, track_id) or (None, None) if no possession is found
        in the lookback window.
        """
        lookback = max(self.scorer_lookback, 210)
        for pf in reversed(self._possession_history):
            if pf.frame > frame:
                continue
            if pf.frame < frame - lookback:
                break
            if pf.state not in self.TEAM_STATES:
                continue
            if pf.team not in (self._team_a, self._team_b):
                continue
            return pf.team, pf.track_id
        return None, None

    def _find_scorer(self, scoring_team: str, frame: int) -> int | None:
        """Last possession touch by the scoring team before the goal."""
        best_id: int | None = None
        best_frame = -1
        for pf in reversed(self._possession_history):
            if pf.frame > frame:
                continue
            if pf.frame < frame - self.scorer_lookback:
                break
            if pf.track_id is None or pf.team != scoring_team:
                continue
            if pf.state not in self.TEAM_STATES:
                continue
            if pf.frame >= best_frame:
                best_frame = pf.frame
                best_id = pf.track_id
        return best_id

    def _expire_notification(self, frame: int) -> None:
        if self._notification and frame > self._notification.expires_frame:
            self._notification = None
