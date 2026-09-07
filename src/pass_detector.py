"""
Automatic football pass detection from possession + bird's-eye coordinates.

A pass is inferred when possession moves from one player to a teammate
(optionally via a brief loose-ball gap). Pass geometry (distance, direction,
start/end zones on the pitch) drives classification into:

    short | long | through_ball | cross | back

Success is determined by whether a teammate receives and retains the ball;
failure by interception (opponent gains possession) or a misplaced ball
(extended loose phase after release).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from . import field_config as fc
from .possession_analyzer import PossessionAnalyzer, PossessionSegment, PossessionState


class PassType(str, Enum):
    SHORT = "short"
    LONG = "long"
    THROUGH_BALL = "through_ball"
    CROSS = "cross"
    BACK = "back"


CM_PER_M = 100.0

# Distance thresholds (metres).
SHORT_MAX_M = 20.0
LONG_MIN_M = 25.0

# Pitch zone boundaries (cm).
W = fc.PITCH_WIDTH
L = fc.PITCH_LENGTH
WING_LOW = 0.20 * W
WING_HIGH = 0.80 * W
BOX_X_NEAR = fc.PENALTY_BOX_LENGTH
BOX_X_FAR = L - fc.PENALTY_BOX_LENGTH
BOX_Y_LO = (W - fc.PENALTY_BOX_WIDTH) / 2
BOX_Y_HI = (W + fc.PENALTY_BOX_WIDTH) / 2


@dataclass
class PassEvent:
    """A single detected pass."""

    pass_id: int
    team: str
    from_track_id: int
    to_track_id: int | None
    successful: bool
    primary_type: str
    types: list[str]
    distance_m: float
    forward_m: float
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    start_x_cm: float
    start_y_cm: float
    end_x_cm: float | None = None
    end_y_cm: float | None = None
    failure_reason: str | None = None  # intercepted | misplaced

    def to_dict(self) -> dict:
        return {
            "pass_id": self.pass_id,
            "team": self.team,
            "from_track_id": self.from_track_id,
            "to_track_id": self.to_track_id,
            "successful": self.successful,
            "primary_type": self.primary_type,
            "types": self.types,
            "distance_m": round(self.distance_m, 2),
            "forward_m": round(self.forward_m, 2),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_sec": round(self.start_time_sec, 3),
            "end_time_sec": round(self.end_time_sec, 3),
            "start_x_cm": round(self.start_x_cm, 1),
            "start_y_cm": round(self.start_y_cm, 1),
            "end_x_cm": round(self.end_x_cm, 1) if self.end_x_cm is not None else None,
            "end_y_cm": round(self.end_y_cm, 1) if self.end_y_cm is not None else None,
            "failure_reason": self.failure_reason,
        }


@dataclass
class PassStats:
    """Aggregate pass statistics."""

    total: int = 0
    successful: int = 0
    unsuccessful: int = 0
    short: int = 0
    long: int = 0
    through_ball: int = 0
    cross: int = 0
    back: int = 0
    by_player: dict[int, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        rate = (self.successful / self.total * 100) if self.total else 0.0
        return {
            "total": self.total,
            "successful": self.successful,
            "unsuccessful": self.unsuccessful,
            "success_rate_pct": round(rate, 1),
            "short": self.short,
            "long": self.long,
            "through_ball": self.through_ball,
            "cross": self.cross,
            "back": self.back,
            "by_player": {str(k): v for k, v in self.by_player.items()},
        }


class PassDetector:
    """Detects and classifies passes from possession + coordinate frames."""

    TEAM_STATES = {PossessionState.TEAM_A.value, PossessionState.TEAM_B.value}
    GAP_STATES = {
        PossessionState.LOOSE.value,
        PossessionState.CONTESTED.value,
        PossessionState.UNKNOWN.value,
    }

    def __init__(
        self,
        fps: float = 25.0,
        max_loose_gap_frames: int = 45,
        min_receive_frames: int = 2,
        min_passer_hold_frames: int = 2,
        team_names: tuple[str, str] = ("Team A", "Team B"),
    ) -> None:
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 25.0
        self.max_loose_gap = max_loose_gap_frames
        self.min_receive_frames = min_receive_frames
        self.min_passer_hold_frames = min_passer_hold_frames
        self.team_names = team_names

        self._passes: list[PassEvent] = []
        self._player_positions: dict[int, list[tuple[float, float]]] = {}
        self._frame_players: dict[int, dict[int, tuple[float, float]]] = {}
        self._frame_ball: dict[int, tuple[float, float]] = {}
        self._team_attack_dir: dict[str, int] = {}

    def analyze(
        self,
        frames: list[dict],
        possession: PossessionAnalyzer | None = None,
    ) -> list[PassEvent]:
        """
        Detect passes from bird's-eye frame records.

        If `possession` is provided its timeline is used; otherwise possession
        is computed internally from the same frames.
        """
        if possession is None:
            possession = PossessionAnalyzer(fps=self.fps)
            for rec in frames:
                possession.update(rec)
            possession.finalize()

        self._index_positions(frames)
        self._compute_attack_directions(possession)

        self._passes = []
        self._scan_segments(possession.timeline())
        return self._passes

    def stats(self, team: str | None = None) -> PassStats:
        """Aggregate statistics, optionally filtered by team."""
        passes = [p for p in self._passes if team is None or p.team == team]

        s = PassStats()
        s.total = len(passes)
        for p in passes:
            if p.successful:
                s.successful += 1
            else:
                s.unsuccessful += 1
            for t in p.types:
                if t == PassType.SHORT.value:
                    s.short += 1
                elif t == PassType.LONG.value:
                    s.long += 1
                elif t == PassType.THROUGH_BALL.value:
                    s.through_ball += 1
                elif t == PassType.CROSS.value:
                    s.cross += 1
                elif t == PassType.BACK.value:
                    s.back += 1

            fp = s.by_player.setdefault(
                p.from_track_id,
                {"passes": 0, "completed": 0, "to": {}},
            )
            fp["passes"] += 1
            if p.successful:
                fp["completed"] += 1
            if p.to_track_id is not None:
                fp["to"][p.to_track_id] = fp["to"].get(p.to_track_id, 0) + 1
        return s

    @property
    def passes(self) -> list[PassEvent]:
        return list(self._passes)

    # ------------------------------------------------------------------ #
    def _index_positions(self, frames: list[dict]) -> None:
        """Index per-frame player and ball positions on the pitch."""
        self._player_positions.clear()
        self._frame_players.clear()
        self._frame_ball.clear()

        for rec in frames:
            if not rec.get("homography_available"):
                continue
            frame = rec["frame"]
            frame_map: dict[int, tuple[float, float]] = {}
            for p in rec.get("players", []):
                tid = p.get("track_id")
                x, y = p.get("x_cm"), p.get("y_cm")
                if tid is None or x is None or y is None:
                    continue
                pos = (float(x), float(y))
                frame_map[tid] = pos
                self._player_positions.setdefault(tid, []).append(pos)
            if frame_map:
                self._frame_players[frame] = frame_map

            ball = rec.get("ball")
            if ball and ball.get("x_cm") is not None and ball.get("y_cm") is not None:
                self._frame_ball[frame] = (float(ball["x_cm"]), float(ball["y_cm"]))

    def _compute_attack_directions(self, possession: PossessionAnalyzer) -> None:
        """Estimate each team's attacking direction (+x or -x)."""
        team_x: dict[str, list[float]] = {t: [] for t in self.team_names}
        for pf in possession.history:
            if pf.track_id is None or pf.team not in team_x:
                continue
            pos = self._player_pos_at(pf.track_id, pf.frame)
            if pos:
                team_x[pf.team].append(pos[0])

        for team in self.team_names:
            xs = team_x[team]
            if not xs:
                self._team_attack_dir[team] = 1
                continue
            median_x = sorted(xs)[len(xs) // 2]
            self._team_attack_dir[team] = 1 if median_x < L / 2 else -1

    def _scan_segments(self, segments: list[PossessionSegment]) -> None:
        """Walk possession segments and emit pass events."""
        pass_id = 1
        i = 0
        n = len(segments)

        while i < n:
            seg = segments[i]
            if not self._is_team_segment(seg):
                i += 1
                continue

            hold_frames = seg.end_frame - seg.start_frame + 1
            if hold_frames < self.min_passer_hold_frames:
                i += 1
                continue

            passer_id = seg.track_id
            team = seg.team
            assert passer_id is not None and team is not None

            release_frame = seg.end_frame
            release_time = seg.end_time_sec
            start_pos = self._release_pos(passer_id, release_frame)

            j = i + 1
            gap_frames = 0
            receiver_seg: PossessionSegment | None = None
            opponent_seg: PossessionSegment | None = None

            while j < n:
                cur = segments[j]
                if self._is_team_segment(cur) and cur.track_id == passer_id:
                    j += 1
                    continue

                if cur.state in self.GAP_STATES:
                    gap_frames += cur.end_frame - cur.start_frame + 1
                    if gap_frames > self.max_loose_gap:
                        break
                    j += 1
                    continue

                if self._is_team_segment(cur):
                    if cur.team == team and cur.track_id != passer_id:
                        receiver_seg = cur
                    else:
                        opponent_seg = cur
                    break
                j += 1

            if receiver_seg is not None:
                recv_hold = receiver_seg.end_frame - receiver_seg.start_frame + 1
                if recv_hold >= self.min_receive_frames:
                    end_pos = self._player_pos_at(
                        receiver_seg.track_id, receiver_seg.start_frame
                    ) or self._ball_pos_at(receiver_seg.start_frame)
                    if end_pos and start_pos:
                        self._passes.append(
                            self._make_pass(
                                pass_id, team, passer_id, receiver_seg.track_id,
                                True, start_pos, end_pos, release_frame,
                                receiver_seg.start_frame, release_time,
                                receiver_seg.start_time_sec, None,
                            )
                        )
                        pass_id += 1
                    i = j
                    continue

            if opponent_seg is not None and start_pos:
                end_pos = self._player_pos_at(
                    opponent_seg.track_id, opponent_seg.start_frame
                ) or self._ball_pos_at(opponent_seg.start_frame) or start_pos
                self._passes.append(
                    self._make_pass(
                        pass_id, team, passer_id, None,
                        False, start_pos, end_pos, release_frame,
                        opponent_seg.start_frame, release_time,
                        opponent_seg.start_time_sec, "intercepted",
                    )
                )
                pass_id += 1
                i = j
                continue

            if gap_frames > self.max_loose_gap and start_pos:
                end_pos = self._ball_pos_at(release_frame) or start_pos
                self._passes.append(
                    self._make_pass(
                        pass_id, team, passer_id, None,
                        False, start_pos, end_pos, release_frame,
                        release_frame + gap_frames, release_time,
                        release_time + gap_frames * self.dt, "misplaced",
                    )
                )
                pass_id += 1

            i += 1

    def _make_pass(
        self,
        pass_id: int,
        team: str,
        from_id: int,
        to_id: int | None,
        successful: bool,
        start: tuple[float, float],
        end: tuple[float, float],
        start_frame: int,
        end_frame: int,
        start_time: float,
        end_time: float,
        fail_reason: str | None,
    ) -> PassEvent:
        distance, forward = self._pass_geometry(start, end, team)
        types, primary = self._classify_pass(start, end, distance, forward, team)
        return PassEvent(
            pass_id=pass_id,
            team=team,
            from_track_id=from_id,
            to_track_id=to_id,
            successful=successful,
            primary_type=primary,
            types=types,
            distance_m=distance,
            forward_m=forward,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time_sec=start_time,
            end_time_sec=end_time,
            start_x_cm=start[0],
            start_y_cm=start[1],
            end_x_cm=end[0],
            end_y_cm=end[1],
            failure_reason=fail_reason,
        )

    @staticmethod
    def _is_team_segment(seg: PossessionSegment) -> bool:
        return (
            seg.track_id is not None
            and seg.team is not None
            and seg.state in PassDetector.TEAM_STATES
        )

    def _release_pos(self, track_id: int, frame: int) -> tuple[float, float] | None:
        """Ball position at release, falling back to passer position."""
        return self._ball_pos_at(frame) or self._player_pos_at(track_id, frame)

    def _player_pos_at(self, track_id: int, frame: int) -> tuple[float, float] | None:
        frame_map = self._frame_players.get(frame)
        if frame_map and track_id in frame_map:
            return frame_map[track_id]
        # Nearest frame fallback.
        for delta in range(1, 6):
            for f in (frame - delta, frame + delta):
                fm = self._frame_players.get(f)
                if fm and track_id in fm:
                    return fm[track_id]
        return None

    def _ball_pos_at(self, frame: int) -> tuple[float, float] | None:
        if frame in self._frame_ball:
            return self._frame_ball[frame]
        for delta in range(1, 6):
            for f in (frame - delta, frame + delta):
                if f in self._frame_ball:
                    return self._frame_ball[f]
        return None

    def _pass_geometry(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        team: str,
    ) -> tuple[float, float]:
        dx = (end[0] - start[0]) / CM_PER_M
        dy = (end[1] - start[1]) / CM_PER_M
        distance = math.hypot(dx, dy)
        attack_dir = self._team_attack_dir.get(team, 1)
        forward = dx * attack_dir
        return distance, forward

    def _classify_pass(
        self,
        start: tuple[float, float],
        end: tuple[float, float] | None,
        distance: float,
        forward: float,
        team: str,
    ) -> tuple[list[str], str]:
        types: list[str] = []

        if distance < SHORT_MAX_M:
            types.append(PassType.SHORT.value)
        elif distance >= LONG_MIN_M:
            types.append(PassType.LONG.value)
        else:
            types.append(PassType.SHORT.value)

        if end is None:
            return types, types[0]

        sx, sy = start
        ex, ey = end

        if forward < -3.0:
            types.append(PassType.BACK.value)

        from_wing = sy < WING_LOW or sy > WING_HIGH
        into_box = BOX_X_NEAR <= ex <= BOX_X_FAR and BOX_Y_LO <= ey <= BOX_Y_HI
        if from_wing and into_box and distance >= 15.0:
            types.append(PassType.CROSS.value)

        attack_dir = self._team_attack_dir.get(team, 1)
        in_final_third = (ex > 0.66 * L) if attack_dir > 0 else (ex < 0.34 * L)
        if forward >= 15.0 and distance >= 20.0 and in_final_third:
            types.append(PassType.THROUGH_BALL.value)

        priority = [
            PassType.CROSS.value,
            PassType.THROUGH_BALL.value,
            PassType.BACK.value,
            PassType.LONG.value,
            PassType.SHORT.value,
        ]
        primary = types[0]
        for p in priority:
            if p in types:
                primary = p
                break
        return types, primary

    def average_player_positions(self) -> dict[int, tuple[float, float]]:
        """Mean pitch position per tracked player (for network layout)."""
        out: dict[int, tuple[float, float]] = {}
        for tid, positions in self._player_positions.items():
            if not positions:
                continue
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            out[tid] = (sum(xs) / len(xs), sum(ys) / len(ys))
        return out
