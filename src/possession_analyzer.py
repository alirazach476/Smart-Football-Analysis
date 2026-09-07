"""
Automatic football possession analysis from bird's-eye coordinates.

Each frame with homography + ball position:
  1. Find the nearest player (pitch cm, Euclidean distance).
  2. If within the possession radius -> assign to that player/team.
  3. If two opponents are within a contested margin -> mark contested.
  4. Otherwise -> loose ball.

Temporal stickiness prevents flicker: the current possessor keeps the ball
until another player is meaningfully closer.

Accumulates team possession %, per-player possession time, and a segment
timeline for export and live display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class PossessionState(str, Enum):
    TEAM_A = "Team A"
    TEAM_B = "Team B"
    LOOSE = "loose"
    CONTESTED = "contested"
    UNKNOWN = "unknown"  # no homography / no ball


@dataclass
class PossessionFrame:
    """Possession snapshot for one video frame."""

    frame: int
    timestamp_sec: float
    state: str
    track_id: int | None = None
    team: str | None = None
    nearest_track_id: int | None = None
    nearest_distance_m: float | None = None
    second_nearest_distance_m: float | None = None

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "state": self.state,
            "track_id": self.track_id,
            "team": self.team,
            "nearest_track_id": self.nearest_track_id,
            "nearest_distance_m": round(self.nearest_distance_m, 2)
            if self.nearest_distance_m is not None
            else None,
            "second_nearest_distance_m": round(self.second_nearest_distance_m, 2)
            if self.second_nearest_distance_m is not None
            else None,
        }


@dataclass
class PossessionSegment:
    """Contiguous stretch of the same possession state."""

    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    state: str
    track_id: int | None = None
    team: str | None = None

    def to_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_sec": round(self.start_time_sec, 3),
            "end_time_sec": round(self.end_time_sec, 3),
            "duration_sec": round(self.duration_sec, 2),
            "state": self.state,
            "track_id": self.track_id,
            "team": self.team,
        }


@dataclass
class PlayerPossessionStats:
    track_id: int
    class_name: str
    team: str | None
    possession_time_sec: float = 0.0
    possession_pct: float = 0.0  # % of total assigned possession time
    touches: int = 0               # number of separate possession segments

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "team": self.team,
            "possession_time_sec": round(self.possession_time_sec, 2),
            "possession_pct": round(self.possession_pct, 1),
            "touches": self.touches,
        }


@dataclass
class TeamPossessionStats:
    team: str
    possession_time_sec: float = 0.0
    possession_pct: float = 0.0  # % of team-on-team time (excludes loose)

    def to_dict(self) -> dict:
        return {
            "team": self.team,
            "possession_time_sec": round(self.possession_time_sec, 2),
            "possession_pct": round(self.possession_pct, 1),
        }


class PossessionAnalyzer:
    """Tracks ball possession from bird's-eye frame records."""

    TRACKABLE = {"player", "goalkeeper"}

    def __init__(
        self,
        fps: float = 25.0,
        possession_radius_m: float = 5.5,
        contested_margin_m: float = 0.5,
        stickiness_margin_m: float = 0.8,
        switch_confirm_frames: int = 4,
        team_names: tuple[str, str] = ("Team A", "Team B"),
    ) -> None:
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 25.0
        self.possession_radius_cm = possession_radius_m * 100.0
        self.contested_margin_cm = contested_margin_m * 100.0
        self.stickiness_margin_cm = stickiness_margin_m * 100.0
        self.switch_confirm_frames = switch_confirm_frames
        self.team_names = team_names

        self._history: list[PossessionFrame] = []
        self._segments: list[PossessionSegment] = []
        self._current: PossessionFrame | None = None

        # Running time accumulators (seconds).
        self._team_time: dict[str, float] = {team_names[0]: 0.0, team_names[1]: 0.0}
        self._loose_time = 0.0
        self._contested_time = 0.0
        self._unknown_time = 0.0
        self._player_time: dict[int, float] = {}
        self._player_meta: dict[int, dict] = {}
        self._player_touches: dict[int, int] = {}

        # Stickiness state.
        self._holder_id: int | None = None
        self._holder_team: str | None = None

        # Temporal debounce — require N consecutive frames before switching.
        self._stable: tuple[str, int | None, str | None] | None = None
        self._pending: tuple[str, int | None, str | None] | None = None
        self._pending_count = 0

        # Open segment builder.
        self._seg_start_frame = 0
        self._seg_start_time = 0.0
        self._seg_state = PossessionState.UNKNOWN.value
        self._seg_track_id: int | None = None
        self._seg_team: str | None = None

    # ------------------------------------------------------------------ #
    def update(self, record: dict) -> PossessionFrame:
        """Process one bird's-eye frame record and return possession state."""
        frame = record["frame"]
        ts = record.get("timestamp_sec", frame * self.dt)

        if not record.get("homography_available"):
            pf = PossessionFrame(frame, ts, PossessionState.UNKNOWN.value)
            self._commit(pf)
            return pf

        ball = record.get("ball")
        players = record.get("players", [])

        if not ball or "x_cm" not in ball:
            pf = PossessionFrame(frame, ts, PossessionState.LOOSE.value)
            self._commit(pf)
            return pf

        bx, by = ball["x_cm"], ball["y_cm"]
        candidates = self._candidates(players, bx, by)

        if not candidates:
            pf = PossessionFrame(frame, ts, PossessionState.LOOSE.value)
            self._commit(pf)
            return pf

        # Sort by distance to ball.
        candidates.sort(key=lambda c: c["dist_cm"])
        nearest = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None

        dist_m = nearest["dist_cm"] / 100.0
        second_dist_m = second["dist_cm"] / 100.0 if second else None

        state, track_id, team = self._decide(nearest, second, candidates)
        state, track_id, team = self._debounce(state, track_id, team)

        if track_id is not None:
            self._player_meta[track_id] = {
                "class_name": nearest.get("class_name", "player"),
                "team": team,
            }

        pf = PossessionFrame(
            frame=frame,
            timestamp_sec=ts,
            state=state,
            track_id=track_id,
            team=team,
            nearest_track_id=nearest["track_id"],
            nearest_distance_m=dist_m,
            second_nearest_distance_m=second_dist_m,
        )
        self._commit(pf)
        return pf

    @property
    def current(self) -> PossessionFrame | None:
        return self._current

    def team_stats(self) -> list[TeamPossessionStats]:
        """Team possession percentages (excludes loose/contested from denominator)."""
        assigned = self._team_time[self.team_names[0]] + self._team_time[self.team_names[1]]
        stats = []
        for name in self.team_names:
            t = self._team_time[name]
            pct = (t / assigned * 100.0) if assigned > 0 else 0.0
            stats.append(TeamPossessionStats(team=name, possession_time_sec=t, possession_pct=pct))
        return stats

    def player_stats(self) -> list[PlayerPossessionStats]:
        """Per-player possession time and touch count."""
        total_player = sum(self._player_time.values())
        result = []
        for tid, t in sorted(self._player_time.items()):
            meta = self._player_meta.get(tid, {})
            pct = (t / total_player * 100.0) if total_player > 0 else 0.0
            result.append(
                PlayerPossessionStats(
                    track_id=tid,
                    class_name=meta.get("class_name", "player"),
                    team=meta.get("team"),
                    possession_time_sec=t,
                    possession_pct=pct,
                    touches=self._player_touches.get(tid, 0),
                )
            )
        return result

    def timeline(self) -> list[PossessionSegment]:
        return list(self._segments)

    @property
    def history(self) -> list[PossessionFrame]:
        return list(self._history)

    def summary(self) -> dict:
        assigned = self._team_time[self.team_names[0]] + self._team_time[self.team_names[1]]
        total = assigned + self._loose_time + self._contested_time + self._unknown_time
        return {
            "frames_analyzed": len(self._history),
            "team_a_pct": round(
                self._team_time[self.team_names[0]] / assigned * 100, 1
            )
            if assigned > 0
            else 0.0,
            "team_b_pct": round(
                self._team_time[self.team_names[1]] / assigned * 100, 1
            )
            if assigned > 0
            else 0.0,
            "loose_pct": round(self._loose_time / total * 100, 1) if total > 0 else 0.0,
            "contested_pct": round(self._contested_time / total * 100, 1)
            if total > 0
            else 0.0,
            "total_possession_sec": round(assigned, 1),
            "loose_sec": round(self._loose_time, 1),
        }

    def finalize(self) -> None:
        """Close the last open timeline segment."""
        if self._history:
            last = self._history[-1]
            self._close_segment(last.frame, last.timestamp_sec)

    def load_from_json(self, path: str) -> None:
        """Rebuild possession from a birdseye_coordinates.json file."""
        import json

        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if "fps" in payload:
            self.fps = payload["fps"]
            self.dt = 1.0 / self.fps
        for record in payload.get("frames", []):
            self.update(record)
        self.finalize()

    # ------------------------------------------------------------------ #
    def _candidates(
        self, players: list[dict], bx: float, by: float
    ) -> list[dict]:
        """Players with pitch position and distance to ball, excluding referees."""
        out = []
        for p in players:
            if p.get("class_name") not in self.TRACKABLE:
                continue
            tid = p.get("track_id")
            x, y = p.get("x_cm"), p.get("y_cm")
            if tid is None or x is None or y is None:
                continue
            dist = math.hypot(x - bx, y - by)
            out.append(
                {
                    "track_id": tid,
                    "team": p.get("team"),
                    "class_name": p.get("class_name", "player"),
                    "x_cm": x,
                    "y_cm": y,
                    "dist_cm": dist,
                }
            )
        return out

    def _decide(
        self,
        nearest: dict,
        second: dict | None,
        candidates: list[dict],
    ) -> tuple[str, int | None, str | None]:
        """Determine possession state from distance rankings."""
        dist = nearest["dist_cm"]
        tid = nearest["track_id"]
        team = nearest.get("team")

        # Contested: two players from different teams within contested margin.
        if second and team and second.get("team") and team != second["team"]:
            gap = second["dist_cm"] - dist
            if gap < self.contested_margin_cm and dist < self.possession_radius_cm * 1.5:
                self._holder_id = None
                self._holder_team = None
                return PossessionState.CONTESTED.value, None, None

        # Outside possession radius -> loose ball.
        if dist > self.possession_radius_cm:
            # Stickiness: keep holder if they are still nearest and within extended range.
            if self._holder_id is not None and tid == self._holder_id:
                if dist < self.possession_radius_cm * 1.5:
                    return self._state_for_holder()
            self._holder_id = None
            self._holder_team = None
            return PossessionState.LOOSE.value, None, None

        # Stickiness: current holder keeps ball unless challenger is closer by margin.
        if self._holder_id is not None and tid != self._holder_id:
            holder = next((c for c in candidates if c["track_id"] == self._holder_id), None)
            if holder:
                if holder["dist_cm"] - dist < self.stickiness_margin_cm:
                    tid = self._holder_id
                    team = self._holder_team
                    nearest = holder

        # Assign possession.
        self._holder_id = tid
        self._holder_team = team

        if team == self.team_names[0]:
            return PossessionState.TEAM_A.value, tid, team
        if team == self.team_names[1]:
            return PossessionState.TEAM_B.value, tid, team
        # Player without team label: still track by player, state = team name or unknown
        return team or "unassigned", tid, team

    def _state_for_holder(self) -> tuple[str, int | None, str | None]:
        tid = self._holder_id
        team = self._holder_team
        if team == self.team_names[0]:
            return PossessionState.TEAM_A.value, tid, team
        if team == self.team_names[1]:
            return PossessionState.TEAM_B.value, tid, team
        return team or PossessionState.LOOSE.value, tid, team

    def _debounce(
        self, state: str, track_id: int | None, team: str | None
    ) -> tuple[str, int | None, str | None]:
        """
        Hold the current possessor until a new assignment is stable for
        several consecutive frames (reduces 1-frame flicker from jitter).
        """
        candidate = (state, track_id, team)
        if self._stable is None:
            self._stable = candidate
            return candidate

        if candidate == self._stable:
            self._pending = None
            self._pending_count = 0
            return candidate

        # Loose/contested can take over immediately when the ball is clearly free.
        if state in (PossessionState.LOOSE.value, PossessionState.CONTESTED.value):
            if self._stable[0] in (
                PossessionState.LOOSE.value,
                PossessionState.CONTESTED.value,
            ):
                self._stable = candidate
                return candidate
            if self._pending == candidate:
                self._pending_count += 1
            else:
                self._pending = candidate
                self._pending_count = 1
            if self._pending_count >= max(2, self.switch_confirm_frames // 2):
                self._stable = candidate
                self._holder_id = None
                self._holder_team = None
                return candidate
            return self._stable

        if self._pending == candidate:
            self._pending_count += 1
        else:
            self._pending = candidate
            self._pending_count = 1

        if self._pending_count >= self.switch_confirm_frames:
            self._stable = candidate
            self._pending = None
            self._pending_count = 0
            if track_id is not None:
                self._holder_id = track_id
                self._holder_team = team
            return candidate

        return self._stable

    def _commit(self, pf: PossessionFrame) -> None:
        """Record frame, accumulate times, update timeline segments."""
        self._history.append(pf)
        self._current = pf

        # Accumulate time for this frame.
        if pf.state == PossessionState.TEAM_A.value:
            self._team_time[self.team_names[0]] += self.dt
        elif pf.state == PossessionState.TEAM_B.value:
            self._team_time[self.team_names[1]] += self.dt
        elif pf.state == PossessionState.LOOSE.value:
            self._loose_time += self.dt
        elif pf.state == PossessionState.CONTESTED.value:
            self._contested_time += self.dt
        else:
            self._unknown_time += self.dt

        if pf.track_id is not None:
            self._player_time[pf.track_id] = (
                self._player_time.get(pf.track_id, 0.0) + self.dt
            )
            if pf.track_id not in self._player_meta:
                self._player_meta[pf.track_id] = {
                    "class_name": "player",
                    "team": pf.team,
                }

        # Timeline segment tracking.
        key = (pf.state, pf.track_id, pf.team)
        prev_key = (self._seg_state, self._seg_track_id, self._seg_team)
        if key != prev_key:
            if self._history and len(self._history) > 1:
                self._close_segment(pf.frame - 1, pf.timestamp_sec - self.dt)
            self._seg_start_frame = pf.frame
            self._seg_start_time = pf.timestamp_sec
            self._seg_state = pf.state
            self._seg_track_id = pf.track_id
            self._seg_team = pf.team
            if (
                pf.track_id is not None
                and pf.state in (PossessionState.TEAM_A.value, PossessionState.TEAM_B.value)
            ):
                self._player_touches[pf.track_id] = (
                    self._player_touches.get(pf.track_id, 0) + 1
                )

    def _close_segment(self, end_frame: int, end_time: float) -> None:
        if end_frame < self._seg_start_frame:
            return
        duration = end_time - self._seg_start_time
        if duration < 0:
            duration = 0.0
        self._segments.append(
            PossessionSegment(
                start_frame=self._seg_start_frame,
                end_frame=end_frame,
                start_time_sec=self._seg_start_time,
                end_time_sec=end_time,
                duration_sec=duration,
                state=self._seg_state,
                track_id=self._seg_track_id,
                team=self._seg_team,
            )
        )
