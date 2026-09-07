"""
Player speed estimation from bird's-eye pitch coordinates.

All positions are in pitch centimetres (from the homography module). Speeds are
reported in m/s and km/h; distances in metres. Per tracked player we maintain:

    position history  ->  frame-to-frame distance
                     ->  instantaneous speed (smoothed)
                     ->  acceleration
                     ->  running average / maximum speed
                     ->  sprint event detection

Sprint detection uses a configurable speed threshold (default 7.0 m/s, ~25 km/h,
a common elite-sprint cutoff in football analytics) with a minimum duration so
brief homography spikes are not counted as sprints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


CM_PER_M = 100.0
# Reject frame-to-frame jumps above this speed (homography glitches / ID swaps).
MAX_PLAUSIBLE_SPEED_M_S = 12.0  # ~43 km/h, above any realistic football sprint


@dataclass
class SpeedSample:
    """Per-frame speed state for one player."""

    frame: int
    timestamp_sec: float
    x_cm: float
    y_cm: float
    distance_step_m: float       # distance moved since last frame
    speed_m_s: float             # smoothed instantaneous speed
    speed_kmh: float
    acceleration_m_s2: float
    total_distance_m: float
    is_sprinting: bool

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "x_cm": round(self.x_cm, 1),
            "y_cm": round(self.y_cm, 1),
            "distance_step_m": round(self.distance_step_m, 3),
            "speed_m_s": round(self.speed_m_s, 2),
            "speed_kmh": round(self.speed_kmh, 1),
            "acceleration_m_s2": round(self.acceleration_m_s2, 2),
            "total_distance_m": round(self.total_distance_m, 2),
            "is_sprinting": self.is_sprinting,
        }


@dataclass
class SprintEvent:
    """A contiguous period where speed stayed above the sprint threshold."""

    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    distance_m: float
    max_speed_m_s: float
    max_speed_kmh: float

    def to_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_sec": round(self.start_time_sec, 3),
            "end_time_sec": round(self.end_time_sec, 3),
            "duration_sec": round(self.duration_sec, 2),
            "distance_m": round(self.distance_m, 2),
            "max_speed_m_s": round(self.max_speed_m_s, 2),
            "max_speed_kmh": round(self.max_speed_kmh, 1),
        }


@dataclass
class PlayerSpeedStats:
    """Aggregate speed statistics for one tracked player."""

    track_id: int
    class_name: str
    team: str | None = None
    total_distance_m: float = 0.0
    avg_speed_m_s: float = 0.0
    max_speed_m_s: float = 0.0
    max_speed_kmh: float = 0.0
    sprint_count: int = 0
    total_sprint_distance_m: float = 0.0
    total_sprint_time_sec: float = 0.0
    sprints: list[SprintEvent] = field(default_factory=list)
    history: list[SpeedSample] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "team": self.team,
            "total_distance_m": round(self.total_distance_m, 2),
            "avg_speed_m_s": round(self.avg_speed_m_s, 2),
            "avg_speed_kmh": round(self.avg_speed_m_s * 3.6, 1),
            "max_speed_m_s": round(self.max_speed_m_s, 2),
            "max_speed_kmh": round(self.max_speed_kmh, 1),
            "sprint_count": self.sprint_count,
            "total_sprint_distance_m": round(self.total_sprint_distance_m, 2),
            "total_sprint_time_sec": round(self.total_sprint_time_sec, 2),
            "sprints": [s.to_dict() for s in self.sprints],
            "history": [h.to_dict() for h in self.history],
        }


class _PlayerTrack:
    """Internal per-track state used by SpeedEstimator."""

    def __init__(
        self,
        track_id: int,
        class_name: str,
        team: str | None,
        fps: float,
        sprint_threshold_m_s: float,
        min_sprint_sec: float,
        speed_smoothing: float,
    ) -> None:
        self.track_id = track_id
        self.class_name = class_name
        self.team = team
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 25.0
        self.sprint_threshold = sprint_threshold_m_s
        self.min_sprint_sec = min_sprint_sec
        self.speed_smoothing = speed_smoothing

        self._prev_x: float | None = None
        self._prev_y: float | None = None
        self._prev_speed: float = 0.0
        self._smoothed_speed: float = 0.0

        self.total_distance_m = 0.0
        self._speed_sum = 0.0
        self._speed_count = 0
        self.max_speed_m_s = 0.0
        self.history: list[SpeedSample] = []

        # Sprint state machine.
        self._in_sprint = False
        self._sprint_start_frame = 0
        self._sprint_start_time = 0.0
        self._sprint_distance = 0.0
        self._sprint_max_speed = 0.0
        self.sprints: list[SprintEvent] = []

    def update(
        self, frame: int, timestamp_sec: float, x_cm: float, y_cm: float
    ) -> SpeedSample:
        """Ingest a new bird's-eye position and return the frame's speed sample."""
        dist_step_m = 0.0
        raw_speed = 0.0
        accel = 0.0

        if self._prev_x is not None:
            dx = x_cm - self._prev_x
            dy = y_cm - self._prev_y
            dist_step_m = math.hypot(dx, dy) / CM_PER_M
            raw_speed = dist_step_m / self.dt if self.dt > 0 else 0.0
            # Discard implausible spikes from homography jitter or track swaps.
            if raw_speed > MAX_PLAUSIBLE_SPEED_M_S:
                dist_step_m = 0.0
                raw_speed = self._prev_speed
            accel = (raw_speed - self._prev_speed) / self.dt if self.dt > 0 else 0.0

        # Exponential smoothing damps homography jitter.
        alpha = 1.0 - self.speed_smoothing
        if self._smoothed_speed == 0.0 and raw_speed > 0:
            self._smoothed_speed = raw_speed
        else:
            self._smoothed_speed = (
                self.speed_smoothing * self._smoothed_speed + alpha * raw_speed
            )

        speed = self._smoothed_speed
        self.total_distance_m += dist_step_m
        self._speed_sum += speed
        self._speed_count += 1
        self.max_speed_m_s = max(self.max_speed_m_s, speed)

        is_sprinting = speed >= self.sprint_threshold
        self._update_sprint(
            frame, timestamp_sec, is_sprinting, dist_step_m, speed
        )

        sample = SpeedSample(
            frame=frame,
            timestamp_sec=timestamp_sec,
            x_cm=x_cm,
            y_cm=y_cm,
            distance_step_m=dist_step_m,
            speed_m_s=speed,
            speed_kmh=speed * 3.6,
            acceleration_m_s2=accel,
            total_distance_m=self.total_distance_m,
            is_sprinting=is_sprinting,
        )
        self.history.append(sample)

        self._prev_x, self._prev_y = x_cm, y_cm
        self._prev_speed = speed
        return sample

    def _update_sprint(
        self,
        frame: int,
        timestamp_sec: float,
        is_sprinting: bool,
        dist_step_m: float,
        speed: float,
    ) -> None:
        if is_sprinting and not self._in_sprint:
            self._in_sprint = True
            self._sprint_start_frame = frame
            self._sprint_start_time = timestamp_sec
            self._sprint_distance = 0.0
            self._sprint_max_speed = speed
        elif is_sprinting and self._in_sprint:
            self._sprint_distance += dist_step_m
            self._sprint_max_speed = max(self._sprint_max_speed, speed)
        elif not is_sprinting and self._in_sprint:
            duration = timestamp_sec - self._sprint_start_time
            if duration >= self.min_sprint_sec:
                self.sprints.append(
                    SprintEvent(
                        start_frame=self._sprint_start_frame,
                        end_frame=frame - 1,
                        start_time_sec=self._sprint_start_time,
                        end_time_sec=timestamp_sec,
                        duration_sec=duration,
                        distance_m=self._sprint_distance,
                        max_speed_m_s=self._sprint_max_speed,
                        max_speed_kmh=self._sprint_max_speed * 3.6,
                    )
                )
            self._in_sprint = False

    def finalize(self, frame: int, timestamp_sec: float) -> None:
        """Close an open sprint at end of video."""
        if self._in_sprint:
            duration = timestamp_sec - self._sprint_start_time
            if duration >= self.min_sprint_sec:
                self.sprints.append(
                    SprintEvent(
                        start_frame=self._sprint_start_frame,
                        end_frame=frame,
                        start_time_sec=self._sprint_start_time,
                        end_time_sec=timestamp_sec,
                        duration_sec=duration,
                        distance_m=self._sprint_distance,
                        max_speed_m_s=self._sprint_max_speed,
                        max_speed_kmh=self._sprint_max_speed * 3.6,
                    )
                )
            self._in_sprint = False

    def stats(self) -> PlayerSpeedStats:
        avg = self._speed_sum / self._speed_count if self._speed_count else 0.0
        sprint_dist = sum(s.distance_m for s in self.sprints)
        sprint_time = sum(s.duration_sec for s in self.sprints)
        return PlayerSpeedStats(
            track_id=self.track_id,
            class_name=self.class_name,
            team=self.team,
            total_distance_m=self.total_distance_m,
            avg_speed_m_s=avg,
            max_speed_m_s=self.max_speed_m_s,
            max_speed_kmh=self.max_speed_m_s * 3.6,
            sprint_count=len(self.sprints),
            total_sprint_distance_m=sprint_dist,
            total_sprint_time_sec=sprint_time,
            sprints=list(self.sprints),
            history=list(self.history),
        )


class SpeedEstimator:
    """
    Tracks speed for all players using bird's-eye pitch coordinates.

    Only `player` and `goalkeeper` classes are tracked for speed analytics;
    referees are excluded.
    """

    TRACKABLE_CLASSES = {"player", "goalkeeper"}

    def __init__(
        self,
        fps: float = 25.0,
        sprint_threshold_m_s: float = 7.0,
        min_sprint_sec: float = 1.0,
        speed_smoothing: float = 0.4,
    ) -> None:
        self.fps = fps
        self.sprint_threshold_m_s = sprint_threshold_m_s
        self.min_sprint_sec = min_sprint_sec
        self.speed_smoothing = speed_smoothing
        self._tracks: dict[int, _PlayerTrack] = {}
        # Latest per-frame sample keyed by track_id (for live overlay).
        self._current: dict[int, SpeedSample] = {}

    def update(
        self,
        frame: int,
        timestamp_sec: float,
        players: list[dict],
    ) -> dict[int, SpeedSample]:
        """
        Update all players with their bird's-eye positions for this frame.

        `players` is the list of dicts from the bird's-eye pipeline, each with
        track_id, class_name, team, x_cm, y_cm.

        Returns the current-frame speed samples keyed by track_id.
        """
        self._current.clear()
        seen: set[int] = set()

        for p in players:
            tid = p.get("track_id")
            if tid is None:
                continue
            cls = p.get("class_name", "player")
            if cls not in self.TRACKABLE_CLASSES:
                continue

            track = self._tracks.get(tid)
            if track is None:
                track = _PlayerTrack(
                    track_id=tid,
                    class_name=cls,
                    team=p.get("team"),
                    fps=self.fps,
                    sprint_threshold_m_s=self.sprint_threshold_m_s,
                    min_sprint_sec=self.min_sprint_sec,
                    speed_smoothing=self.speed_smoothing,
                )
                self._tracks[tid] = track
            elif p.get("team") and track.team is None:
                track.team = p["team"]

            sample = track.update(
                frame, timestamp_sec, p["x_cm"], p["y_cm"]
            )
            self._current[tid] = sample
            seen.add(tid)

        return dict(self._current)

    def current_speed(self, track_id: int) -> SpeedSample | None:
        """Latest speed sample for a track (for live video overlay)."""
        return self._current.get(track_id)

    def finalize(self, last_frame: int, last_timestamp_sec: float) -> None:
        """Close open sprint events at end of processing."""
        for track in self._tracks.values():
            track.finalize(last_frame, last_timestamp_sec)

    def all_stats(self) -> list[PlayerSpeedStats]:
        """Aggregate statistics for every tracked player."""
        return [t.stats() for t in self._tracks.values()]

    def summary(self) -> dict:
        """High-level summary for the run log."""
        stats = self.all_stats()
        if not stats:
            return {"players_tracked": 0}
        max_player = max(stats, key=lambda s: s.max_speed_m_s)
        return {
            "players_tracked": len(stats),
            "fastest_player_id": max_player.track_id,
            "fastest_max_speed_kmh": round(max_player.max_speed_kmh, 1),
            "total_sprints": sum(s.sprint_count for s in stats),
        }
