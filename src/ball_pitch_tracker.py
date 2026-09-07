"""
Pitch-space ball position tracker with jump rejection and coasting.

Broadcast footage often produces wild homography projections when the ball
moves quickly or the camera pans. This module keeps a stable pitch position
for overlays/possession while exposing raw segments for goal-line crossing
checks (with sub-frame interpolation across implausible jumps).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BallPitchSample:
    """One accepted pitch-space ball observation."""

    frame: int
    x_cm: float
    y_cm: float
    raw: bool = True  # False when coasted/interpolated


class BallPitchTracker:
    """Reject homography teleports; coast briefly; interpolate for crossings."""

    def __init__(
        self,
        max_jump_cm: float = 900.0,
        smoothing: float = 0.6,
        coast_frames: int = 6,
        trail_max: int = 40,
        trail_gap_frames: int = 8,
        gap_reset_frames: int = 4,
        reseat_confirm: int = 3,
        reseat_cluster_cm: float = 700.0,
    ) -> None:
        self.max_jump_cm = max_jump_cm
        self.smoothing = smoothing
        self.coast_frames = coast_frames
        self.trail_max = trail_max
        self.trail_gap_frames = trail_gap_frames
        # A discontinuity larger than this (in frames) forces a re-seat so we
        # do not coast from a stale pre-gap position.
        self.gap_reset_frames = gap_reset_frames
        # If this many consecutive rejected readings cluster together, snap
        # to them (the ball genuinely moved / homography re-acquired).
        self.reseat_confirm = reseat_confirm
        self.reseat_cluster_cm = reseat_cluster_cm

        self._pos: tuple[float, float] | None = None
        self._miss = 0
        self._last_frame = -1
        self.trail: list[tuple[float, float]] = []
        self._last_trail_frame = -1

        # Buffer of recently rejected raw readings (for re-seat detection).
        self._reject_buf: list[tuple[float, float]] = []

    def update(
        self, frame: int, x_cm: float | None, y_cm: float | None
    ) -> BallPitchSample | None:
        """Accept, coast, or reject a pitch-space ball reading."""
        # Long discontinuity (e.g. homography loss / camera cut): drop the
        # stale position so the next reading re-seats instead of coasting.
        if self._pos is not None and frame - self._last_frame > self.gap_reset_frames:
            self._pos = None
            self._reject_buf.clear()

        if x_cm is None or y_cm is None:
            return self._on_miss(frame)

        if self._pos is None:
            self._pos = (x_cm, y_cm)
            self._miss = 0
            self._last_frame = frame
            self._reject_buf.clear()
            self._append_trail(frame, self._pos)
            return BallPitchSample(frame, self._pos[0], self._pos[1])

        jump = math.hypot(x_cm - self._pos[0], y_cm - self._pos[1])
        if jump > self.max_jump_cm:
            # Possibly a homography glitch — but if several rejected readings
            # cluster in a new spot, the ball really moved: snap to it.
            if self._should_reseat(x_cm, y_cm):
                self._pos = (x_cm, y_cm)
                self._miss = 0
                self._last_frame = frame
                self._reject_buf.clear()
                # Re-seat is a teleport for the trail: start a fresh path so we
                # don't draw a long line across the pitch.
                self.trail.clear()
                self._append_trail(frame, self._pos)
                return BallPitchSample(frame, x_cm, y_cm, raw=True)
            return self._on_miss(frame)

        self._reject_buf.clear()
        a = self.smoothing
        sx = a * self._pos[0] + (1 - a) * x_cm
        sy = a * self._pos[1] + (1 - a) * y_cm
        self._pos = (sx, sy)
        self._miss = 0
        self._last_frame = frame
        self._append_trail(frame, self._pos)
        return BallPitchSample(frame, sx, sy, raw=True)

    def _should_reseat(self, x_cm: float, y_cm: float) -> bool:
        """True when recent rejected readings cluster around this new point."""
        buf = self._reject_buf
        if buf:
            cx = sum(p[0] for p in buf) / len(buf)
            cy = sum(p[1] for p in buf) / len(buf)
            if math.hypot(x_cm - cx, y_cm - cy) > self.reseat_cluster_cm:
                # New reading disagrees with the cluster; restart the cluster.
                self._reject_buf = [(x_cm, y_cm)]
                return False
        self._reject_buf.append((x_cm, y_cm))
        return len(self._reject_buf) >= self.reseat_confirm

    def crossing_segments(
        self, prev: dict, cur: dict, pitch_length_cm: float
    ) -> list[tuple[dict, dict]]:
        """
        Return one or more (a, b) segments for goal-line crossing tests.

        Large jumps are split into interpolated sub-steps so a fast shot
        that projects as a single-frame teleport can still cross the line.
        """
        ax, ay = prev["x"], prev["y"]
        bx, by = cur["x"], cur["y"]
        jump = math.hypot(bx - ax, by - ay)
        if jump <= self.max_jump_cm * 1.5:
            return [(prev, cur)]

        steps = max(2, int(math.ceil(jump / self.max_jump_cm)))
        segs: list[tuple[dict, dict]] = []
        last = prev
        for i in range(1, steps + 1):
            t = i / steps
            mid = {
                "frame": cur["frame"],
                "ts": prev["ts"] + t * (cur["ts"] - prev["ts"]),
                "x": ax + t * (bx - ax),
                "y": ay + t * (by - ay),
            }
            segs.append((last, mid))
            last = mid
        return segs

    def reset(self) -> None:
        self._pos = None
        self._miss = 0
        self._last_frame = -1
        self.trail.clear()
        self._last_trail_frame = -1
        self._reject_buf.clear()

    # ------------------------------------------------------------------ #
    def _on_miss(self, frame: int) -> BallPitchSample | None:
        self._miss += 1
        if self._pos is not None and self._miss <= self.coast_frames:
            self._last_frame = frame
            self._append_trail(frame, self._pos, coasted=True)
            return BallPitchSample(frame, self._pos[0], self._pos[1], raw=False)
        if self._miss > self.coast_frames:
            self._pos = None
        return None

    def _append_trail(
        self, frame: int, pos: tuple[float, float], coasted: bool = False
    ) -> None:
        gap = (
            self._last_trail_frame >= 0
            and frame - self._last_trail_frame > self.trail_gap_frames
        )
        # Also break the trail on any non-physical jump from the last point.
        big_jump = bool(self.trail) and (
            math.hypot(pos[0] - self.trail[-1][0], pos[1] - self.trail[-1][1])
            > self.max_jump_cm
        )
        if gap or big_jump:
            self.trail.clear()
        self.trail.append(pos)
        if len(self.trail) > self.trail_max:
            self.trail = self.trail[-self.trail_max :]
        self._last_trail_frame = frame
