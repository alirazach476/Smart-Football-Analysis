"""
Automatic highlight generation.

Aggregates match events (goals, shots, fouls, cards, penalties, corners) into a
single ranked timeline, extracts a clip window around each from the source
video, merges the windows (deduplicating overlaps), overlays an event label,
and writes a single `highlights.mp4`.

The highlight timeline is also exported to JSON so the clip list is inspectable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import cv2

from .event_detector import EventType


# Relative importance for ordering / optional filtering.
EVENT_PRIORITY = {
    "goal": 100,
    "penalty": 90,
    "red_card": 80,
    "shot": 70,
    "yellow_card": 60,
    "foul": 40,
    "corner": 30,
}

# Per-type label colour (BGR).
EVENT_COLOR = {
    "goal": (60, 220, 60),
    "penalty": (0, 215, 255),
    "red_card": (60, 60, 230),
    "yellow_card": (40, 220, 240),
    "shot": (230, 180, 60),
    "foul": (180, 180, 180),
    "corner": (230, 140, 40),
}


@dataclass
class HighlightItem:
    """One entry in the highlight timeline."""

    type: str
    frame: int
    timestamp_sec: float
    label: str
    team: str | None = None
    priority: int = 0

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "frame": self.frame,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "label": self.label,
            "team": self.team,
            "priority": self.priority,
        }


@dataclass
class _Window:
    start_frame: int
    end_frame: int
    items: list[HighlightItem] = field(default_factory=list)


class HighlightGenerator:
    """Builds a highlight reel from detected match events."""

    def __init__(
        self,
        source: str,
        output_path: str = "output/highlights.mp4",
        timeline_json: str = "output/highlights.json",
        pre_seconds: float = 4.0,
        post_seconds: float = 3.0,
        title_cards: bool = True,
    ) -> None:
        self.source = source
        self.output_path = output_path
        self.timeline_json = timeline_json
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.title_cards = title_cards

        self._items: list[HighlightItem] = []

    # ------------------------------------------------------------------ #
    def add_goals(self, goals: list) -> None:
        for g in goals:
            scorer = f" — #{g.scorer_track_id}" if g.scorer_track_id else ""
            self._items.append(HighlightItem(
                type="goal",
                frame=g.frame,
                timestamp_sec=g.timestamp_sec,
                label=f"GOAL! {g.scoring_team}{scorer} ({g.score_team_a}-{g.score_team_b})",
                team=g.scoring_team,
                priority=EVENT_PRIORITY["goal"],
            ))

    def add_shots(self, shots: list) -> None:
        for s in shots:
            self._items.append(HighlightItem(
                type="shot",
                frame=s.start_frame,
                timestamp_sec=s.start_time_sec,
                label=f"Shot {s.outcome.replace('_', ' ')} — {s.team} ({s.speed_kmh:.0f} km/h)",
                team=s.team,
                priority=EVENT_PRIORITY["shot"],
            ))

    def add_events(self, events: list) -> None:
        for e in events:
            etype = e.type
            label = {
                EventType.CORNER.value: "Corner",
                EventType.PENALTY.value: "Penalty",
                EventType.FOUL.value: "Foul",
                EventType.YELLOW_CARD.value: "Yellow card",
                EventType.RED_CARD.value: "Red card",
            }.get(etype, etype.title())
            if e.team:
                label = f"{label} — {e.team}"
            self._items.append(HighlightItem(
                type=etype,
                frame=e.frame,
                timestamp_sec=e.timestamp_sec,
                label=label,
                team=e.team,
                priority=EVENT_PRIORITY.get(etype, 10),
            ))

    # ------------------------------------------------------------------ #
    def generate(self) -> dict:
        """Extract and merge clips into a single highlights video."""
        if not os.path.exists(self.source):
            raise FileNotFoundError(f"Source video not found: {self.source}")

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        items = sorted(self._items, key=lambda it: it.frame)
        windows = self._build_windows(items, fps, total)

        self._export_timeline(items, windows, fps)

        if not windows:
            cap.release()
            return {
                "output_path": self.output_path,
                "timeline_json": self.timeline_json,
                "clips": 0,
                "events": len(items),
                "duration_sec": 0.0,
            }

        _ensure_parent(self.output_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open writer: {self.output_path}")

        frames_written = 0
        try:
            for win in windows:
                cap.set(cv2.CAP_PROP_POS_FRAMES, win.start_frame)
                label = self._window_label(win)
                color = EVENT_COLOR.get(win.items[0].type, (255, 255, 255)) if win.items else (255, 255, 255)
                for fidx in range(win.start_frame, win.end_frame + 1):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    self._draw_label(frame, label, color)
                    writer.write(frame)
                    frames_written += 1
        finally:
            cap.release()
            writer.release()

        return {
            "output_path": self.output_path,
            "timeline_json": self.timeline_json,
            "clips": len(windows),
            "events": len(items),
            "duration_sec": round(frames_written / fps, 1) if fps else 0.0,
        }

    # ------------------------------------------------------------------ #
    def _build_windows(
        self, items: list[HighlightItem], fps: float, total: int
    ) -> list[_Window]:
        pre = int(self.pre_seconds * fps)
        post = int(self.post_seconds * fps)
        limit = (total - 1) if total > 0 else None

        windows: list[_Window] = []
        for it in items:
            start = max(0, it.frame - pre)
            end = it.frame + post
            if limit is not None:
                end = min(end, limit)
            if windows and start <= windows[-1].end_frame:
                windows[-1].end_frame = max(windows[-1].end_frame, end)
                windows[-1].items.append(it)
            else:
                windows.append(_Window(start, end, [it]))
        return windows

    def _window_label(self, win: _Window) -> str:
        top = max(win.items, key=lambda it: it.priority)
        return top.label

    def _draw_label(self, frame, label: str, color) -> None:
        h, w = frame.shape[:2]
        bar_h = 44
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, h - bar_h), (12, h), color, -1)
        frame_blend = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
        frame[h - bar_h:h, :] = frame_blend[h - bar_h:h, :]
        cv2.putText(
            frame, label, (24, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

    def _export_timeline(
        self, items: list[HighlightItem], windows: list[_Window], fps: float
    ) -> None:
        _ensure_parent(self.timeline_json)
        payload = {
            "source": self.source,
            "fps": fps,
            "pre_seconds": self.pre_seconds,
            "post_seconds": self.post_seconds,
            "event_count": len(items),
            "clip_count": len(windows),
            "events": [it.to_dict() for it in items],
            "clips": [
                {
                    "start_frame": w.start_frame,
                    "end_frame": w.end_frame,
                    "start_sec": round(w.start_frame / fps, 2) if fps else 0.0,
                    "end_sec": round(w.end_frame / fps, 2) if fps else 0.0,
                    "events": [it.to_dict() for it in w.items],
                }
                for w in windows
            ],
        }
        with open(self.timeline_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
