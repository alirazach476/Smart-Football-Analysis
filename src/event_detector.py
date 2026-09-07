"""
Heuristic match-event detection from bird's-eye coordinates.

Beyond goals and shots (handled by their own modules), this detector infers
set-piece and disciplinary events from ball/player geometry:

  * corner   - ball settles in a corner arc, then is delivered
  * penalty  - ball settles on a penalty spot
  * foul     - opposing players converge tightly and play stops (ball dead)
  * card     - a referee lingers by a player cluster during a stoppage

IMPORTANT — accuracy note
-------------------------
Corners and penalties are *geometric* and reliable. Fouls and cards are
*heuristic*: positional data alone cannot see contact or a raised card, so these
are best-effort proxies (converging opponents + dead ball; referee proximity
during a stoppage). Card colour (yellow vs red) cannot be derived from
coordinates, so card events default to `yellow_card` and expose the mechanism;
supply visual card recognition to distinguish colours reliably.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from . import field_config as fc


class EventType(str, Enum):
    CORNER = "corner"
    PENALTY = "penalty"
    FOUL = "foul"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"


CM_PER_M = 100.0
L = fc.PITCH_LENGTH
W = fc.PITCH_WIDTH

CORNERS = [(0.0, 0.0), (0.0, W), (float(L), 0.0), (float(L), W)]
PENALTY_SPOTS = [
    (fc.PENALTY_SPOT_DISTANCE, W / 2.0),
    (L - fc.PENALTY_SPOT_DISTANCE, W / 2.0),
]

# Thresholds.
CORNER_RADIUS_CM = 350.0
PENALTY_RADIUS_CM = 150.0
DEAD_BALL_SPEED_MS = 1.0        # ball considered stationary below this
SETTLE_FRAMES = 8               # frames the ball must dwell to count a set-piece
FOUL_PROXIMITY_CM = 160.0       # opposing players this close = challenge
REF_PROXIMITY_CM = 400.0        # referee near a player = potential card
CARD_LINGER_FRAMES = 20         # referee lingers this long during a stoppage
EVENT_DEBOUNCE_FRAMES = 60      # min gap between same-type events


@dataclass
class MatchEvent:
    """A detected match event."""

    event_id: int
    type: str
    frame: int
    timestamp_sec: float
    team: str | None
    player_track_id: int | None
    x_cm: float | None
    y_cm: float | None
    detail: str

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "type": self.type,
            "frame": self.frame,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "team": self.team,
            "player_track_id": self.player_track_id,
            "x_cm": round(self.x_cm, 1) if self.x_cm is not None else None,
            "y_cm": round(self.y_cm, 1) if self.y_cm is not None else None,
            "detail": self.detail,
        }


class EventDetector:
    """Detects corners, penalties, fouls, and cards from frame records."""

    def __init__(
        self,
        fps: float = 25.0,
        detect_fouls: bool = True,
        detect_cards: bool = True,
        card_default_type: str = EventType.YELLOW_CARD.value,
        team_names: tuple[str, str] = ("Team A", "Team B"),
    ) -> None:
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 25.0
        self.detect_fouls = detect_fouls
        self.detect_cards = detect_cards
        self.card_default_type = card_default_type
        self.team_names = team_names

        self._events: list[MatchEvent] = []
        self._ball: list[dict] = []
        self._frame_players: dict[int, list[dict]] = {}

    def analyze(self, frames: list[dict]) -> list[MatchEvent]:
        """Detect all heuristic events from bird's-eye frame records."""
        self._index(frames)
        self._events = []

        next_id = 1
        last_by_type: dict[str, int] = {}

        # Track sustained states for fouls/cards.
        foul_active_since: int | None = None
        card_ref_since: int | None = None

        for idx, b in enumerate(self._ball):
            frame = b["frame"]
            ts = b["ts"]
            speed = self._ball_speed(idx)
            players = self._frame_players.get(frame, [])

            # --- Corner --------------------------------------------------
            corner = self._near_any(b, CORNERS, CORNER_RADIUS_CM)
            if corner is not None and speed < DEAD_BALL_SPEED_MS:
                if self._settled(idx, CORNERS, CORNER_RADIUS_CM):
                    if self._debounced(EventType.CORNER.value, frame, last_by_type):
                        self._events.append(MatchEvent(
                            next_id, EventType.CORNER.value, frame, ts,
                            None, None, b["x"], b["y"],
                            "Ball delivered from the corner arc",
                        ))
                        last_by_type[EventType.CORNER.value] = frame
                        next_id += 1

            # --- Penalty -------------------------------------------------
            spot = self._near_any(b, PENALTY_SPOTS, PENALTY_RADIUS_CM)
            if spot is not None and speed < DEAD_BALL_SPEED_MS:
                if self._settled(idx, PENALTY_SPOTS, PENALTY_RADIUS_CM):
                    if self._debounced(EventType.PENALTY.value, frame, last_by_type):
                        self._events.append(MatchEvent(
                            next_id, EventType.PENALTY.value, frame, ts,
                            None, None, b["x"], b["y"],
                            "Ball placed on the penalty spot",
                        ))
                        last_by_type[EventType.PENALTY.value] = frame
                        next_id += 1

            # --- Foul (heuristic) ---------------------------------------
            challenge = self._challenge_at(players)
            dead_ball = speed < DEAD_BALL_SPEED_MS
            if self.detect_fouls and challenge is not None and dead_ball:
                if foul_active_since is None:
                    foul_active_since = frame
                    if self._debounced(EventType.FOUL.value, frame, last_by_type):
                        team, tid = challenge
                        self._events.append(MatchEvent(
                            next_id, EventType.FOUL.value, frame, ts,
                            team, tid, b["x"], b["y"],
                            "Challenge between opponents with the ball dead",
                        ))
                        last_by_type[EventType.FOUL.value] = frame
                        next_id += 1
            else:
                foul_active_since = None

            # --- Card (heuristic) ---------------------------------------
            if self.detect_cards:
                ref_near = self._referee_near_players(players)
                stoppage = speed < DEAD_BALL_SPEED_MS
                if ref_near is not None and stoppage:
                    if card_ref_since is None:
                        card_ref_since = frame
                    elif frame - card_ref_since >= CARD_LINGER_FRAMES:
                        if self._debounced(self.card_default_type, frame, last_by_type):
                            team, tid, x, y = ref_near
                            self._events.append(MatchEvent(
                                next_id, self.card_default_type, frame, ts,
                                team, tid, x, y,
                                "Referee lingering by a player during a stoppage",
                            ))
                            last_by_type[self.card_default_type] = frame
                            next_id += 1
                        card_ref_since = None
                else:
                    card_ref_since = None

        return self._events

    @property
    def events(self) -> list[MatchEvent]:
        return list(self._events)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self._events:
            out[e.type] = out.get(e.type, 0) + 1
        return out

    # ------------------------------------------------------------------ #
    def _index(self, frames: list[dict]) -> None:
        self._ball = []
        self._frame_players = {}
        for rec in sorted(frames, key=lambda r: r["frame"]):
            if not rec.get("homography_available"):
                continue
            frame = rec["frame"]
            ts = rec.get("timestamp_sec")
            if ts is None:
                ts = frame * self.dt
            self._frame_players[frame] = [
                {
                    "track_id": p.get("track_id"),
                    "team": p.get("team"),
                    "class_name": p.get("class_name", "player"),
                    "x_cm": float(p["x_cm"]),
                    "y_cm": float(p["y_cm"]),
                }
                for p in rec.get("players", [])
                if p.get("x_cm") is not None and p.get("y_cm") is not None
            ]
            ball = rec.get("ball")
            if ball and ball.get("x_cm") is not None and ball.get("y_cm") is not None:
                self._ball.append({
                    "frame": frame,
                    "ts": ts,
                    "x": float(ball["x_cm"]),
                    "y": float(ball["y_cm"]),
                })

    def _ball_speed(self, idx: int) -> float:
        if idx == 0 or idx >= len(self._ball):
            return 0.0
        a, b = self._ball[idx - 1], self._ball[idx]
        dt = (b["ts"] - a["ts"]) or self.dt
        if dt <= 0:
            dt = self.dt
        return math.hypot(b["x"] - a["x"], b["y"] - a["y"]) / CM_PER_M / dt

    @staticmethod
    def _near_any(
        b: dict, points: list[tuple[float, float]], radius: float
    ) -> tuple[float, float] | None:
        for px, py in points:
            if math.hypot(b["x"] - px, b["y"] - py) <= radius:
                return (px, py)
        return None

    def _settled(
        self, idx: int, points: list[tuple[float, float]], radius: float
    ) -> bool:
        """True if the ball dwelt near one of the points for SETTLE_FRAMES."""
        count = 0
        for k in range(max(0, idx - SETTLE_FRAMES), idx + 1):
            if self._near_any(self._ball[k], points, radius) is not None:
                count += 1
        return count >= SETTLE_FRAMES

    def _challenge_at(
        self, players: list[dict]
    ) -> tuple[str | None, int | None] | None:
        """Return (team, track_id) of the closest opposing-player pair, if tight."""
        outfield = [p for p in players if p["class_name"] in ("player", "goalkeeper")]
        best: tuple[str | None, int | None] | None = None
        best_dist = FOUL_PROXIMITY_CM
        for i in range(len(outfield)):
            for j in range(i + 1, len(outfield)):
                a, c = outfield[i], outfield[j]
                if a["team"] and c["team"] and a["team"] != c["team"]:
                    d = math.hypot(a["x_cm"] - c["x_cm"], a["y_cm"] - c["y_cm"])
                    if d < best_dist:
                        best_dist = d
                        best = (a["team"], a["track_id"])
        return best

    def _referee_near_players(
        self, players: list[dict]
    ) -> tuple[str | None, int | None, float, float] | None:
        """Return (team, track_id, x, y) of a player a referee is standing near."""
        refs = [p for p in players if p["class_name"] == "referee"]
        outfield = [p for p in players if p["class_name"] in ("player", "goalkeeper")]
        if not refs or not outfield:
            return None
        for ref in refs:
            for p in outfield:
                d = math.hypot(ref["x_cm"] - p["x_cm"], ref["y_cm"] - p["y_cm"])
                if d <= REF_PROXIMITY_CM:
                    return (p["team"], p["track_id"], p["x_cm"], p["y_cm"])
        return None

    @staticmethod
    def _debounced(event_type: str, frame: int, last: dict[str, int]) -> bool:
        prev = last.get(event_type)
        return prev is None or (frame - prev) >= EVENT_DEBOUNCE_FRAMES
