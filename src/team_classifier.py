"""
Automatic team classification via jersey-colour K-Means.

Pipeline for each tracked player:

    player crop  ->  jersey region  ->  HSV  ->  mask out grass
                 ->  hue-aware colour feature  ->  accumulate per track ID

Once enough evidence has been collected, all player tracks are clustered with
**K-Means (k=2)** into two teams. The clusters are labelled "Team A" / "Team B"
deterministically (by hue), so the assignment is fully automatic - there is no
manual colour selection anywhere.

Referees and goalkeepers are *not* clustered here: the detector already emits
them as their own classes, so they are simply left out of the team clustering
and keep their own labels/colours downstream.

K-Means is run with OpenCV (`cv2.kmeans`), avoiding any extra dependency.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .detector import Detection


# Feature layout: [cos(hue), sin(hue), saturation, value].
# Encoding hue as (cos, sin) makes the circular hue space behave correctly
# under the Euclidean distance K-Means uses (so red ~0 and red ~179 are close).
_FEATURE_DIM = 4


class TeamClassifier:
    """Clusters tracked players into two teams by jersey colour."""

    def __init__(
        self,
        warmup_frames: int = 30,
        refit_interval: int = 30,
        min_track_samples: int = 3,
        team_names: tuple[str, str] = ("Team A", "Team B"),
    ) -> None:
        # Wait this many frames before the first clustering, so we have colour
        # evidence from most players before committing to teams.
        self.warmup_frames = warmup_frames
        # Re-cluster every N frames to incorporate newly seen players and more
        # samples per player (cheap: runs on a handful of per-track means).
        self.refit_interval = refit_interval
        # Ignore tracks we have barely seen; they are usually noisy.
        self.min_track_samples = min_track_samples
        self.team_names = team_names

        # Per-track running accumulation of the colour feature:
        #   track_id -> [sum_vector (float64, _FEATURE_DIM), count]
        self._sum: dict[int, np.ndarray] = {}
        self._count: dict[int, int] = {}

        # Fitted state.
        self._fitted = False
        self._centers: np.ndarray | None = None          # (2, _FEATURE_DIM)
        self._center_to_team: list[int] = [0, 1]         # cluster idx -> team idx
        self._team_colors: list[tuple[int, int, int]] = []  # BGR per team

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #
    @staticmethod
    def jersey_feature(crop_bgr: np.ndarray) -> np.ndarray | None:
        """
        Turn a player crop into a jersey colour feature vector, or None.

        Steps: isolate the torso (jersey) region, convert to HSV, drop grass-
        green pixels, then summarise the remaining pixels with a circular-mean
        hue plus mean saturation/value.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        h, w = crop_bgr.shape[:2]
        if h < 12 or w < 6:
            return None  # too small to be reliable

        # Jersey/torso region: skip the head and legs, and trim the sides to
        # avoid background bleeding in around the shoulders.
        y1, y2 = int(0.15 * h), int(0.55 * h)
        x1, x2 = int(0.20 * w), int(0.80 * w)
        region = crop_bgr[y1:y2, x1:x2]
        if region.size == 0:
            return None

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].astype(np.float32)      # 0..179 (OpenCV hue range)
        sat = hsv[:, :, 1].astype(np.float32)      # 0..255
        val = hsv[:, :, 2].astype(np.float32)      # 0..255

        # Grass mask: green hue band with enough saturation/brightness.
        grass = (hue >= 35) & (hue <= 85) & (sat > 40) & (val > 40)
        keep = ~grass

        # If masking grass removed almost everything (e.g. a green goalkeeper),
        # fall back to using all pixels so we still get a colour.
        if keep.sum() < 20:
            keep = np.ones_like(grass, dtype=bool)

        hue_k = hue[keep]
        sat_k = sat[keep]
        val_k = val[keep]
        if hue_k.size == 0:
            return None

        # Circular mean of hue: average the unit vectors, not the raw angles.
        angles = hue_k * (2.0 * math.pi / 180.0)  # OpenCV hue 0..179 -> radians
        cos_mean = float(np.mean(np.cos(angles)))
        sin_mean = float(np.mean(np.sin(angles)))
        sat_mean = float(np.mean(sat_k)) / 255.0
        val_mean = float(np.mean(val_k)) / 255.0

        return np.array([cos_mean, sin_mean, sat_mean, val_mean], dtype=np.float32)

    @staticmethod
    def feature_from_frame(
        frame: np.ndarray, det: Detection
    ) -> np.ndarray | None:
        """Crop a detection's box out of the frame and extract its feature."""
        h, w = frame.shape[:2]
        x1 = max(det.x1, 0)
        y1 = max(det.y1, 0)
        x2 = min(det.x2, w)
        y2 = min(det.y2, h)
        if x2 <= x1 or y2 <= y1:
            return None
        return TeamClassifier.jersey_feature(frame[y1:y2, x1:x2])

    # ------------------------------------------------------------------ #
    # Accumulation + fitting
    # ------------------------------------------------------------------ #
    def add_sample(self, track_id: int, feature: np.ndarray) -> None:
        """Accumulate one colour observation for a player track."""
        if track_id not in self._sum:
            self._sum[track_id] = np.zeros(_FEATURE_DIM, dtype=np.float64)
            self._count[track_id] = 0
        self._sum[track_id] += feature
        self._count[track_id] += 1

    def _track_mean(self, track_id: int) -> np.ndarray | None:
        count = self._count.get(track_id, 0)
        if count == 0:
            return None
        return (self._sum[track_id] / count).astype(np.float32)

    def maybe_refit(self, frame_index: int) -> None:
        """Fit or refit K-Means when appropriate for the current frame."""
        past_warmup = frame_index >= self.warmup_frames
        due_refit = (frame_index % self.refit_interval == 0) or not self._fitted
        if past_warmup and due_refit:
            self.fit()

    def fit(self) -> bool:
        """
        Run K-Means (k=2) over the per-track mean jersey features.

        Returns True if a model was fitted. Requires at least two eligible
        tracks (players seen at least `min_track_samples` times).
        """
        track_ids = [
            tid
            for tid, c in self._count.items()
            if c >= self.min_track_samples
        ]
        if len(track_ids) < 2:
            return False

        data = np.stack([self._track_mean(t) for t in track_ids]).astype(np.float32)

        # OpenCV K-Means. We use several attempts with different seeds and keep
        # the most compact result for stability.
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            50,
            1e-4,
        )
        _compactness, _labels, centers = cv2.kmeans(
            data,
            2,
            None,
            criteria,
            attempts=10,
            flags=cv2.KMEANS_PP_CENTERS,
        )

        # Deterministically map clusters to Team A / Team B by hue angle, so the
        # same jersey colour always maps to the same team name across refits.
        hues = [math.atan2(c[1], c[0]) % (2.0 * math.pi) for c in centers]
        order = np.argsort(hues)  # smaller hue -> Team A
        center_to_team = [0, 0]
        center_to_team[int(order[0])] = 0
        center_to_team[int(order[1])] = 1

        self._centers = centers
        self._center_to_team = center_to_team
        self._team_colors = [
            self._center_bgr(centers[int(order[0])]),  # Team A colour
            self._center_bgr(centers[int(order[1])]),  # Team B colour
        ]
        self._fitted = True
        return True

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(self, track_id: int) -> str | None:
        """Return the team name for a player track, or None if not yet known."""
        if not self._fitted or self._centers is None:
            return None
        mean = self._track_mean(track_id)
        if mean is None:
            return None
        # Nearest centre in feature space.
        dists = np.linalg.norm(self._centers - mean, axis=1)
        cluster = int(np.argmin(dists))
        team_idx = self._center_to_team[cluster]
        return self.team_names[team_idx]

    def team_color(self, team_name: str) -> tuple[int, int, int]:
        """Representative BGR colour for a team (falls back to grey)."""
        if not self._team_colors or team_name not in self.team_names:
            return (200, 200, 200)
        return self._team_colors[self.team_names.index(team_name)]

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def summary(self) -> dict[str, int]:
        """Count how many unique player tracks fell into each team."""
        counts = {name: 0 for name in self.team_names}
        if not self._fitted:
            return counts
        for tid in self._count:
            team = self.predict(tid)
            if team is not None:
                counts[team] += 1
        return counts

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _center_bgr(center: np.ndarray) -> tuple[int, int, int]:
        """Reconstruct a displayable BGR colour from a cluster centre."""
        angle = math.atan2(center[1], center[0]) % (2.0 * math.pi)
        hue = int((angle * 180.0 / (2.0 * math.pi)))          # back to 0..179
        sat = int(np.clip(center[2] * 255.0, 0, 255))
        val = int(np.clip(center[3] * 255.0, 0, 255))
        # Boost a little so the drawn colour is vivid even for muted jerseys.
        sat = max(sat, 120)
        val = max(val, 120)
        hsv = np.uint8([[[hue, sat, val]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])
