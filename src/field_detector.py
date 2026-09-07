"""
YOLOv8 pose wrapper for football field (pitch) keypoint detection.

Loads the trained `football-field-detection.pt` pose model and, for each frame,
returns the 32 pitch keypoints as typed `FieldKeypoint` objects annotated with
their semantic label and group (from `field_config`). Keypoints below a
confidence threshold are marked not-visible so downstream code can ignore them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from . import field_config as fc


@dataclass
class FieldKeypoint:
    """One pitch keypoint detected in a frame."""

    index: int
    label: str
    group: str
    x: float
    y: float
    confidence: float
    visible: bool

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "group": self.group,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "confidence": round(self.confidence, 4),
            "visible": self.visible,
        }


@dataclass
class FieldDetection:
    """All pitch keypoints for a single frame, plus the pitch box confidence."""

    keypoints: list[FieldKeypoint]
    pitch_confidence: float

    @property
    def visible_keypoints(self) -> list[FieldKeypoint]:
        return [k for k in self.keypoints if k.visible]

    def by_group(self) -> dict[str, list[FieldKeypoint]]:
        """Group visible keypoints by their semantic group."""
        grouped: dict[str, list[FieldKeypoint]] = {}
        for k in self.visible_keypoints:
            grouped.setdefault(k.group, []).append(k)
        return grouped


class FieldDetector:
    """Thin wrapper around the Ultralytics YOLOv8 pose model."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        kpt_conf_threshold: float = 0.5,
        device: str | None = None,
        imgsz: int = 1280,
    ) -> None:
        self.conf_threshold = conf_threshold
        # Per-keypoint confidence needed to treat a keypoint as "visible".
        self.kpt_conf_threshold = kpt_conf_threshold
        self.imgsz = imgsz
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = YOLO(model_path)

        # Sanity check: this pipeline is written for the 32-keypoint pose model.
        if getattr(self.model, "task", None) != "pose":
            raise ValueError(
                f"Expected a YOLOv8 pose model, got task={self.model.task!r}"
            )

    def detect(self, frame: np.ndarray) -> FieldDetection:
        """
        Detect pitch keypoints in a single BGR frame.

        Returns a `FieldDetection`. If no pitch is found, its keypoint list is
        empty. Only the highest-confidence pitch instance is used (there is one
        pitch per frame).
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        result = results[0]

        # No pitch detected / no keypoints in this frame.
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return FieldDetection(keypoints=[], pitch_confidence=0.0)

        # Pick the most confident pitch instance.
        box_confs = result.boxes.conf.cpu().numpy()
        best = int(np.argmax(box_confs))
        pitch_conf = float(box_confs[best])

        # keypoints.data: (num_instances, num_kpts, 3) -> (x, y, conf)
        kpts = result.keypoints.data.cpu().numpy()[best]

        keypoints: list[FieldKeypoint] = []
        for idx in range(min(len(kpts), fc.NUM_KEYPOINTS)):
            x, y, conf = float(kpts[idx][0]), float(kpts[idx][1]), float(kpts[idx][2])
            # A keypoint at (0,0) with zero confidence means "not predicted".
            visible = conf >= self.kpt_conf_threshold and (x > 0 or y > 0)
            keypoints.append(
                FieldKeypoint(
                    index=idx,
                    label=fc.label_for(idx),
                    group=fc.group_for(idx),
                    x=x,
                    y=y,
                    confidence=conf,
                    visible=visible,
                )
            )

        return FieldDetection(keypoints=keypoints, pitch_confidence=pitch_conf)
