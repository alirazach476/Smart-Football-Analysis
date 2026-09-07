"""
YOLOv8 detector wrapper.

Encapsulates loading the trained model and running inference on a single
frame, returning a clean list of `Detection` objects that the rest of the
pipeline can consume without knowing anything about Ultralytics internals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    """A single detected (and optionally tracked) object in one frame."""

    class_id: int
    class_name: str
    confidence: float
    # Bounding box in pixel coordinates (top-left / bottom-right).
    x1: int
    y1: int
    x2: int
    y2: int
    # Persistent tracking ID assigned by ByteTrack. `None` when running in
    # pure detection mode, or when the tracker has not yet confirmed a track.
    track_id: int | None = None
    # Team label ("Team A" / "Team B") assigned by the team classifier.
    # Only ever set for the "player" class; None otherwise.
    team: str | None = None
    # Representative BGR colour for the assigned team (for drawing). Excluded
    # from serialisation - it is a display concern only.
    team_color: tuple[int, int, int] | None = None

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        """Bounding-box centre (cx, cy) - handy for trajectories."""
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    def to_dict(self) -> dict:
        """Serialisable representation used for JSON/CSV export."""
        cx, cy = self.center
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "team": self.team,
            "confidence": round(self.confidence, 4),
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "cx": cx,
            "cy": cy,
        }


class FootballDetector:
    """Thin, well-typed wrapper around an Ultralytics YOLO model."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        ball_conf_threshold: float = 0.10,
        iou_threshold: float = 0.45,
        device: str | None = None,
        imgsz: int = 1280,
        tracker_config: str = "bytetrack.yaml",
    ) -> None:
        self.conf_threshold = conf_threshold
        self.ball_conf_threshold = ball_conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        # Path to the ByteTrack YAML used by `track()`.
        self.tracker_config = tracker_config

        # Resolve the device automatically when not explicitly provided.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load the trained weights. This also loads the class-name mapping
        # that was baked into the model during training.
        self.model = YOLO(model_path)
        self.class_names: dict[int, str] = self.model.names

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single BGR frame.

        Returns a list of `Detection` objects (may be empty).
        """
        # `verbose=False` keeps the console clean during video processing.
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections: list[Detection] = []

        # predict() returns a list (one entry per image); we passed one frame.
        result = results[0]
        if result.boxes is None:
            return detections

        # Move box data to CPU/numpy once for cheap iteration.
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls_id in zip(
            boxes_xyxy, confidences, class_ids
        ):
            detections.append(
                Detection(
                    class_id=int(cls_id),
                    class_name=self.class_names.get(int(cls_id), str(cls_id)),
                    confidence=float(conf),
                    x1=int(round(x1)),
                    y1=int(round(y1)),
                    x2=int(round(x2)),
                    y2=int(round(y2)),
                )
            )

        return detections

    def track(self, frame: np.ndarray) -> list[Detection]:
        """
        Run detection + ByteTrack on a single BGR frame.

        Unlike `detect()`, this is **stateful**: `persist=True` tells
        Ultralytics to keep the tracker's state between successive calls, so
        IDs remain consistent across frames of the same video. Call it
        sequentially, frame by frame, on the same detector instance.

        ByteTrack associates low- and high-confidence boxes and uses a Kalman
        motion model, letting it bridge short occlusions and re-attach a
        reappearing object to its existing ID (within the tracker's
        `track_buffer` window - tuned in the tracker YAML).

        Returns a list of `Detection` objects, each carrying a `track_id`
        (which may be `None` for a not-yet-confirmed track).
        """
        results = self.model.track(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.imgsz,
            tracker=self.tracker_config,
            persist=True,      # keep tracker state across frames
            verbose=False,
        )

        detections: list[Detection] = []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        # `boxes.id` is None on frames where no track has been confirmed yet.
        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [None] * len(boxes_xyxy)

        for (x1, y1, x2, y2), conf, cls_id, tid in zip(
            boxes_xyxy, confidences, class_ids, track_ids
        ):
            detections.append(
                Detection(
                    class_id=int(cls_id),
                    class_name=self.class_names.get(int(cls_id), str(cls_id)),
                    confidence=float(conf),
                    x1=int(round(x1)),
                    y1=int(round(y1)),
                    x2=int(round(x2)),
                    y2=int(round(y2)),
                    track_id=int(tid) if tid is not None else None,
                )
            )

        # Balls are small and often missed at the player conf threshold.
        # Supplement with a lower-confidence pass when no ball was tracked.
        if self.ball_conf_threshold < self.conf_threshold:
            has_ball = any(d.class_name == "ball" for d in detections)
            if not has_ball:
                detections.extend(self._supplement_ball(frame))

        return detections

    def _supplement_ball(self, frame: np.ndarray) -> list[Detection]:
        """Low-confidence ball-only detection pass for frames where track() missed it."""
        results = self.model.predict(
            source=frame,
            conf=self.ball_conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        out: list[Detection] = []
        result = results[0]
        if result.boxes is None:
            return out
        for (x1, y1, x2, y2), conf, cls_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
        ):
            name = self.class_names.get(int(cls_id), str(cls_id))
            if name != "ball":
                continue
            out.append(
                Detection(
                    class_id=int(cls_id),
                    class_name=name,
                    confidence=float(conf),
                    x1=int(round(x1)),
                    y1=int(round(y1)),
                    x2=int(round(x2)),
                    y2=int(round(y2)),
                    track_id=None,
                )
            )
        return out
