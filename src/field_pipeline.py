"""
Football field detection pipeline.

Runs the YOLOv8 pose field model over an image or a video, draws the pitch
visualization, and exports the detected field coordinates to JSON.

Two input modes are auto-detected from the file extension:
  * image  -> writes one annotated image + a single-frame coordinates JSON
  * video  -> writes an annotated video + a per-frame coordinates JSON
"""

from __future__ import annotations

import json
import os
import time

import cv2

from . import field_config as fc
from .field_annotator import FieldAnnotator
from .field_detector import FieldDetection, FieldDetector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


class FieldPipeline:
    """Detects, visualizes and exports football pitch keypoints."""

    def __init__(
        self,
        model_path: str,
        source: str,
        output_path: str = "output/field_annotated.mp4",
        coordinates_json: str = "output/field_coordinates.json",
        conf_threshold: float = 0.25,
        kpt_conf_threshold: float = 0.5,
        device: str = "cpu",
        imgsz: int = 1280,
        draw_labels: bool = True,
        show_preview: bool = False,
    ) -> None:
        self.source = source
        self.output_path = output_path
        self.coordinates_json = coordinates_json
        self.show_preview = show_preview

        self.detector = FieldDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            kpt_conf_threshold=kpt_conf_threshold,
            device=device if device != "cpu" else None,
            imgsz=imgsz,
        )
        self.annotator = FieldAnnotator(draw_labels=draw_labels)

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Dispatch to image or video processing based on the source type."""
        if not os.path.exists(self.source):
            raise FileNotFoundError(f"Input not found: {self.source}")

        ext = os.path.splitext(self.source)[1].lower()
        print(f"[info] device : {self.detector.device}")
        print(f"[info] source : {self.source}")

        if ext in IMAGE_EXTS:
            self._run_image()
        elif ext in VIDEO_EXTS:
            self._run_video()
        else:
            raise ValueError(
                f"Unsupported input extension '{ext}'. "
                f"Images: {sorted(IMAGE_EXTS)}; videos: {sorted(VIDEO_EXTS)}"
            )

    # ------------------------------------------------------------------ #
    def _run_image(self) -> None:
        """Process a single still image."""
        frame = cv2.imread(self.source)
        if frame is None:
            raise RuntimeError(f"Could not read image: {self.source}")

        detection = self.detector.detect(frame)
        self.annotator.annotate(frame, detection)
        self.annotator.draw_summary(frame, detection)

        # Default the output to a .png for images if a video path was given.
        out_path = self.output_path
        if os.path.splitext(out_path)[1].lower() not in IMAGE_EXTS:
            out_path = os.path.splitext(out_path)[0] + ".png"
        _ensure_parent(out_path)
        cv2.imwrite(out_path, frame)

        payload = {
            "source": self.source,
            "type": "image",
            "num_keypoints": fc.NUM_KEYPOINTS,
            "frame": self._frame_payload(0, None, detection),
            "pitch_template_cm": self._template_payload(),
        }
        self._write_json(payload)

        print(f"[done] keypoints found : {len(detection.visible_keypoints)}/{fc.NUM_KEYPOINTS}")
        print(f"       annotated image : {out_path}")
        print(f"       coordinates     : {self.coordinates_json}")

    # ------------------------------------------------------------------ #
    def _run_video(self) -> None:
        """Process a full video frame by frame."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[info] resolution : {width}x{height} @ {fps:.2f} fps")
        print(f"[info] frames     : {total if total > 0 else 'unknown'}")

        writer = self._create_writer(self.output_path, fps, width, height)
        frames_payload: list[dict] = []
        frame_index = 0
        start = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                detection = self.detector.detect(frame)
                self.annotator.annotate(frame, detection)
                self.annotator.draw_summary(frame, detection)
                writer.write(frame)

                timestamp = round(frame_index / fps, 3) if fps else None
                frames_payload.append(
                    self._frame_payload(frame_index, timestamp, detection)
                )

                if self.show_preview:
                    cv2.imshow("Field Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[info] preview interrupted by user")
                        break

                frame_index += 1
                if frame_index % 30 == 0:
                    if total > 0:
                        pct = 100.0 * frame_index / total
                        print(f"[info] processed {frame_index}/{total} ({pct:.1f}%)")
                    else:
                        print(f"[info] processed {frame_index} frames")
        finally:
            cap.release()
            writer.release()
            if self.show_preview:
                cv2.destroyAllWindows()

        elapsed = time.time() - start
        avg_fps = frame_index / elapsed if elapsed > 0 else 0.0

        payload = {
            "source": self.source,
            "type": "video",
            "fps": fps,
            "num_keypoints": fc.NUM_KEYPOINTS,
            "total_frames": frame_index,
            "frames": frames_payload,
            "pitch_template_cm": self._template_payload(),
        }
        self._write_json(payload)

        print("\n[done] field detection complete")
        print(f"       frames processed : {frame_index}")
        print(f"       average fps      : {avg_fps:.2f}")
        print(f"       annotated video  : {self.output_path}")
        print(f"       coordinates      : {self.coordinates_json}")

    # ------------------------------------------------------------------ #
    # Payload builders
    # ------------------------------------------------------------------ #
    @staticmethod
    def _frame_payload(
        frame_index: int, timestamp: float | None, detection: FieldDetection
    ) -> dict:
        """Serialise one frame's keypoints, plus a grouped view of features."""
        grouped = {
            group: [k.to_dict() for k in kps]
            for group, kps in detection.by_group().items()
        }
        return {
            "frame": frame_index,
            "timestamp_sec": timestamp,
            "pitch_confidence": round(detection.pitch_confidence, 4),
            "num_visible": len(detection.visible_keypoints),
            "keypoints": [k.to_dict() for k in detection.keypoints],
            "features": grouped,
        }

    @staticmethod
    def _template_payload() -> list[dict]:
        """The real-world pitch template landmarks (cm) for homography use."""
        return [
            {
                "index": i,
                "label": fc.label_for(i),
                "group": fc.group_for(i),
                "x_cm": round(float(x), 1),
                "y_cm": round(float(y), 1),
            }
            for i, (x, y) in enumerate(fc.TEMPLATE_VERTICES)
        ]

    def _write_json(self, payload: dict) -> None:
        _ensure_parent(self.coordinates_json)
        with open(self.coordinates_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _create_writer(
        path: str, fps: float, width: int, height: int
    ) -> cv2.VideoWriter:
        _ensure_parent(path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for: {path}")
        return writer


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
