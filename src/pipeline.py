"""
End-to-end video processing pipeline.

Ties together the detector, annotator and exporter:
  read frame -> detect -> annotate -> write frame -> record detections.
"""

from __future__ import annotations

import os
import time

import cv2

from .annotator import Annotator
from .ball_tracker import BallTracker
from .config import PipelineConfig
from .detector import FootballDetector
from .exporter import DetectionExporter
from .team_classifier import TeamClassifier
from .tracking_exporter import TrackingExporter


class FootballPipeline:
    """Runs detection over a whole video and produces all output artifacts."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

        # Team classification relies on stable per-player crops, so it requires
        # tracking. Enable tracking automatically if teams were requested.
        if config.enable_teams and not config.enable_tracking:
            print("[info] team classification requires tracking; enabling ByteTrack")
            config.enable_tracking = True

        self.detector = FootballDetector(
            model_path=config.model_path,
            conf_threshold=config.conf_threshold,
            iou_threshold=config.iou_threshold,
            device=config.device if config.device != "cpu" else None,
            imgsz=config.imgsz,
            tracker_config=config.tracker_config,
        )
        self.annotator = Annotator(config)

        self.team_classifier = (
            TeamClassifier(
                warmup_frames=config.team_warmup_frames,
                refit_interval=config.team_refit_interval,
                min_track_samples=config.team_min_samples,
            )
            if config.enable_teams
            else None
        )

        self.ball_tracker: BallTracker | None = None  # created in run() once fps known

    def run(self) -> None:
        """Process the input video and write all outputs."""
        cfg = self.config

        if not os.path.exists(cfg.source_video):
            raise FileNotFoundError(f"Input video not found: {cfg.source_video}")

        cap = cv2.VideoCapture(cfg.source_video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {cfg.source_video}")

        # Read source video properties.
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        mode = "track + detect (ByteTrack)" if cfg.enable_tracking else "detect"
        if cfg.enable_ball_tracking:
            mode += " + ball (Kalman)"
        print(f"[info] mode        : {mode}")
        print(f"[info] device      : {self.detector.device}")
        print(f"[info] input       : {cfg.source_video}")
        print(f"[info] resolution  : {width}x{height} @ {source_fps:.2f} fps")
        print(f"[info] total frames: {total_frames if total_frames > 0 else 'unknown'}")

        writer = self._create_writer(cfg.output_video, source_fps, width, height)

        # Choose the exporter(s) based on the run mode. In tracking mode we also
        # keep the plain detection exporter so users still get detections.*.
        exporter = DetectionExporter(fps=source_fps)
        tracking_exporter = (
            TrackingExporter(fps=source_fps) if cfg.enable_tracking else None
        )

        if cfg.enable_ball_tracking:
            self.ball_tracker = BallTracker(
                fps=source_fps,
                ball_conf_threshold=cfg.ball_conf_threshold,
                gate_radius=cfg.ball_gate_radius,
                max_coast_frames=cfg.ball_max_coast_frames,
                trajectory_length=cfg.ball_trajectory_length,
            )

        frame_index = 0
        processing_start = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break  # end of video

                # --- Detect / Track ----------------------------------------
                frame_start = time.time()
                if cfg.enable_tracking:
                    # Stateful call: keeps ByteTrack IDs consistent frame-to-frame.
                    detections = self.detector.track(frame)
                else:
                    detections = self.detector.detect(frame)
                # Instantaneous FPS = 1 / time spent on this frame.
                elapsed = time.time() - frame_start
                fps = 1.0 / elapsed if elapsed > 0 else 0.0

                # --- Classify teams ----------------------------------------
                if self.team_classifier is not None:
                    self._classify_teams(frame, detections, frame_index)

                # --- Track ball (Kalman) -----------------------------------
                ball_state = None
                if self.ball_tracker is not None:
                    ball_state = self.ball_tracker.update(frame_index, detections)

                # --- Annotate ----------------------------------------------
                self.annotator.draw_detections(frame, detections)
                if self.ball_tracker is not None:
                    self.annotator.draw_ball_tracking(
                        frame,
                        ball_state,
                        self.ball_tracker.trajectory_points(),
                    )
                self.annotator.draw_fps(frame, fps)

                # --- Persist -----------------------------------------------
                writer.write(frame)
                exporter.add_frame(frame_index, detections)
                if tracking_exporter is not None:
                    tracking_exporter.add_frame(frame_index, detections)

                if cfg.show_preview:
                    cv2.imshow("Football Detection", frame)
                    # Press 'q' to abort the preview early.
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[info] preview interrupted by user")
                        break

                frame_index += 1
                self._log_progress(frame_index, total_frames)

        finally:
            # Always release resources, even if an error occurs mid-run.
            cap.release()
            writer.release()
            if cfg.show_preview:
                cv2.destroyAllWindows()

        total_time = time.time() - processing_start
        avg_fps = frame_index / total_time if total_time > 0 else 0.0

        # Write detection exports.
        exporter.save_json(cfg.detections_json)
        exporter.save_csv(cfg.detections_csv)

        # Write tracking exports (if tracking was enabled).
        if tracking_exporter is not None:
            tracking_exporter.save_json(cfg.tracking_json)
            tracking_exporter.save_csv(cfg.tracking_csv)

        if self.ball_tracker is not None:
            self.ball_tracker.save_json(cfg.ball_trajectory_json)

        # Final summary.
        print("\n[done] processing complete")
        print(f"       frames processed : {frame_index}")
        print(f"       total detections : {exporter.total_detections}")
        if tracking_exporter is not None:
            print(f"       unique tracks    : {tracking_exporter.unique_tracks}")
        print(f"       average fps      : {avg_fps:.2f}")
        print(f"       annotated video  : {cfg.output_video}")
        print(f"       json export      : {cfg.detections_json}")
        print(f"       csv export       : {cfg.detections_csv}")
        if tracking_exporter is not None:
            print(f"       tracking json    : {cfg.tracking_json}")
            print(f"       tracking csv     : {cfg.tracking_csv}")
        if self.team_classifier is not None:
            counts = self.team_classifier.summary()
            summary = ", ".join(f"{name}: {n}" for name, n in counts.items())
            print(f"       team split       : {summary}")
        if self.ball_tracker is not None:
            stats = self.ball_tracker.summary()
            if stats["total_points"] > 0:
                print(
                    f"       ball points      : {stats['total_points']} "
                    f"({stats['detected_points']} detected)"
                )
                print(f"       max ball speed   : {stats['max_speed_px_per_sec']:.0f} px/s")
                print(f"       ball trajectory  : {cfg.ball_trajectory_json}")

    def _classify_teams(self, frame, detections, frame_index: int) -> None:
        """Extract jersey colours, (re)fit K-Means, and tag players with teams."""
        classifier = self.team_classifier
        assert classifier is not None  # guarded by caller

        # 1) Gather a jersey-colour sample for every tracked player this frame.
        for det in detections:
            if det.class_name == "player" and det.track_id is not None:
                feature = classifier.feature_from_frame(frame, det)
                if feature is not None:
                    classifier.add_sample(det.track_id, feature)

        # 2) Fit / refit the 2-team K-Means model when appropriate.
        classifier.maybe_refit(frame_index)

        # 3) Assign the resolved team (name + colour) back onto each player.
        if classifier.is_fitted:
            for det in detections:
                if det.class_name == "player" and det.track_id is not None:
                    team = classifier.predict(det.track_id)
                    if team is not None:
                        det.team = team
                        det.team_color = classifier.team_color(team)

    @staticmethod
    def _create_writer(
        path: str, fps: float, width: int, height: int
    ) -> cv2.VideoWriter:
        """Create the output video writer, ensuring the folder exists."""
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

        # mp4v is broadly available; produces an .mp4 the pipeline can write.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for: {path}")
        return writer

    @staticmethod
    def _log_progress(frame_index: int, total_frames: int) -> None:
        """Print a lightweight progress line every 30 frames."""
        if frame_index % 30 != 0:
            return
        if total_frames > 0:
            pct = 100.0 * frame_index / total_frames
            print(f"[info] processed {frame_index}/{total_frames} frames ({pct:.1f}%)")
        else:
            print(f"[info] processed {frame_index} frames")
