"""
Bird's-eye view pipeline (homography-based tactical minimap).

For every frame this pipeline:
  1. detects the pitch keypoints (YOLOv8 pose)      -> field_detector
  2. estimates the frame->pitch homography           -> homography
  3. detects & tracks players / ball / refs / GKs    -> detector (+ tracking)
  4. optionally classifies players into two teams     -> team_classifier
  5. projects each object's ground position onto the  -> homography.transform
     real pitch plane and draws it on a top-down pitch -> pitch (PitchView)
  6. composites [annotated frame | minimap] side by    -> synchronized output
     side and writes the synchronized video
  7. exports all transformed pitch coordinates to JSON

The side-by-side composite keeps the minimap perfectly in sync with the source
video because both panels are produced from the very same frame.
"""

from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np

from . import field_config as fc
from .annotator import Annotator
from .config import PipelineConfig
from .ball_detector import BallDetector
from .ball_pitch_tracker import BallPitchTracker
from .detector import Detection, FootballDetector
from .field_detector import FieldDetector
from .heatmap_collector import HeatmapCollector
from .heatmap_exporter import HeatmapExporter
from .homography import HomographyEstimator
from .pitch import PitchView
from .pass_detector import PassDetector
from .pass_exporter import PassExporter
from .shot_detector import ShotDetector
from .shot_exporter import ShotExporter
from .possession_analyzer import PossessionAnalyzer
from .possession_exporter import PossessionExporter, draw_possession_overlay
from .event_detector import EventDetector
from .highlight_generator import HighlightGenerator
from .goal_detector import GoalDetector
from .goal_exporter import GoalExporter, draw_goal_overlay
from .speed_estimator import SpeedEstimator
from .speed_exporter import SpeedExporter, draw_speed_overlay
from .team_classifier import TeamClassifier


class BirdseyePipeline:
    """Produces a synchronized bird's-eye tactical view from a match video."""

    def __init__(
        self,
        field_model: str,
        object_model: str,
        source: str,
        output_path: str = "output/birdseye.mp4",
        coordinates_json: str = "output/birdseye_coordinates.json",
        conf_threshold: float = 0.25,
        ball_conf_threshold: float = 0.08,
        kpt_conf_threshold: float = 0.5,
        device: str = "cpu",
        imgsz: int = 1280,
        enable_teams: bool = True,
        pitch_scale: float = 0.06,
        smoothing: float = 0.5,
        show_preview: bool = False,
        enable_speed: bool = True,
        speed_json: str = "output/player_speed.json",
        speed_csv: str = "output/player_speed.csv",
        sprint_threshold_m_s: float = 7.0,
        min_sprint_sec: float = 1.0,
        speed_smoothing: float = 0.4,
        enable_heatmaps: bool = False,
        heatmap_dir: str = "output/heatmaps",
        heatmap_json: str = "output/heatmap_data.json",
        heatmap_html: str = "output/heatmaps.html",
        enable_possession: bool = True,
        possession_json: str = "output/possession.json",
        possession_csv: str = "output/possession.csv",
        possession_timeline_csv: str = "output/possession_timeline.csv",
        possession_radius_m: float = 5.5,
        enable_passes: bool = True,
        passes_json: str = "output/passes.json",
        passes_csv: str = "output/passes.csv",
        pass_statistics_csv: str = "output/pass_statistics.csv",
        pass_network_dir: str = "output/pass_networks",
        enable_shots: bool = True,
        shots_json: str = "output/shots.json",
        shots_csv: str = "output/shots.csv",
        shot_statistics_csv: str = "output/shot_statistics.csv",
        shot_map_dir: str = "output/shot_maps",
        enable_goals: bool = True,
        goals_json: str = "output/goals.json",
        goals_csv: str = "output/goals.csv",
        goal_statistics_csv: str = "output/goal_statistics.csv",
        goal_notifications_path: str = "output/goal_notifications.txt",
        enable_highlights: bool = True,
        highlights_path: str = "output/highlights.mp4",
        highlights_json: str = "output/highlights.json",
        highlight_pre_sec: float = 4.0,
        highlight_post_sec: float = 3.0,
        enable_ball_model: bool = True,
        ball_model: str = "football-ball-detection/football-ball-detection.pt",
        position_smoothing: float = 0.6,
        progress_callback=None,
    ) -> None:
        self.source = source
        self.output_path = output_path
        self.coordinates_json = coordinates_json
        self.show_preview = show_preview
        self.enable_teams = enable_teams
        self.enable_speed = enable_speed
        self.enable_heatmaps = enable_heatmaps
        self.heatmap_dir = heatmap_dir
        self.heatmap_json = heatmap_json
        self.heatmap_html = heatmap_html
        self.enable_passes = enable_passes
        self.enable_shots = enable_shots
        self.enable_goals = enable_goals
        self.enable_highlights = enable_highlights
        self.enable_possession = (
            enable_possession or enable_passes or enable_shots or enable_goals
        )
        self.possession_json = possession_json
        self.possession_csv = possession_csv
        self.possession_timeline_csv = possession_timeline_csv
        self.possession_radius_m = possession_radius_m
        self.passes_json = passes_json
        self.passes_csv = passes_csv
        self.pass_statistics_csv = pass_statistics_csv
        self.pass_network_dir = pass_network_dir
        self.shots_json = shots_json
        self.shots_csv = shots_csv
        self.shot_statistics_csv = shot_statistics_csv
        self.shot_map_dir = shot_map_dir
        self.goals_json = goals_json
        self.goals_csv = goals_csv
        self.goal_statistics_csv = goal_statistics_csv
        self.goal_notifications_path = goal_notifications_path
        self.highlights_path = highlights_path
        self.highlights_json = highlights_json
        self.highlight_pre_sec = highlight_pre_sec
        self.highlight_post_sec = highlight_post_sec
        self.progress_callback = progress_callback
        # Per-track exponential smoothing of pitch positions (reduces jitter).
        # 0 = off, higher = smoother/laggier. Keyed by (class_name, track_id).
        self.position_smoothing = position_smoothing
        self._pos_ema: dict[tuple, tuple[float, float]] = {}
        self.speed_json = speed_json
        self.speed_csv = speed_csv
        self.sprint_threshold_m_s = sprint_threshold_m_s
        self._pitch_scale = pitch_scale

        dev = device if device != "cpu" else None
        self.field_detector = FieldDetector(
            model_path=field_model,
            conf_threshold=conf_threshold,
            kpt_conf_threshold=kpt_conf_threshold,
            device=dev,
            imgsz=imgsz,
        )
        self.object_detector = FootballDetector(
            model_path=object_model,
            conf_threshold=conf_threshold,
            ball_conf_threshold=ball_conf_threshold,
            device=dev,
            imgsz=imgsz,
        )
        # Dedicated trained ball detector (football-ball-detection.pt). Optional:
        # if the weights can't be loaded, it disables itself and we fall back
        # to the custom players model's ball detections.
        self.ball_detector: BallDetector | None = None
        if enable_ball_model:
            self.ball_detector = BallDetector(
                model_path=ball_model,
                conf_threshold=ball_conf_threshold,
                device=dev,
                imgsz=imgsz,
            )
            if not self.ball_detector.available:
                self.ball_detector = None
        # Recent ball positions on the pitch (cm) for the trajectory overlay.
        self._ball_pitch = BallPitchTracker(
            max_jump_cm=1100.0,
            smoothing=max(0.45, min(0.7, position_smoothing)),
            coast_frames=12,
            trail_max=40,
            trail_gap_frames=18,
            gap_reset_frames=8,
            reseat_confirm=2,
        )
        # Pitch left/right orientation vs the broadcast. The homography maps to
        # a fixed template that can be mirrored relative to the camera, which
        # would place the on-screen right-hand goal at pitch x=0 (left). We
        # detect this and flip x so the analysis matches the video.
        self._flip_x: bool | None = None
        self._frame_w: int = 0
        self._frame_h: int = 0
        # Global left/right orientation is decided from where play actually
        # concentrates during a short warmup, then locked. Homography solutions
        # can be internally mirrored, so a geometric per-frame test is
        # unreliable; the distribution of ball positions is far more robust.
        self._orient_lock_frame: int = 180
        self._orient_ball_x: list[float] = []
        self._flip_locked: bool = False
        self._orient_locked_at: int = -1
        # Held minimap during brief homography dropouts (anti-blink).
        self._last_minimap: np.ndarray | None = None
        self._minimap_hold: int = 0
        self.homography = HomographyEstimator(smoothing=smoothing)
        self.pitch = PitchView(scale=pitch_scale)

        # Reuse the object-detection annotator for the left (source) panel.
        self._obj_config = PipelineConfig(
            model_path=object_model,
            source_video=source,
            enable_tracking=True,
            enable_teams=enable_teams,
        )
        self.annotator = Annotator(self._obj_config)

        self.team_classifier = (
            TeamClassifier() if enable_teams else None
        )

        self.speed_estimator: SpeedEstimator | None = None
        self.speed_exporter: SpeedExporter | None = None
        self._speed_smoothing = speed_smoothing
        self._min_sprint_sec = min_sprint_sec
        self.heatmap_collector: HeatmapCollector | None = (
            HeatmapCollector() if enable_heatmaps else None
        )
        self.possession_analyzer: PossessionAnalyzer | None = None
        self.possession_exporter: PossessionExporter | None = None
        self.goal_detector: GoalDetector | None = None

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        if not os.path.exists(self.source):
            raise FileNotFoundError(f"Input video not found: {self.source}")

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_w, self._frame_h = width, height

        print(f"[info] device      : {self.object_detector.device}")
        print(f"[info] source      : {self.source}")
        print(f"[info] resolution  : {width}x{height} @ {fps:.2f} fps")
        print(f"[info] total frames: {total if total > 0 else 'unknown'}")
        if self.enable_speed:
            print(f"[info] speed       : enabled (sprint >= {self.sprint_threshold_m_s:.1f} m/s)")
        if self.enable_heatmaps:
            print("[info] heatmaps    : enabled")
        if self.enable_possession:
            print(f"[info] possession  : enabled (radius {self.possession_radius_m:.1f} m)")
        if self.enable_passes:
            print("[info] passes      : enabled")
        if self.enable_shots:
            print("[info] shots       : enabled")
        if self.enable_goals:
            print("[info] goals       : enabled")
        if self.enable_highlights:
            print("[info] highlights  : enabled")

        if self.enable_speed:
            self.speed_estimator = SpeedEstimator(
                fps=fps,
                sprint_threshold_m_s=self.sprint_threshold_m_s,
                min_sprint_sec=self._min_sprint_sec,
                speed_smoothing=self._speed_smoothing,
            )
            self.speed_exporter = SpeedExporter(
                json_path=self.speed_json,
                csv_path=self.speed_csv,
                sprint_threshold_m_s=self.sprint_threshold_m_s,
            )

        if self.enable_possession:
            self.possession_analyzer = PossessionAnalyzer(
                fps=fps,
                possession_radius_m=self.possession_radius_m,
            )
            self.possession_exporter = PossessionExporter(
                json_path=self.possession_json,
                csv_path=self.possession_csv,
                timeline_csv_path=self.possession_timeline_csv,
            )

        if self.enable_goals:
            self.goal_detector = GoalDetector(fps=fps)
            self.goal_detector.set_attack_directions({
                "Team A": 1,
                "Team B": -1,
            })

        # Output is the full-size annotated frame; the bird's-eye pitch (with
        # the ball trajectory) is overlaid small on the top-right corner.
        out_w, out_h = width, height
        writer = self._create_writer(self.output_path, fps, out_w, out_h)

        frames_payload: list[dict] = []
        frame_index = 0
        start = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # 1) Field keypoints -> homography.
                field_det = self.field_detector.detect(frame)
                H = self.homography.update(field_det)

                # 2) Objects (tracked for stable identities).
                detections = self.object_detector.track(frame)

                # 2b) Dedicated ball detection (pretrained COCO model).
                ball_px: tuple[float, float] | None = None
                if self.ball_detector is not None:
                    ball_px = self.ball_detector.detect(frame)

                # 3) Optional team classification.
                if self.team_classifier is not None:
                    self._classify_teams(frame, detections, frame_index)

                # 4) Draw the source (left) panel.
                self.annotator.draw_detections(frame, detections)

                # 5) Build the minimap (right) panel + collect coordinates.
                minimap, record = self._build_minimap(
                    detections, H, frame_index, fps, ball_px
                )
                frames_payload.append(record)

                # Lock the global left/right orientation after a short warmup,
                # then keep the heatmap consistent with the locked coordinates.
                if not self._flip_locked and frame_index >= self._orient_lock_frame:
                    self._lock_orientation(frames_payload)

                if (
                    self.heatmap_collector is not None
                    and self._flip_locked
                    and frame_index > self._orient_locked_at
                ):
                    self.heatmap_collector.add_frame(record)

                # 6) Speed estimation from bird's-eye coordinates.
                speed_samples: dict = {}
                if self.speed_estimator is not None and record["homography_available"]:
                    ts = frame_index / fps if fps else 0.0
                    speed_samples = self.speed_estimator.update(
                        frame_index, ts, record["players"]
                    )
                    draw_speed_overlay(frame, detections, speed_samples)

                pf = None
                # 7) Possession analysis.
                if self.possession_analyzer is not None:
                    pf = self.possession_analyzer.update(record)
                    draw_possession_overlay(frame, self.possession_analyzer)

                # 8) Goal detection + live scoreboard.
                if self.goal_detector is not None:
                    if frame_index == 60 and len(frames_payload) >= 30:
                        dirs = GoalDetector.infer_attack_directions(frames_payload)
                        self.goal_detector.set_attack_directions(dirs)
                    new_goals = self.goal_detector.update(record, pf)
                    for g in new_goals:
                        scorer = f"#{g.scorer_track_id}" if g.scorer_track_id else "Unknown"
                        print(
                            f"[goal] {g.scoring_team} scores ({scorer}) "
                            f"{g.score_team_a}-{g.score_team_b} @ {g.timestamp_sec:.1f}s"
                        )
                    draw_goal_overlay(frame, self.goal_detector)

                # 9) Overlay the bird's-eye pitch (small, top-right) onto the
                #    full-size annotated frame. When homography drops briefly,
                #    hold the last good minimap so the overlay does not blink to
                #    an empty pitch; clear it only after a sustained loss.
                if record["homography_available"]:
                    self._last_minimap = minimap
                    self._minimap_hold = 0
                elif self._last_minimap is not None and self._minimap_hold < 20:
                    minimap = self._last_minimap
                    self._minimap_hold += 1
                composite = self._composite(frame, minimap, height)
                writer.write(composite)

                if self.show_preview:
                    cv2.imshow("Bird's-eye view", composite)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[info] preview interrupted by user")
                        break

                frame_index += 1
                if frame_index % 30 == 0:
                    if total > 0:
                        pct = 100.0 * frame_index / total
                        print(f"[info] processed {frame_index}/{total} ({pct:.1f}%)")
                        if self.progress_callback is not None:
                            # Reserve 0.05..0.90 for the frame loop; the tail
                            # (exports, highlights) advances to 1.0 on completion.
                            frac = 0.05 + 0.85 * min(1.0, frame_index / total)
                            try:
                                self.progress_callback(frac)
                            except Exception:  # noqa: BLE001 - progress must never break the run
                                pass
                    else:
                        print(f"[info] processed {frame_index} frames")
        finally:
            cap.release()
            writer.release()
            if self.show_preview:
                cv2.destroyAllWindows()

        elapsed = time.time() - start
        avg_fps = frame_index / elapsed if elapsed > 0 else 0.0

        # Short clip that never reached the warmup lock: decide now so exports
        # (passes/shots/heatmap) are oriented consistently.
        if not self._flip_locked and frames_payload:
            self._lock_orientation(frames_payload)

        self._export(frames_payload, fps)

        speed_summary: dict = {}
        if self.speed_estimator is not None and self.speed_exporter is not None:
            last_ts = (frame_index - 1) / fps if fps and frame_index > 0 else 0.0
            self.speed_estimator.finalize(frame_index - 1, last_ts)
            stats = self.speed_estimator.all_stats()
            self.speed_exporter.save(stats, fps, self.source)
            speed_summary = self.speed_estimator.summary()

        heatmap_summary: dict = {}
        if self.heatmap_collector is not None:
            hm_data = self.heatmap_collector.finalize()
            hm_exporter = HeatmapExporter(
                output_dir=self.heatmap_dir,
                json_path=self.heatmap_json,
                html_path=self.heatmap_html,
                pitch_scale=self._pitch_scale,
            )
            heatmap_summary = hm_exporter.export(hm_data, source=self.source)

        possession_summary: dict = {}
        if self.possession_analyzer is not None and self.possession_exporter is not None:
            self.possession_analyzer.finalize()
            self.possession_exporter.save(
                self.possession_analyzer, self.source, fps
            )
            possession_summary = self.possession_analyzer.summary()

        pass_summary: dict = {}
        if (
            self.enable_passes
            and self.possession_analyzer is not None
            and frames_payload
        ):
            pass_detector = PassDetector(fps=fps)
            pass_detector.analyze(frames_payload, self.possession_analyzer)
            pass_exporter = PassExporter(
                json_path=self.passes_json,
                csv_path=self.passes_csv,
                summary_csv_path=self.pass_statistics_csv,
                network_dir=self.pass_network_dir,
                pitch_scale=self._pitch_scale,
            )
            pass_summary = pass_exporter.save(pass_detector, self.source, fps)

        shot_summary: dict = {}
        detected_shots: list = []
        if (
            self.enable_shots
            and self.possession_analyzer is not None
            and frames_payload
        ):
            shot_detector = ShotDetector(fps=fps)
            detected_shots = shot_detector.analyze(frames_payload, self.possession_analyzer)
            shot_exporter = ShotExporter(
                json_path=self.shots_json,
                csv_path=self.shots_csv,
                summary_csv_path=self.shot_statistics_csv,
                shot_map_dir=self.shot_map_dir,
                pitch_scale=self._pitch_scale,
            )
            shot_summary = shot_exporter.save(shot_detector, self.source, fps)

        goal_summary: dict = {}
        if self.enable_goals and self.goal_detector is not None:
            goal_exporter = GoalExporter(
                json_path=self.goals_json,
                csv_path=self.goals_csv,
                summary_csv_path=self.goal_statistics_csv,
                notifications_path=self.goal_notifications_path,
            )
            goal_summary = goal_exporter.save(self.goal_detector, self.source, fps)

        highlight_summary: dict = {}
        if self.enable_highlights and frames_payload:
            event_detector = EventDetector(fps=fps)
            events = event_detector.analyze(frames_payload)
            generator = HighlightGenerator(
                source=self.source,
                output_path=self.highlights_path,
                timeline_json=self.highlights_json,
                pre_seconds=self.highlight_pre_sec,
                post_seconds=self.highlight_post_sec,
            )
            if self.goal_detector is not None:
                generator.add_goals(self.goal_detector.goals)
            generator.add_shots(detected_shots)
            generator.add_events(events)
            highlight_summary = generator.generate()
            highlight_summary["event_counts"] = event_detector.counts()

        homography_frames = sum(1 for f in frames_payload if f["homography_available"])
        print("\n[done] bird's-eye view complete")
        print(f"       frames processed  : {frame_index}")
        print(f"       homography frames : {homography_frames}")
        print(f"       average fps       : {avg_fps:.2f}")
        print(f"       output video      : {self.output_path}")
        print(f"       coordinates       : {self.coordinates_json}")
        if self.speed_estimator is not None and self.speed_exporter is not None:
            print(f"       speed json          : {self.speed_json}")
            print(f"       speed csv           : {self.speed_csv}")
            if speed_summary.get("players_tracked", 0) > 0:
                print(
                    f"       players tracked   : {speed_summary['players_tracked']} "
                    f"(max {speed_summary['fastest_max_speed_kmh']} km/h, "
                    f"{speed_summary['total_sprints']} sprints)"
                )
        if heatmap_summary:
            print(f"       heatmap pngs      : {heatmap_summary['png_count']} in {heatmap_summary['output_dir']}")
            print(f"       heatmap json      : {heatmap_summary['json_path']}")
            print(f"       interactive html  : {heatmap_summary['html_path']}")
        if possession_summary:
            print(
                f"       possession        : "
                f"Team A {possession_summary['team_a_pct']:.0f}% | "
                f"Team B {possession_summary['team_b_pct']:.0f}% "
                f"(loose {possession_summary['loose_pct']:.0f}%)"
            )
            print(f"       possession json   : {self.possession_json}")
            print(f"       possession csv    : {self.possession_csv}")
        if pass_summary:
            print(
                f"       passes            : "
                f"{pass_summary['total_passes']} detected "
                f"({pass_summary['success_rate_pct']:.0f}% success)"
            )
            print(f"       passes json       : {pass_summary['json_path']}")
            print(f"       pass statistics   : {pass_summary['summary_csv_path']}")
            print(f"       pass networks     : {pass_summary['network_dir']}")
        if shot_summary:
            print(
                f"       shots             : "
                f"{shot_summary['total_shots']} detected "
                f"({shot_summary['on_target']} on target, "
                f"{shot_summary['accuracy_pct']:.0f}% accuracy)"
            )
            print(f"       shots json        : {shot_summary['json_path']}")
            print(f"       shot statistics   : {shot_summary['summary_csv_path']}")
            print(f"       shot maps         : {shot_summary['shot_map_dir']}")
        if goal_summary:
            fs = goal_summary["final_score"]
            teams = list(fs.keys())
            print(
                f"       goals             : "
                f"{goal_summary['total_goals']} detected "
                f"(final {teams[0]} {fs[teams[0]]} - {fs[teams[1]]} {fs[teams[1]]})"
            )
            print(f"       goals json        : {goal_summary['json_path']}")
            print(f"       goal statistics   : {goal_summary['summary_csv_path']}")
            print(f"       notifications     : {goal_summary['notifications_path']}")
        if highlight_summary:
            print(
                f"       highlights        : "
                f"{highlight_summary['clips']} clips from "
                f"{highlight_summary['events']} events "
                f"({highlight_summary['duration_sec']}s)"
            )
            print(f"       highlights video  : {highlight_summary['output_path']}")
            print(f"       highlights json   : {highlight_summary['timeline_json']}")

    # ------------------------------------------------------------------ #
    def _build_minimap(
        self,
        detections: list[Detection],
        H: np.ndarray | None,
        frame_index: int,
        fps: float,
        ball_px: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Project detections onto the pitch and render the minimap."""
        canvas = self.pitch.blank()
        players: list[dict] = []
        ball: dict | None = None
        ball_cm: tuple[float, float] | None = None
        seen_keys: set[tuple] = set()

        ball_raw_cm: tuple[float, float] | None = None

        if H is not None:
            for det in detections:
                # The dedicated ball detector (if present) is authoritative for
                # the ball; skip the custom model's own ball boxes to avoid
                # duplicate/competing markers.
                if det.class_name == "ball" and self.ball_detector is not None:
                    continue

                # Ground contact point: bottom-centre for people, centre for ball.
                if det.class_name == "ball":
                    fx, fy = det.center
                else:
                    fx, fy = (det.x1 + det.x2) / 2.0, float(det.y2)

                pitch_xy = self.homography.transform_point(fx, fy, H)
                if pitch_xy is None:
                    continue
                x_cm, y_cm = self._orient(*pitch_xy)

                if det.class_name == "ball":
                    if self._on_pitch(x_cm, y_cm, margin_cm=600):
                        ball_raw_cm = (x_cm, y_cm)
                else:
                    if not self._on_pitch(x_cm, y_cm, margin_cm=400):
                        continue
                    key = (det.class_name, det.track_id)
                    if det.track_id is not None:
                        x_cm, y_cm = self._smooth_position(key, x_cm, y_cm)
                        seen_keys.add(key)
                    color = self._marker_color(det)
                    label = str(det.track_id) if det.track_id is not None else None
                    self.pitch.draw_player(canvas, x_cm, y_cm, color, label)
                    players.append(
                        {
                            "track_id": det.track_id,
                            "class_name": det.class_name,
                            "team": det.team,
                            "x_cm": round(x_cm, 1),
                            "y_cm": round(y_cm, 1),
                        }
                    )

            # Ball from the dedicated trained detector (pixel -> pitch). This
            # is authoritative and overrides any custom-model ball above.
            if ball_px is not None:
                pitch_xy = self.homography.transform_point(ball_px[0], ball_px[1], H)
                if pitch_xy is not None:
                    oriented = self._orient(*pitch_xy)
                    if self._on_pitch(*oriented, margin_cm=600):
                        ball_raw_cm = oriented

            # Accumulate ball x for the warmup orientation decision.
            if ball_raw_cm is not None and not self._flip_locked:
                self._orient_ball_x.append(ball_raw_cm[0])

            sample = self._ball_pitch.update(
                frame_index,
                ball_raw_cm[0] if ball_raw_cm else None,
                ball_raw_cm[1] if ball_raw_cm else None,
            )
            if sample is not None:
                ball_cm = (sample.x_cm, sample.y_cm)
                ball = {"x_cm": round(sample.x_cm, 1), "y_cm": round(sample.y_cm, 1)}

            trail = self._ball_pitch.trail
            if len(trail) >= 2:
                self.pitch.draw_trajectory(canvas, trail)
            if ball_cm is not None:
                self.pitch.draw_ball(canvas, ball_cm[0], ball_cm[1])

            # Forget smoothing state for tracks not seen this frame, so a
            # reappearing/re-identified track does not glide from a stale spot.
            # Ball pitch smoothing is handled separately (_ball_pitch).
            self._prune_ema(seen_keys)
        else:
            self._ball_pitch.update(frame_index, None, None)

        record = {
            "frame": frame_index,
            "timestamp_sec": round(frame_index / fps, 3) if fps else None,
            "homography_available": H is not None,
            "players": players,
            "ball": ball,
            "ball_raw": (
                {"x_cm": round(ball_raw_cm[0], 1), "y_cm": round(ball_raw_cm[1], 1)}
                if ball_raw_cm is not None
                else None
            ),
        }
        return canvas, record

    def _smooth_position(
        self, key: tuple, x_cm: float, y_cm: float
    ) -> tuple[float, float]:
        """Exponentially smooth a track's pitch position to reduce jitter."""
        a = self.position_smoothing
        prev = self._pos_ema.get(key)
        if prev is None or a <= 0.0:
            self._pos_ema[key] = (x_cm, y_cm)
            return x_cm, y_cm
        # Large real movement (e.g. a long ball) should not be over-damped:
        # snap if the jump is big, otherwise blend.
        import math as _math

        jump = _math.hypot(x_cm - prev[0], y_cm - prev[1])
        if jump > 1500:  # cm — teleport/large move, trust the new position
            sx, sy = x_cm, y_cm
        else:
            sx = a * prev[0] + (1 - a) * x_cm
            sy = a * prev[1] + (1 - a) * y_cm
        self._pos_ema[key] = (sx, sy)
        return sx, sy

    def _prune_ema(self, seen_keys: set[tuple]) -> None:
        """Drop smoothing state for tracks absent this frame."""
        if not self._pos_ema:
            return
        stale = [k for k in self._pos_ema if k not in seen_keys]
        for k in stale:
            del self._pos_ema[k]

    @staticmethod
    def _on_pitch(x_cm: float, y_cm: float, margin_cm: float = 600) -> bool:
        """True if a projected point is plausibly on/near the pitch."""
        return (
            -margin_cm <= x_cm <= fc.PITCH_LENGTH + margin_cm
            and -margin_cm <= y_cm <= fc.PITCH_WIDTH + margin_cm
        )

    def _orient(self, x_cm: float, y_cm: float) -> tuple[float, float]:
        """Apply the locked left/right flip so pitch x matches the broadcast."""
        if self._flip_x:
            return fc.PITCH_LENGTH - x_cm, y_cm
        return x_cm, y_cm

    def _lock_orientation(self, frames_payload: list[dict]) -> None:
        """
        Decide and lock the global left/right orientation from warmup data.

        Football play (and the goal) concentrates toward the attacked goal. If
        that activity sits on the left half of the projected pitch we flip x so
        it renders on the right, matching how the action reads in the broadcast.
        Possession/speed use relative distances and are unaffected by the flip.
        """
        L = fc.PITCH_LENGTH
        samples = self._orient_ball_x
        if len(samples) < 15:
            # Too little ball data; fall back to outfield player distribution.
            samples = [
                p["x_cm"]
                for rec in frames_payload
                for p in rec.get("players", [])
                if p.get("x_cm") is not None
            ]

        flip = False
        if samples:
            med = sorted(samples)[len(samples) // 2]
            flip = med < L / 2.0

        self._flip_x = flip
        self._flip_locked = True
        self._orient_locked_at = frames_payload[-1]["frame"]

        if flip:
            for rec in frames_payload:
                for p in rec.get("players", []):
                    if p.get("x_cm") is not None:
                        p["x_cm"] = round(L - p["x_cm"], 1)
                for key in ("ball", "ball_raw"):
                    b = rec.get(key)
                    if b and b.get("x_cm") is not None:
                        b["x_cm"] = round(L - b["x_cm"], 1)

            # Positional smoothing / trail state was built pre-flip; reset it so
            # nothing glides across the pitch at the switch.
            self._pos_ema.clear()
            self._ball_pitch.reset()
            if self.goal_detector is not None:
                self.goal_detector._prev_ball = None
                self.goal_detector._zone_line = None
                self.goal_detector._zone_run = 0
                self.goal_detector._zone_last_pos = None

        # Rebuild the heatmap from all (now oriented) frames collected so far.
        # Frames strictly after the lock are added by the normal loop path.
        if self.heatmap_collector is not None:
            self.heatmap_collector = HeatmapCollector()
            for rec in frames_payload:
                self.heatmap_collector.add_frame(rec)

        print(f"[info] orientation  : {'flipped' if flip else 'normal'} "
              f"(locked at frame {frames_payload[-1]['frame']})")

    def _detect_flip(self, H: np.ndarray) -> bool | None:
        """
        Decide whether the projected pitch is mirrored vs the broadcast.

        Projects a point on the left of the frame and one on the right (near
        the bottom, where the near touchline is). If the on-screen right side
        maps to a *smaller* pitch x than the left side, the template is
        mirrored and we must flip x so the right-hand goal stays on the right.

        Returns None (defer) when the projection is implausible, so we lock the
        orientation only once the homography is trustworthy.
        """
        w = self._frame_w or 1280
        h = self._frame_h or 720
        y = h * 0.85
        left = self.homography.transform_point(w * 0.2, y, H)
        right = self.homography.transform_point(w * 0.8, y, H)
        if left is None or right is None:
            return None
        # Both probe points should land on/near the pitch, and be meaningfully
        # separated along x; otherwise defer the decision.
        if not self._on_pitch(*left, margin_cm=1500) or not self._on_pitch(
            *right, margin_cm=1500
        ):
            return None
        if abs(right[0] - left[0]) < 500:
            return None
        # right-of-screen should have the larger pitch x; if not, flip.
        return right[0] < left[0]

    def _marker_color(self, det: Detection) -> tuple[int, int, int]:
        """Minimap marker colour: team colour if known, else class colour."""
        if det.team_color is not None:
            return det.team_color
        return self._obj_config.color_for(det.class_name)

    def _classify_teams(
        self, frame: np.ndarray, detections: list[Detection], frame_index: int
    ) -> None:
        """Same jersey-colour clustering used by the object pipeline."""
        classifier = self.team_classifier
        assert classifier is not None
        for det in detections:
            if det.class_name == "player" and det.track_id is not None:
                feature = classifier.feature_from_frame(frame, det)
                if feature is not None:
                    classifier.add_sample(det.track_id, feature)
        classifier.maybe_refit(frame_index)
        if classifier.is_fitted:
            for det in detections:
                if det.class_name == "player" and det.track_id is not None:
                    team = classifier.predict(det.track_id)
                    if team is not None:
                        det.team = team
                        det.team_color = classifier.team_color(team)

    @staticmethod
    def _composite(
        frame: np.ndarray, minimap: np.ndarray, target_h: int
    ) -> np.ndarray:
        """Overlay the minimap (small) on the top-right of the full frame."""
        fh, fw = frame.shape[:2]
        mh, mw = minimap.shape[:2]

        # Scale the pitch overlay to ~30% of the frame width.
        overlay_w = max(160, int(fw * 0.30))
        scale = overlay_w / mw
        overlay_h = int(mh * scale)
        overlay = cv2.resize(minimap, (overlay_w, overlay_h))

        margin = max(8, int(fw * 0.012))
        x0 = fw - overlay_w - margin
        y0 = margin
        x1, y1 = x0 + overlay_w, y0 + overlay_h

        out = frame.copy()
        # Semi-transparent backing so the overlay reads over any background.
        roi = out[y0:y1, x0:x1]
        blended = cv2.addWeighted(overlay, 0.85, roi, 0.15, 0)
        out[y0:y1, x0:x1] = blended
        # Border around the overlay.
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (0, 215, 255), 2, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------ #
    def _export(self, frames_payload: list[dict], fps: float) -> None:
        _ensure_parent(self.coordinates_json)
        payload = {
            "source": self.source,
            "fps": fps,
            "pitch_length_cm": fc.PITCH_LENGTH,
            "pitch_width_cm": fc.PITCH_WIDTH,
            "total_frames": len(frames_payload),
            "frames": frames_payload,
        }
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
