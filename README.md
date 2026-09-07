# Football Object Detection Pipeline

A clean, modular pipeline that runs a trained **YOLOv8** model over a football
match video to detect **players, goalkeepers, referees, and the ball**. It
draws colour-coded bounding boxes with labels + confidence, overlays live FPS,
saves an annotated video, and exports every detection to **JSON** and **CSV**.

It also supports **multi-object tracking with ByteTrack** (`--track`): every
player, referee and the ball is assigned a persistent ID that survives short
occlusions and reappearances, drawn on the video and exported with full
per-track trajectories to `tracking.json` / `tracking.csv`.

Finally, it can **automatically classify players into Team A / Team B**
(`--teams`) from their jersey colour using HSV + K-Means clustering - no manual
colour picking. Referees and goalkeepers are detected as their own classes and
kept separate from the two teams.

For the ball specifically, an **advanced Kalman-filter tracker** (`--ball-track`)
keeps a continuous estimate through fast movement, motion blur, and brief
disappearances, draws the trajectory on video, computes speed, and exports
`ball_trajectory.json`.

A separate **field detection pipeline** (`detect_field.py`) uses the trained
YOLOv8 **pose** model to locate the 32 pitch keypoints, reconstruct the
sidelines, goal lines, penalty boxes, goal boxes, halfway line and centre
circle, save a visualization, and export the field coordinates to JSON. See
[Field detection](#field-detection-pitch-keypoints) below.

A **bird's-eye view pipeline** (`birdseye.py`) combines both models: it uses the
field keypoints to compute a **homography**, projects players and the ball onto
a top-down pitch, and renders that tactical minimap side by side with (and in
sync with) the source video, exporting the transformed pitch coordinates. See
[Bird's-eye view](#birds-eye-view-homography) below.

## Detected classes

| Class ID | Class name  | Box colour |
|----------|-------------|------------|
| 0        | ball        | yellow     |
| 1        | goalkeeper  | orange     |
| 2        | player      | green      |
| 3        | referee     | red        |

## Project structure

```
.
├── detect.py                 # object detection / tracking / teams / ball CLI
├── detect_field.py           # field (pitch) detection CLI
├── birdseye.py               # bird's-eye homography view CLI
├── heatmaps.py               # heatmap generation CLI (from coordinates JSON)
├── possession.py             # possession analysis CLI (from coordinates JSON)
├── passes.py                 # pass detection CLI (from coordinates JSON)
├── shots.py                  # shot detection CLI (from coordinates JSON)
├── goals.py                  # goal detection CLI (from coordinates JSON)
├── highlights.py             # highlight reel generation CLI
├── run_api.py                # FastAPI backend launcher
├── dashboard/                # React analytics dashboard (FIFA WC 2026 theme)
├── bytetrack.yaml            # ByteTrack config tuned for football
├── requirements.txt
├── src/
│   ├── config.py             # class/ID colours + run configuration
│   ├── detector.py           # YOLOv8 wrapper: detect() and track()
│   ├── annotator.py          # draws boxes, IDs, labels, FPS
│   ├── exporter.py           # writes detections.json / detections.csv
│   ├── tracking_exporter.py  # writes tracking.json / tracking.csv (+ history)
│   ├── team_classifier.py    # jersey HSV + K-Means -> Team A / Team B
│   ├── ball_tracker.py       # Kalman ball tracker + trajectory JSON
│   ├── pipeline.py           # orchestrates the object-detection run
│   ├── field_config.py       # 32 pitch keypoints: labels, groups, edges
│   ├── field_detector.py     # YOLOv8 pose wrapper -> FieldDetection
│   ├── field_annotator.py    # draws pitch lines, circle, keypoints
│   ├── field_pipeline.py     # field image/video run + coordinate export
│   ├── pitch.py              # top-down pitch renderer + cm<->px mapping
│   ├── homography.py         # frame->pitch homography estimator
│   ├── birdseye_pipeline.py  # homography tactical view + coord export
│   ├── speed_estimator.py    # player speed / distance / sprint detection
│   ├── speed_exporter.py     # speed stats JSON/CSV + live overlay
│   ├── heatmap_collector.py  # accumulate transformed coords into grids
│   ├── heatmap_renderer.py   # PNG heatmap rendering on pitch
│   ├── heatmap_exporter.py   # PNG + JSON + interactive HTML export
│   ├── possession_analyzer.py # nearest-player possession + timeline
│   ├── possession_exporter.py # possession JSON/CSV + live overlay
│   ├── pass_detector.py      # automatic pass detection + classification
│   ├── pass_network.py       # pass network visualization on pitch
│   ├── pass_exporter.py      # pass JSON/CSV + network PNG export
│   ├── shot_detector.py      # shot detection + on/off target/blocked
│   ├── shot_map.py           # shot map visualization on pitch
│   ├── shot_exporter.py      # shot JSON/CSV + shot map PNG export
│   ├── goal_detector.py      # goal line crossing + scorer attribution
│   ├── goal_exporter.py      # goal JSON/CSV + scoreboard + notifications
│   ├── event_detector.py     # corners/penalties/fouls/cards (heuristic)
│   └── highlight_generator.py # clip extraction + merge into highlights.mp4
└── output/                   # generated artifacts (video, json, csv)
```

## Installation

```bash
pip install -r requirements.txt
```

> Requires Python 3.9+. A CUDA-capable GPU is optional but recommended for
> faster inference (use `--device cuda` or `--device 0`).

## Usage

Minimal detection (uses the bundled model by default):

```bash
python detect.py --source match.mp4
```

Detection **+ ByteTrack tracking** (persistent IDs + `tracking.json`/`tracking.csv`):

```bash
python detect.py --source match.mp4 --track
```

Detection + tracking **+ automatic team classification** (Team A / Team B):

```bash
python detect.py --source match.mp4 --teams
```

Detection **+ Kalman ball tracking** (trajectory, speed, `ball_trajectory.json`):

```bash
python detect.py --source match.mp4 --ball-track
```

Full control:

```bash
python detect.py \
    --model football-players-detection/football-players-detection.pt \
    --source match.mp4 \
    --output output/annotated.mp4 \
    --json output/detections.json \
    --csv output/detections.csv \
    --track \
    --tracker bytetrack.yaml \
    --tracking-json output/tracking.json \
    --tracking-csv output/tracking.csv \
    --conf 0.25 \
    --iou 0.45 \
    --imgsz 1280 \
    --device cpu \
    --show
```

### CLI options

| Flag              | Default                                                        | Description                              |
|-------------------|----------------------------------------------------------------|------------------------------------------|
| `--model`         | `football-players-detection/football-players-detection.pt`     | Path to the trained `.pt` weights        |
| `--source`        | *(required)*                                                   | Input football match video               |
| `--output`        | `output/annotated.mp4`                                         | Annotated output video path              |
| `--json`          | `output/detections.json`                                      | Detections JSON export path              |
| `--csv`           | `output/detections.csv`                                       | Detections CSV export path               |
| `--conf`          | `0.25`                                                          | Confidence threshold                     |
| `--iou`           | `0.45`                                                          | NMS IoU threshold                        |
| `--imgsz`         | `1280`                                                          | Inference image size                     |
| `--device`        | `cpu`                                                          | `cpu`, `cuda`, or GPU index (`0`)        |
| `--show`          | off                                                            | Live preview window (press `q` to stop)  |
| `--track`         | off                                                            | Enable ByteTrack tracking (persistent IDs) |
| `--tracker`       | `bytetrack.yaml`                                              | Path to the ByteTrack config             |
| `--tracking-json` | `output/tracking.json`                                        | Tracking JSON export path                |
| `--tracking-csv`  | `output/tracking.csv`                                         | Tracking CSV export path                 |
| `--class-colors`  | off                                                            | Colour boxes per class instead of per ID |
| `--teams`         | off                                                            | Auto team classification (implies `--track`) |
| `--team-warmup`   | `30`                                                           | Frames of colour evidence before first clustering |
| `--ball-track`    | off                                                            | Kalman ball tracking with trajectory + speed |
| `--ball-json`     | `output/ball_trajectory.json`                                 | Ball trajectory JSON export path |
| `--ball-conf`     | `0.10`                                                         | Ball confidence floor (blur-tolerant) |
| `--ball-coast`    | `20`                                                           | Max frames to predict through disappearance |

## Output formats

**`detections.json`** — grouped by frame, with timestamps:

```json
{
  "fps": 25.0,
  "total_frames": 750,
  "total_detections": 15230,
  "frames": [
    {
      "frame": 0,
      "timestamp_sec": 0.0,
      "detections": [
        {
          "class_id": 2, "class_name": "player", "confidence": 0.91,
          "x1": 120, "y1": 340, "x2": 165, "y2": 430,
          "width": 45, "height": 90
        }
      ]
    }
  ]
}
```

**`detections.csv`** — flat, one row per detection:

```
frame,timestamp_sec,class_id,class_name,confidence,x1,y1,x2,y2,width,height
0,0.0,2,player,0.91,120,340,165,430,45,90
```

## Tracking (ByteTrack)

Enable with `--track`. The pipeline uses Ultralytics' built-in **ByteTrack**,
which is a *motion-only* tracker (no appearance/ReID model), keeping it fast
enough for real-time use.

- **Persistent IDs** — each player, referee and the ball keeps the same ID for
  as long as it is tracked.
- **Occlusion handling** — ByteTrack's two-stage association also matches
  *low-confidence* boxes (partially hidden players) against a Kalman motion
  model, so an ID isn't dropped the moment an object is occluded.
- **Reappearance** — a lost track is retained for `track_buffer` frames (see
  `bytetrack.yaml`, default 90 ≈ 3.6 s at 25 fps). If the object reappears
  within that window it re-attaches to the **same ID**.
- **Unique colours** — boxes are coloured per track ID by default (use
  `--class-colors` to revert to per-class colours).

Tune behaviour in [`bytetrack.yaml`](bytetrack.yaml): raise `track_buffer` for
longer occlusions, adjust `track_high_thresh` / `track_low_thresh` for how
aggressively low-confidence boxes are rescued.

### `tracking.json`

```json
{
  "fps": 25.0,
  "total_frames": 750,
  "unique_tracks": 33,
  "unique_tracks_per_class": { "player": 32, "referee": 1 },
  "tracks": [
    {
      "track_id": 1, "class_name": "referee",
      "first_frame": 0, "last_frame": 24,
      "frames_seen": 25, "frame_span": 25, "avg_confidence": 0.77,
      "trajectory": [ { "frame": 0, "timestamp_sec": 0.0, "cx": 1322, "cy": 1885 } ]
    }
  ],
  "frames": [
    { "frame": 0, "timestamp_sec": 0.0, "tracks": [ { "track_id": 1, "class_name": "referee", "confidence": 0.77, "x1": 1315, "y1": 1862, "x2": 1330, "y2": 1909, "cx": 1322, "cy": 1885 } ] }
  ]
}
```

> `frame_span > frames_seen` indicates the track was occluded/lost for some
> frames but kept the same ID on reappearance.

### `tracking.csv`

```
frame,timestamp_sec,track_id,class_id,class_name,confidence,x1,y1,x2,y2,width,height,cx,cy
0,0.0,1,3,referee,0.7748,1315,1862,1330,1909,15,47,1322,1885
```

## Team classification

Enable with `--teams` (this implies `--track`, since it works on tracked player
crops). Everything is automatic - there is **no manual colour selection**.

For every tracked **player** each frame:

1. **Jersey region** — the torso is cropped out of the player box (skipping head
   and legs, trimming the sides) so the colour comes from the shirt, not the
   background.
2. **HSV** — the region is converted to HSV, which separates colour (hue) from
   lighting (value) far better than RGB.
3. **Grass removal** — green pixels are masked out so the pitch does not
   contaminate the jersey colour.
4. **Colour feature** — the remaining pixels are summarised as a hue-aware
   feature `[cos(H), sin(H), S, V]` (encoding hue on a circle so red ≈ 0° and
   red ≈ 179° are treated as close), accumulated per track ID across frames for
   stability.
5. **K-Means (k=2)** — after a short warm-up, all player tracks are clustered
   into two groups with `cv2.kmeans`. Clusters are labelled **Team A / Team B**
   deterministically by hue, and re-fit periodically as more evidence arrives.

**Referees and goalkeepers** are separate detector classes, so they are excluded
from the team clustering and keep their own labels/colours - they are never
assigned to a team (their `team` field stays empty in the exports).

Output:

- **Team-coloured boxes** — each player's box is drawn in its team's
  representative jersey colour; referees/goalkeepers keep their class colour.
- **Team names on-screen** — e.g. `Team A #7 0.88` is drawn above each player.
- **Exports** — the `team` column/field is added to `detections.*` and
  `tracking.*`, and the run prints a final `Team A: n, Team B: m` split.

Tune with `--team-warmup` (frames before the first clustering) and, in
`config.py`, `team_refit_interval` and `team_min_samples`.

## Ball tracking (Kalman filter)

Enable with `--ball-track`. This is a dedicated tracker for the football only,
separate from ByteTrack (which handles players/referees). YOLOv8 supplies
intermittent ball detections; the Kalman filter keeps a smooth, continuous
estimate between them.

Per frame:

1. **Predict** — constant-velocity Kalman step `[x, y, vx, vy]` extrapolates
   the ball position, handling fast movement between frames.
2. **Associate** — the nearest YOLO ball detection within a spatial gate is
   matched to the prediction. Association uses *position*, not confidence alone,
   so low-confidence (motion-blurred) detections are still accepted when they
   land near the predicted spot.
3. **Correct or coast** — on a match the filter corrects with the measurement;
   on a miss it keeps predicting for up to `--ball-coast` frames, bridging
   brief disappearances (occlusion, blur, out of frame).
4. **Speed** — velocity from the filter state gives `speed_px_per_sec` each frame.

Visual output:

- **Trajectory trail** — fading yellow polyline of recent positions.
- **Ball marker** — solid circle when detected, hollow ring when coasting
  (Kalman prediction).
- **Label** — coordinates and speed, e.g. `ball (DET) (450,320) 620px/s`.

### `ball_trajectory.json`

```json
{
  "fps": 25.0,
  "total_points": 750,
  "detected_points": 680,
  "predicted_points": 70,
  "max_speed_px_per_sec": 890.5,
  "avg_speed_px_per_sec": 145.2,
  "trajectory": [
    {
      "frame": 0, "timestamp_sec": 0.0,
      "x": 450.2, "y": 320.1,
      "vx": 290.7, "vy": -12.4,
      "speed_px_per_sec": 291.0,
      "detected": true, "confidence": 0.87
    }
  ]
}
```

Tune with `--ball-conf` (lower accepts blurrier detections), `--ball-coast`
(longer disappearance tolerance), and `ball_gate_radius` in `config.py`.

## Field detection (pitch keypoints)

The `football-field-detection.pt` model is a YOLOv8 **pose** model (single class
`pitch`, 32 keypoints) following Roboflow's `SoccerPitchConfiguration` layout.
`detect_field.py` runs it on an **image or video** and reconstructs the full
pitch geometry.

```bash
# Single image  -> annotated .png + single-frame JSON
python detect_field.py --source pitch.jpg

# Match video   -> annotated .mp4 + per-frame JSON
python detect_field.py --source match.mp4 --output output/field.mp4
```

Each of the 32 keypoints is given a **semantic label and group** in
`field_config.py`, and the pitch lines are reconstructed from the known edge
topology. Elements are colour-coded by feature:

| Feature        | Colour   | How it is derived                                    |
|----------------|----------|------------------------------------------------------|
| Sidelines      | white    | edges along `y = 0` and `y = width` (touchlines)     |
| Goal lines     | green    | edges along `x = 0` and `x = length` (end lines)     |
| Penalty boxes  | orange   | 18-yard box edges, both ends                         |
| Goal boxes     | magenta  | 6-yard box edges, both ends                          |
| Halfway line   | yellow   | edges across the pitch at `x = length/2`             |
| Centre circle  | red      | ellipse fit through the 4 circle keypoints           |
| Penalty spots  | azure    | penalty-spot keypoints                               |

A pitch line is only drawn when **both** its endpoint keypoints are visible
(above `--kpt-conf`), so partial views (typical broadcast footage) render
correctly. The centre circle is fit with `minAreaRect` for robustness under
perspective.

### CLI options

| Flag         | Default                                                     | Description                              |
|--------------|-------------------------------------------------------------|------------------------------------------|
| `--model`    | `football-field-detection/football-field-detection.pt`      | Trained pose `.pt` model                 |
| `--source`   | *(required)*                                               | Input image or video                     |
| `--output`   | `output/field_annotated.mp4`                               | Visualization path (images auto -> .png) |
| `--json`     | `output/field_coordinates.json`                            | Field coordinates JSON                   |
| `--conf`     | `0.25`                                                      | Pitch detection confidence               |
| `--kpt-conf` | `0.5`                                                       | Per-keypoint visibility threshold        |
| `--imgsz`    | `1280`                                                      | Inference image size                     |
| `--device`   | `cpu`                                                       | `cpu`, `cuda`, or GPU index (`0`)        |
| `--no-labels`| off                                                        | Hide keypoint index labels               |
| `--show`     | off                                                        | Live preview window (video)              |

### `field_coordinates.json`

```json
{
  "source": "match.mp4",
  "type": "video",
  "num_keypoints": 32,
  "frames": [
    {
      "frame": 0,
      "timestamp_sec": 0.0,
      "pitch_confidence": 0.98,
      "num_visible": 7,
      "keypoints": [
        { "index": 14, "label": "centre_circle_top", "group": "centre_circle",
          "x": 690.1, "y": 322.4, "confidence": 0.91, "visible": true }
      ],
      "features": {
        "centre_circle": [ /* ... */ ],
        "penalty_box":  [ /* ... */ ],
        "halfway_line": [ /* ... */ ]
      }
    }
  ],
  "pitch_template_cm": [
    { "index": 8, "label": "left_penalty_spot", "group": "penalty_spot",
      "x_cm": 1100.0, "y_cm": 3500.0 }
  ]
}
```

The `keypoints` list carries the raw per-index detections; `features` regroups
the visible ones by pitch element (sidelines, goal lines, penalty boxes, etc.).
`pitch_template_cm` gives the real-world landmark coordinates (centimetres) of
all 32 points — the target plane for a **homography** if you want a top-down
minimap: pair the visible frame keypoints with their template points and solve
`cv2.findHomography`. That is exactly what the bird's-eye pipeline does next.

## Bird's-eye view (homography)

`birdseye.py` fuses **both** models to build a live tactical minimap:

```bash
# Both bundled models, teams coloured, synchronized side-by-side output
python birdseye.py --source match.mp4

# Custom output, disable team colouring
python birdseye.py --source match.mp4 --output output/tactical.mp4 --no-teams
```

Per frame the pipeline:

1. Detects the **32 pitch keypoints** (pose model).
2. Computes the **frame -> pitch homography** `H` from the visible keypoints and
   their known real-world template positions (`cv2.findHomography`, RANSAC).
3. Detects & **tracks** players / ball / referees / goalkeepers (object model),
   optionally clustering players into teams for colouring.
4. Projects each object's **ground contact point** (players: bottom-centre of
   the box; ball: centre) through `H` into pitch centimetres.
5. Draws a **top-down pitch** and places players (team/class-coloured dots with
   track IDs) and the ball on it.
6. Composites `[ source video | minimap ]` side by side and writes the video —
   the two panels are always **in sync** because both come from the same frame.
7. Exports every transformed coordinate to `birdseye_coordinates.json`.

Robustness for live video:

- **RANSAC** rejects mismatched keypoints when solving `H`.
- On frames with **too few keypoints** (camera zoomed into a corner), the last
  good `H` is reused for up to 30 frames so the minimap does not flicker.
- **Temporal smoothing** (`--smoothing`, default 0.5) blends `H` across frames
  to reduce jitter.

### CLI options

| Flag             | Default                                                      | Description                              |
|------------------|--------------------------------------------------------------|------------------------------------------|
| `--field-model`  | `football-field-detection/football-field-detection.pt`       | Trained field (pose) model               |
| `--object-model` | `football-players-detection/football-players-detection.pt`   | Trained players/ball model               |
| `--source`       | *(required)*                                                | Input match video                        |
| `--output`       | `output/birdseye.mp4`                                       | Synchronized side-by-side output video   |
| `--json`         | `output/birdseye_coordinates.json`                         | Transformed coordinates export           |
| `--conf`         | `0.25`                                                       | Detection confidence                     |
| `--kpt-conf`     | `0.5`                                                        | Field keypoint visibility threshold      |
| `--imgsz`        | `1280`                                                       | Inference image size                     |
| `--device`       | `cpu`                                                        | `cpu`, `cuda`, or GPU index (`0`)        |
| `--scale`        | `0.06`                                                       | Minimap scale (pixels per cm)            |
| `--smoothing`    | `0.5`                                                        | Homography smoothing in `[0,1)`          |
| `--no-teams`     | off                                                         | Disable jersey-colour team colouring     |
| `--show`         | off                                                         | Live preview window                      |

### `birdseye_coordinates.json`

```json
{
  "source": "match.mp4",
  "fps": 25.0,
  "pitch_length_cm": 12000,
  "pitch_width_cm": 7000,
  "total_frames": 750,
  "frames": [
    {
      "frame": 0,
      "timestamp_sec": 0.0,
      "homography_available": true,
      "players": [
        { "track_id": 7, "class_name": "player", "team": "Team A",
          "x_cm": 3994.7, "y_cm": 4072.8 }
      ],
      "ball": { "x_cm": 1735.2, "y_cm": 5437.5 }
    }
  ]
}
```

Coordinates are in **pitch centimetres** (`x` in `[0, 12000]` goal-to-goal,
`y` in `[0, 7000]` touchline-to-touchline), ready for downstream analytics like
distance covered, team shape, or heatmaps.

### Player speed estimation

Enabled by default in `birdseye.py`. Once players are projected onto the pitch,
the speed module computes **real-world** metrics from bird's-eye coordinates:

| Metric | Description |
|--------|-------------|
| Distance | Cumulative metres covered per player |
| Instantaneous speed | Smoothed frame-to-frame speed (km/h on video) |
| Average speed | Mean speed over all tracked frames |
| Maximum speed | Peak speed reached |
| Acceleration | m/s² between consecutive frames |
| Sprint events | Periods above `--sprint-threshold` (default 7.0 m/s ≈ 25 km/h) |

Live overlay: each player's speed in **km/h** is drawn below their bounding box
(cyan normally, red + `SPRINT` label when sprinting).

Exports:
- `player_speed.json` — full per-player history, sprint events, aggregates
- `player_speed.csv` — flat summary table (one row per player)

```bash
python birdseye.py --source match.mp4
python birdseye.py --source match.mp4 --sprint-threshold 6.5 --min-sprint 0.8
python birdseye.py --source match.mp4 --no-speed   # disable speed module
```

| Flag | Default | Description |
|------|---------|-------------|
| `--no-speed` | off | Disable speed estimation |
| `--speed-json` | `output/player_speed.json` | Full statistics JSON |
| `--speed-csv` | `output/player_speed.csv` | Summary CSV |
| `--sprint-threshold` | `7.0` | Sprint cutoff in m/s |
| `--min-sprint` | `1.0` | Min sprint duration (seconds) |

### Heatmap generation

Build tactical heatmaps from **bird's-eye transformed coordinates** (pitch cm).
Five map types are produced:

| Map | Description |
|-----|-------------|
| **Player heatmaps** | Per-track density (one PNG per player) |
| **Team heatmaps** | Aggregated density for Team A and Team B |
| **Ball heatmap** | Where the ball spent time on the pitch |
| **Zone control** | 12×7 tactical zones — blue = Team A, red = Team B |
| **Pressure map** | Player density weighted by proximity to the ball |

```bash
# Generate during bird's-eye processing
python birdseye.py --source match.mp4 --heatmaps

# Or standalone from saved coordinates
python heatmaps.py --coordinates output/birdseye_coordinates.json
```

Outputs:
- **PNG images** in `output/heatmaps/` (one per player/team + ball + zone + pressure)
- **`heatmap_data.json`** — full grid data for every layer
- **`heatmaps.html`** — interactive dashboard (layer switching, hover values, player picker)

Open `output/heatmaps.html` in any browser to explore maps interactively.

| Flag | Default | Description |
|------|---------|-------------|
| `--heatmaps` | off | Enable in `birdseye.py` |
| `--heatmap-dir` | `output/heatmaps` | PNG output directory |
| `--heatmap-json` | `output/heatmap_data.json` | Grid data JSON |
| `--heatmap-html` | `output/heatmaps.html` | Interactive HTML |

### Possession analysis

Automatic possession tracking from **bird's-eye coordinates**. Each frame:

1. Finds the **nearest player** to the ball (pitch cm distance).
2. Assigns possession if within `--possession-radius` (default **2.0 m**).
3. Marks **contested** when opponents are equidistant.
4. Marks **loose ball** when no player is close enough.
5. Uses **stickiness** so possession does not flicker frame-to-frame.

```bash
# Enabled by default in bird's-eye processing
python birdseye.py --source match.mp4

# Standalone from saved coordinates
python possession.py --coordinates output/birdseye_coordinates.json
```

**Live overlay:** top bar shows Team A % | Team B %; label shows current holder,
`LOOSE BALL`, or `CONTESTED`.

**Exports:**
- `possession.json` — team stats, player stats, full timeline, per-frame log
- `possession.csv` — team + player summary table
- `possession_timeline.csv` — segment list (start/end, duration, holder)

| Flag | Default | Description |
|------|---------|-------------|
| `--no-possession` | off | Disable in `birdseye.py` |
| `--possession-json` | `output/possession.json` | Full statistics JSON |
| `--possession-csv` | `output/possession.csv` | Summary CSV |
| `--possession-timeline` | `output/possession_timeline.csv` | Timeline CSV |
| `--possession-radius` | `2.0` | Max ball-player distance (m) |

### Pass detection

Automatic pass detection from **possession transitions** and bird's-eye
coordinates. When one player releases the ball and a teammate receives it
(within a short loose-ball window), a pass event is recorded and classified:

| Type | Criteria |
|------|----------|
| **Short** | Distance &lt; 20 m |
| **Long** | Distance ≥ 25 m |
| **Through ball** | Long forward pass into the attacking third |
| **Cross** | From wide channel into the penalty box |
| **Back** | Significant backward component relative to attack direction |

**Success** is recorded when a teammate receives and holds the ball; **failure**
when an opponent intercepts or the ball stays loose too long.

```bash
# Enabled by default in bird's-eye processing
python birdseye.py --source match.mp4

# Standalone from saved coordinates
python passes.py --coordinates output/birdseye_coordinates.json
```

**Exports:**
- `passes.json` — all pass events + team/overall statistics
- `passes.csv` — flat per-pass table
- `pass_statistics.csv` — summary by team (totals, success rate, type counts)
- `output/pass_networks/` — pass network PNGs per team + pass trajectory maps

| Flag | Default | Description |
|------|---------|-------------|
| `--no-passes` | off | Disable in `birdseye.py` |
| `--passes-json` | `output/passes.json` | Pass events JSON |
| `--passes-csv` | `output/passes.csv` | Per-pass CSV |
| `--pass-statistics` | `output/pass_statistics.csv` | Summary CSV |
| `--pass-network-dir` | `output/pass_networks` | Network PNG directory |

### Shot detection

Automatic shot detection from the **ball trajectory** and bird's-eye
coordinates. A shot is inferred when the ball leaves an attacking player and
travels fast toward the opponent's goal from within shooting range. Each shot
is located, speed-rated, and classified:

| Outcome | Criteria |
|---------|----------|
| **On target** | Projected trajectory crosses inside the goal mouth, uninterrupted |
| **Off target** | Trajectory projects wide/over the goal mouth |
| **Blocked** | A defender/keeper sits on the ball path before the goal and the ball stops short |

For every shot the module also estimates:

- **Speed** — peak ball speed in m/s and km/h from consecutive pitch positions
- **Location** — pitch coordinates, distance to goal (m), and angle to goal centre
- **Target goal** — which goal (`left` / `right`) the attempt was aimed at

```bash
# Enabled by default in bird's-eye processing
python birdseye.py --source match.mp4

# Standalone from saved coordinates
python shots.py --coordinates output/birdseye_coordinates.json
```

**Exports:**
- `shots.json` — all shot events + team/overall statistics
- `shots.csv` — flat per-shot table
- `shot_statistics.csv` — summary by team (totals, on/off target, blocked, accuracy, speeds)
- `output/shot_maps/` — shot map PNGs (marker colour = outcome, size = speed)

| Flag | Default | Description |
|------|---------|-------------|
| `--no-shots` | off | Disable in `birdseye.py` |
| `--shots-json` | `output/shots.json` | Shot events JSON |
| `--shots-csv` | `output/shots.csv` | Per-shot CSV |
| `--shot-statistics` | `output/shot_statistics.csv` | Summary CSV |
| `--shot-map-dir` | `output/shot_maps` | Shot map PNG directory |

Standalone (`shots.py`) also accepts `--min-speed` (m/s toward goal, default
6.0) and `--max-distance` (m from goal, default 35.0) to tune sensitivity.

### Goal detection

Automatic goal detection from the **ball trajectory** crossing a goal line.
When the ball passes between consecutive pitch positions through x = 0 or
x = pitch length with y inside the goal mouth (7.32 m) and sufficient velocity
toward the goal, a goal event is recorded:

| Feature | Description |
|---------|-------------|
| **Goal line crossing** | Interpolated crossing point on left/right goal line |
| **Scorer** | Last attacking-team possession touch within 3 s before crossing |
| **Timestamp** | Frame number and `timestamp_sec` of the crossing |
| **Live scoreboard** | Top-right overlay: `Team A  2  -  1  Team B` |
| **Goal notification** | Full-width green banner for 4 s: `GOAL! Team A — #7 (1-0) @ 12.3s` |

```bash
# Enabled by default in bird's-eye processing (live scoreboard on video)
python birdseye.py --source match.mp4

# Standalone from saved coordinates
python goals.py --coordinates output/birdseye_coordinates.json
```

**Exports:**
- `goals.json` — all goal events, final score, notifications list
- `goals.csv` — flat per-goal table (scorer, timestamp, scoreline)
- `goal_statistics.csv` — goals per team with first/last goal times
- `goal_notifications.txt` — human-readable goal alert log

| Flag | Default | Description |
|------|---------|-------------|
| `--no-goals` | off | Disable in `birdseye.py` |
| `--goals-json` | `output/goals.json` | Goal events JSON |
| `--goals-csv` | `output/goals.csv` | Per-goal CSV |
| `--goal-statistics` | `output/goal_statistics.csv` | Summary CSV |
| `--goal-notifications` | `output/goal_notifications.txt` | Notifications log |

### Highlight generation

Automatically builds a single highlight reel from all detected match events.
The generator aggregates events into a ranked timeline, extracts a clip window
around each from the source video, merges overlapping windows, overlays an
event label, and writes `highlights.mp4`.

Events included:

| Event | Source | Reliability |
|-------|--------|-------------|
| **Goals** | `goal_detector` (ball crosses goal line) | robust |
| **Shots** | `shot_detector` (fast ball toward goal) | robust |
| **Corners** | ball settles in the corner arc | geometric |
| **Penalties** | ball settles on a penalty spot | geometric |
| **Fouls** | opposing players converge + ball dead | heuristic |
| **Yellow / red cards** | referee lingers by a player during a stoppage | heuristic |

> **Accuracy note.** Goals, shots, corners and penalties are derived from
> reliable ball/pitch geometry. **Fouls and cards are best-effort heuristics** —
> positional data alone cannot see contact or a raised card. Card *colour*
> (yellow vs red) cannot be inferred from coordinates, so cards default to
> `yellow_card`; plug in visual card recognition to distinguish colours.

```bash
# Enabled by default in bird's-eye processing (writes output/highlights.mp4)
python birdseye.py --source match.mp4

# Standalone — needs the source video (for clips) + saved coordinates
python highlights.py --source match.mp4 \
    --coordinates output/birdseye_coordinates.json
```

**Exports:**
- `highlights.mp4` — merged highlight video with per-clip event labels
- `highlights.json` — timeline of events and the merged clip windows

| Flag | Default | Description |
|------|---------|-------------|
| `--no-highlights` | off | Disable in `birdseye.py` |
| `--highlights` | `output/highlights.mp4` | Merged highlight video |
| `--highlights-json` | `output/highlights.json` | Highlight timeline JSON |
| `--highlight-pre` | `4.0` | Seconds of footage before each event |
| `--highlight-post` | `3.0` | Seconds of footage after each event |

Standalone (`highlights.py`) also accepts `--no-fouls`, `--no-cards`, and
`--no-shots` to filter which events feed the reel.

## REST API backend (FastAPI)

A modular, production-ready **FastAPI** backend (`app/`) wraps the bird's-eye
pipeline as a web service: upload match videos, process them **asynchronously**,
store analytics in a database, and serve JSON analytics, processed videos, and
downloadable reports.

### Run

```bash
pip install -r requirements.txt

# Development (auto-reload)
python run_api.py --reload
# or: uvicorn app.main:app --reload

# Production (bind all interfaces)
python run_api.py --host 0.0.0.0 --port 8000
```

Interactive docs are served at `http://127.0.0.1:8000/docs` (Swagger UI) and
`/redoc`.

### Architecture

```
app/
├── main.py                 # app factory: lifespan, CORS, routers, error handling
├── config.py               # env-driven settings (SFA_* vars, .env supported)
├── database.py             # SQLAlchemy engine/session (SQLite by default)
├── models.py               # MatchJob ORM model
├── repository.py           # job CRUD + lifecycle transitions
├── schemas.py              # Pydantic request/response models
├── enums.py                # JobStatus, AnalyticsType, VideoKind
├── worker.py               # ThreadPoolExecutor async job queue
├── dependencies.py         # shared FastAPI dependencies
├── services/
│   ├── analysis_service.py # runs BirdseyePipeline, collects artifacts + summary
│   └── report_service.py   # HTML/JSON match reports
└── routers/
    ├── health.py           # /health
    ├── matches.py          # upload / list / detail / delete
    ├── analytics.py        # summary + typed analytics JSON
    └── downloads.py        # processed videos + reports
```

Uploads and all generated artifacts live under `data/jobs/{job_id}/`; job
metadata, options, the artifact manifest, and the analytics summary are stored
in `data/app.db` (SQLite).

### Processing lifecycle

1. `POST /matches` saves the upload, creates a `queued` job, and submits it to
   the background queue.
2. A worker thread transitions the job to `processing`, runs the full
   `BirdseyePipeline`, then records the artifact manifest + summary and marks it
   `completed` (or `failed` with an error).
3. Clients poll `GET /matches/{id}` for `status`/`progress`, then fetch
   analytics, videos, and reports once `completed`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v1/health` | Service health + whether model weights are present |
| `POST` | `/api/v1/matches` | Upload a video (`file`, optional `options` JSON), queue analysis |
| `GET`  | `/api/v1/matches` | List jobs (`status`, `limit`, `offset`) |
| `GET`  | `/api/v1/matches/{id}` | Job status, progress, summary, artifacts |
| `DELETE` | `/api/v1/matches/{id}` | Delete a job and its files |
| `GET`  | `/api/v1/matches/{id}/analytics` | Consolidated analytics summary |
| `GET`  | `/api/v1/matches/{id}/analytics/{type}` | Typed JSON: `possession`, `passes`, `shots`, `goals`, `highlights`, `speed`, `heatmaps`, `coordinates` |
| `GET`  | `/api/v1/matches/{id}/video?kind=birdseye\|highlights` | Download a processed video |
| `GET`  | `/api/v1/matches/{id}/report?format=html\|json` | Download a match report |

### Example

```bash
# Upload a match with custom processing options
curl -F "file=@match.mp4" \
     -F 'options={"enable_heatmaps":true,"device":"cpu"}' \
     http://127.0.0.1:8000/api/v1/matches
# -> {"id":"<job_id>","status":"queued", ...}

# Poll status
curl http://127.0.0.1:8000/api/v1/matches/<job_id>

# Fetch analytics once completed
curl http://127.0.0.1:8000/api/v1/matches/<job_id>/analytics/shots

# Download the highlights reel and an HTML report
curl -OJ "http://127.0.0.1:8000/api/v1/matches/<job_id>/video?kind=highlights"
curl -OJ "http://127.0.0.1:8000/api/v1/matches/<job_id>/report?format=html"
```

### Configuration (environment variables)

All settings use the `SFA_` prefix (and an optional `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `SFA_DATA_DIR` | `./data` | Root for uploads, artifacts, and the database |
| `SFA_DATABASE_URL` | *(SQLite in data dir)* | Any SQLAlchemy URL (e.g. Postgres) |
| `SFA_FIELD_MODEL_PATH` | bundled pose model | Field (pitch) model weights |
| `SFA_OBJECT_MODEL_PATH` | bundled detector | Players/ball model weights |
| `SFA_MAX_WORKERS` | `1` | Concurrent analysis jobs |
| `SFA_DEVICE` | `cpu` | Inference device (`cpu`, `cuda`, `0`) |
| `SFA_IMGSZ` | `1280` | Inference image size |
| `SFA_MAX_UPLOAD_BYTES` | `2147483648` | Upload size limit (2 GB) |
| `SFA_CORS_ORIGINS` | `*` | Comma-separated allowed origins |

> For heavier production loads, swap the in-process thread queue for a broker
> (Celery/RQ) and point `SFA_DATABASE_URL` at Postgres — the service layer and
> repository are already decoupled for this.

## React Analytics Dashboard (FIFA WC 2026)

A modern **React** dashboard (`dashboard/`) connects to the FastAPI backend for
a full sports-analytics experience:

| Feature | Description |
|---------|-------------|
| **Upload** | Drag-and-drop match video upload |
| **Video** | Stream processed bird's-eye & highlights videos |
| **Possession** | Animated pie chart + progress bars |
| **Player stats** | Table with possession, touches, distance, sprints |
| **Speed** | Bar chart (avg vs max km/h per player) |
| **Pass network** | Interactive SVG pitch network + server PNGs |
| **Shot map** | SVG shot locations by outcome |
| **Heatmaps** | Switchable heatmap layers |
| **Goal timeline** | Chronological goals with scoreline |

**Stack:** React 19, TypeScript, Vite, Tailwind CSS 4, Framer Motion, Recharts.

```bash
# Terminal 1 — API
python run_api.py --reload

# Terminal 2 — Dashboard
cd dashboard && npm install && npm run dev
# → http://localhost:5173
```

See [`dashboard/README.md`](dashboard/README.md) for full details.

## Real-time performance tips

- Use a GPU: `--device 0` (install a CUDA build of PyTorch).
- Lower `--imgsz` (e.g. `960` or `640`) for a big speedup at some accuracy cost.
- ByteTrack itself adds negligible overhead (no ReID network), so throughput is
  dominated by YOLOv8 inference.

## How it works

1. **`detector.py`** loads the YOLOv8 weights and exposes `detect()` (stateless)
   and `track()` (stateful ByteTrack) returning typed `Detection` objects.
2. **`annotator.py`** draws colour-coded boxes, IDs, labels (`ID:n name conf`),
   and an FPS overlay.
3. **`exporter.py`** writes `detections.json` / `detections.csv`.
4. **`tracking_exporter.py`** builds per-track history (trajectories, lifespan)
   and writes `tracking.json` / `tracking.csv`.
5. **`team_classifier.py`** extracts jersey colour per player and clusters
   tracks into Team A / Team B with K-Means.
6. **`ball_tracker.py`** runs a Kalman filter on ball detections for continuous
   tracking, speed, and trajectory export.
7. **`pipeline.py`** reads the video frame by frame, wires the above together,
   and writes the annotated video.
8. **`field_detector.py` / `field_pipeline.py`** detect the 32 pitch keypoints
   and export field coordinates.
9. **`homography.py` / `pitch.py` / `birdseye_pipeline.py`** compute the
   frame->pitch homography and render the synchronized bird's-eye tactical view.
10. **`speed_estimator.py` / `speed_exporter.py`** derive distance, speed,
    acceleration and sprint events from bird's-eye coordinates.
11. **`heatmap_collector.py` / `heatmap_renderer.py` / `heatmap_exporter.py`**
    build player, team, ball, zone-control and pressure heatmaps with PNG,
    JSON and interactive HTML output.
12. **`possession_analyzer.py` / `possession_exporter.py`** determine nearest
    player possession, team %, player time, timeline, and live overlay.
13. **`pass_detector.py` / `pass_network.py` / `pass_exporter.py`** detect passes
    from possession transitions, classify by distance/direction, render pass
    networks, and export statistics.
14. **`shot_detector.py` / `shot_map.py` / `shot_exporter.py`** detect shots from
    the ball trajectory, classify on/off target/blocked, estimate speed and
    location, render shot maps, and export statistics.
15. **`goal_detector.py` / `goal_exporter.py`** detect goal-line crossings,
    attribute scorers, maintain a live scoreboard, generate goal notifications,
    and export goal statistics.
16. **`event_detector.py` / `highlight_generator.py`** detect corners, penalties,
    fouls and cards (heuristic), then extract and merge clips around every event
    into a single `highlights.mp4`.
