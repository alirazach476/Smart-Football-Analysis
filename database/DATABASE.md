# Smart Football Match Analysis — PostgreSQL Database Design

Production-ready relational schema for storing match videos, per-frame tracking, and derived analytics from the YOLOv8 + homography pipeline.

## Quick start

```bash
# Create database
createdb football_analytics

# Apply schema
psql -U postgres -d football_analytics -f database/schema.sql

# Run sample queries (replace UUID placeholders with real match ids)
psql -U postgres -d football_analytics -f database/sample_queries.sql
```

**Requirements:** PostgreSQL 14+ (`gen_random_uuid()`, `EXECUTE FUNCTION` triggers)

**Connection string (FastAPI):**

```
postgresql+asyncpg://user:pass@localhost:5432/football_analytics
```

---

## Entity–relationship diagram

```mermaid
erDiagram
    matches ||--o{ teams : "has 2"
    matches ||--o{ players : "tracks"
    matches ||--o{ tracking : "frames"
    matches ||--o{ possession : "segments"
    matches ||--o{ passes : "events"
    matches ||--o{ shots : "events"
    matches ||--o{ goals : "events"
    matches ||--o{ events : "set-piece / cards"
    matches ||--o{ heatmaps : "layers"
    matches ||--o{ highlights : "clips"
    matches ||--o{ artifacts : "files"

    teams ||--o{ players : "roster"
    teams ||--o{ possession : "holder"
    teams ||--o{ passes : "attacking"
    teams ||--o{ shots : "attacking"
    teams ||--o{ goals : "scoring"
    teams ||--o{ events : "involved"

    players ||--o{ tracking : "positions"
    players ||--o{ possession : "holder"
    players ||--o{ passes : "from / to"
    players ||--o{ shots : "shooter"
    players ||--o{ goals : "scorer"
    players ||--o{ events : "involved"
    players ||--o{ heatmaps : "player layer"

    matches {
        uuid id PK
        varchar external_job_id UK
        match_status status
        jsonb summary
        jsonb processing_options
    }

    teams {
        serial id PK
        uuid match_id FK
        varchar name
        team_side side
        numeric possession_pct
        int goals_scored
    }

    players {
        serial id PK
        uuid match_id FK
        int team_id FK
        int track_id
        entity_class class_name
        numeric total_distance_m
        int passes_made
        int goals
    }

    tracking {
        bigserial id PK
        uuid match_id FK
        int player_id FK
        int frame
        entity_class entity_class
        numeric x_cm
        numeric y_cm
    }

    possession {
        bigserial id PK
        uuid match_id FK
        int team_id FK
        int player_id FK
        possession_state state
        numeric start_time_sec
        numeric end_time_sec
        numeric duration_sec "generated"
    }

    passes {
        bigserial id PK
        uuid match_id FK
        int from_player_id FK
        int to_player_id FK
        boolean successful
        pass_type primary_type
        numeric distance_m
    }

    shots {
        bigserial id PK
        uuid match_id FK
        int shooter_player_id FK
        shot_outcome outcome
        goal_line_side target_goal
        numeric location_x_cm
        numeric location_y_cm
    }

    goals {
        bigserial id PK
        uuid match_id FK
        int scoring_team_id FK
        int scorer_player_id FK
        goal_line_side goal_line
        int score_team_a
        int score_team_b
    }

    events {
        bigserial id PK
        uuid match_id FK
        match_event_type event_type
        int frame
        numeric timestamp_sec
    }

    heatmaps {
        serial id PK
        uuid match_id FK
        heatmap_layer layer
        varchar layer_key UK
        jsonb grid_data
        text png_path
    }

    highlights {
        serial id PK
        uuid match_id FK
        int clip_index
        numeric start_time_sec
        numeric end_time_sec
        text label
        text merged_video_path
    }

    artifacts {
        serial id PK
        uuid match_id FK
        artifact_kind kind
        text file_path
    }
```

---

## Table reference

| Table | Purpose | Cardinality per match |
|-------|---------|----------------------|
| `matches` | Job root: video metadata, status, denormalized `summary` | 1 |
| `teams` | Team A / Team B (or named sides) with possession % and goals | 2 |
| `players` | ByteTrack identities + aggregated player stats | ~22–30 |
| `tracking` | Per-frame bird's-eye positions (cm) | **High volume** (fps × entities) |
| `possession` | Contiguous possession segments | Hundreds |
| `passes` | Pass events with type, distance, success | Tens–hundreds |
| `shots` | Shot attempts with outcome and location | Single digits–tens |
| `goals` | Goal-line crossings with scoreline snapshot | 0–10 |
| `events` | Corners, penalties, fouls, cards | Variable |
| `heatmaps` | Density grids + PNG paths per layer | ~5–25 |
| `highlights` | Clip windows for `highlights.mp4` | ~5–50 |
| `artifacts` | File manifest (videos, JSON, reports) | ~10–20 |

---

## Relationships

### Hub: `matches`

Every analytics table references `matches(id)` with `ON DELETE CASCADE`. Deleting a match removes all derived data and artifacts metadata.

```
matches (1) ──< teams (2)
matches (1) ──< players (N)
matches (1) ──< tracking (N×frames)
matches (1) ──< possession, passes, shots, goals, events, heatmaps, highlights, artifacts
```

### Teams ↔ players

- `players.team_id → teams.id` (`ON DELETE SET NULL`) — team assignment can be unknown early in processing.
- `UNIQUE (match_id, track_id)` — track ids are scoped to one match only.

### Event tables → players / teams

| Child table | FK to teams | FK to players |
|-------------|---------------|---------------|
| `possession` | `team_id` (holder) | `player_id` (holder) |
| `passes` | `team_id` | `from_player_id`, `to_player_id` |
| `shots` | `team_id` | `shooter_player_id` |
| `goals` | `scoring_team_id`, `conceded_team_id` | `scorer_player_id` |
| `events` | `team_id` | `player_id` |
| `heatmaps` | — | `player_id` (player layer only) |

### Tracking

- `tracking.player_id` links player/referee rows; ball frames use `entity_class = 'ball'` with nullable `track_id`.
- `UNIQUE (match_id, frame, entity_class, COALESCE(track_id, -1))` prevents duplicate positions per frame.

---

## Enumerated types

| Type | Values | Used in |
|------|--------|---------|
| `match_status` | queued, processing, completed, failed | `matches.status` |
| `team_side` | team_a, team_b, home, away, unknown | `teams.side` |
| `entity_class` | player, goalkeeper, referee, ball | `players`, `tracking` |
| `possession_state` | team_a, team_b, loose, contested, unknown | `possession.state` |
| `pass_type` | short, long, through_ball, cross, back | `passes` |
| `shot_outcome` | on_target, off_target, blocked | `shots.outcome` |
| `goal_line_side` | left, right | `shots`, `goals` |
| `match_event_type` | corner, penalty, foul, yellow_card, red_card | `events` |
| `heatmap_layer` | team_a, team_b, ball, zone_control, pressure, player | `heatmaps` |
| `artifact_kind` | source_video, birdseye_video, highlights_video, *_json, *_png, report_* | `artifacts` |

---

## Indexes

All indexes are defined in `schema.sql`. Summary by access pattern:

| Pattern | Index |
|---------|-------|
| Job queue / list | `idx_matches_status`, `idx_matches_created_at`, `idx_matches_status_created` |
| Match detail FK lookups | `idx_*_match_id` on every child table |
| Timeline queries | `idx_*_match_time` on possession, passes, shots, goals, events, highlights |
| Tracking by frame | `idx_tracking_match_frame`, partial `idx_tracking_ball` |
| Player trajectories | `idx_tracking_match_player_frame` |
| Pass network | `idx_passes_from_player`, `idx_passes_to_player`, `idx_passes_team_success` |
| Shot map / accuracy | `idx_shots_outcome`, `idx_shots_shooter` |
| Goal timeline | `idx_goals_scorer` |
| Dashboard JSON | GIN on `matches.summary`, `heatmaps.grid_data` |

---

## Coordinate system

Pitch coordinates match `field_config.py`:

- **Width:** 12,000 cm (120 m)
- **Height:** 7,000 cm (70 m)
- **Goal width:** 732 cm
- Origin and axes follow the bird's-eye homography output used by the pipeline.

---

## Denormalization strategy

| Location | What's stored | Why |
|----------|---------------|-----|
| `matches.summary` | Possession %, pass totals, shot counts | Fast list/detail API without joins |
| `teams.possession_pct`, `goals_scored` | Team aggregates | Scoreboard and possession chart |
| `players.*` stats columns | Distance, speed, passes, shots, goals | Player stats table without aggregation |
| `goals.score_team_a/b` | Score at moment of goal | Historical scoreline even if later goals added |

Event tables (`passes`, `shots`, `goals`) remain normalized for analytics queries and exports.

---

## Production considerations

### 1. Partition `tracking` by `match_id`

At 25 fps × 90 min × ~25 entities ≈ **3.4M rows per match**. Use declarative partitioning:

```sql
-- Example: convert tracking to partitioned table (migration)
CREATE TABLE tracking_partitioned (LIKE tracking INCLUDING ALL)
    PARTITION BY HASH (match_id);
```

Or archive tracking to object storage (Parquet) after processing and keep only aggregated stats in Postgres.

### 2. Connection pooling

Use **PgBouncer** (transaction mode) in front of the FastAPI async workers. Set `pool_size` conservatively; bulk inserts for tracking should use `COPY` or batched `executemany`.

### 3. Retention

```sql
DELETE FROM matches
WHERE status = 'failed' AND created_at < NOW() - INTERVAL '30 days';
-- CASCADE removes all child rows.
```

### 4. Backups

- **Logical:** `pg_dump` for schema + metadata tables.
- **Large tracking:** consider excluding `tracking` from nightly dumps; regenerate from `coordinates.json` artifacts if needed.

### 5. Read replicas

Route dashboard analytics (`v_match_overview`, pass network, shot map) to a read replica. Write path: upload → `matches` (queued) → worker bulk-inserts → `matches` (completed).

### 6. Security

- Application role: `GRANT SELECT, INSERT, UPDATE, DELETE` on all tables; no `SUPERUSER`.
- Separate read-only role for reporting.
- Store `source_video_path` / `file_path` as server-local paths, not user input.

### 7. Integration with FastAPI

Current API uses SQLite `match_jobs`. Migration path:

1. Set `SFA_DATABASE_URL=postgresql+asyncpg://...`
2. Replace SQLAlchemy models with tables from this schema.
3. After `BirdseyePipeline` completes, ETL JSON artifacts into normalized rows (loader script).

---

## Sample queries

See [`sample_queries.sql`](sample_queries.sql) for 20 ready-to-run examples:

- Recent matches with scoreline
- Top players by distance
- Pass network aggregation
- Shot map coordinates
- Goal timeline
- Possession validation
- Unified event chronology
- Highlight clips
- Processing queue
- Cross-match goal leaderboard

---

## Files

| File | Description |
|------|-------------|
| `schema.sql` | Full DDL: types, tables, indexes, triggers, views |
| `sample_queries.sql` | Analytics and operational query examples |
| `DATABASE.md` | This document |

---

## Logical data flow

```
Video upload
    → matches (queued)
    → Pipeline: detection + homography
    → tracking (bulk insert)
    → Derived: teams, players, possession, passes, shots, goals, events
    → Visuals: heatmaps, highlights, artifacts
    → matches (completed, summary JSONB)
    → Dashboard / API reads v_match_overview + child tables
```
