-- =============================================================================
-- Smart Football Match Analysis System — PostgreSQL Schema
-- =============================================================================
-- Production-ready relational design for match videos, tracking data, and
-- derived analytics (possession, passes, shots, goals, events, heatmaps,
-- highlights).
--
-- Run order:
--   psql -U postgres -d football_analytics -f schema.sql
--
-- Requires: PostgreSQL 14+
-- =============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS "btree_gist"; -- range overlap queries (optional)

-- ---------------------------------------------------------------------------
-- Enumerated types
-- ---------------------------------------------------------------------------
CREATE TYPE match_status AS ENUM (
    'queued',
    'processing',
    'completed',
    'failed'
);

CREATE TYPE team_side AS ENUM (
    'team_a',
    'team_b',
    'home',
    'away',
    'unknown'
);

CREATE TYPE entity_class AS ENUM (
    'player',
    'goalkeeper',
    'referee',
    'ball'
);

CREATE TYPE possession_state AS ENUM (
    'team_a',
    'team_b',
    'loose',
    'contested',
    'unknown'
);

CREATE TYPE pass_type AS ENUM (
    'short',
    'long',
    'through_ball',
    'cross',
    'back'
);

CREATE TYPE shot_outcome AS ENUM (
    'on_target',
    'off_target',
    'blocked'
);

CREATE TYPE goal_line_side AS ENUM (
    'left',
    'right'
);

CREATE TYPE match_event_type AS ENUM (
    'corner',
    'penalty',
    'foul',
    'yellow_card',
    'red_card'
);

CREATE TYPE heatmap_layer AS ENUM (
    'team_a',
    'team_b',
    'ball',
    'zone_control',
    'pressure',
    'player'
);

CREATE TYPE artifact_kind AS ENUM (
    'source_video',
    'birdseye_video',
    'highlights_video',
    'coordinates_json',
    'possession_json',
    'passes_json',
    'shots_json',
    'goals_json',
    'speed_json',
    'heatmap_json',
    'heatmap_png',
    'pass_network_png',
    'shot_map_png',
    'report_html',
    'report_json'
);

-- ---------------------------------------------------------------------------
-- Core: matches
-- ---------------------------------------------------------------------------
CREATE TABLE matches (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_job_id     VARCHAR(64) UNIQUE,          -- legacy API job id (hex)
    title               VARCHAR(512),
    original_filename   VARCHAR(512) NOT NULL,
    source_video_path   TEXT,
    fps                 NUMERIC(8, 3),
    total_frames        INTEGER,
    duration_sec        NUMERIC(10, 3),
    status              match_status NOT NULL DEFAULT 'queued',
    progress            NUMERIC(5, 4) NOT NULL DEFAULT 0
                        CHECK (progress >= 0 AND progress <= 1),
    processing_options  JSONB NOT NULL DEFAULT '{}',
    summary             JSONB NOT NULL DEFAULT '{}', -- denormalized quick stats
    error_message       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at          TIMESTAMPTZ,
    finished_at         TIMESTAMPTZ,
    CONSTRAINT matches_finished_after_started
        CHECK (finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at)
);

COMMENT ON TABLE matches IS
    'A football match uploaded for AI analysis. One row per video/job.';
COMMENT ON COLUMN matches.summary IS
    'Denormalized aggregates for API/dashboard (possession %, pass totals, etc.).';

-- ---------------------------------------------------------------------------
-- Teams (two per match, linked to detector labels Team A / Team B)
-- ---------------------------------------------------------------------------
CREATE TABLE teams (
    id                  SERIAL PRIMARY KEY,
    match_id            UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    name                VARCHAR(128) NOT NULL,       -- e.g. "Team A", "Brazil"
    side                team_side NOT NULL DEFAULT 'unknown',
    attack_direction    SMALLINT CHECK (attack_direction IN (-1, 1)), -- +x or -x
    possession_pct      NUMERIC(5, 2),
    possession_time_sec NUMERIC(10, 2),
    goals_scored        INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (match_id, name),
    UNIQUE (match_id, side) -- at most one row per side enum per match
);

COMMENT ON TABLE teams IS
    'Teams participating in a match. Populated after team classification.';

-- ---------------------------------------------------------------------------
-- Players (ByteTrack identities scoped to a single match)
-- ---------------------------------------------------------------------------
CREATE TABLE players (
    id                  SERIAL PRIMARY KEY,
    match_id            UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    team_id             INTEGER REFERENCES teams(id) ON DELETE SET NULL,
    track_id            INTEGER NOT NULL,            -- detector track id
    class_name          entity_class NOT NULL DEFAULT 'player',
    display_name        VARCHAR(128),
    jersey_number       SMALLINT,
    -- Aggregated stats (denormalized for fast dashboard queries)
    possession_time_sec NUMERIC(10, 2) DEFAULT 0,
    possession_pct      NUMERIC(5, 2) DEFAULT 0,
    touches             INTEGER DEFAULT 0,
    total_distance_m    NUMERIC(10, 2) DEFAULT 0,
    avg_speed_kmh       NUMERIC(6, 2) DEFAULT 0,
    max_speed_kmh       NUMERIC(6, 2) DEFAULT 0,
    sprint_count        INTEGER DEFAULT 0,
    passes_made         INTEGER DEFAULT 0,
    passes_completed    INTEGER DEFAULT 0,
    shots               INTEGER DEFAULT 0,
    goals               INTEGER DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (match_id, track_id)
);

COMMENT ON TABLE players IS
    'Tracked individuals in a match. track_id is the ByteTrack id within the video.';

-- ---------------------------------------------------------------------------
-- Tracking (per-frame positions on the pitch plane, centimetres)
-- ---------------------------------------------------------------------------
CREATE TABLE tracking (
    id                  BIGSERIAL PRIMARY KEY,
    match_id            UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    player_id           INTEGER REFERENCES players(id) ON DELETE CASCADE,
    frame               INTEGER NOT NULL CHECK (frame >= 0),
    timestamp_sec       NUMERIC(10, 3) NOT NULL,
    entity_class        entity_class NOT NULL,
    track_id            INTEGER,                     -- nullable for anonymous ball frames
    x_cm                NUMERIC(10, 2) NOT NULL,
    y_cm                NUMERIC(10, 2) NOT NULL,
    confidence          NUMERIC(5, 4),
    homography_valid    BOOLEAN NOT NULL DEFAULT TRUE,
    UNIQUE (match_id, frame, entity_class, COALESCE(track_id, -1))
);

COMMENT ON TABLE tracking IS
    'Bird''s-eye pitch coordinates per frame. High volume — consider partitioning by match_id.';

-- ---------------------------------------------------------------------------
-- Possession (timeline segments)
-- ---------------------------------------------------------------------------
CREATE TABLE possession (
    id                  BIGSERIAL PRIMARY KEY,
    match_id            UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    team_id             INTEGER REFERENCES teams(id) ON DELETE SET NULL,
    player_id           INTEGER REFERENCES players(id) ON DELETE SET NULL,
    state               possession_state NOT NULL,
    start_frame         INTEGER NOT NULL,
    end_frame           INTEGER NOT NULL,
    start_time_sec      NUMERIC(10, 3) NOT NULL,
    end_time_sec        NUMERIC(10, 3) NOT NULL,
    duration_sec        NUMERIC(10, 3) GENERATED ALWAYS AS (end_time_sec - start_time_sec) STORED,
    CHECK (end_frame >= start_frame),
    CHECK (end_time_sec >= start_time_sec)
);

COMMENT ON TABLE possession IS
    'Contiguous possession segments derived from nearest-player ball proximity.';

-- ---------------------------------------------------------------------------
-- Passes
-- ---------------------------------------------------------------------------
CREATE TABLE passes (
    id                  BIGSERIAL PRIMARY KEY,
    match_id            UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    team_id             INTEGER REFERENCES teams(id) ON DELETE SET NULL,
    from_player_id      INTEGER REFERENCES players(id) ON DELETE SET NULL,
    to_player_id        INTEGER REFERENCES players(id) ON DELETE SET NULL,
    successful          BOOLEAN NOT NULL,
    primary_type        pass_type NOT NULL,
    types               pass_type[] NOT NULL DEFAULT '{}',
    distance_m          NUMERIC(8, 2) NOT NULL,
    forward_m           NUMERIC(8, 2),
    start_frame         INTEGER NOT NULL,
    end_frame           INTEGER NOT NULL,
    start_time_sec      NUMERIC(10, 3) NOT NULL,
    end_time_sec        NUMERIC(10, 3) NOT NULL,
    start_x_cm          NUMERIC(10, 2) NOT NULL,
    start_y_cm          NUMERIC(10, 2) NOT NULL,
    end_x_cm            NUMERIC(10, 2),
    end_y_cm            NUMERIC(10, 2),
    failure_reason      VARCHAR(32)                    -- intercepted | misplaced
);

COMMENT ON TABLE passes IS
    'Detected pass events between teammates from possession transitions.';

-- ---------------------------------------------------------------------------
-- Shots
-- ---------------------------------------------------------------------------
CREATE TABLE shots (
    id                  BIGSERIAL PRIMARY KEY,
    match_id            UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    team_id             INTEGER REFERENCES teams(id) ON DELETE SET NULL,
    shooter_player_id   INTEGER REFERENCES players(id) ON DELETE SET NULL,
    outcome             shot_outcome NOT NULL,
    target_goal         goal_line_side NOT NULL,
    speed_ms            NUMERIC(8, 2),
    speed_kmh           NUMERIC(8, 2),
    location_x_cm       NUMERIC(10, 2) NOT NULL,
    location_y_cm       NUMERIC(10, 2) NOT NULL,
    distance_to_goal_m  NUMERIC(8, 2),
    angle_deg           NUMERIC(6, 2),
    projected_goal_y_cm NUMERIC(10, 2),
    start_frame         INTEGER NOT NULL,
    end_frame           INTEGER NOT NULL,
    start_time_sec      NUMERIC(10, 3) NOT NULL,
    end_time_sec        NUMERIC(10, 3) NOT NULL
);

COMMENT ON TABLE shots IS
    'Shot attempts inferred from fast ball movement toward goal.';

-- ---------------------------------------------------------------------------
-- Goals
-- ---------------------------------------------------------------------------
CREATE TABLE goals (
    id                  BIGSERIAL PRIMARY KEY,
    match_id            UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    scoring_team_id     INTEGER REFERENCES teams(id) ON DELETE SET NULL,
    conceded_team_id    INTEGER REFERENCES teams(id) ON DELETE SET NULL,
    scorer_player_id    INTEGER REFERENCES players(id) ON DELETE SET NULL,
    goal_line           goal_line_side NOT NULL,
    cross_y_cm          NUMERIC(10, 2),
    frame               INTEGER NOT NULL,
    timestamp_sec       NUMERIC(10, 3) NOT NULL,
    speed_ms            NUMERIC(8, 2),
    score_team_a        INTEGER NOT NULL,
    score_team_b        INTEGER NOT NULL,
    notification_text   TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE goals IS
    'Goals detected when the ball crosses the goal line inside the goal mouth.';

-- ---------------------------------------------------------------------------
-- Events (corners, penalties, fouls, cards)
-- ---------------------------------------------------------------------------
CREATE TABLE events (
    id                  BIGSERIAL PRIMARY KEY,
    match_id            UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    team_id             INTEGER REFERENCES teams(id) ON DELETE SET NULL,
    player_id           INTEGER REFERENCES players(id) ON DELETE SET NULL,
    event_type          match_event_type NOT NULL,
    frame               INTEGER NOT NULL,
    timestamp_sec       NUMERIC(10, 3) NOT NULL,
    x_cm                NUMERIC(10, 2),
    y_cm                NUMERIC(10, 2),
    detail              TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE events IS
    'Set-piece and disciplinary events (corners/penalties geometric; fouls/cards heuristic).';

-- ---------------------------------------------------------------------------
-- Heatmaps
-- ---------------------------------------------------------------------------
CREATE TABLE heatmaps (
    id                  SERIAL PRIMARY KEY,
    match_id            UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    player_id           INTEGER REFERENCES players(id) ON DELETE CASCADE,
    layer               heatmap_layer NOT NULL,
    layer_key           VARCHAR(64) NOT NULL,        -- e.g. team_a, player_7, ball
    grid_rows           INTEGER,
    grid_cols           INTEGER,
    grid_data           JSONB,                       -- numeric density grid
    png_path            TEXT,
    frames_sampled      INTEGER,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (match_id, layer_key)
);

COMMENT ON TABLE heatmaps IS
    'Positional density grids and PNG paths per layer (team, player, ball, zone).';

-- ---------------------------------------------------------------------------
-- Highlights (clip windows merged into highlights.mp4)
-- ---------------------------------------------------------------------------
CREATE TABLE highlights (
    id                  SERIAL PRIMARY KEY,
    match_id            UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    clip_index          INTEGER NOT NULL,
    start_frame         INTEGER NOT NULL,
    end_frame           INTEGER NOT NULL,
    start_time_sec      NUMERIC(10, 3) NOT NULL,
    end_time_sec        NUMERIC(10, 3) NOT NULL,
    duration_sec        NUMERIC(10, 3) GENERATED ALWAYS AS (end_time_sec - start_time_sec) STORED,
    label               VARCHAR(256) NOT NULL,
    priority            INTEGER NOT NULL DEFAULT 0,
    source_event_types  TEXT[] NOT NULL DEFAULT '{}', -- goal, shot, corner, etc.
    merged_video_path   TEXT,                        -- path to full highlights.mp4 (per match)
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (match_id, clip_index),
    CHECK (end_frame >= start_frame)
);

COMMENT ON TABLE highlights IS
    'Highlight clip windows extracted around detected events.';

-- ---------------------------------------------------------------------------
-- Artifacts (file manifest — videos, JSON exports, reports)
-- ---------------------------------------------------------------------------
CREATE TABLE artifacts (
    id                  SERIAL PRIMARY KEY,
    match_id            UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    kind                artifact_kind NOT NULL,
    file_path           TEXT NOT NULL,
    file_size_bytes     BIGINT,
    mime_type           VARCHAR(128),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (match_id, kind, file_path)
);

COMMENT ON TABLE artifacts IS
    'On-disk outputs produced by the analysis pipeline.';

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

-- Matches
CREATE INDEX idx_matches_status ON matches (status);
CREATE INDEX idx_matches_created_at ON matches (created_at DESC);
CREATE INDEX idx_matches_status_created ON matches (status, created_at DESC);

-- Teams
CREATE INDEX idx_teams_match_id ON teams (match_id);

-- Players
CREATE INDEX idx_players_match_id ON players (match_id);
CREATE INDEX idx_players_team_id ON players (team_id);
CREATE INDEX idx_players_match_track ON players (match_id, track_id);

-- Tracking (high volume)
CREATE INDEX idx_tracking_match_frame ON tracking (match_id, frame);
CREATE INDEX idx_tracking_match_player_frame ON tracking (match_id, player_id, frame)
    WHERE player_id IS NOT NULL;
CREATE INDEX idx_tracking_match_time ON tracking (match_id, timestamp_sec);
CREATE INDEX idx_tracking_ball ON tracking (match_id, frame)
    WHERE entity_class = 'ball';

-- Possession
CREATE INDEX idx_possession_match_id ON possession (match_id);
CREATE INDEX idx_possession_match_time ON possession (match_id, start_time_sec);
CREATE INDEX idx_possession_player_id ON possession (player_id) WHERE player_id IS NOT NULL;

-- Passes
CREATE INDEX idx_passes_match_id ON passes (match_id);
CREATE INDEX idx_passes_match_time ON passes (match_id, start_time_sec);
CREATE INDEX idx_passes_from_player ON passes (from_player_id);
CREATE INDEX idx_passes_to_player ON passes (to_player_id);
CREATE INDEX idx_passes_team_success ON passes (match_id, team_id, successful);

-- Shots
CREATE INDEX idx_shots_match_id ON shots (match_id);
CREATE INDEX idx_shots_match_time ON shots (match_id, start_time_sec);
CREATE INDEX idx_shots_outcome ON shots (match_id, outcome);
CREATE INDEX idx_shots_shooter ON shots (shooter_player_id);

-- Goals
CREATE INDEX idx_goals_match_id ON goals (match_id);
CREATE INDEX idx_goals_match_time ON goals (match_id, timestamp_sec);
CREATE INDEX idx_goals_scorer ON goals (scorer_player_id);

-- Events
CREATE INDEX idx_events_match_id ON events (match_id);
CREATE INDEX idx_events_match_type ON events (match_id, event_type);
CREATE INDEX idx_events_match_time ON events (match_id, timestamp_sec);

-- Heatmaps
CREATE INDEX idx_heatmaps_match_id ON heatmaps (match_id);
CREATE INDEX idx_heatmaps_layer ON heatmaps (match_id, layer);

-- Highlights
CREATE INDEX idx_highlights_match_id ON highlights (match_id);
CREATE INDEX idx_highlights_match_time ON highlights (match_id, start_time_sec);

-- Artifacts
CREATE INDEX idx_artifacts_match_id ON artifacts (match_id);
CREATE INDEX idx_artifacts_kind ON artifacts (match_id, kind);

-- JSONB (summary, grid_data, processing_options)
CREATE INDEX idx_matches_summary_gin ON matches USING GIN (summary jsonb_path_ops);
CREATE INDEX idx_heatmaps_grid_gin ON heatmaps USING GIN (grid_data jsonb_path_ops)
    WHERE grid_data IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Triggers: auto-update updated_at on matches
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_matches_updated_at
    BEFORE UPDATE ON matches
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- Views (convenience for API / reporting)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_match_overview AS
SELECT
    m.id,
    m.title,
    m.original_filename,
    m.status,
    m.fps,
    m.duration_sec,
    m.progress,
    m.summary,
    m.created_at,
    m.finished_at,
    ta.name AS team_a_name,
    tb.name AS team_b_name,
    ta.possession_pct AS team_a_possession_pct,
    tb.possession_pct AS team_b_possession_pct,
    ta.goals_scored AS team_a_goals,
    tb.goals_scored AS team_b_goals,
    (SELECT COUNT(*) FROM players p WHERE p.match_id = m.id) AS player_count,
    (SELECT COUNT(*) FROM passes ps WHERE ps.match_id = m.id) AS pass_count,
    (SELECT COUNT(*) FROM shots sh WHERE sh.match_id = m.id) AS shot_count,
    (SELECT COUNT(*) FROM goals g WHERE g.match_id = m.id) AS goal_count
FROM matches m
LEFT JOIN teams ta ON ta.match_id = m.id AND ta.side = 'team_a'
LEFT JOIN teams tb ON tb.match_id = m.id AND tb.side = 'team_b';

COMMENT ON VIEW v_match_overview IS
    'Single-row match summary for dashboards and list endpoints.';

COMMIT;
