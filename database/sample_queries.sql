-- =============================================================================
-- Smart Football Match Analysis System — Sample Queries
-- =============================================================================
-- Illustrative SQL for dashboards, reports, and operational tasks.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. List recent completed matches with scoreline
-- ---------------------------------------------------------------------------
SELECT
    m.id,
    m.original_filename,
    m.finished_at,
    ta.name AS team_a,
    tb.name AS team_b,
    ta.goals_scored AS score_a,
    tb.goals_scored AS score_b,
    ta.possession_pct AS possession_a,
    tb.possession_pct AS possession_b
FROM matches m
JOIN teams ta ON ta.match_id = m.id AND ta.side = 'team_a'
JOIN teams tb ON tb.match_id = m.id AND tb.side = 'team_b'
WHERE m.status = 'completed'
ORDER BY m.finished_at DESC
LIMIT 20;


-- ---------------------------------------------------------------------------
-- 2. Match overview (use the convenience view)
-- ---------------------------------------------------------------------------
SELECT * FROM v_match_overview
WHERE id = '00000000-0000-0000-0000-000000000001';


-- ---------------------------------------------------------------------------
-- 3. Top players by distance covered in a match
-- ---------------------------------------------------------------------------
SELECT
    p.track_id,
    p.display_name,
    t.name AS team,
    p.total_distance_m,
    p.avg_speed_kmh,
    p.max_speed_kmh,
    p.sprint_count
FROM players p
JOIN teams t ON t.id = p.team_id
WHERE p.match_id = '00000000-0000-0000-0000-000000000001'
ORDER BY p.total_distance_m DESC
LIMIT 11;


-- ---------------------------------------------------------------------------
-- 4. Pass network: passer → receiver counts
-- ---------------------------------------------------------------------------
SELECT
    fp.track_id AS from_track,
    tp.track_id AS to_track,
    tm.name AS team,
    COUNT(*) AS pass_count,
    SUM(CASE WHEN ps.successful THEN 1 ELSE 0 END) AS completed,
    ROUND(100.0 * SUM(CASE WHEN ps.successful THEN 1 ELSE 0 END) / COUNT(*), 1) AS completion_pct
FROM passes ps
JOIN teams tm ON tm.id = ps.team_id
JOIN players fp ON fp.id = ps.from_player_id
JOIN players tp ON tp.id = ps.to_player_id
WHERE ps.match_id = '00000000-0000-0000-0000-000000000001'
GROUP BY fp.track_id, tp.track_id, tm.name
ORDER BY pass_count DESC;


-- ---------------------------------------------------------------------------
-- 5. Pass breakdown by type
-- ---------------------------------------------------------------------------
SELECT
    t.name AS team,
    ps.primary_type,
    COUNT(*) AS total,
    SUM(CASE WHEN ps.successful THEN 1 ELSE 0 END) AS successful
FROM passes ps
JOIN teams t ON t.id = ps.team_id
WHERE ps.match_id = '00000000-0000-0000-0000-000000000001'
GROUP BY t.name, ps.primary_type
ORDER BY t.name, total DESC;


-- ---------------------------------------------------------------------------
-- 6. Shot map data (for frontend SVG rendering)
-- ---------------------------------------------------------------------------
SELECT
    sh.id,
    t.name AS team,
    sh.outcome,
    sh.location_x_cm,
    sh.location_y_cm,
    sh.speed_kmh,
    sh.distance_to_goal_m,
    sh.start_time_sec
FROM shots sh
LEFT JOIN teams t ON t.id = sh.team_id
WHERE sh.match_id = '00000000-0000-0000-0000-000000000001'
ORDER BY sh.start_time_sec;


-- ---------------------------------------------------------------------------
-- 7. Goal timeline with scorer
-- ---------------------------------------------------------------------------
SELECT
    g.timestamp_sec,
    FLOOR(g.timestamp_sec / 60) AS minute,
    MOD(FLOOR(g.timestamp_sec)::INT, 60) AS second,
    st.name AS scoring_team,
    p.track_id AS scorer_track_id,
    g.score_team_a || '-' || g.score_team_b AS scoreline,
    g.goal_line
FROM goals g
JOIN teams st ON st.id = g.scoring_team_id
LEFT JOIN players p ON p.id = g.scorer_player_id
WHERE g.match_id = '00000000-0000-0000-0000-000000000001'
ORDER BY g.timestamp_sec;


-- ---------------------------------------------------------------------------
-- 8. Possession timeline (segments)
-- ---------------------------------------------------------------------------
SELECT
    pos.start_time_sec,
    pos.end_time_sec,
    pos.duration_sec,
    pos.state,
    t.name AS team,
    p.track_id AS holder_track_id
FROM possession pos
LEFT JOIN teams t ON t.id = pos.team_id
LEFT JOIN players p ON p.id = pos.player_id
WHERE pos.match_id = '00000000-0000-0000-0000-000000000001'
ORDER BY pos.start_time_sec;


-- ---------------------------------------------------------------------------
-- 9. Team possession % from segments (validation query)
-- ---------------------------------------------------------------------------
SELECT
    t.name,
    ROUND(SUM(pos.duration_sec)::NUMERIC, 1) AS possession_sec,
    ROUND(
        100.0 * SUM(pos.duration_sec) /
        NULLIF(SUM(SUM(pos.duration_sec)) OVER (), 0),
        1
    ) AS possession_pct
FROM possession pos
JOIN teams t ON t.id = pos.team_id
WHERE pos.match_id = '00000000-0000-0000-0000-000000000001'
  AND pos.state IN ('team_a', 'team_b')
GROUP BY t.name;


-- ---------------------------------------------------------------------------
-- 10. All match events chronologically (goals, shots, corners, cards…)
-- ---------------------------------------------------------------------------
SELECT event_time, event_type, label, team
FROM (
    SELECT g.timestamp_sec AS event_time, 'goal' AS event_type,
           'Goal — ' || st.name AS label, st.name AS team
    FROM goals g
    JOIN teams st ON st.id = g.scoring_team_id
    WHERE g.match_id = '00000000-0000-0000-0000-000000000001'

    UNION ALL

    SELECT sh.start_time_sec, 'shot',
           'Shot (' || sh.outcome::TEXT || ')', t.name
    FROM shots sh
    LEFT JOIN teams t ON t.id = sh.team_id
    WHERE sh.match_id = '00000000-0000-0000-0000-000000000001'

    UNION ALL

    SELECT e.timestamp_sec, e.event_type::TEXT,
           INITCAP(REPLACE(e.event_type::TEXT, '_', ' ')), t.name
    FROM events e
    LEFT JOIN teams t ON t.id = e.team_id
    WHERE e.match_id = '00000000-0000-0000-0000-000000000001'
) unified
ORDER BY event_time;


-- ---------------------------------------------------------------------------
-- 11. Highlight clips for a match
-- ---------------------------------------------------------------------------
SELECT
    clip_index,
    start_time_sec,
    end_time_sec,
    duration_sec,
    label,
    source_event_types,
    merged_video_path
FROM highlights
WHERE match_id = '00000000-0000-0000-0000-000000000001'
ORDER BY clip_index;


-- ---------------------------------------------------------------------------
-- 12. Available heatmap layers
-- ---------------------------------------------------------------------------
SELECT layer, layer_key, png_path, grid_rows, grid_cols
FROM heatmaps
WHERE match_id = '00000000-0000-0000-0000-000000000001'
ORDER BY layer, layer_key;


-- ---------------------------------------------------------------------------
-- 13. Player trajectory for a specific frame range (tracking)
-- ---------------------------------------------------------------------------
SELECT
    tr.frame,
    tr.timestamp_sec,
    tr.x_cm,
    tr.y_cm
FROM tracking tr
JOIN players p ON p.id = tr.player_id
WHERE tr.match_id = '00000000-0000-0000-0000-000000000001'
  AND p.track_id = 7
  AND tr.frame BETWEEN 1000 AND 1500
ORDER BY tr.frame;


-- ---------------------------------------------------------------------------
-- 14. Ball positions during a goal sequence
-- ---------------------------------------------------------------------------
SELECT tr.frame, tr.timestamp_sec, tr.x_cm, tr.y_cm
FROM tracking tr
JOIN goals g ON g.match_id = tr.match_id
WHERE tr.match_id = '00000000-0000-0000-0000-000000000001'
  AND tr.entity_class = 'ball'
  AND tr.frame BETWEEN g.frame - 50 AND g.frame + 10
ORDER BY tr.frame;


-- ---------------------------------------------------------------------------
-- 15. Processing queue (operational)
-- ---------------------------------------------------------------------------
SELECT id, original_filename, status, progress, created_at, started_at
FROM matches
WHERE status IN ('queued', 'processing')
ORDER BY created_at;


-- ---------------------------------------------------------------------------
-- 16. Artifact manifest for downloads API
-- ---------------------------------------------------------------------------
SELECT kind, file_path, file_size_bytes, mime_type
FROM artifacts
WHERE match_id = '00000000-0000-0000-0000-000000000001'
ORDER BY kind;


-- ---------------------------------------------------------------------------
-- 17. Leaderboard: most goals across all matches (by track_id proxy)
-- ---------------------------------------------------------------------------
SELECT
    p.track_id,
    p.display_name,
    t.name AS team,
    COUNT(g.id) AS goals_scored,
    COUNT(DISTINCT g.match_id) AS matches_played
FROM goals g
JOIN players p ON p.id = g.scorer_player_id
JOIN teams t ON t.id = p.team_id
GROUP BY p.track_id, p.display_name, t.name
ORDER BY goals_scored DESC
LIMIT 20;


-- ---------------------------------------------------------------------------
-- 18. Shot accuracy by team (aggregate)
-- ---------------------------------------------------------------------------
SELECT
    t.name AS team,
    COUNT(*) AS total_shots,
    SUM(CASE WHEN sh.outcome = 'on_target' THEN 1 ELSE 0 END) AS on_target,
    ROUND(
        100.0 * SUM(CASE WHEN sh.outcome = 'on_target' THEN 1 ELSE 0 END) / COUNT(*),
        1
    ) AS accuracy_pct
FROM shots sh
JOIN teams t ON t.id = sh.team_id
WHERE sh.match_id = '00000000-0000-0000-0000-000000000001'
GROUP BY t.name;


-- ---------------------------------------------------------------------------
-- 19. Insert pattern: new match from API upload
-- ---------------------------------------------------------------------------
/*
INSERT INTO matches (external_job_id, original_filename, source_video_path, status, processing_options)
VALUES ('abc123', 'world_cup_final.mp4', '/data/jobs/abc123/source.mp4', 'queued', '{"enable_heatmaps": true}'::jsonb)
RETURNING id;
*/

-- ---------------------------------------------------------------------------
-- 20. Bulk delete old failed jobs (retention policy)
-- ---------------------------------------------------------------------------
/*
DELETE FROM matches
WHERE status = 'failed'
  AND created_at < NOW() - INTERVAL '30 days';
-- Cascades to all child tables.
*/
