export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed'

export interface JobSummary {
  id: string
  original_filename: string
  status: JobStatus
  progress: number
  created_at?: string
  updated_at?: string
}

export interface JobDetail extends JobSummary {
  options: Record<string, unknown>
  summary: AnalyticsSummary
  artifacts: string[]
  error?: string | null
  started_at?: string | null
  finished_at?: string | null
}

export interface JobListResponse {
  total: number
  limit: number
  offset: number
  items: JobSummary[]
}

export interface UploadResponse {
  id: string
  status: JobStatus
  message: string
}

export interface AnalyticsSummary {
  possession?: { team_a_pct?: number; team_b_pct?: number; loose_pct?: number }
  passes?: { total?: number; successful?: number; success_rate_pct?: number }
  shots?: { total?: number; on_target?: number; accuracy_pct?: number }
  goals?: { total?: number; final_score?: Record<string, number> }
  highlights?: { event_count?: number; clip_count?: number }
  speed?: { players_tracked?: number; fastest_max_speed_kmh?: number }
}

export interface AnalyticsBundle {
  possession?: Record<string, unknown>
  passes?: Record<string, unknown>
  shots?: Record<string, unknown>
  goals?: Record<string, unknown>
  speed?: Record<string, unknown>
  heatmaps?: Record<string, unknown>
}

export interface VisualItem {
  id: string
  category: 'heatmap' | 'pass_network' | 'shot_map'
  name: string
  filename: string
}

export interface PlayerPossession {
  track_id: number
  class_name: string
  team: string | null
  possession_time_sec: number
  possession_pct: number
  touches: number
}

export interface PlayerSpeed {
  track_id: number
  class_name: string
  team: string | null
  total_distance_m: number
  avg_speed_kmh: number
  max_speed_kmh: number
  sprint_count: number
}

export interface GoalEvent {
  goal_id: number
  scoring_team: string
  scorer_track_id: number | null
  timestamp_sec: number
  scoreline: string
  goal_line: string
}

export interface PassEvent {
  pass_id: number
  team: string
  from_track_id: number
  to_track_id: number | null
  successful: boolean
  primary_type: string
  distance_m: number
  start_x_cm?: number
  start_y_cm?: number
  end_x_cm?: number
  end_y_cm?: number
}

export interface ShotEvent {
  shot_id: number
  team: string
  outcome: string
  speed_kmh: number
  distance_to_goal_m: number
  location_x_cm: number
  location_y_cm: number
}
