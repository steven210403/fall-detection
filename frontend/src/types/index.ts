/**
 * Domain types mirroring the backend's `schemas/api.py`, field for field.
 */

/** States of the per-person fall state machine (THESIS_PLAN §5). */
export type FallState = "unknown" | "normal" | "falling" | "impact" | "fallen";

export const FALL_STATES: readonly FallState[] = [
  "unknown",
  "normal",
  "falling",
  "impact",
  "fallen",
];

/** The three fall parameters for one person on one frame; `null` means not measurable. */
export interface FallFeatures {
  aspect_ratio: number | null;
  torso_angle_deg: number | null;
  torso_side: "both" | "partial" | null;
  hip_center: [number, number] | null;
  body_length_px: number | null;
  visible_keypoints: number;
  vy_px_s: number | null;
  vy_norm: number | null;
  height_ratio?: number | null;
  time_source: "source" | "monotonic";
}

export interface FallStateSnapshot {
  state: FallState;
  state_duration_s: number;
  long_lie_remaining_s: number | null;
  event_id: string | null;
  frame_seq: number;
  features: FallFeatures;
}

export interface TrackSnapshot {
  track_uuid: string;
  display_id: number;
  state: string;
  bbox: [number, number, number, number] | null;
  age_s: number;
  time_since_last_observation_s: number;
  stale: boolean;
  confidence: number | null;
  fall: FallStateSnapshot | null;
}

export interface PreviewFrame {
  frame_seq: number;
  sequence: number;
  width: number;
  height: number;
  source_width: number;
  source_height: number;
  jpeg_bytes: number;
  source_timestamp: number | null;
  captured_monotonic: number;
  captured_utc: number;
}

export interface LatencyStage {
  mean_ms: number;
  p95_ms: number;
}

export interface PerformanceSnapshot {
  fps?: number | null;
  window_s?: number;
  frames?: number;
  latency?: Record<string, LatencyStage>;
}

/** One stored fall event. `*_utc` are observation times; `active` while the alarm is open. */
export interface FallEvent {
  event_id: string;
  track_uuid: string;
  camera_id: string;
  stream_session_id: string;
  display_id: number | null;
  falling_utc: number;
  impact_utc: number;
  alert_utc: number;
  ended_utc: number | null;
  end_reason: string | null;
  active: boolean;
  response_time_s: number;
  peak_vy: number | null;
  min_torso_angle_deg: number | null;
  max_aspect_ratio: number | null;
  alert_source_timestamp: number | null;
  acknowledged_utc: number | null;
  created_utc?: number | null;
  schema_version?: number;
}

/** What the live feed sends when an alarm opens, closes, or is acknowledged. */
export interface FallEventMessage {
  kind: "opened" | "closed" | "acknowledged";
  persisted: boolean;
  event: Partial<FallEvent> & { event_id: string };
}

export interface FallEventPage {
  events: FallEvent[];
  next_cursor: string | null;
  limit: number;
  total: number;
}

export interface LiveEnvelope {
  schema_version: number;
  kind: "hello" | "snapshot" | "event" | "status";
  sequence: number;
  emitted_monotonic: number;
  emitted_utc: number;
  camera_id?: string | null;
  stream_session_id?: string | null;
  frame_seq?: number | null;
  source_timestamp?: number | null;
  receive_monotonic?: number | null;
  receive_utc?: number | null;
  preview?: PreviewFrame | null;
  tracks?: TrackSnapshot[];
  active_alarms?: number | null;
  metrics?: PerformanceSnapshot & Record<string, unknown>;
  event?: FallEventMessage | null;
  status?: StatusPayload | null;
  reason?: string;
  state?: string;
  latest_cursor?: string | null;
}

export interface Camera {
  camera_id: string;
  name: string;
  source_type: string;
  source_ref: string;
  state: string;
  stream_session_id: string | null;
  attempt: number | null;
  stream: Record<string, unknown> | null;
  source: Record<string, unknown> | null;
}

export interface FallThresholds {
  ar_threshold?: number;
  angle_lying_deg?: number;
  angle_upright_deg?: number;
  vy_threshold?: number;
  impact_window_s?: number;
  long_lie_s?: number;
  require_both?: boolean;
  recovery_hold_s?: number;
  calibrated?: boolean;
  calibration_path?: string | null;
}

export interface StatusPayload {
  schema_version: number;
  camera: {
    camera_id: string;
    state: string;
    name: string;
    source_type: string;
    stream_session_id: string | null;
    attempt: number | null;
  };
  pipeline: Record<string, unknown>;
  detection: Record<string, unknown>;
  tracking: Record<string, unknown>;
  fall: {
    enabled?: boolean;
    people?: number;
    states?: Partial<Record<FallState, number>>;
    active_alarms?: number;
    stats?: Record<string, number>;
    thresholds?: FallThresholds;
  };
  performance: PerformanceSnapshot;
  events: { persistent?: boolean; stats?: Record<string, number> };
  streaming: {
    preview?: {
      enabled?: boolean;
      preview_fps?: number;
      stats?: Record<string, number>;
    };
    live?: {
      clients?: { client_id: string; delivered: number; dropped: number }[];
      stats?: Record<string, number>;
    };
  };
  resources: {
    rss_bytes?: number | null;
    threads?: number;
    load_average?: number[] | null;
    device?: string;
    note?: string;
    [key: string]: unknown;
  };
}

export interface ApiErrorBody {
  code: string;
  message: string;
  detail?: string | null;
}

export interface ApiErrorResponse {
  error: ApiErrorBody;
}
