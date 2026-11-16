import type { FallEvent, TrackSnapshot } from "../src/types";

export function fallEvent(overrides: Partial<FallEvent> = {}): FallEvent {
  return {
    event_id: "e1",
    track_uuid: "t1",
    camera_id: "c1",
    stream_session_id: "s1",
    display_id: 3,
    falling_utc: 1_700_000_000,
    impact_utc: 1_700_000_000.5,
    alert_utc: 1_700_000_003.5,
    ended_utc: null,
    end_reason: null,
    active: true,
    response_time_s: 3,
    peak_vy: 2.1,
    min_torso_angle_deg: 8,
    max_aspect_ratio: 2.4,
    alert_source_timestamp: null,
    acknowledged_utc: null,
    ...overrides,
  };
}

export function track(overrides: Partial<TrackSnapshot> = {}): TrackSnapshot {
  return {
    track_uuid: "t1",
    display_id: 3,
    state: "active",
    bbox: [0, 0, 10, 10],
    age_s: 5,
    time_since_last_observation_s: 0,
    stale: false,
    confidence: 0.9,
    fall: {
      state: "normal",
      state_duration_s: 4.2,
      long_lie_remaining_s: null,
      event_id: null,
      frame_seq: 10,
      features: {
        aspect_ratio: 0.31,
        torso_angle_deg: 84.4,
        torso_side: "both",
        hip_center: [5, 5],
        body_length_px: 170,
        visible_keypoints: 17,
        vy_px_s: 4,
        vy_norm: 0.02,
        time_source: "source",
      },
    },
    ...overrides,
  };
}
