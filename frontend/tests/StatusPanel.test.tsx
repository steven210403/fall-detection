import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StatusPanel } from "../src/components/StatusPanel";
import type { StatusPayload } from "../src/types";

function status(overrides: Partial<StatusPayload> = {}): StatusPayload {
  return {
    schema_version: 1,
    camera: {
      camera_id: "c1",
      state: "live",
      name: "room.mp4",
      source_type: "mp4",
      stream_session_id: "s1",
      attempt: 1,
    },
    pipeline: { stats: { frames_read: 1200, frames_dropped: 3 } },
    detection: { enabled: true, ready: true, model: { model_name: "yolov8n-pose.pt" } },
    tracking: { enabled: true, tracker: "bytetrack" },
    fall: {
      enabled: true,
      active_alarms: 1,
      stats: { handovers: 2 },
      thresholds: {
        ar_threshold: 1,
        angle_lying_deg: 30,
        vy_threshold: 1,
        long_lie_s: 3,
        calibrated: false,
      },
    },
    performance: {
      fps: 24.6,
      latency: {
        total_ms: { mean_ms: 38.2, p95_ms: 45.1 },
        detection_ms: { mean_ms: 30.0, p95_ms: 36.0 },
        tracking_ms: { mean_ms: 2.0, p95_ms: 3.0 },
        fall_ms: { mean_ms: 0.4, p95_ms: 0.6 },
      },
    },
    events: { persistent: true, stats: { opened: 3 } },
    streaming: { live: { stats: { dropped: 2, clients_connected: 1 } } },
    resources: { rss_bytes: 314572800, threads: 12, device: "mps" },
    ...overrides,
  };
}

const base = {
  connection: "live",
  missedMessages: 0,
  recoveredEvents: 0,
  lastSequence: 42,
  error: null,
  loading: false,
};

describe("StatusPanel", () => {
  it("shows a loading state before the first payload", () => {
    render(<StatusPanel {...base} status={null} loading />);
    expect(screen.getByRole("status")).toHaveTextContent("Membaca status");
  });

  it("shows FPS and the latency of every stage", () => {
    render(<StatusPanel {...base} status={status()} />);
    expect(screen.getByText("24.6")).toBeInTheDocument();
    expect(screen.getByText("38.2 ms (p95 45.1)")).toBeInTheDocument();
    expect(screen.getByText("30.0 ms (p95 36.0)")).toBeInTheDocument();
    expect(screen.getByText("0.4 ms (p95 0.6)")).toBeInTheDocument();
  });

  it("names the model and says whether the thresholds are calibrated", () => {
    render(<StatusPanel {...base} status={status()} />);
    expect(screen.getByText("yolov8n-pose.pt")).toBeInTheDocument();
    expect(screen.getByText(/AR≥1 · θ<30° · Vy>1\/dtk · long-lie 3 dtk/)).toBeInTheDocument();
    expect(screen.getByText("nilai literatur")).toBeInTheDocument();
  });

  it("names the tracker when the payload sends an object", () => {
    render(<StatusPanel {...base} status={status({ tracking: { enabled: true, tracker: { name: "bytetrack" } } })} />);
    expect(screen.getByText("bytetrack")).toBeInTheDocument();
  });

  it("shows a missing FPS as a dash, not zero", () => {
    render(<StatusPanel {...base} status={status({ performance: {} })} />);
    const row = screen.getByText("FPS").closest("div");
    expect(row).toHaveTextContent("—");
  });

  it("reports the live-feed gap counters", () => {
    render(<StatusPanel {...base} missedMessages={5} recoveredEvents={5} status={status()} />);
    expect(screen.getByText(/5 terlewat · 5 dipulihkan/)).toBeInTheDocument();
  });

  it("surfaces an error", () => {
    render(<StatusPanel {...base} status={status()} error="could not read the status" />);
    expect(screen.getByRole("alert")).toHaveTextContent("could not read the status");
  });
});
