import { describe, expect, it } from "vitest";

import { activeAlarms, applyEventMessage, mergeEvents } from "../src/hooks/useLiveFeed";
import { fallEvent } from "./fixtures";

describe("fall-event merging", () => {
  it("orders by alarm time, newest first, without duplicates", () => {
    const older = fallEvent({ event_id: "a", alert_utc: 10 });
    const newer = fallEvent({ event_id: "b", alert_utc: 20 });
    const merged = mergeEvents([older], [newer, older]);
    expect(merged.map((event) => event.event_id)).toEqual(["b", "a"]);
  });

  it("patches an event when it closes", () => {
    const open = fallEvent();
    const closed = applyEventMessage([open], {
      kind: "closed",
      persisted: true,
      event: { event_id: "e1", ended_utc: 1_700_000_020, end_reason: "recovered" },
    });
    expect(closed[0].active).toBe(false);
    expect(closed[0].end_reason).toBe("recovered");
    expect(closed[0].display_id).toBe(3);
  });

  it("ignores a closure for an event the client never saw", () => {
    const result = applyEventMessage([], {
      kind: "closed",
      persisted: true,
      event: { event_id: "unknown", ended_utc: 1, end_reason: "stopped" },
    });
    expect(result).toEqual([]);
  });

  it("adds an opened alarm", () => {
    const result = applyEventMessage([], { kind: "opened", persisted: true, event: fallEvent() });
    expect(result).toHaveLength(1);
    expect(result[0].active).toBe(true);
  });

  it("keeps a polled alarm the socket missed, and drops one that ended in recovery", () => {
    const missed = fallEvent({ event_id: "missed" });
    const stale = fallEvent({ event_id: "stale" });
    const live = [fallEvent({ event_id: "stale", active: false, end_reason: "recovered" })];
    const alarms = activeAlarms(live, [missed, stale]);
    expect(alarms.map((event) => event.event_id)).toEqual(["missed"]);
  });

  it("keeps an unacknowledged alarm whose person left the view", () => {
    const lost = fallEvent({ event_id: "lost", active: false, end_reason: "track_lost" });
    expect(activeAlarms([lost], []).map((event) => event.event_id)).toEqual(["lost"]);
    const acknowledged = { ...lost, acknowledged_utc: 1_700_000_020 };
    expect(activeAlarms([acknowledged], [])).toEqual([]);
  });
});
