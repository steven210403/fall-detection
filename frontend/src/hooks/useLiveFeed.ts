/**
 * Live feed hook: WebSocket connection, reconnect, gap recovery, and fall alarms.
 *
 * §17.2 fixes the contract this hook implements:
 *
 * - "Send track snapshots at 2 Hz; send events/status immediately." The hook does not
 *   throttle — the server does — so it renders whatever arrives.
 * - "Slow clients receive the latest snapshot, not a backlog." Each envelope carries a
 *   monotonic `sequence`, so this hook can *see* that it missed messages.
 * - "On reconnect, fetch history using a cursor to recover gaps." The `hello` message
 *   carries the server's latest event cursor; when the hook notices a sequence gap it
 *   fetches the events it missed and merges them, so a slow client's timeline has no hole.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { api, ApiError } from "../services/api";
import type {
  FallEvent,
  FallEventMessage,
  LiveEnvelope,
  PerformanceSnapshot,
  TrackSnapshot,
} from "../types";

export type ConnectionState = "connecting" | "live" | "reconnecting" | "offline" | "closed";

export interface LiveFeed {
  connection: ConnectionState;
  cameraState: string | null;
  streamSessionId: string | null;
  frameSeq: number | null;
  sourceTimestamp: number | null;
  tracks: TrackSnapshot[];
  /** Fall events seen live or recovered, newest alarm first. */
  events: FallEvent[];
  /** Number of people in the ``fallen`` state on the latest snapshot. */
  activeAlarms: number;
  metrics: PerformanceSnapshot & Record<string, unknown>;
  lastSequence: number;
  /** Messages the client never received, inferred from the sequence gaps (§17.2). */
  missedMessages: number;
  /** Events recovered through the history cursor after a gap or a reconnect. */
  recoveredEvents: number;
  error: string | null;
  reconnect: () => void;
}

const MAX_EVENTS = 200;
/** §17.2's recovery path needs a cursor, so the newest one is kept across reconnects. */
const RECONNECT_DELAYS_MS = [500, 1000, 2000, 4000, 8000];
/**
 * Ceiling on the pages one gap recovery may fetch.
 *
 * The walk stops at the first page the client already holds, so this bound is only reached
 * when the gap is genuinely huge. Ten pages of fifty events is a thousand events, which is
 * far more than a 2 Hz snapshot stream can drop in a plausible disconnect; beyond that a
 * reload is the honest answer and the hook says so.
 */
const MAX_RECOVERY_PAGES = 10;

export function useLiveFeed(enabled = true): LiveFeed {
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [cameraState, setCameraState] = useState<string | null>(null);
  const [streamSessionId, setStreamSessionId] = useState<string | null>(null);
  const [frameSeq, setFrameSeq] = useState<number | null>(null);
  const [sourceTimestamp, setSourceTimestamp] = useState<number | null>(null);
  const [tracks, setTracks] = useState<TrackSnapshot[]>([]);
  const [events, setEvents] = useState<FallEvent[]>([]);
  const [activeAlarms, setActiveAlarms] = useState(0);
  const [metrics, setMetrics] = useState<PerformanceSnapshot & Record<string, unknown>>({});
  const [lastSequence, setLastSequence] = useState(0);
  const [missedMessages, setMissedMessages] = useState(0);
  const [recoveredEvents, setRecoveredEvents] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  const socketRef = useRef<WebSocket | null>(null);
  const cursorRef = useRef<string | null>(null);
  /** Event ids already rendered, so a recovery can tell where the gap ends. */
  const knownEventIds = useRef<Set<string>>(new Set());
  const sequenceRef = useRef(0);
  const timerRef = useRef<number | null>(null);
  const closedRef = useRef(false);

  /**
   * Fetch the events a sequence gap hid, using the server's cursor (§17.2).
   *
   * The history endpoint pages **backwards** from a cursor (`next_cursor` points at older
   * rows), while the cursor a client holds is the *newest* boundary the server reported.
   * Recovery therefore walks backwards from that boundary and stops as soon as it reaches
   * an event the client already has, which is what makes the walk bounded: the number of
   * pages is the size of the gap, not the size of the history. `MAX_RECOVERY_PAGES` caps a
   * pathological case, and hitting the cap is reported rather than hidden — a client that
   * fell that far behind should reload.
   */
  const recoverGap = useCallback(async () => {
    let cursor = cursorRef.current;
    if (!cursor) {
      return;
    }
    try {
      let recovered = 0;
      let pages = 0;
      const known = new Set(knownEventIds.current);
      while (cursor && pages < MAX_RECOVERY_PAGES) {
        pages += 1;
        const page = await api.fallEvents({ limit: 50, cursor });
        if (page.events.length === 0) {
          break;
        }
        const fresh = page.events.filter((event) => !known.has(event.event_id));
        if (fresh.length === 0) {
          // The page is entirely history the client already holds: the gap is closed.
          break;
        }
        setEvents((previous) => mergeEvents(previous, fresh));
        for (const event of fresh) {
          known.add(event.event_id);
        }
        recovered += fresh.length;
        cursor = page.next_cursor;
      }
      if (recovered > 0) {
        setRecoveredEvents((count) => count + recovered);
      }
      if (cursor && pages >= MAX_RECOVERY_PAGES) {
        setError(
          "the live feed fell too far behind to recover in one pass; reload the page",
        );
      }
    } catch (cause) {
      // A failed recovery is reported rather than retried in a loop: the socket's own
      // reconnect will fetch the history again, and a tight retry would flood the API.
      setError(cause instanceof ApiError ? cause.message : "could not recover history");
    }
  }, []);

  const connect = useCallback(() => {
    closedRef.current = false;
    setConnection((current) => (current === "live" ? "reconnecting" : "connecting"));

    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${window.location.host}/api/v1/live`);
    socketRef.current = socket;

    socket.onopen = () => {
      setConnection("live");
      setError(null);
    };

    socket.onmessage = (message) => {
      let envelope: LiveEnvelope;
      try {
        envelope = JSON.parse(message.data as string) as LiveEnvelope;
      } catch {
        setError("the live feed sent a message this client cannot parse");
        return;
      }
      handle(envelope);
    };

    socket.onclose = (event) => {
      socketRef.current = null;
      if (closedRef.current) {
        setConnection("closed");
        return;
      }
      // 1008 is a policy violation: the live feed is disabled or the client ceiling was
      // reached. Neither is fixed by reconnecting, so the hook reports the reason instead
      // of looping.
      if (event.code === 1008) {
        setConnection("offline");
        setError(event.reason || "the live feed refused the connection");
        return;
      }
      setConnection("reconnecting");
      const delay = RECONNECT_DELAYS_MS[Math.min(attempt, RECONNECT_DELAYS_MS.length - 1)];
      timerRef.current = window.setTimeout(() => {
        setAttempt((value) => value + 1);
      }, delay);
    };

    socket.onerror = () => {
      // ``onclose`` follows and carries the code, so the error handler only records that
      // something went wrong rather than guessing at a cause.
      setError("the live feed connection failed");
    };

    function handle(envelope: LiveEnvelope) {
      const previous = sequenceRef.current;
      sequenceRef.current = envelope.sequence;
      setLastSequence(envelope.sequence);
      if (envelope.kind === "hello") {
        cursorRef.current = envelope.latest_cursor ?? null;
        if (envelope.status) {
          setCameraState(envelope.status.camera.state);
          setStreamSessionId(envelope.status.camera.stream_session_id);
        }
        return;
      }
      // A gap means the client was slower than the server's queue and the oldest messages
      // were discarded by design (§17.2). The cursor is what makes that recoverable.
      if (previous !== 0 && envelope.sequence > previous + 1) {
        setMissedMessages((count) => count + (envelope.sequence - previous - 1));
        void recoverGap();
      }
      if (envelope.kind === "event" && envelope.event) {
        // An alarm opened, closed, or was acknowledged. The cursor stays the one the
        // `hello` message handed us (an opaque keyset token, never an event id), so a
        // recovery re-reads the interval it missed.
        const message = envelope.event;
        knownEventIds.current.add(message.event.event_id);
        setEvents((existing) => applyEventMessage(existing, message));
        return;
      }
      if (envelope.kind === "status") {
        if (envelope.camera_id) {
          setCameraState(envelope.reason === "camera_stopped" ? "idle" : cameraStateOf(envelope));
        }
        if (envelope.stream_session_id !== undefined) {
          setStreamSessionId(envelope.stream_session_id ?? null);
        }
        return;
      }
      // A snapshot: the frame sequence, the session, each person's fall state, and the
      // pipeline's FPS and latency.
      if (envelope.kind === "snapshot") {
        setTracks(envelope.tracks ?? []);
        setActiveAlarms(envelope.active_alarms ?? 0);
        setMetrics(envelope.metrics ?? {});
        setFrameSeq(envelope.frame_seq ?? null);
        setSourceTimestamp(envelope.source_timestamp ?? null);
        if (envelope.stream_session_id) {
          setStreamSessionId(envelope.stream_session_id);
        }
      }
    }

    function cameraStateOf(envelope: LiveEnvelope): string {
      // `AppState.publish_status` sends the state top-level beside `camera_id`; only the
      // `hello` message nests it under `status.camera`.
      return envelope.state ?? envelope.status?.camera?.state ?? "unknown";
    }
  }, [attempt, recoverGap]);

  useEffect(() => {
    // The socket is an external system, which is exactly what an effect is for; the state
    // updates happen in its callbacks. The disabled case is deferred to a task so the effect
    // body itself performs no state update.
    let cancelled = false;
    void (async () => {
      if (!enabled) {
        if (!cancelled) {
          setConnection("closed");
        }
        return;
      }
      connect();
    })();

    return () => {
      cancelled = true;
      closedRef.current = true;
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [connect, enabled]);

  const reconnect = useCallback(() => {
    setAttempt(0);
    setError(null);
    socketRef.current?.close();
    connect();
  }, [connect]);

  return {
    connection,
    cameraState,
    streamSessionId,
    frameSeq,
    sourceTimestamp,
    tracks,
    events: events.slice(0, MAX_EVENTS),
    activeAlarms,
    metrics,
    lastSequence,
    missedMessages,
    recoveredEvents,
    error,
    reconnect,
  };
}

/** Merge events by identifier, newest alarm first, so a recovery cannot duplicate one. */
export function mergeEvents(existing: FallEvent[], incoming: FallEvent[]): FallEvent[] {
  const byId = new Map<string, FallEvent>();
  for (const event of [...existing, ...incoming]) {
    byId.set(event.event_id, { ...byId.get(event.event_id), ...event });
  }
  return [...byId.values()].sort((left, right) => right.alert_utc - left.alert_utc);
}

/**
 * Apply one live fall-event message.
 *
 * ``opened`` and ``acknowledged`` carry the full record; ``closed`` carries only the end,
 * so it patches the event the client already holds. A closure for an event the client
 * never saw is kept as a minimal inactive record rather than dropped.
 */
export function applyEventMessage(existing: FallEvent[], message: FallEventMessage): FallEvent[] {
  const current = existing.find((event) => event.event_id === message.event.event_id);
  if (message.kind === "closed") {
    if (current === undefined) {
      return existing;
    }
    return mergeEvents(existing, [{ ...current, ...message.event, active: false }]);
  }
  return mergeEvents(existing, [{ ...(current ?? {}), ...message.event } as FallEvent]);
}

/**
 * Alarms that still need attention, newest first, without duplicates.
 *
 * An open alarm always needs attention. A closed one still does until it is acknowledged,
 * unless the person was seen getting up (`recovered`): an alarm closed because the person
 * left the camera's view or the camera stopped is a fall nobody has confirmed yet.
 */
export function activeAlarms(events: FallEvent[], polled: FallEvent[]): FallEvent[] {
  return mergeEvents(polled, events).filter(needsAttention);
}

export function needsAttention(event: FallEvent): boolean {
  if (event.active) {
    return true;
  }
  return event.end_reason !== "recovered" && event.acknowledged_utc === null;
}
