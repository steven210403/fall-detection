/**
 * Status hook: poll `GET /api/v1/status` for the composed subsystem state.
 *
 * §17.2 lists the endpoint as "Camera, model, sampling, and resources", and §8 asks the
 * interface to show "throughput, latency, drop rate, and memory usage". Status changes
 * arrive over the WebSocket as well, but the resource and configuration figures do not
 * change per frame, so polling is the right shape for them: it keeps the live feed free of
 * data nobody needs at 2 Hz, and it means the panel still populates when the socket is down
 * — which is exactly when an operator needs it most.
 *
 * A failed poll is *reported*, never swallowed: a status panel that silently keeps showing
 * the last good numbers would hide the problem it exists to reveal.
 */

import { useCallback, useEffect, useState } from "react";

import { api, ApiError } from "../services/api";
import type { StatusPayload } from "../types";

export interface StatusFeed {
  status: StatusPayload | null;
  error: string | null;
  loading: boolean;
  refresh: () => void;
}

export function useStatus(intervalMs = 2000, enabled = true): StatusFeed {
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);

  const refresh = useCallback(() => setNonce((value) => value + 1), []);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    // The whole body is one async task: the effect starts work and subscribes to a timer,
    // and every state update happens in a callback rather than during the render pass the
    // effect belongs to.
    void (async () => {
      const load = async () => {
        try {
          const payload = await api.status();
          if (!cancelled) {
            setStatus(payload);
            setError(null);
          }
        } catch (cause) {
          if (!cancelled) {
            setError(cause instanceof ApiError ? cause.message : "could not read the status");
          }
        } finally {
          if (!cancelled) {
            setLoading(false);
          }
        }
      };

      if (!enabled) {
        setLoading(false);
        return;
      }
      await load();
      timer = window.setInterval(() => void load(), intervalMs);
    })();

    return () => {
      cancelled = true;
      if (timer !== undefined) {
        window.clearInterval(timer);
      }
    };
  }, [enabled, intervalMs, nonce]);

  return { status, error, loading, refresh };
}
