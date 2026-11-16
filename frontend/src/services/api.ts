/**
 * API client.
 *
 * The API has no login, so a request carries no credential, cookie, or token. Errors carry
 * a stable `code` (§17.2), so `ApiError` exposes the code and the UI switches on it rather
 * than on a message that may be reworded.
 *
 * The module holds no state: the dashboard's own state lives in hooks.
 */

import type { ApiErrorResponse, Camera, FallEvent, FallEventPage, StatusPayload } from "../types";

/** The API's error, carrying the stable code §17.2 requires. */
export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly detail?: string | null;

  constructor(status: number, code: string, message: string, detail?: string | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

async function parseError(response: Response): Promise<ApiError> {
  let code = "http_error";
  let message = `${response.status} ${response.statusText}`;
  let detail: string | null = null;
  try {
    const body = (await response.json()) as ApiErrorResponse;
    if (body?.error) {
      code = body.error.code;
      message = body.error.message;
      detail = body.error.detail ?? null;
    }
  } catch {
    // A response without the documented envelope is still an error; the status line is the
    // most honest description available, and inventing a code would hide the contract breach.
  }
  return new ApiError(response.status, code, message, detail);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    throw await parseError(response);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

/** A read request. */
export function get<T>(path: string): Promise<T> {
  return request<T>(path);
}

/** A mutating request: sends JSON. */
export function mutate<T>(
  method: "POST" | "PUT" | "DELETE",
  path: string,
  body?: unknown,
): Promise<T> {
  return request<T>(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export const api = {
  cameras: () => get<Camera[]>("/api/v1/cameras"),
  startCamera: (cameraId: string) =>
    mutate<{ state: string }>("POST", `/api/v1/cameras/${cameraId}/start`),
  stopCamera: (cameraId: string) =>
    mutate<{ state: string }>("POST", `/api/v1/cameras/${cameraId}/stop`),

  status: () => get<StatusPayload>("/api/v1/status"),

  fallEvents: (params: { limit?: number; cursor?: string | null; active?: boolean } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.cursor) search.set("cursor", params.cursor);
    if (params.active) search.set("active", "true");
    const query = search.toString();
    return get<FallEventPage>(`/api/v1/fall-events${query ? `?${query}` : ""}`);
  },
  fallEvent: (eventId: string) => get<FallEvent>(`/api/v1/fall-events/${eventId}`),
  acknowledge: (eventId: string) =>
    mutate<FallEvent>("POST", `/api/v1/fall-events/${eventId}/acknowledge`),

  snapshotUrl: (cameraId: string) => `/api/v1/cameras/${cameraId}/snapshot.jpg`,
  previewUrl: (cameraId: string) => `/api/v1/cameras/${cameraId}/preview.mjpeg`,
};
