/**
 * The only place in the app that calls `fetch`.
 *
 * A refusal is not an error. `declined_incomplete` arrives as HTTP 200 with the
 * questions attached, and treating it as a failure here would push the UI
 * towards retrying past a gate that is working correctly. Only genuine
 * transport and server faults raise.
 */

import type {
  EvidenceResponse,
  Health,
  SessionResponse,
  TurnResponse,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    // The backend not running is the most common failure in development, and
    // "Failed to fetch" tells the reader nothing about which thing is down.
    throw new ApiError("Cannot reach the backend on 127.0.0.1:8010", 0);
  }

  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new ApiError(describe(detail) ?? res.statusText, res.status);
  }
  return res.json() as Promise<T>;
}

/** FastAPI puts the useful part in `detail`, sometimes as an object. */
function describe(body: unknown): string | null {
  if (!body || typeof body !== "object" || !("detail" in body)) return null;
  const detail = (body as { detail: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object" && "question_for_user" in detail) {
    return String((detail as { question_for_user: unknown }).question_for_user);
  }
  return null;
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, {
    method: "POST",
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });

export const api = {
  health: () => request<Health>("/health"),
  evidence: () => request<EvidenceResponse>("/evidence"),

  createSession: () => post<TurnResponse>("/assistant/sessions"),
  sendMessage: (sessionId: string, message: string) =>
    post<TurnResponse>(`/assistant/sessions/${sessionId}/messages`, { message }),
  getSession: (sessionId: string) =>
    request<SessionResponse>(`/assistant/sessions/${sessionId}`),
  deleteSession: (sessionId: string) =>
    request<{ deleted: string }>(`/assistant/sessions/${sessionId}`, {
      method: "DELETE",
    }),
};
