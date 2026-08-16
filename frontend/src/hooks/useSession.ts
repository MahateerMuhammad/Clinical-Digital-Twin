import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { TurnResponse } from "../api/types";

export interface Exchange {
  /** Absent on the opening capability message, which answers nothing. */
  question?: string;
  turn: TurnResponse;
}

/**
 * One conversation.
 *
 * Holds the exchanges rather than a flat message list, because every panel on
 * the case screen reads from the *latest turn* — facts, gate state, model
 * output, evidence all arrive together and must never be shown from different
 * turns.
 */
export function useSession() {
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionId = useRef<string | null>(null);
  const started = useRef(false);

  useEffect(() => {
    // StrictMode mounts twice in development; without this guard the second
    // mount opens a second session and the first is orphaned server-side.
    if (started.current) return;
    started.current = true;

    api
      .createSession()
      .then((turn) => {
        sessionId.current = turn.session_id;
        setExchanges([{ turn }]);
      })
      .catch((e: unknown) =>
        setError(e instanceof Error ? e.message : String(e)),
      );
  }, []);

  const send = useCallback(async (message: string) => {
    const id = sessionId.current;
    if (!id || !message.trim()) return;

    setPending(true);
    setError(null);
    try {
      const turn = await api.sendMessage(id, message);
      setExchanges((prev) => [...prev, { question: message, turn }]);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
    }
  }, []);

  const reset = useCallback(async () => {
    const id = sessionId.current;
    if (id) await api.deleteSession(id).catch(() => undefined);
    const turn = await api.createSession();
    sessionId.current = turn.session_id;
    setExchanges([{ turn }]);
    setError(null);
  }, []);

  const latest = exchanges.at(-1)?.turn ?? null;
  return { exchanges, latest, pending, error, send, reset };
}
