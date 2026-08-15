import { useEffect, useState } from "react";

export interface Async<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/**
 * Run a fetch once on mount and expose the three states a screen must render.
 *
 * Generic on purpose: `useHealth` and `useEvidence` differ only in which
 * function they call, and writing each one out separately would be three copies
 * of the same loading/error bookkeeping waiting to drift apart.
 */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []): Async<T> {
  const [state, setState] = useState<Async<T>>({
    data: null,
    error: null,
    loading: true,
  });

  useEffect(() => {
    let live = true;
    setState((s) => ({ ...s, loading: true }));
    fn()
      .then((data) => live && setState({ data, error: null, loading: false }))
      .catch(
        (e: unknown) =>
          live &&
          setState({
            data: null,
            error: e instanceof Error ? e.message : String(e),
            loading: false,
          }),
      );
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}
