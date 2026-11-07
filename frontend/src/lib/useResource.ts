import { useCallback, useEffect, useRef, useState } from "react";
import { isAbort } from "./api/client";

export interface Resource<T> {
  data: T | undefined;
  error: unknown;
  loading: boolean;
  reload: () => void;
}

interface Settled<T> {
  key: string;
  tick: number;
  data: T | undefined;
  error: unknown;
}

/** Loads `fetcher` whenever `key` changes (null = nothing to load) or `reload()` is called. Earlier requests
 * are aborted, so a slow stale answer can never replace a newer one. With `keepPrevious`, the last data stays
 * visible while the next key loads (e.g. the previous catalog page while the next one arrives). */
export function useResource<T>(
  key: string | null,
  fetcher: (signal: AbortSignal) => Promise<T>,
  keepPrevious = false,
): Resource<T> {
  const [settled, setSettled] = useState<Settled<T> | null>(null);
  const [tick, setTick] = useState(0);
  const fetcherRef = useRef(fetcher);
  useEffect(() => {
    fetcherRef.current = fetcher;
  });

  useEffect(() => {
    if (key === null) return;
    const ctl = new AbortController();
    fetcherRef.current(ctl.signal).then(
      (data) => {
        if (!ctl.signal.aborted) setSettled({ key, tick, data, error: undefined });
      },
      (error: unknown) => {
        if (ctl.signal.aborted || isAbort(error)) return;
        setSettled((s) => ({ key, tick, data: s?.key === key ? s.data : undefined, error }));
      },
    );
    return () => ctl.abort();
  }, [key, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  if (key === null) return { data: undefined, error: undefined, loading: false, reload };
  const sameKey = settled?.key === key;
  const current = sameKey && settled.tick === tick;
  return {
    data: sameKey || keepPrevious ? settled?.data : undefined,
    error: current ? settled.error : undefined,
    loading: !current,
    reload,
  };
}

export function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}
