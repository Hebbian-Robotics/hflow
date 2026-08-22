import { useEffect, useState } from "react";

/**
 * Wall-clock milliseconds, refreshed on an interval. `null` stops the clock —
 * a finished run needs no ticking, so nothing re-renders once the run's last
 * task has ended.
 */
export function useNowTick(intervalMs: number | null): number {
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    if (intervalMs === null) return;
    setNowMs(Date.now());
    const timer = window.setInterval(() => setNowMs(Date.now()), intervalMs);
    return () => window.clearInterval(timer);
  }, [intervalMs]);
  return nowMs;
}
