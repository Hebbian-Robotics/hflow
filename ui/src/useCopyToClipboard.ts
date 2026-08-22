import { useEffect, useRef, useState } from "react";

/** Copy feedback: idle -> copied/failed -> (after the reset delay) idle. */
export type CopyState = "idle" | "copied" | "failed";

const COPY_RESET_DELAY_MS = 1600;

/**
 * Clipboard write with self-resetting feedback state — the one owner of the
 * write/flip/reset-timer/unmount-cleanup dance, shared by every copy
 * affordance (SQL footer, version chips) so their timing never drifts apart.
 */
export function useCopyToClipboard(): {
  copyState: CopyState;
  copyText: (text: string) => Promise<void>;
} {
  const [copyState, setCopyState] = useState<CopyState>("idle");
  const resetTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (resetTimerRef.current !== null) window.clearTimeout(resetTimerRef.current);
    };
  }, []);

  async function copyText(text: string): Promise<void> {
    let nextState: CopyState;
    try {
      await navigator.clipboard.writeText(text);
      nextState = "copied";
    } catch {
      nextState = "failed";
    }
    setCopyState(nextState);
    if (resetTimerRef.current !== null) window.clearTimeout(resetTimerRef.current);
    resetTimerRef.current = window.setTimeout(() => setCopyState("idle"), COPY_RESET_DELAY_MS);
  }

  return { copyState, copyText };
}
