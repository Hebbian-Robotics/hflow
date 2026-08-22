import { runStateChipClass } from "../runState";

/** A run/task state as a chip, in the shared state vocabulary (runState.ts). */
export function RunStateChip({ state }: { state: string | null | undefined }) {
  return <span className={runStateChipClass(state)}>{state ?? "no state"}</span>;
}
