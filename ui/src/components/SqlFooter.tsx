import { useEffect, useRef, useState } from "react";
import { CheckIcon, ChevronDownIcon, CopyIcon } from "../icons";

type CopyState = "idle" | "copied" | "failed";

/**
 * The compiled-query readout: the server returns the exact SELECT it ran for
 * the current filters, and this strip keeps it visible, expandable, copyable.
 */
export function SqlFooter({
  sql,
  emptyMessage = "compiled query appears here once episodes load",
}: {
  sql: string | undefined;
  emptyMessage?: string;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [copyState, setCopyState] = useState<CopyState>("idle");
  const resetTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (resetTimerRef.current !== null) window.clearTimeout(resetTimerRef.current);
    };
  }, []);

  async function copySql() {
    if (!sql) return;
    let nextState: CopyState;
    try {
      await navigator.clipboard.writeText(sql);
      nextState = "copied";
    } catch {
      nextState = "failed";
    }
    setCopyState(nextState);
    if (resetTimerRef.current !== null) window.clearTimeout(resetTimerRef.current);
    resetTimerRef.current = window.setTimeout(() => setCopyState("idle"), 1600);
  }

  const copyLabel =
    copyState === "copied" ? "Copied" : copyState === "failed" ? "Copy failed" : "Copy";

  return (
    <footer className="sql-footer">
      <button
        type="button"
        className="sql-toggle"
        onClick={() => setIsExpanded((previous) => !previous)}
        aria-expanded={isExpanded}
        title={isExpanded ? "Collapse the compiled query" : "Expand the compiled query"}
      >
        <ChevronDownIcon className={isExpanded ? "chevron is-open" : "chevron"} />
        <span>SQL</span>
      </button>
      {sql ? (
        isExpanded ? (
          <pre className="sql-full">{sql}</pre>
        ) : (
          <code
            className="sql-preview"
            title="Query compiled by the server for the current filters"
          >
            {sql}
          </code>
        )
      ) : (
        <code className="sql-preview sql-empty">{emptyMessage}</code>
      )}
      <button
        type="button"
        className="btn btn-ghost sql-copy"
        onClick={copySql}
        disabled={!sql}
        title="Copy the compiled SQL"
      >
        {copyState === "copied" ? <CheckIcon /> : <CopyIcon />}
        <span>{copyLabel}</span>
      </button>
    </footer>
  );
}
