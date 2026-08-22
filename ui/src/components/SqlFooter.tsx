import { useState } from "react";
import { CheckIcon, ChevronDownIcon, CopyIcon } from "../icons";
import { useCopyToClipboard } from "../useCopyToClipboard";

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
  const { copyState, copyText } = useCopyToClipboard();

  const copySql = () => {
    if (sql) void copyText(sql);
  };

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
