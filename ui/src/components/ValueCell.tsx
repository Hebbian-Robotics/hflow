import { formatNumber, formatTimestamp, looksLikeIsoTimestamp } from "../format";
import { StatusChip } from "./StatusChip";

const LONG_TEXT_TITLE_THRESHOLD = 32;

function isIdentifierColumn(columnName: string): boolean {
  return (
    columnName === "episode_id" ||
    columnName.endsWith("_id") ||
    columnName.endsWith("fingerprint") ||
    columnName === "uri"
  );
}

/** Generic display for one catalog value; column name only refines styling. */
export function ValueCell({ columnName, value }: { columnName: string; value: unknown }) {
  if (value === null || value === undefined) return <span className="cell-null">—</span>;
  if (columnName === "status" && typeof value === "string") return <StatusChip status={value} />;
  if (typeof value === "boolean") {
    return <span className="cell-mono">{value ? "true" : "false"}</span>;
  }
  if (typeof value === "number") return <span className="cell-num">{formatNumber(value)}</span>;
  const text = typeof value === "string" ? value : JSON.stringify(value);
  if (looksLikeIsoTimestamp(text)) {
    return (
      <span className="cell-mono" title={text}>
        {formatTimestamp(text)}
      </span>
    );
  }
  return (
    <span
      className={isIdentifierColumn(columnName) ? "cell-mono" : undefined}
      title={text.length > LONG_TEXT_TITLE_THRESHOLD ? text : undefined}
    >
      {text}
    </span>
  );
}
