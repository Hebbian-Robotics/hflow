import { Check, Copy } from "lucide-react";
import { shortFingerprint } from "../format";
import { useCopyToClipboard } from "../useCopyToClipboard";

/**
 * A step's content-hash version: truncated for density (the same
 * shortFingerprint rule the episode dossier uses), full value on hover, one
 * click copies the whole hash (diffing pipelines needs the exact string).
 */
export function VersionChip({ version }: { version: string | null }) {
  const { copyState, copyText } = useCopyToClipboard();
  // Some served versions are nullable (a check_runs LEFT JOIN that matched
  // nothing). There is no hash to show or copy, so it reads as absent.
  if (version === null) return <span className="cell-null">—</span>;

  const title =
    copyState === "copied"
      ? "Copied"
      : copyState === "failed"
        ? "Copy failed"
        : `${version} — click to copy`;

  return (
    <button
      type="button"
      className="version-chip"
      onClick={() => void copyText(version)}
      title={title}
    >
      <span className="version-chip-text">{shortFingerprint(version)}</span>
      {copyState === "copied" ? (
        <Check className="version-chip-icon" />
      ) : (
        <Copy className="version-chip-icon" />
      )}
    </button>
  );
}
