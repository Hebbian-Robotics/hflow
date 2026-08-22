import { shortFingerprint } from "../format";
import { CheckIcon, CopyIcon } from "../icons";
import { useCopyToClipboard } from "../useCopyToClipboard";

/**
 * A step's content-hash version: truncated for density (the same
 * shortFingerprint rule the episode dossier uses), full value on hover, one
 * click copies the whole hash (diffing pipelines needs the exact string).
 */
export function VersionChip({ version }: { version: string }) {
  const { copyState, copyText } = useCopyToClipboard();

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
        <CheckIcon className="version-chip-icon" />
      ) : (
        <CopyIcon className="version-chip-icon" />
      )}
    </button>
  );
}
