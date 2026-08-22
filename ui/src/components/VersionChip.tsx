import { useEffect, useRef, useState } from "react";
import { CheckIcon, CopyIcon } from "../icons";

const VERSION_DISPLAY_LENGTH = 10;

/**
 * A step's content-hash version: truncated for density, full value on hover,
 * one click copies the whole hash (diffing pipelines needs the exact string).
 */
export function VersionChip({ version }: { version: string }) {
  const [isCopied, setIsCopied] = useState(false);
  const resetTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (resetTimerRef.current !== null) window.clearTimeout(resetTimerRef.current);
    };
  }, []);

  async function copyVersion() {
    try {
      await navigator.clipboard.writeText(version);
      setIsCopied(true);
      if (resetTimerRef.current !== null) window.clearTimeout(resetTimerRef.current);
      resetTimerRef.current = window.setTimeout(() => setIsCopied(false), 1600);
    } catch {
      // Clipboard unavailable (permissions, insecure context): leave the chip as-is.
    }
  }

  const displayText =
    version.length > VERSION_DISPLAY_LENGTH ? version.slice(0, VERSION_DISPLAY_LENGTH) : version;

  return (
    <button
      type="button"
      className="version-chip"
      onClick={copyVersion}
      title={isCopied ? "Copied" : `${version} — click to copy`}
    >
      <span className="version-chip-text">{displayText}</span>
      {isCopied ? (
        <CheckIcon className="version-chip-icon" />
      ) : (
        <CopyIcon className="version-chip-icon" />
      )}
    </button>
  );
}
