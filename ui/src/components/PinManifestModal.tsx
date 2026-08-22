import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { describeApiError, pinManifest } from "../api";
import { formatFractionAsPercent } from "../format";
import { CloseIcon, PinIcon } from "../icons";

// Pins an immutable Parquet manifest of the editor's SQL under
// <data_root>/manifests/ (never the engine's mutable default manifest.parquet).
// On success it links straight to the Manifests tab.

export function PinManifestModal({
  sql,
  onClose,
  onOpenManifests,
}: {
  sql: string;
  onClose: () => void;
  onOpenManifests: () => void;
}) {
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const nameInputRef = useRef<HTMLInputElement | null>(null);

  const pinMutation = useMutation({
    mutationFn: pinManifest,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["manifests"] });
    },
  });

  useEffect(() => {
    nameInputRef.current?.focus();
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const pinnedEntry = pinMutation.data;
  const partialCoverageCount = pinnedEntry
    ? pinnedEntry.coverage.filter((entry) => entry.fraction < 1).length
    : 0;

  return (
    <div className="modal-overlay">
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="pin-modal-title">
        <header className="modal-header">
          <h2 id="pin-modal-title" className="modal-title">
            {pinnedEntry ? "Manifest pinned" : "Pin manifest"}
          </h2>
          <button
            type="button"
            className="btn btn-ghost btn-tiny"
            onClick={onClose}
            aria-label="Close"
            title="Close"
          >
            <CloseIcon />
          </button>
        </header>

        {pinnedEntry ? (
          <div className="modal-body">
            <p className="modal-success">
              Pinned <strong>{pinnedEntry.name}</strong> — {pinnedEntry.row_count} rows over{" "}
              {pinnedEntry.total_episodes} catalog episodes.
            </p>
            <p className="modal-path" title={pinnedEntry.manifest_path}>
              {pinnedEntry.manifest_path}
            </p>
            {partialCoverageCount > 0 ? (
              <p className="modal-coverage-warning">
                {partialCoverageCount} check{partialCoverageCount === 1 ? "" : "s"} covered only
                part of this cut (lowest:{" "}
                {formatFractionAsPercent(
                  Math.min(...pinnedEntry.coverage.map((entry) => entry.fraction)),
                )}
                ) — see the coverage column in the Manifests tab.
              </p>
            ) : null}
            <footer className="modal-actions">
              <button type="button" className="btn" onClick={onClose}>
                Done
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => {
                  onOpenManifests();
                  onClose();
                }}
              >
                View in Manifests
              </button>
            </footer>
          </div>
        ) : (
          <form
            className="modal-body"
            onSubmit={(event) => {
              event.preventDefault();
              const trimmedName = name.trim();
              if (!trimmedName) return;
              const trimmedDescription = description.trim();
              pinMutation.mutate({
                sql,
                name: trimmedName,
                description: trimmedDescription === "" ? undefined : trimmedDescription,
              });
            }}
          >
            <p className="modal-hint">
              Freezes this cut as an immutable Parquet manifest under the workspace’s{" "}
              <code>manifests/</code> directory and records it in the registry.
            </p>
            <label className="modal-field">
              <span className="meta-label">Name</span>
              <input
                ref={nameInputRef}
                className="input"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="e.g. clean pick_place v1"
                required
              />
            </label>
            <label className="modal-field">
              <span className="meta-label">Description (optional)</span>
              <textarea
                className="input modal-textarea"
                rows={2}
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                placeholder="What this cut is for"
              />
            </label>
            <details className="modal-sql">
              <summary>SQL to pin</summary>
              <pre className="sql-block">{sql}</pre>
            </details>
            {pinMutation.isError ? (
              <p className="form-error" role="alert">
                {describeApiError(pinMutation.error)}
              </p>
            ) : null}
            <footer className="modal-actions">
              <button type="button" className="btn" onClick={onClose}>
                Cancel
              </button>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={!name.trim() || pinMutation.isPending}
              >
                <PinIcon />
                <span>{pinMutation.isPending ? "Pinning…" : "Pin manifest"}</span>
              </button>
            </footer>
          </form>
        )}
      </div>
    </div>
  );
}
