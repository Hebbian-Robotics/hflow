import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Pin, X } from "lucide-react";
import { Dialog } from "radix-ui";
import { type RefObject, useRef, useState } from "react";
import { describeApiError, pinManifest } from "../api";
import { formatFractionAsPercent } from "../format";

// Pins an immutable Parquet manifest of the editor's SQL under
// <data_root>/manifests/ (never the engine's mutable default manifest.parquet).
// On success it links straight to the Manifests tab.
//
// Radix Dialog owns the overlay behaviour that used to be hand-rolled here:
// the focus trap, aria-hidden on everything behind the dialog, the body
// scroll lock, outside-click and Escape dismissal, and — the reason the
// trigger lives INSIDE this component rather than in the toolbar — returning
// focus to the opener on close. Radix restores focus to `Dialog.Trigger`, so
// the trigger has to BE a Radix trigger; a plain toolbar button would leave
// Radix with nothing to focus and drop focus on <body>, which is the bug this
// replaced.

function PinManifestBody({
  sql,
  nameInputRef,
  onViewInManifests,
}: {
  sql: string;
  nameInputRef: RefObject<HTMLInputElement | null>;
  onViewInManifests: () => void;
}) {
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  const pinMutation = useMutation({
    mutationFn: pinManifest,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["manifests"] });
    },
  });

  const pinnedEntry = pinMutation.data;
  const partialCoverageCount = pinnedEntry
    ? pinnedEntry.coverage.filter((entry) => entry.fraction < 1).length
    : 0;

  return (
    <>
      <header className="modal-header">
        <Dialog.Title className="modal-title">
          {pinnedEntry ? "Manifest pinned" : "Pin manifest"}
        </Dialog.Title>
        <Dialog.Close asChild>
          <button type="button" className="btn btn-ghost btn-tiny" aria-label="Close" title="Close">
            <X />
          </button>
        </Dialog.Close>
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
              {partialCoverageCount} check{partialCoverageCount === 1 ? "" : "s"} covered only part
              of this cut (lowest:{" "}
              {formatFractionAsPercent(
                Math.min(...pinnedEntry.coverage.map((entry) => entry.fraction)),
              )}
              ) — see the coverage column in the Manifests tab.
            </p>
          ) : null}
          <footer className="modal-actions">
            <Dialog.Close asChild>
              <button type="button" className="btn">
                Done
              </button>
            </Dialog.Close>
            <button type="button" className="btn btn-primary" onClick={onViewInManifests}>
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
            <Dialog.Close asChild>
              <button type="button" className="btn">
                Cancel
              </button>
            </Dialog.Close>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={!name.trim() || pinMutation.isPending}
            >
              <Pin />
              <span>{pinMutation.isPending ? "Pinning…" : "Pin manifest"}</span>
            </button>
          </footer>
        </form>
      )}
    </>
  );
}

export function PinManifestModal({
  disabled,
  currentSql,
  onOpenManifests,
}: {
  disabled: boolean;
  /** Reads the editor's full document at the moment the dialog opens. */
  currentSql: () => string;
  /** Switches CuratePage to the Manifests tab AND focuses that tab. */
  onOpenManifests: () => void;
}) {
  const [sql, setSql] = useState<string | null>(null);
  const nameInputRef = useRef<HTMLInputElement | null>(null);
  const handOffFocusToManifestsRef = useRef(false);

  return (
    <Dialog.Root
      open={sql !== null}
      onOpenChange={(open) => {
        // Snapshot the SQL on open: the dialog pins exactly what the editor
        // held when it was asked to, never a later edit.
        setSql(open ? currentSql() || null : null);
      }}
    >
      <Dialog.Trigger asChild>
        <button
          type="button"
          className="btn"
          disabled={disabled}
          title="Freeze this cut as an immutable Parquet manifest"
        >
          <Pin />
          <span>Pin manifest</span>
        </button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="modal-overlay" />
        <Dialog.Content
          className="modal"
          // The dialog has no single summary paragraph; opt out explicitly
          // rather than let Radix warn about a missing description.
          aria-describedby={undefined}
          onOpenAutoFocus={(event) => {
            // Radix would take the first tabbable node, which is the header's
            // close button. The name field is what the reader came to fill in.
            const nameInput = nameInputRef.current;
            if (!nameInput) return;
            event.preventDefault();
            nameInput.focus();
          }}
          onCloseAutoFocus={(event) => {
            if (!handOffFocusToManifestsRef.current) return;
            handOffFocusToManifestsRef.current = false;
            // "View in Manifests" hides the studio pane, and the trigger goes
            // with it — focusing a display:none element silently no-ops and
            // drops focus on <body>. Switch tabs from here instead, once the
            // focus trap has let go, and let CuratePage focus the tab itself.
            event.preventDefault();
            onOpenManifests();
          }}
        >
          {sql === null ? null : (
            <PinManifestBody
              sql={sql}
              nameInputRef={nameInputRef}
              onViewInManifests={() => {
                handOffFocusToManifestsRef.current = true;
                setSql(null);
              }}
            />
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
