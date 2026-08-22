import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, Trash2 } from "lucide-react";
import { DropdownMenu } from "radix-ui";
import { useEffect, useRef, useState } from "react";
import {
  createSavedQuery,
  deleteSavedQuery,
  describeApiError,
  type SavedQuery,
  updateSavedQuery,
} from "../api";
import { formatTimestamp } from "../format";

// Saved-queries dropdown: load always works; save / save-as / delete are
// hidden entirely when the workspace is read-only (server 403s them anyway).
//
// Radix DropdownMenu supplies the roving arrow-key focus, typeahead, Escape
// and outside-click dismissal, portalling and focus return that this used to
// half-implement with two document listeners.

export function SavedQueriesMenu({
  queries,
  isPending,
  error,
  readOnly,
  loadedQuery,
  hasText,
  onLoad,
  onLoadedQueryChange,
  currentSql,
}: {
  queries: SavedQuery[] | undefined;
  isPending: boolean;
  error: unknown;
  readOnly: boolean;
  loadedQuery: SavedQuery | null;
  hasText: boolean;
  onLoad: (query: SavedQuery) => void;
  onLoadedQueryChange: (query: SavedQuery | null) => void;
  currentSql: () => string;
}) {
  const queryClient = useQueryClient();
  const [isOpen, setIsOpen] = useState(false);
  const [isSaveAsOpen, setIsSaveAsOpen] = useState(false);
  const [saveAsName, setSaveAsName] = useState("");
  const saveAsInputRef = useRef<HTMLInputElement | null>(null);

  // The save-as field is not a menu item, so arrow keys never reach it; put
  // the caret there the moment it appears instead.
  useEffect(() => {
    if (isSaveAsOpen) saveAsInputRef.current?.focus();
  }, [isSaveAsOpen]);

  const invalidateSavedQueries = () => {
    void queryClient.invalidateQueries({ queryKey: ["saved-queries"] });
  };

  const createMutation = useMutation({
    mutationFn: createSavedQuery,
    onSuccess: (created) => {
      invalidateSavedQueries();
      onLoadedQueryChange(created);
      setIsSaveAsOpen(false);
      setSaveAsName("");
    },
  });
  const updateMutation = useMutation({
    mutationFn: (input: { queryId: string; sql: string }) =>
      updateSavedQuery(input.queryId, { sql: input.sql }),
    onSuccess: (updated) => {
      invalidateSavedQueries();
      onLoadedQueryChange(updated);
    },
  });
  const deleteMutation = useMutation({
    mutationFn: deleteSavedQuery,
    onSuccess: (_result, deletedQueryId) => {
      invalidateSavedQueries();
      if (loadedQuery?.id === deletedQueryId) onLoadedQueryChange(null);
    },
  });

  const saveToLoaded = () => {
    const sqlText = currentSql();
    if (!loadedQuery || !sqlText) return;
    updateMutation.mutate({ queryId: loadedQuery.id, sql: sqlText });
  };

  const submitSaveAs = () => {
    const sqlText = currentSql();
    const trimmedName = saveAsName.trim();
    if (!trimmedName || !sqlText) return;
    createMutation.mutate({ name: trimmedName, sql: sqlText });
  };

  const writeError = createMutation.isError
    ? createMutation.error
    : updateMutation.isError
      ? updateMutation.error
      : deleteMutation.isError
        ? deleteMutation.error
        : null;

  return (
    <DropdownMenu.Root
      open={isOpen}
      onOpenChange={(open) => {
        setIsOpen(open);
        if (!open) setIsSaveAsOpen(false);
      }}
    >
      <DropdownMenu.Trigger className="btn" title="Saved queries">
        <span className="menu-toggle-label">
          {loadedQuery ? loadedQuery.name : "Saved queries"}
        </span>
        <ChevronDown className={isOpen ? "chevron is-open" : "chevron"} />
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="menu-panel" align="start" sideOffset={4}>
          {isPending ? (
            <p className="menu-note">Loading saved queries…</p>
          ) : error ? (
            <p className="menu-note menu-error">{describeApiError(error)}</p>
          ) : (queries ?? []).length === 0 ? (
            <p className="menu-note">
              No saved queries yet.{readOnly ? "" : " Use “Save as…” to keep the current SQL."}
            </p>
          ) : (
            <div className="menu-list">
              {(queries ?? []).map((savedQuery) => (
                <div key={savedQuery.id} className="menu-row">
                  <DropdownMenu.Item
                    className={
                      savedQuery.id === loadedQuery?.id ? "menu-item is-active" : "menu-item"
                    }
                    onSelect={() => onLoad(savedQuery)}
                    title={savedQuery.sql}
                  >
                    <span className="menu-item-name">{savedQuery.name}</span>
                    <span className="menu-item-time">{formatTimestamp(savedQuery.updated_at)}</span>
                  </DropdownMenu.Item>
                  {readOnly ? null : (
                    <DropdownMenu.Item
                      className="btn btn-ghost btn-tiny"
                      disabled={deleteMutation.isPending}
                      onSelect={(event) => {
                        // Deleting must not close the menu — the point is to
                        // prune several entries in one visit.
                        event.preventDefault();
                        deleteMutation.mutate(savedQuery.id);
                      }}
                      title={`Delete "${savedQuery.name}"`}
                      aria-label={`Delete saved query ${savedQuery.name}`}
                    >
                      <Trash2 />
                    </DropdownMenu.Item>
                  )}
                </div>
              ))}
            </div>
          )}
          {readOnly ? null : (
            <div className="menu-actions">
              <DropdownMenu.Item
                className="btn btn-tiny"
                disabled={!loadedQuery || !hasText || updateMutation.isPending}
                onSelect={(event) => {
                  event.preventDefault();
                  saveToLoaded();
                }}
                title={
                  loadedQuery
                    ? `Overwrite "${loadedQuery.name}" with the editor's SQL`
                    : "Load a query first, or use Save as…"
                }
              >
                {updateMutation.isPending ? "Saving…" : "Save"}
              </DropdownMenu.Item>
              <DropdownMenu.Item
                className="btn btn-tiny"
                disabled={!hasText}
                onSelect={(event) => {
                  event.preventDefault();
                  setIsSaveAsOpen((previous) => !previous);
                }}
                title="Save the editor's SQL as a new named query"
              >
                Save as…
              </DropdownMenu.Item>
            </div>
          )}
          {isSaveAsOpen && !readOnly ? (
            <form
              className="menu-saveas"
              onSubmit={(event) => {
                event.preventDefault();
                submitSaveAs();
              }}
            >
              {/* A text field inside a menu has to opt out of the menu's own
                  keyboard handling — Radix swallows Tab and reads printable
                  keys as typeahead. Escape is left alone so the dismiss layer
                  still closes the menu from inside the field. */}
              <input
                className="input"
                placeholder="Query name"
                value={saveAsName}
                onChange={(event) => setSaveAsName(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key !== "Escape") event.stopPropagation();
                }}
                ref={saveAsInputRef}
                aria-label="New saved query name"
              />
              <button
                type="submit"
                className="btn btn-tiny"
                disabled={!saveAsName.trim() || createMutation.isPending}
              >
                {createMutation.isPending ? "Saving…" : "Save"}
              </button>
            </form>
          ) : null}
          {writeError ? (
            <p className="menu-note menu-error">{describeApiError(writeError)}</p>
          ) : null}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
