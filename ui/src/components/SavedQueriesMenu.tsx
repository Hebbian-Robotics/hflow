import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import {
  createSavedQuery,
  deleteSavedQuery,
  describeApiError,
  type SavedQuery,
  updateSavedQuery,
} from "../api";
import { formatTimestamp } from "../format";
import { ChevronDownIcon, TrashIcon } from "../icons";

// Saved-queries dropdown: load always works; save / save-as / delete are
// hidden entirely when the workspace is read-only (server 403s them anyway).

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
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    const handlePointerDown = (event: MouseEvent) => {
      const container = containerRef.current;
      if (container && event.target instanceof Node && !container.contains(event.target)) {
        setIsOpen(false);
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIsOpen(false);
    };
    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen]);

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
    <div className="menu-anchor" ref={containerRef}>
      <button
        type="button"
        className="btn"
        onClick={() => setIsOpen((previous) => !previous)}
        aria-expanded={isOpen}
        title="Saved queries"
      >
        <span className="menu-toggle-label">
          {loadedQuery ? loadedQuery.name : "Saved queries"}
        </span>
        <ChevronDownIcon className={isOpen ? "chevron is-open" : "chevron"} />
      </button>
      {isOpen ? (
        <div className="menu-panel">
          {isPending ? (
            <p className="menu-note">Loading saved queries…</p>
          ) : error ? (
            <p className="menu-note menu-error">{describeApiError(error)}</p>
          ) : (queries ?? []).length === 0 ? (
            <p className="menu-note">
              No saved queries yet.{readOnly ? "" : " Use “Save as…” to keep the current SQL."}
            </p>
          ) : (
            <ul className="menu-list">
              {(queries ?? []).map((savedQuery) => (
                <li key={savedQuery.id} className="menu-row">
                  <button
                    type="button"
                    className={
                      savedQuery.id === loadedQuery?.id ? "menu-item is-active" : "menu-item"
                    }
                    onClick={() => {
                      onLoad(savedQuery);
                      setIsOpen(false);
                    }}
                    title={savedQuery.sql}
                  >
                    <span className="menu-item-name">{savedQuery.name}</span>
                    <span className="menu-item-time">{formatTimestamp(savedQuery.updated_at)}</span>
                  </button>
                  {readOnly ? null : (
                    <button
                      type="button"
                      className="btn btn-ghost btn-tiny"
                      onClick={() => deleteMutation.mutate(savedQuery.id)}
                      disabled={deleteMutation.isPending}
                      title={`Delete "${savedQuery.name}"`}
                      aria-label={`Delete saved query ${savedQuery.name}`}
                    >
                      <TrashIcon />
                    </button>
                  )}
                </li>
              ))}
            </ul>
          )}
          {readOnly ? null : (
            <div className="menu-actions">
              <button
                type="button"
                className="btn btn-tiny"
                onClick={saveToLoaded}
                disabled={!loadedQuery || !hasText || updateMutation.isPending}
                title={
                  loadedQuery
                    ? `Overwrite "${loadedQuery.name}" with the editor's SQL`
                    : "Load a query first, or use Save as…"
                }
              >
                {updateMutation.isPending ? "Saving…" : "Save"}
              </button>
              <button
                type="button"
                className="btn btn-tiny"
                onClick={() => setIsSaveAsOpen((previous) => !previous)}
                disabled={!hasText}
                title="Save the editor's SQL as a new named query"
              >
                Save as…
              </button>
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
              <input
                className="input"
                placeholder="Query name"
                value={saveAsName}
                onChange={(event) => setSaveAsName(event.target.value)}
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
        </div>
      ) : null}
    </div>
  );
}
