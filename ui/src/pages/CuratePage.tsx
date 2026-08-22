import { useMutation, useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  fetchCatalogTables,
  fetchSavedQueries,
  fetchWorkspaceConfig,
  runCurationPreview,
  runCurationReport,
  type SavedQuery,
} from "../api";
import { CatalogTree } from "../components/CatalogTree";
import { ColumnStatsPanel } from "../components/ColumnStatsPanel";
import { ManifestsPanel } from "../components/ManifestsPanel";
import { PinManifestModal } from "../components/PinManifestModal";
import { EmptyPanel, ErrorPanel, LoadingPanel } from "../components/QueryStates";
import { ReportPanel } from "../components/ReportPanel";
import { ResultGrid } from "../components/ResultGrid";
import { SavedQueriesMenu } from "../components/SavedQueriesMenu";
import { SqlEditor, type SqlEditorHandle, type SqlSchema } from "../components/SqlEditor";
import { SqlFooter } from "../components/SqlFooter";
import { PinIcon, PlayIcon, ReportIcon, StatsIcon } from "../icons";

type CurateTab = "studio" | "manifests";

// The curation studio: catalog tree | SQL editor + results | column stats.
// The SPA only renders what the server computed — the preview/report/pin
// endpoints do all SQL work over a constrained DuckDB connection.

function StudioPane({
  readOnly,
  onOpenManifests,
}: {
  readOnly: boolean;
  onOpenManifests: () => void;
}) {
  const editorRef = useRef<SqlEditorHandle | null>(null);
  const [hasText, setHasText] = useState(false);
  const [hasSelection, setHasSelection] = useState(false);
  const [isStatsOpen, setIsStatsOpen] = useState(true);
  const [lastPreviewSql, setLastPreviewSql] = useState<string | null>(null);
  const [pinModalSql, setPinModalSql] = useState<string | null>(null);
  const [loadedQuery, setLoadedQuery] = useState<SavedQuery | null>(null);

  const catalogQuery = useQuery({ queryKey: ["catalog-tables"], queryFn: fetchCatalogTables });
  const savedQueriesQuery = useQuery({ queryKey: ["saved-queries"], queryFn: fetchSavedQueries });

  const editorSchema = useMemo<SqlSchema>(() => {
    const schema: SqlSchema = {};
    for (const table of catalogQuery.data ?? []) {
      schema[table.name] = table.columns.map((column) => column.name);
    }
    return schema;
  }, [catalogQuery.data]);

  const previewMutation = useMutation({ mutationFn: runCurationPreview });
  const reportMutation = useMutation({ mutationFn: runCurationReport });
  const { mutate: mutatePreview } = previewMutation;
  const { mutate: mutateReport } = reportMutation;

  // Cmd/Ctrl+Enter and the Preview button: run the selection when one exists.
  const runPreview = useCallback(() => {
    const editor = editorRef.current;
    if (!editor) return;
    const sqlText = editor.currentSql(true);
    if (!sqlText) return;
    setLastPreviewSql(sqlText);
    mutatePreview({ sql: sqlText, stats: isStatsOpen });
  }, [mutatePreview, isStatsOpen]);

  const toggleStats = () => {
    const nextStatsOpen = !isStatsOpen;
    setIsStatsOpen(nextStatsOpen);
    // Opening the panel after a stats-less run: re-run the same SQL to fill it.
    if (nextStatsOpen && lastPreviewSql && previewMutation.data?.column_stats === null) {
      mutatePreview({ sql: lastPreviewSql, stats: true });
    }
  };

  // Report and pin always take the full document — never a silent partial cut.
  const runReport = () => {
    const sqlText = editorRef.current?.currentSql(false) ?? "";
    if (sqlText) mutateReport(sqlText);
  };

  const openPinModal = () => {
    const sqlText = editorRef.current?.currentSql(false) ?? "";
    if (sqlText) setPinModalSql(sqlText);
  };

  const loadSavedQuery = (savedQuery: SavedQuery) => {
    editorRef.current?.setDocument(savedQuery.sql);
    setLoadedQuery(savedQuery);
  };

  const previewData = previewMutation.data;

  return (
    <>
      <div className="toolbar studio-toolbar">
        <SavedQueriesMenu
          queries={savedQueriesQuery.data}
          isPending={savedQueriesQuery.isPending}
          error={savedQueriesQuery.isError ? savedQueriesQuery.error : null}
          readOnly={readOnly}
          loadedQuery={loadedQuery}
          hasText={hasText}
          onLoad={loadSavedQuery}
          onLoadedQueryChange={setLoadedQuery}
          currentSql={() => editorRef.current?.currentSql(false) ?? ""}
        />
        <div className="toolbar-spacer" />
        <button
          type="button"
          className="btn btn-primary"
          onClick={runPreview}
          disabled={!hasText || previewMutation.isPending}
          title="Cmd/Ctrl+Enter — runs only the selection when text is selected"
        >
          <PlayIcon />
          <span>{hasSelection ? "Preview selection" : "Preview"}</span>
        </button>
        <button
          type="button"
          className="btn"
          onClick={runReport}
          disabled={!hasText || reportMutation.isPending}
          title="Row count, episode total, and per-check coverage for this cut"
        >
          <ReportIcon />
          <span>Report</span>
        </button>
        {readOnly ? null : (
          <button
            type="button"
            className="btn"
            onClick={openPinModal}
            disabled={!hasText}
            title="Freeze this cut as an immutable Parquet manifest"
          >
            <PinIcon />
            <span>Pin manifest</span>
          </button>
        )}
        <button
          type="button"
          className={isStatsOpen ? "btn is-toggled" : "btn"}
          onClick={toggleStats}
          aria-pressed={isStatsOpen}
          title="Toggle the column-stats panel"
        >
          <StatsIcon />
          <span>Stats</span>
        </button>
      </div>

      <div className={isStatsOpen ? "studio-body has-stats" : "studio-body"}>
        <CatalogTree
          tables={catalogQuery.data}
          isPending={catalogQuery.isPending}
          error={catalogQuery.isError ? catalogQuery.error : null}
          onRetry={() => {
            void catalogQuery.refetch();
          }}
          onInsertText={(text) => editorRef.current?.insertText(text)}
        />

        <div className="studio-center">
          <div className="editor-shell">
            <SqlEditor
              ref={editorRef}
              initialDoc=""
              schema={editorSchema}
              onDocChanged={(doc) => setHasText(doc.trim().length > 0)}
              onSelectionChanged={setHasSelection}
              onRunShortcut={runPreview}
            />
          </div>

          <ReportPanel
            report={reportMutation.data}
            isPending={reportMutation.isPending}
            error={reportMutation.isError ? reportMutation.error : null}
            onRetry={runReport}
          />

          <div className="studio-results">
            {previewMutation.isPending ? (
              <LoadingPanel label="Running preview…" />
            ) : previewMutation.isError ? (
              <ErrorPanel error={previewMutation.error} onRetry={runPreview} />
            ) : previewData ? (
              <>
                {previewData.truncated ? (
                  <div className="truncation-banner" role="status">
                    Preview truncated — showing the first {previewData.rows.length} of{" "}
                    {previewData.row_count} rows.
                  </div>
                ) : (
                  <div className="result-meta">
                    {previewData.row_count} row{previewData.row_count === 1 ? "" : "s"}
                  </div>
                )}
                <ResultGrid columns={previewData.columns} rows={previewData.rows} />
              </>
            ) : (
              <EmptyPanel
                title="No preview yet."
                hint="Write a SELECT over the catalog views, then press Cmd/Ctrl+Enter or Preview. Select text to run only the selection."
              />
            )}
          </div>

          <SqlFooter
            sql={previewData?.sql}
            emptyMessage="the exact query the server ran appears here after a preview"
          />
        </div>

        {isStatsOpen ? (
          <ColumnStatsPanel stats={previewData ? previewData.column_stats : undefined} />
        ) : null}
      </div>

      {pinModalSql !== null && !readOnly ? (
        <PinManifestModal
          sql={pinModalSql}
          onClose={() => setPinModalSql(null)}
          onOpenManifests={onOpenManifests}
        />
      ) : null}
    </>
  );
}

export function CuratePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const activeTab: CurateTab = searchParams.get("tab") === "manifests" ? "manifests" : "studio";

  const configQuery = useQuery({ queryKey: ["config"], queryFn: fetchWorkspaceConfig });
  // Until the config arrives (or if it fails), keep write affordances hidden.
  const readOnly = configQuery.data ? configQuery.data.read_only : true;

  useEffect(() => {
    document.title = "Curate · HFlow";
    return () => {
      document.title = "HFlow";
    };
  }, []);

  const selectTab = (tab: CurateTab) => {
    setSearchParams(
      (previous) => {
        const next = new URLSearchParams(previous);
        if (tab === "manifests") next.set("tab", "manifests");
        else next.delete("tab");
        return next;
      },
      { replace: true },
    );
  };

  return (
    <div className="curate-page">
      <div className="curate-tabs" role="tablist" aria-label="Curate sections">
        <button
          type="button"
          role="tab"
          className={activeTab === "studio" ? "curate-tab is-active" : "curate-tab"}
          aria-selected={activeTab === "studio"}
          onClick={() => selectTab("studio")}
        >
          Studio
        </button>
        <button
          type="button"
          role="tab"
          className={activeTab === "manifests" ? "curate-tab is-active" : "curate-tab"}
          aria-selected={activeTab === "manifests"}
          onClick={() => selectTab("manifests")}
        >
          Manifests
        </button>
      </div>

      {/* The studio stays mounted while the Manifests tab shows, so the
          editor document, preview, and report survive tab switches. */}
      <div className={activeTab === "studio" ? "curate-tab-body" : "curate-tab-body is-hidden"}>
        <StudioPane readOnly={readOnly} onOpenManifests={() => selectTab("manifests")} />
      </div>
      {activeTab === "manifests" ? (
        <div className="curate-tab-body">
          <ManifestsPanel />
        </div>
      ) : null}
    </div>
  );
}
