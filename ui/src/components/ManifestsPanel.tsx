import { useQuery } from "@tanstack/react-query";
import { Download } from "lucide-react";
import { Fragment, useState } from "react";
import { fetchManifests, manifestDownloadPath } from "../api";
import { formatFractionAsPercent, formatTimestamp } from "../format";
import { EmptyPanel, ErrorPanel, LoadingPanel } from "./QueryStates";

// The Manifests tab: the registry of pinned cuts, newest first. Each pin is an
// immutable Parquet file under <data_root>/manifests/ plus this registry row.

export function ManifestsPanel() {
  const manifestsQuery = useQuery({ queryKey: ["manifests"], queryFn: fetchManifests });
  const [expandedSqlIds, setExpandedSqlIds] = useState<ReadonlySet<string>>(new Set());

  const toggleSql = (manifestId: string) => {
    setExpandedSqlIds((previous) => {
      const next = new Set(previous);
      if (next.has(manifestId)) next.delete(manifestId);
      else next.add(manifestId);
      return next;
    });
  };

  if (manifestsQuery.isPending) {
    return (
      <div className="manifests-pane">
        <LoadingPanel label="Loading manifests…" />
      </div>
    );
  }
  if (manifestsQuery.isError) {
    return (
      <div className="manifests-pane">
        <ErrorPanel
          error={manifestsQuery.error}
          onRetry={() => {
            void manifestsQuery.refetch();
          }}
        />
      </div>
    );
  }
  const manifests = manifestsQuery.data;
  if (manifests.length === 0) {
    return (
      <div className="manifests-pane">
        <EmptyPanel
          title="No pinned manifests yet."
          hint="Preview a cut in the Studio tab, then use “Pin manifest” to freeze it here."
        />
      </div>
    );
  }

  return (
    <div className="manifests-pane">
      <p className="manifests-note">
        Pinned manifests are immutable Parquet cuts under the workspace’s <code>manifests/</code>{" "}
        directory — the CLI’s mutable default manifest never appears here.
      </p>
      <div className="table-overflow">
        <table className="evidence-table manifests-table">
          <thead>
            <tr>
              <th>name</th>
              <th>description</th>
              <th>rows</th>
              <th>episodes</th>
              <th>min coverage</th>
              <th>created</th>
              <th>download</th>
              <th>sql</th>
            </tr>
          </thead>
          <tbody>
            {manifests.map((manifest) => {
              const isSqlExpanded = expandedSqlIds.has(manifest.id);
              const lowestCoverage =
                manifest.coverage.length > 0
                  ? Math.min(...manifest.coverage.map((entry) => entry.fraction))
                  : null;
              return (
                <Fragment key={manifest.id}>
                  <tr>
                    <td className="manifest-name" title={manifest.manifest_path}>
                      {manifest.name}
                    </td>
                    <td className="cell-dim" title={manifest.description ?? undefined}>
                      {manifest.description ? (
                        manifest.description
                      ) : (
                        <span className="cell-null">—</span>
                      )}
                    </td>
                    <td className="cell-num">{manifest.row_count}</td>
                    <td className="cell-num">{manifest.total_episodes}</td>
                    <td>
                      {lowestCoverage === null ? (
                        <span className="cell-null">—</span>
                      ) : (
                        <span
                          className={
                            lowestCoverage < 1 ? "coverage-percent is-partial" : "coverage-percent"
                          }
                          title="Lowest check coverage over this cut"
                        >
                          {formatFractionAsPercent(lowestCoverage)}
                        </span>
                      )}
                    </td>
                    <td className="cell-mono" title={manifest.created_at}>
                      {formatTimestamp(manifest.created_at)}
                    </td>
                    <td>
                      <a
                        className="btn btn-tiny"
                        href={manifestDownloadPath(manifest.id)}
                        download
                        title={`Download ${manifest.manifest_path}`}
                      >
                        <Download />
                        <span>Parquet</span>
                      </a>
                    </td>
                    <td>
                      <button
                        type="button"
                        className="btn btn-ghost btn-tiny"
                        onClick={() => toggleSql(manifest.id)}
                        aria-expanded={isSqlExpanded}
                      >
                        {isSqlExpanded ? "Hide SQL" : "Show SQL"}
                      </button>
                    </td>
                  </tr>
                  {isSqlExpanded ? (
                    <tr className="error-detail-row">
                      <td colSpan={8}>
                        <pre className="sql-block">{manifest.sql}</pre>
                      </td>
                    </tr>
                  ) : null}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
