import type { CurationReport } from "../api";
import { formatFractionAsPercent } from "../format";
import { ErrorPanel, LoadingPanel } from "./QueryStates";

// Coverage display is mandatory (CATALOG.md): a statistic computed over half a
// delivery must not look like a statistic over all of it, so checks that ran
// on only part of the catalog are highlighted, not hidden.

function CoverageTable({ report }: { report: CurationReport }) {
  if (report.coverage.length === 0) {
    return <p className="empty-note">No check runs recorded — nothing to state coverage over.</p>;
  }
  const partialCount = report.coverage.filter((entry) => entry.fraction < 1).length;
  return (
    <>
      <div className="table-overflow">
        <table className="evidence-table coverage-table">
          <thead>
            <tr>
              <th>check</th>
              <th>episodes ran</th>
              <th>coverage</th>
            </tr>
          </thead>
          <tbody>
            {report.coverage.map((entry) => {
              const isPartial = entry.fraction < 1;
              return (
                <tr key={entry.check_name}>
                  <td className="cell-mono">{entry.check_name}</td>
                  <td className="cell-num">
                    {entry.episodes_ran} / {entry.total_episodes}
                  </td>
                  <td>
                    <div className="coverage-cell">
                      <div
                        className="coverage-bar"
                        role="img"
                        aria-label={`${formatFractionAsPercent(entry.fraction)} of episodes`}
                      >
                        <div
                          className={
                            isPartial ? "coverage-bar-fill is-partial" : "coverage-bar-fill"
                          }
                          style={{ width: `${Math.max(entry.fraction * 100, 1)}%` }}
                        />
                      </div>
                      <span
                        className={isPartial ? "coverage-percent is-partial" : "coverage-percent"}
                      >
                        {formatFractionAsPercent(entry.fraction)}
                      </span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {partialCount > 0 ? (
        <p className="report-note">
          {partialCount} check{partialCount === 1 ? "" : "s"} ran on only part of the catalog —
          statistics they feed do not describe the whole cut.
        </p>
      ) : null}
    </>
  );
}

/**
 * Row count + total episodes + check coverage for the SQL in the editor.
 * Renders nothing until a report has been requested; stays visible afterwards.
 */
export function ReportPanel({
  report,
  isPending,
  error,
  onRetry,
}: {
  report: CurationReport | undefined;
  isPending: boolean;
  error: unknown;
  onRetry: () => void;
}) {
  if (report === undefined && !isPending && !error) return null;
  return (
    <section className="report-panel" aria-label="Curation report">
      <h3 className="facet-label">Report</h3>
      {isPending ? (
        <LoadingPanel label="Computing report…" />
      ) : error ? (
        <ErrorPanel error={error} onRetry={onRetry} />
      ) : report ? (
        <>
          <div className="report-stats">
            <div className="report-stat">
              <span className="meta-label">rows in cut</span>
              <span className="report-stat-value">{report.row_count}</span>
            </div>
            <div className="report-stat">
              <span className="meta-label">catalog episodes</span>
              <span className="report-stat-value">{report.total_episodes}</span>
            </div>
          </div>
          <CoverageTable report={report} />
        </>
      ) : null}
    </section>
  );
}
