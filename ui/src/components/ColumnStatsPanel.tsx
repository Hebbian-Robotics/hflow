import type { SummarizeRow } from "../api";
import { summarizeValueText } from "../format";
import { EmptyPanel } from "./QueryStates";

// One card per result column, straight from the server's SUMMARIZE rows.
// The key set is whatever DuckDB emitted; nothing is derived client-side.

function statLabel(key: string): string {
  return key.replaceAll("_", " ");
}

function StatCard({ statsRow }: { statsRow: SummarizeRow }) {
  const columnName = summarizeValueText(statsRow.column_name);
  const columnType = summarizeValueText(statsRow.column_type);
  const detailEntries = Object.entries(statsRow).filter(
    ([key]) => key !== "column_name" && key !== "column_type",
  );
  return (
    <article className="stat-card">
      <header className="stat-card-header">
        <span className="stat-card-name" title={columnName}>
          {columnName}
        </span>
        <span className="stat-card-type">{columnType}</span>
      </header>
      <dl className="stat-grid">
        {detailEntries.map(([key, value]) => {
          const valueText = summarizeValueText(value);
          return (
            <div key={key} className="stat-pair">
              <dt className="stat-key">{statLabel(key)}</dt>
              <dd className="stat-value" title={valueText}>
                {valueText}
              </dd>
            </div>
          );
        })}
      </dl>
    </article>
  );
}

/**
 * The toggleable right pane of the studio. `stats` is undefined before any
 * preview ran, null when the last preview ran with stats off.
 */
export function ColumnStatsPanel({ stats }: { stats: SummarizeRow[] | null | undefined }) {
  return (
    <aside className="curate-stats" aria-label="Column stats">
      <h3 className="facet-label">Column stats</h3>
      {stats === undefined ? (
        <EmptyPanel
          title="No profile yet."
          hint="Run a preview and each result column gets summarized here."
        />
      ) : stats === null ? (
        <EmptyPanel
          title="Stats were off for the last run."
          hint="Run the preview again while this panel is open."
        />
      ) : stats.length === 0 ? (
        <EmptyPanel title="Nothing to summarize." hint="The preview returned no columns." />
      ) : (
        stats.map((statsRow) => (
          <StatCard key={summarizeValueText(statsRow.column_name)} statsRow={statsRow} />
        ))
      )}
    </aside>
  );
}
