import { useQuery } from "@tanstack/react-query";
import { ChevronDown, Plus, X } from "lucide-react";
import { useState } from "react";
import { type CatalogTable, fetchCatalogTableSummary } from "../api";
import { summarizeValueText } from "../format";
import { EmptyPanel, ErrorPanel, LoadingPanel } from "./QueryStates";

// Left pane of the studio: the registered catalog views with expandable
// columns, plus a per-table profile (row count + SUMMARIZE) on click.

const SUMMARY_PROFILE_KEYS = ["min", "max", "approx_unique", "null_percentage"] as const;

const PLAIN_SQL_IDENTIFIER_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;

/** What click-to-insert puts in the editor: quoted unless the name is a plain
 * identifier — the wide episodes view pivots measurement keys into columns,
 * and those keys are arbitrary strings (e.g. "artifact/wrist_cam"), which
 * DuckDB would otherwise parse as a division of two unknown columns. */
function insertableSqlIdentifier(name: string): string {
  return PLAIN_SQL_IDENTIFIER_PATTERN.test(name) ? name : `"${name.replaceAll('"', '""')}"`;
}

function TableSummarySection({
  tableName,
  onDeselect,
}: {
  tableName: string;
  onDeselect: () => void;
}) {
  const summaryQuery = useQuery({
    queryKey: ["table-summary", tableName],
    queryFn: () => fetchCatalogTableSummary(tableName),
  });
  return (
    <section className="table-summary" aria-label={`Summary of ${tableName}`}>
      <header className="table-summary-header">
        <h3 className="facet-label table-summary-title" title={tableName}>
          {tableName}
        </h3>
        <button
          type="button"
          className="btn btn-ghost btn-tiny"
          onClick={onDeselect}
          title="Close summary"
          aria-label={`Close summary of ${tableName}`}
        >
          <X />
        </button>
      </header>
      {summaryQuery.isPending ? (
        <LoadingPanel label="Profiling…" />
      ) : summaryQuery.isError ? (
        <ErrorPanel
          error={summaryQuery.error}
          onRetry={() => {
            void summaryQuery.refetch();
          }}
        />
      ) : (
        <>
          <p className="table-summary-rowcount">
            <span className="cell-num">{summaryQuery.data.row_count}</span> rows
          </p>
          <div className="table-overflow">
            <table className="evidence-table summary-table">
              <thead>
                <tr>
                  <th>column</th>
                  {SUMMARY_PROFILE_KEYS.map((key) => (
                    <th key={key}>{key.replaceAll("_", " ")}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {summaryQuery.data.columns.map((summarizeRow) => {
                  const columnName = summarizeValueText(summarizeRow.column_name);
                  return (
                    <tr key={columnName}>
                      <td
                        className="cell-mono"
                        title={summarizeValueText(summarizeRow.column_type)}
                      >
                        {columnName}
                      </td>
                      {SUMMARY_PROFILE_KEYS.map((key) => {
                        const valueText = summarizeValueText(summarizeRow[key]);
                        return (
                          <td key={key} className="cell-num" title={valueText}>
                            {valueText}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

export function CatalogTree({
  tables,
  isPending,
  error,
  onRetry,
  onInsertText,
}: {
  tables: CatalogTable[] | undefined;
  isPending: boolean;
  error: unknown;
  onRetry: () => void;
  onInsertText: (text: string) => void;
}) {
  const [expandedTableNames, setExpandedTableNames] = useState<ReadonlySet<string>>(new Set());
  const [selectedTableName, setSelectedTableName] = useState<string | null>(null);

  const toggleExpanded = (tableName: string) => {
    setExpandedTableNames((previous) => {
      const next = new Set(previous);
      if (next.has(tableName)) next.delete(tableName);
      else next.add(tableName);
      return next;
    });
  };

  return (
    <aside className="catalog-pane" aria-label="Catalog tables">
      <h3 className="facet-label">Catalog</h3>
      <div className="catalog-scroll">
        {isPending ? (
          <LoadingPanel label="Loading catalog…" />
        ) : error ? (
          <ErrorPanel error={error} onRetry={onRetry} />
        ) : (tables ?? []).length === 0 ? (
          <EmptyPanel
            title="No catalog views."
            hint="Ingest episodes first — the registered views appear here."
          />
        ) : (
          <ul className="catalog-list">
            {(tables ?? []).map((table) => {
              const isExpanded = expandedTableNames.has(table.name);
              const isSelected = selectedTableName === table.name;
              return (
                <li key={table.name}>
                  <div className="catalog-row">
                    <button
                      type="button"
                      className="tree-toggle"
                      onClick={() => toggleExpanded(table.name)}
                      aria-expanded={isExpanded}
                      aria-label={`${isExpanded ? "Collapse" : "Expand"} columns of ${table.name}`}
                    >
                      <ChevronDown className={isExpanded ? "chevron is-open" : "chevron"} />
                    </button>
                    <button
                      type="button"
                      className={isSelected ? "catalog-name is-selected" : "catalog-name"}
                      onClick={() => setSelectedTableName(isSelected ? null : table.name)}
                      title={`Profile ${table.name} (row count + column summary)`}
                    >
                      {table.name}
                    </button>
                    <span className="catalog-kind">{table.kind}</span>
                    <button
                      type="button"
                      className="btn btn-ghost btn-tiny insert-btn"
                      onClick={() => onInsertText(insertableSqlIdentifier(table.name))}
                      title={`Insert "${table.name}" into the editor`}
                      aria-label={`Insert ${table.name} into the editor`}
                    >
                      <Plus />
                    </button>
                  </div>
                  {isExpanded ? (
                    <ul className="catalog-columns">
                      {table.columns.map((column) => (
                        <li key={column.name}>
                          <button
                            type="button"
                            className="catalog-column"
                            onClick={() => onInsertText(insertableSqlIdentifier(column.name))}
                            title={`${column.type} — insert "${column.name}" into the editor`}
                          >
                            <span className="catalog-column-name">{column.name}</span>
                            <span className="catalog-column-type">{column.type}</span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  ) : null}
                </li>
              );
            })}
          </ul>
        )}
      </div>
      {selectedTableName ? (
        <TableSummarySection
          tableName={selectedTableName}
          onDeselect={() => setSelectedTableName(null)}
        />
      ) : (
        <p className="catalog-hint">Click a table name to profile it.</p>
      )}
    </aside>
  );
}
