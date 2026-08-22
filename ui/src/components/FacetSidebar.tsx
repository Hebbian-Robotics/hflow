import { useState } from "react";
import type { EpisodeFacets, EpisodeStatus, FacetEntry } from "../api";
import { ErrorPanel, LoadingPanel } from "./QueryStates";

export type MultiValueFacetName = "task" | "operator" | "embodiment";

const VISIBLE_ENTRY_LIMIT = 8;

function isEpisodeStatus(value: string): value is EpisodeStatus {
  return value === "ok" || value === "quarantined";
}

function ShowMoreToggle({
  totalCount,
  showAll,
  onToggle,
}: {
  totalCount: number;
  showAll: boolean;
  onToggle: () => void;
}) {
  if (totalCount <= VISIBLE_ENTRY_LIMIT) return null;
  return (
    <button type="button" className="facet-more" onClick={onToggle}>
      {showAll ? "Show fewer" : `Show all ${totalCount}`}
    </button>
  );
}

function CheckboxFacetGroup({
  label,
  entries,
  selectedValues,
  onToggleValue,
}: {
  label: string;
  entries: FacetEntry[];
  selectedValues: string[];
  onToggleValue: (value: string) => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const visibleEntries = showAll ? entries : entries.slice(0, VISIBLE_ENTRY_LIMIT);
  return (
    <section className="facet-group">
      <h3 className="facet-label">{label}</h3>
      {entries.length === 0 ? <p className="facet-empty">none recorded</p> : null}
      <ul className="facet-list">
        {visibleEntries.map((entry) => (
          <li key={entry.value}>
            <label className="facet-row">
              <input
                type="checkbox"
                checked={selectedValues.includes(entry.value)}
                onChange={() => onToggleValue(entry.value)}
              />
              <span className="facet-value" title={entry.value}>
                {entry.value}
              </span>
              <span className="facet-count">{entry.count}</span>
            </label>
          </li>
        ))}
      </ul>
      <ShowMoreToggle
        totalCount={entries.length}
        showAll={showAll}
        onToggle={() => setShowAll((previous) => !previous)}
      />
    </section>
  );
}

// The list endpoint's status param is single-valued, so this group behaves
// like radio buttons where re-checking the active value clears the filter.
function StatusFacetGroup({
  entries,
  selectedStatus,
  onSelectStatus,
}: {
  entries: FacetEntry[];
  selectedStatus: EpisodeStatus | null;
  onSelectStatus: (status: EpisodeStatus | null) => void;
}) {
  return (
    <section className="facet-group">
      <h3 className="facet-label">Status</h3>
      {entries.length === 0 ? <p className="facet-empty">none recorded</p> : null}
      <ul className="facet-list">
        {entries.map((entry) => {
          const filterable = isEpisodeStatus(entry.value);
          return (
            <li key={entry.value}>
              <label className="facet-row">
                <input
                  type="checkbox"
                  checked={selectedStatus === entry.value}
                  disabled={!filterable}
                  onChange={() => {
                    if (!isEpisodeStatus(entry.value)) return;
                    onSelectStatus(selectedStatus === entry.value ? null : entry.value);
                  }}
                />
                <span className="facet-value" title={entry.value}>
                  {entry.value}
                </span>
                <span className="facet-count">{entry.count}</span>
              </label>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

// pipeline_version is a facet in the counts response but not a list filter in
// M0, so it renders as informational counts without checkboxes.
function CountOnlyFacetGroup({ label, entries }: { label: string; entries: FacetEntry[] }) {
  const [showAll, setShowAll] = useState(false);
  const visibleEntries = showAll ? entries : entries.slice(0, VISIBLE_ENTRY_LIMIT);
  return (
    <section className="facet-group" title="Counts only — not a filter in this release">
      <h3 className="facet-label">{label}</h3>
      {entries.length === 0 ? <p className="facet-empty">none recorded</p> : null}
      <ul className="facet-list">
        {visibleEntries.map((entry) => (
          <li key={entry.value} className="facet-row facet-row-static">
            <span className="facet-value" title={entry.value}>
              {entry.value}
            </span>
            <span className="facet-count">{entry.count}</span>
          </li>
        ))}
      </ul>
      <ShowMoreToggle
        totalCount={entries.length}
        showAll={showAll}
        onToggle={() => setShowAll((previous) => !previous)}
      />
    </section>
  );
}

export function FacetSidebar({
  facets,
  isPending,
  error,
  onRetry,
  selectedValues,
  selectedStatus,
  onToggleValue,
  onSelectStatus,
}: {
  facets: EpisodeFacets | undefined;
  isPending: boolean;
  error: unknown;
  onRetry: () => void;
  selectedValues: Record<MultiValueFacetName, string[]>;
  selectedStatus: EpisodeStatus | null;
  onToggleValue: (facet: MultiValueFacetName, value: string) => void;
  onSelectStatus: (status: EpisodeStatus | null) => void;
}) {
  if (isPending) {
    return (
      <aside className="facet-sidebar" aria-label="Episode filters">
        <LoadingPanel label="Loading facets…" />
      </aside>
    );
  }
  if (error) {
    return (
      <aside className="facet-sidebar" aria-label="Episode filters">
        <ErrorPanel error={error} onRetry={onRetry} />
      </aside>
    );
  }
  if (!facets) return null;
  return (
    <aside className="facet-sidebar" aria-label="Episode filters">
      <CheckboxFacetGroup
        label="Task"
        entries={facets.task ?? []}
        selectedValues={selectedValues.task}
        onToggleValue={(value) => onToggleValue("task", value)}
      />
      <CheckboxFacetGroup
        label="Operator"
        entries={facets.operator ?? []}
        selectedValues={selectedValues.operator}
        onToggleValue={(value) => onToggleValue("operator", value)}
      />
      <CheckboxFacetGroup
        label="Embodiment"
        entries={facets.embodiment ?? []}
        selectedValues={selectedValues.embodiment}
        onToggleValue={(value) => onToggleValue("embodiment", value)}
      />
      <StatusFacetGroup
        entries={facets.status ?? []}
        selectedStatus={selectedStatus}
        onSelectStatus={onSelectStatus}
      />
      <CountOnlyFacetGroup label="Pipeline version" entries={facets.pipeline_version ?? []} />
    </aside>
  );
}
