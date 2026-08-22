import { ChartNoAxesColumn } from "lucide-react";
import { Popover } from "radix-ui";
import type { CategoricalColumnStats, EpisodeColumnStats, NumericColumnStats } from "../api";
import { formatNumber } from "../format";

// Column-header mini-distribution: a small popover fed by /episodes/stats,
// always computed over the ACTIVE filter set. For filterable categorical
// columns a value click applies the structured filter param — the server
// keeps compiling the SQL (thin-client rule).
//
// Radix Popover replaces hand-rolled positioning that only ever flipped on the
// right edge, never repositioned on scroll or resize, and rendered in place —
// so the panel was clipped by the table's own overflow:hidden. Radix portals
// it to the body and floating-ui keeps it in view on every axis.
//
// It also collapses the old hover-to-peek / click-to-pin pair into one click:
// once the panel is portaled, hover intent across the gap between trigger and
// panel needs its own timers, and hovering never let a keyboard user in. One
// click, Escape or an outside click to dismiss, focus returned to the trigger.

function NumericHistogram({ stats }: { stats: NumericColumnStats }) {
  if (stats.buckets.length === 0) {
    return <p className="stats-popover-note">no histogram for this column</p>;
  }
  const maxCount = Math.max(...stats.buckets.map((bucket) => bucket.count), 1);
  const rangeMin = stats.min ?? stats.buckets[0]?.lo ?? null;
  const rangeMax = stats.max ?? stats.buckets[stats.buckets.length - 1]?.hi ?? null;
  return (
    <div>
      <div className="histogram">
        {stats.buckets.map((bucket) => (
          <div
            key={`${bucket.lo}-${bucket.hi}`}
            className="histogram-bar"
            style={{ height: `${Math.max(4, (bucket.count / maxCount) * 100)}%` }}
            title={`${formatNumber(bucket.lo)} – ${formatNumber(bucket.hi)}: ${bucket.count}`}
          />
        ))}
      </div>
      <div className="histogram-range">
        <span>{rangeMin === null ? "—" : formatNumber(rangeMin)}</span>
        <span>{rangeMax === null ? "—" : formatNumber(rangeMax)}</span>
      </div>
    </div>
  );
}

function CategoricalValues({
  stats,
  onSelectValue,
  activeValues,
}: {
  stats: CategoricalColumnStats;
  onSelectValue: ((value: string) => void) | null;
  activeValues: readonly string[];
}) {
  if (stats.values.length === 0) {
    return <p className="stats-popover-note">no values for this column</p>;
  }
  const maxCount = Math.max(...stats.values.map((entry) => entry.count), 1);
  const otherCount = stats.other_count ?? 0;
  return (
    <ul className="stats-value-list">
      {stats.values.map((entry) => {
        // "other" is the server's rollup of the long tail, not a real category
        // — never a click-to-filter target.
        const isClickable = onSelectValue !== null && entry.value !== "other";
        const isActive = activeValues.includes(entry.value);
        const rowBody = (
          <>
            <span className="stats-value-name" title={entry.value}>
              {entry.value}
            </span>
            <span className="stats-value-bar" aria-hidden="true">
              <span
                className="stats-value-bar-fill"
                style={{ width: `${(entry.count / maxCount) * 100}%` }}
              />
            </span>
            <span className="stats-value-count">{entry.count}</span>
          </>
        );
        return (
          <li key={entry.value}>
            {isClickable ? (
              <button
                type="button"
                className={isActive ? "stats-value-row is-active" : "stats-value-row"}
                onClick={() => onSelectValue(entry.value)}
                title={
                  isActive
                    ? `Remove the ${stats.name} = ${entry.value} filter`
                    : `Filter to ${stats.name} = ${entry.value}`
                }
              >
                {rowBody}
              </button>
            ) : (
              <span className="stats-value-row stats-value-row-static">{rowBody}</span>
            )}
          </li>
        );
      })}
      {otherCount > 0 ? (
        <li>
          <span className="stats-value-row stats-value-row-static">
            <span className="stats-value-name">other</span>
            <span className="stats-value-bar" aria-hidden="true">
              <span
                className="stats-value-bar-fill"
                style={{ width: `${Math.min(100, (otherCount / maxCount) * 100)}%` }}
              />
            </span>
            <span className="stats-value-count">{otherCount}</span>
          </span>
        </li>
      ) : null}
    </ul>
  );
}

export function HeaderDistribution({
  stats,
  onSelectValue,
  activeValues,
}: {
  stats: EpisodeColumnStats;
  /** null for columns without a matching structured filter param. */
  onSelectValue: ((value: string) => void) | null;
  activeValues: readonly string[];
}) {
  const label = `Distribution of ${stats.name} under the active filters`;
  return (
    <Popover.Root>
      <div className="th-stats">
        <Popover.Trigger className="th-stats-trigger" aria-label={label} title={label}>
          <ChartNoAxesColumn />
        </Popover.Trigger>
      </div>
      <Popover.Portal>
        <Popover.Content
          className="stats-popover"
          side="bottom"
          align="start"
          sideOffset={2}
          collisionPadding={8}
        >
          <div className="stats-popover-header">
            <span className="stats-popover-name">{stats.name}</span>
            <span className="stats-popover-kind">{stats.kind} · active filters</span>
          </div>
          {stats.kind === "numeric" ? (
            <NumericHistogram stats={stats} />
          ) : (
            <CategoricalValues
              stats={stats}
              onSelectValue={onSelectValue}
              activeValues={activeValues}
            />
          )}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
