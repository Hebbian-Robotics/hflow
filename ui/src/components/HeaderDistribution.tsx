import { useEffect, useRef, useState } from "react";
import type { CategoricalColumnStats, EpisodeColumnStats, NumericColumnStats } from "../api";
import { formatNumber } from "../format";
import { DistributionIcon } from "../icons";

// Column-header mini-distribution: a small popover fed by /episodes/stats,
// always computed over the ACTIVE filter set. Opens on hover; a click pins it.
// For filterable categorical columns a value click applies the structured
// filter param — the server keeps compiling the SQL (thin-client rule).

/** Mirrors the .stats-popover width so edge flipping matches the CSS. */
const POPOVER_WIDTH_PX = 236;

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
  // Hover/focus visibility is pure CSS (.th-stats:hover / :focus-within show
  // the popover); state only tracks the click-pin, so no static element needs
  // mouse handlers and keyboard users get the same affordance via the button.
  const [isPinned, setIsPinned] = useState(false);
  const [isRightAligned, setIsRightAligned] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Flip the popover leftward for headers near the scroll container's right
  // edge, where the default left-aligned panel would be clipped.
  const updatePopoverAlignment = () => {
    const container = containerRef.current;
    if (!container) return;
    const scrollParent = container.closest(".table-scroll");
    const rightBound = scrollParent
      ? scrollParent.getBoundingClientRect().right
      : window.innerWidth;
    const triggerLeft = container.getBoundingClientRect().left;
    setIsRightAligned(triggerLeft + POPOVER_WIDTH_PX > rightBound - 8);
  };

  useEffect(() => {
    if (!isPinned) return;
    const closeOnOutsideClick = (event: MouseEvent) => {
      const container = containerRef.current;
      if (container && event.target instanceof Node && !container.contains(event.target)) {
        setIsPinned(false);
      }
    };
    document.addEventListener("mousedown", closeOnOutsideClick);
    return () => document.removeEventListener("mousedown", closeOnOutsideClick);
  }, [isPinned]);

  return (
    <div className={isPinned ? "th-stats is-pinned" : "th-stats"} ref={containerRef}>
      <button
        type="button"
        className="th-stats-trigger"
        onMouseEnter={updatePopoverAlignment}
        onFocus={updatePopoverAlignment}
        onClick={() => {
          updatePopoverAlignment();
          setIsPinned((previous) => !previous);
        }}
        aria-expanded={isPinned}
        aria-label={`Distribution of ${stats.name} under the active filters`}
        title={`Distribution of ${stats.name} under the active filters`}
      >
        <DistributionIcon />
      </button>
      <div className={isRightAligned ? "stats-popover is-right-aligned" : "stats-popover"}>
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
      </div>
    </div>
  );
}
