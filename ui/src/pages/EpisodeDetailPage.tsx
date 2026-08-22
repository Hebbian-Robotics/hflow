import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Download } from "lucide-react";
import { Fragment, type ReactNode, useEffect, useMemo, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import {
  ApiError,
  type EpisodeCheckRun,
  type EpisodeDossier,
  type EpisodeInterval,
  type EpisodeMeasurement,
  type EpisodeMediaItem,
  type EpisodeRow,
  type EpisodeTagRecord,
  type EpisodeTimeline,
  fetchEpisodeDossier,
  fetchEpisodeTimeline,
} from "../api";
import { MeasurementBars, TimelineStrip } from "../components/EpisodeTimeline";
import { EmptyPanel, ErrorPanel, LoadingPanel } from "../components/QueryStates";
import { StatusChip } from "../components/StatusChip";
import { ValueCell } from "../components/ValueCell";
import {
  formatDurationSeconds,
  formatTimestamp,
  historyRowKey,
  measurementDisplayValue,
  nanosecondsToRelativeSeconds,
  shortFingerprint,
} from "../format";

const HEADER_FIELDS = [
  "task",
  "operator",
  "embodiment",
  "success",
  "pipeline_version",
  "hflow_version",
  "recorded_at",
];

function SectionShell({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="section">
      <h2 className="section-title">{title}</h2>
      {children}
    </section>
  );
}

function MediaSection({ media }: { media: EpisodeMediaItem[] }) {
  if (media.length === 0) {
    return (
      <SectionShell title="Contact sheet">
        <p className="empty-note">No media artifacts recorded for this episode.</p>
      </SectionShell>
    );
  }
  return (
    <SectionShell title="Contact sheet">
      <div className="media-row">
        {media.map((item) =>
          item.url ? (
            <a
              key={item.name}
              className="media-item"
              href={item.url}
              target="_blank"
              rel="noreferrer"
              title={`Open ${item.name} full-size`}
            >
              <img src={item.url} alt={item.name} loading="lazy" />
              <span className="media-name">{item.name}</span>
            </a>
          ) : (
            <div key={item.name} className="media-item media-item-unservable">
              <span className="media-name">{item.name}</span>
              <span className="media-uri" title={item.uri}>
                {item.uri}
              </span>
              <span className="empty-note">not servable from this server</span>
            </div>
          ),
        )}
      </div>
    </SectionShell>
  );
}

/** The visual half of the evidence: where things happened in the recording,
 * and how big the numbers are. Both come from /episodes/{id}/timeline, which
 * derives the span server-side. */
function TimelineSections({ timeline }: { timeline: EpisodeTimeline }) {
  return (
    <>
      <SectionShell title="Timeline">
        <TimelineStrip timeline={timeline} />
      </SectionShell>
      <SectionShell title="Measurements at a glance">
        <MeasurementBars measurements={timeline.measurements} />
      </SectionShell>
    </>
  );
}

function VisualEvidence({ episodeId }: { episodeId: string }) {
  const timelineQuery = useQuery({
    queryKey: ["episode-timeline", episodeId],
    queryFn: () => fetchEpisodeTimeline(episodeId),
    // A 404 here means the server predates the timeline endpoint, not a bad
    // episode id (the dossier above already resolved it) — retrying is pointless.
    retry: (failureCount, error) =>
      !(error instanceof ApiError && error.status === 404) && failureCount < 1,
  });

  if (timelineQuery.isPending) {
    return (
      <SectionShell title="Timeline">
        <LoadingPanel label="Loading the episode timeline…" />
      </SectionShell>
    );
  }
  if (timelineQuery.isError) {
    if (timelineQuery.error instanceof ApiError && timelineQuery.error.status === 404) {
      return (
        <SectionShell title="Timeline">
          <p className="empty-note">
            This server does not serve episode timelines yet — upgrade hflow-server to see interval
            bands and measurement bars here.
          </p>
        </SectionShell>
      );
    }
    return (
      <SectionShell title="Timeline">
        <ErrorPanel
          error={timelineQuery.error}
          onRetry={() => {
            void timelineQuery.refetch();
          }}
        />
      </SectionShell>
    );
  }
  return <TimelineSections timeline={timelineQuery.data} />;
}

function CheckRunsSection({ checkRuns }: { checkRuns: EpisodeCheckRun[] }) {
  const [expandedErrorKeys, setExpandedErrorKeys] = useState<ReadonlySet<string>>(new Set());
  if (checkRuns.length === 0) {
    return (
      <SectionShell title="Check runs">
        <p className="empty-note">No quality checks have run against this episode.</p>
      </SectionShell>
    );
  }
  const toggleError = (runKey: string) => {
    setExpandedErrorKeys((previous) => {
      const next = new Set(previous);
      if (next.has(runKey)) next.delete(runKey);
      else next.add(runKey);
      return next;
    });
  };
  return (
    <SectionShell title="Check runs">
      <div className="table-overflow">
        <table className="evidence-table">
          <thead>
            <tr>
              <th>check</th>
              <th>version</th>
              <th>critical</th>
              <th>status</th>
              <th>duration</th>
              <th>recorded</th>
              <th>fingerprint</th>
              <th>error</th>
            </tr>
          </thead>
          <tbody>
            {checkRuns.map((run) => {
              const runKey = `${run.check_name}@${run.check_version}:${run.run_fingerprint}`;
              const isErrorExpanded = expandedErrorKeys.has(runKey);
              return (
                <Fragment key={runKey}>
                  <tr>
                    <td className="cell-mono">{run.check_name}</td>
                    <td className="cell-mono">{run.check_version}</td>
                    <td>
                      {run.critical ? (
                        <span className="chip chip-muted">critical</span>
                      ) : (
                        <span className="cell-null">—</span>
                      )}
                    </td>
                    <td>
                      <StatusChip status={run.status} />
                    </td>
                    <td className="cell-num">{formatDurationSeconds(run.duration_s)}</td>
                    <td className="cell-mono" title={run.recorded_at}>
                      {formatTimestamp(run.recorded_at)}
                    </td>
                    <td className="cell-mono" title={run.run_fingerprint}>
                      {shortFingerprint(run.run_fingerprint)}
                    </td>
                    <td>
                      {run.error ? (
                        <button
                          type="button"
                          className="btn btn-ghost btn-tiny"
                          onClick={() => toggleError(runKey)}
                          aria-expanded={isErrorExpanded}
                        >
                          {isErrorExpanded ? "Hide error" : "Show error"}
                        </button>
                      ) : (
                        <span className="cell-null">—</span>
                      )}
                    </td>
                  </tr>
                  {run.error && isErrorExpanded ? (
                    <tr className="error-detail-row">
                      <td colSpan={8}>
                        <pre className="error-text">{run.error}</pre>
                      </td>
                    </tr>
                  ) : null}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </SectionShell>
  );
}

function MeasurementsSection({ measurements }: { measurements: EpisodeMeasurement[] }) {
  if (measurements.length === 0) {
    return (
      <SectionShell title="Measurements">
        <p className="empty-note">No measurements recorded for this episode.</p>
      </SectionShell>
    );
  }
  return (
    <SectionShell title="Measurements">
      <div className="table-overflow">
        <table className="evidence-table">
          <thead>
            <tr>
              <th>key</th>
              <th>value</th>
              <th>producer</th>
              <th>recorded</th>
            </tr>
          </thead>
          <tbody>
            {measurements.map((measurement) => (
              <tr key={`${measurement.key}:${measurement.check_name}@${measurement.check_version}`}>
                <td className="cell-mono">{measurement.key}</td>
                <td className="cell-num">{measurementDisplayValue(measurement)}</td>
                <td className="cell-mono cell-dim">
                  {measurement.check_name}@{measurement.check_version}
                </td>
                <td className="cell-mono" title={measurement.recorded_at}>
                  {formatTimestamp(measurement.recorded_at)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </SectionShell>
  );
}

function IntervalsSection({ intervals }: { intervals: EpisodeInterval[] }) {
  if (intervals.length === 0) {
    return (
      <SectionShell title="Intervals">
        <p className="empty-note">No intervals recorded for this episode.</p>
      </SectionShell>
    );
  }
  // Times display in seconds relative to the earliest interval start.
  const originNs = Math.min(...intervals.map((interval) => interval.start_ns));
  return (
    <SectionShell title="Intervals">
      <div className="table-overflow">
        <table className="evidence-table">
          <thead>
            <tr>
              <th>label</th>
              <th>start (s)</th>
              <th>end (s)</th>
              <th>length (s)</th>
              <th>producer</th>
            </tr>
          </thead>
          <tbody>
            {intervals.map((interval) => (
              <tr key={`${interval.label}:${interval.start_ns}:${interval.check_name}`}>
                <td className="cell-mono">{interval.label}</td>
                <td className="cell-num">
                  {nanosecondsToRelativeSeconds(interval.start_ns, originNs)}
                </td>
                <td className="cell-num">
                  {nanosecondsToRelativeSeconds(interval.end_ns, originNs)}
                </td>
                <td className="cell-num">
                  {nanosecondsToRelativeSeconds(interval.end_ns, interval.start_ns)}
                </td>
                <td className="cell-mono cell-dim">
                  {/* check_version comes from a LEFT JOIN and can be null. */}
                  {interval.check_version
                    ? `${interval.check_name}@${interval.check_version}`
                    : interval.check_name}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </SectionShell>
  );
}

function TagsSection({ tags }: { tags: EpisodeTagRecord[] }) {
  if (tags.length === 0) {
    return (
      <SectionShell title="Tags">
        <p className="empty-note">No tags recorded for this episode.</p>
      </SectionShell>
    );
  }
  return (
    <SectionShell title="Tags">
      <div className="tag-row">
        {tags.map((tagRecord) => (
          <span
            key={`${tagRecord.tag}:${tagRecord.check_name}`}
            className="chip chip-muted"
            title={`by ${tagRecord.check_name} at ${tagRecord.recorded_at}`}
          >
            {tagRecord.tag}
          </span>
        ))}
      </div>
    </SectionShell>
  );
}

function HistorySection({ history }: { history: EpisodeRow[] }) {
  const columnNames = useMemo(() => {
    const firstRow = history[0];
    return firstRow ? Object.keys(firstRow) : [];
  }, [history]);
  // Content-derived keys, deduplicated in order, so rows never key off indexes.
  const keyedHistoryRows = useMemo(() => {
    const seenCounts = new Map<string, number>();
    return history.map((row) => {
      const baseKey = historyRowKey(row);
      const seenCount = seenCounts.get(baseKey) ?? 0;
      seenCounts.set(baseKey, seenCount + 1);
      return { rowKey: seenCount === 0 ? baseKey : `${baseKey}#${seenCount}`, row };
    });
  }, [history]);
  if (history.length === 0) {
    return (
      <SectionShell title="History">
        <p className="empty-note">No raw catalog rows for this episode.</p>
      </SectionShell>
    );
  }
  return (
    <SectionShell title={`History · ${history.length} raw row${history.length === 1 ? "" : "s"}`}>
      <div className="table-overflow">
        <table className="evidence-table">
          <thead>
            <tr>
              {columnNames.map((name) => (
                <th key={name}>{name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {keyedHistoryRows.map(({ rowKey, row }) => (
              <tr key={rowKey}>
                {columnNames.map((name) => (
                  <td key={name}>
                    <ValueCell columnName={name} value={row[name]} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </SectionShell>
  );
}

function DossierView({ dossier }: { dossier: EpisodeDossier }) {
  const episodeIdText =
    typeof dossier.episode.episode_id === "string" ? dossier.episode.episode_id : "episode";
  return (
    <>
      <header className="episode-header">
        <div className="episode-title-row">
          <h1 className="episode-title">{episodeIdText}</h1>
          <StatusChip status={dossier.episode.status} />
          {dossier.episode.quarantine_tags.map((tag) => (
            <span key={tag} className="chip chip-warn">
              {tag}
            </span>
          ))}
          <div className="toolbar-spacer" />
          {dossier.canonical_url ? (
            <a className="btn" href={dossier.canonical_url} download>
              <Download />
              <span>Download canonical</span>
            </a>
          ) : null}
        </div>
        <dl className="meta-grid">
          {HEADER_FIELDS.map((field) => {
            const value = dossier.episode[field];
            if (value === null || value === undefined) return null;
            return (
              <div key={field} className="meta-item">
                <dt className="meta-label">{field}</dt>
                <dd className="meta-value">
                  <ValueCell columnName={field} value={value} />
                </dd>
              </div>
            );
          })}
        </dl>
      </header>

      <MediaSection media={dossier.media} />

      <VisualEvidence episodeId={episodeIdText} />

      <CheckRunsSection checkRuns={dossier.check_runs} />
      <MeasurementsSection measurements={dossier.measurements} />
      <IntervalsSection intervals={dossier.intervals} />
      <TagsSection tags={dossier.tags} />
      <HistorySection history={dossier.history} />

      <details className="raw-json">
        <summary>Raw JSON</summary>
        <pre>{JSON.stringify(dossier, null, 2)}</pre>
      </details>
    </>
  );
}

export function EpisodeDetailPage() {
  const { episodeId } = useParams<{ episodeId: string }>();
  // Row navigation carried the Episodes URL's search string onto this route,
  // so the back link can restore the exact filters/sort/page.
  const location = useLocation();
  const dossierQuery = useQuery({
    queryKey: ["episode", episodeId],
    queryFn: () => fetchEpisodeDossier(episodeId ?? ""),
    enabled: episodeId !== undefined,
  });

  useEffect(() => {
    document.title = episodeId ? `${episodeId} · HFlow` : "HFlow";
    return () => {
      document.title = "HFlow";
    };
  }, [episodeId]);

  return (
    <div className="episode-page">
      <Link to={{ pathname: "/", search: location.search }} className="back-link">
        <ArrowLeft />
        <span>Episodes</span>
      </Link>
      {episodeId === undefined ? (
        <EmptyPanel title="No episode selected." hint="Pick an episode from the list." />
      ) : dossierQuery.isPending ? (
        <LoadingPanel label="Loading episode…" />
      ) : dossierQuery.isError ? (
        <ErrorPanel
          error={dossierQuery.error}
          onRetry={() => {
            void dossierQuery.refetch();
          }}
        />
      ) : (
        <DossierView dossier={dossierQuery.data} />
      )}
    </div>
  );
}
