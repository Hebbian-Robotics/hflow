import { useQuery } from "@tanstack/react-query";
import { Fragment, useEffect } from "react";
import { ApiError, fetchPipeline, type PipelineManifest, type PipelineStepManifest } from "../api";
import { EmptyPanel, ErrorPanel, LoadingPanel } from "../components/QueryStates";
import { VersionChip } from "../components/VersionChip";
import { formatTimestamp } from "../format";
import { ChevronDownIcon } from "../icons";

// Stage lanes mirror hflow.steps.Stage: sync runs the canonical transform and
// derived channels, meta runs checks + catalog registration, labels runs
// enrichments, media renders derived media (the engine's contact sheets).

interface LaneCard {
  name: string;
  kind: string;
  version: string | null;
  critical: boolean;
  uses: string | null;
  isEngineOwned: boolean;
}

interface StageLane {
  stage: string;
  title: string;
  description: string;
  cards: LaneCard[];
}

function cardFromStep(step: PipelineStepManifest): LaneCard {
  return {
    name: step.name,
    kind: step.kind,
    version: step.version,
    critical: step.critical,
    uses: step.uses,
    isEngineOwned: false,
  };
}

function engineCard(name: string, kind: string): LaneCard {
  return { name, kind, version: null, critical: false, uses: null, isEngineOwned: true };
}

function buildStageLanes(manifest: PipelineManifest): StageLane[] {
  const transformCard: LaneCard = {
    name: "canonical transform",
    kind: manifest.has_transform_override ? "transform override" : "engine transform",
    version: null,
    // The transform is the critical path: a failure quarantines the episode.
    critical: true,
    uses: null,
    isEngineOwned: !manifest.has_transform_override,
  };
  const derivedCards = manifest.derived_channels.map(
    (channel): LaneCard => ({
      name: channel.topic,
      kind: "derived channel",
      version: channel.version,
      critical: false,
      uses: null,
      isEngineOwned: false,
    }),
  );
  return [
    {
      stage: "sync",
      title: "Transform & sync",
      description: "canonical transform + derived channels (critical path)",
      cards: [transformCard, ...derivedCards],
    },
    {
      stage: "meta",
      title: "Metadata",
      description: "checks + catalog registration",
      cards: [...manifest.checks.map(cardFromStep), engineCard("catalog registration", "engine")],
    },
    {
      stage: "labels",
      title: "Labels & artifacts",
      description: "enrichments (non-critical)",
      cards: manifest.enrichments.map(cardFromStep),
    },
    {
      stage: "media",
      title: "Media",
      description: "derived media artifacts",
      cards: [engineCard("media/contact_sheet", "engine")],
    },
  ];
}

function StepCardView({ card }: { card: LaneCard }) {
  return (
    <div className={card.isEngineOwned ? "step-card step-card-engine" : "step-card"}>
      <div className="step-card-top">
        <span className="step-card-name" title={card.name}>
          {card.name}
        </span>
        {card.critical ? <span className="chip chip-err">critical</span> : null}
      </div>
      <div className="step-card-meta">
        <span className="step-card-kind">{card.kind}</span>
        {card.version ? <VersionChip version={card.version} /> : null}
        {card.uses ? (
          <span className="chip chip-accent" title={`declares endpoint alias ${card.uses}`}>
            {card.uses}
          </span>
        ) : null}
      </div>
    </div>
  );
}

export function PipelinePage() {
  useEffect(() => {
    document.title = "Pipeline · HFlow";
    return () => {
      document.title = "HFlow";
    };
  }, []);

  const pipelineQuery = useQuery({
    queryKey: ["pipeline"],
    queryFn: fetchPipeline,
    // 409 means "no --pipeline configured" — a stable state, not a transient failure.
    retry: (failureCount, error) =>
      !(error instanceof ApiError && error.status === 409) && failureCount < 1,
  });

  if (pipelineQuery.isPending) {
    return (
      <div className="pipeline-page">
        <LoadingPanel label="Loading the pipeline manifest…" />
      </div>
    );
  }

  if (pipelineQuery.isError) {
    const queryError = pipelineQuery.error;
    if (queryError instanceof ApiError && queryError.status === 409) {
      return (
        <div className="pipeline-page">
          <EmptyPanel title="No pipeline to show." hint={queryError.detail}>
            <p className="state-detail">
              Start the server with <code>hflow ui --pipeline path/to/pipeline.py</code> (append{" "}
              <code>:app</code> if the App object is not named <code>app</code>) to see its stage
              graph here. The file is imported once at startup, in your own environment.
            </p>
          </EmptyPanel>
        </div>
      );
    }
    return (
      <div className="pipeline-page">
        <ErrorPanel
          error={queryError}
          onRetry={() => {
            void pipelineQuery.refetch();
          }}
        />
      </div>
    );
  }

  const { manifest, observed, stale } = pipelineQuery.data;
  const stageLanes = buildStageLanes(manifest);

  return (
    <div className="pipeline-page">
      <div className="page-title-row">
        <h1 className="page-title">{manifest.pipeline_name}</h1>
        <VersionChip version={manifest.pipeline_version} />
      </div>
      <div className="meta-grid pipeline-meta">
        <div className="meta-item">
          <div className="meta-label">hflow version</div>
          <p className="meta-value">{manifest.hflow_version}</p>
        </div>
        <div className="meta-item">
          <div className="meta-label">schema version</div>
          <p className="meta-value">{manifest.schema_version}</p>
        </div>
        <div className="meta-item">
          <div className="meta-label">endpoint aliases</div>
          <p className="meta-value">
            {manifest.endpoint_aliases.length > 0 ? manifest.endpoint_aliases.join(", ") : "—"}
          </p>
        </div>
      </div>

      {stale && stale.count > 0 ? (
        <div className="stale-banner" role="status">
          <strong>
            {stale.count} episode{stale.count === 1 ? " is" : "s are"} stale
          </strong>{" "}
          against pipeline version <code className="cell-mono">{stale.pipeline_version}</code> —
          last processed by an older pipeline. Re-run ingest (profile <code>metadata_backfill</code>{" "}
          or <code>relabel</code>) to refresh them.
        </div>
      ) : null}

      <section className="section">
        <h2 className="section-title">Stage graph</h2>
        <div className="pipeline-lanes">
          {stageLanes.map((lane, laneIndex) => (
            <Fragment key={lane.stage}>
              {laneIndex > 0 ? (
                <div className="lane-arrow" aria-hidden="true">
                  <ChevronDownIcon className="lane-arrow-icon" />
                </div>
              ) : null}
              <div className="pipeline-lane">
                <div className="lane-header">
                  <span className="lane-stage">{lane.stage}</span>
                  <span className="lane-title">{lane.title}</span>
                </div>
                <p className="lane-description">{lane.description}</p>
                {lane.cards.length === 0 ? (
                  <p className="empty-note">nothing registered</p>
                ) : (
                  <div className="lane-cards">
                    {lane.cards.map((card) => (
                      <StepCardView key={card.name} card={card} />
                    ))}
                  </div>
                )}
              </div>
            </Fragment>
          ))}
        </div>
      </section>

      <section className="section">
        <h2 className="section-title">Observed versions</h2>
        {observed.length === 0 ? (
          <p className="empty-note">
            No check runs recorded in this catalog yet — observed step versions appear after an
            ingest.
          </p>
        ) : (
          <div className="table-overflow">
            <table className="evidence-table">
              <thead>
                <tr>
                  <th>check</th>
                  <th>version</th>
                  <th>first seen</th>
                  <th>last seen</th>
                  <th>runs</th>
                </tr>
              </thead>
              <tbody>
                {observed.map((entry) => (
                  <tr key={`${entry.check_name}@${entry.check_version}`}>
                    <td className="cell-mono">{entry.check_name}</td>
                    <td>
                      <VersionChip version={entry.check_version} />
                    </td>
                    <td className="cell-mono">{formatTimestamp(entry.first_seen)}</td>
                    <td className="cell-mono">{formatTimestamp(entry.last_seen)}</td>
                    <td className="cell-num">{entry.run_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
