import { useQuery } from "@tanstack/react-query";
import { ChevronDown } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  ApiError,
  type DagTopology,
  fetchPipeline,
  fetchPipelineGraph,
  type PipelineGraphResponse,
  type PipelineGraphStage,
  type PipelineResponse,
  type QuarantineGate,
} from "../api";
import {
  contentKey,
  DagGraph,
  type DagGraphEdge,
  type DagGraphNode,
  type DagGraphNodeContent,
} from "../components/DagGraph";
import { DetailBlock, DetailRow, DetailsPanel } from "../components/DetailsPanel";
import { EmptyPanel, ErrorPanel, LoadingPanel } from "../components/QueryStates";
import { VersionChip } from "../components/VersionChip";
import { formatTimestamp } from "../format";

// Two nested layers meet on this page and the UI must not conflate them:
//   1. ORCHESTRATION — a real DAG with real edges (the master chain, and each
//      stage's plan -> process_batch -> budget gate sub-DAG).
//   2. USER STEPS — registered checks/enrichments that run INSIDE one
//      process_batch task with NO edges between them, ordered only by the
//      engine's two-tier cheap-first policy. Drawing arrows between them would
//      be a lie, so this page draws groups and says so in words.
// The one real cross-step edge is the quarantine gate, and it IS drawn.

type PipelineSelection =
  | { kind: "master-task"; taskId: string }
  | { kind: "stage-task"; stage: string; taskId: string }
  | { kind: "user-step"; stage: string; name: string }
  | { kind: "engine-step"; stage: string; name: string }
  | { kind: "quarantine" };

function topologyNodes(topology: DagTopology): DagGraphNode[] {
  return topology.tasks.map((task) => ({
    id: task.task_id,
    label: task.task_id,
    summary: task.summary,
    mapped: task.mapped,
    deferred: task.deferred,
  }));
}

function topologyEdges(topology: DagTopology): DagGraphEdge[] {
  return topology.edges.map(([from, to]) => ({ from, to }));
}

/** The one dynamically mapped task in a stage sub-DAG: the node whose insides
 * are the steps. Taken from the served topology rather than named here. */
function mappedTaskId(stage: PipelineGraphStage): string | null {
  return stage.dag.tasks.find((task) => task.mapped)?.task_id ?? null;
}

/** The quarantine gate as connectors: from the stage that runs the critical
 * checks into the gate task of every stage its verdict can skip. */
function quarantineEdges(
  graph: PipelineGraphResponse,
  gate: QuarantineGate | null,
): DagGraphEdge[] {
  if (!gate) return [];
  const sourceStage = graph.stages.find((stage) => stage.stage === gate.from_stage);
  if (!sourceStage) return [];
  return gate.to_stages.flatMap((stageName) => {
    const target = graph.stages.find((stage) => stage.stage === stageName);
    if (!target) return [];
    return [
      {
        from: sourceStage.trigger_task_id,
        to: target.gate_task_id,
        kind: "gate" as const,
        label: "quarantine gate",
      },
    ];
  });
}

function stageOfMasterTask(
  graph: PipelineGraphResponse,
  taskId: string,
): PipelineGraphStage | undefined {
  return graph.stages.find(
    (stage) => stage.gate_task_id === taskId || stage.trigger_task_id === taskId,
  );
}

function StageLane({
  stage,
  graph,
  isExpanded,
  onToggle,
  selection,
  onSelect,
  derivedChannels,
}: {
  stage: PipelineGraphStage;
  graph: PipelineGraphResponse;
  isExpanded: boolean;
  onToggle: () => void;
  selection: PipelineSelection | null;
  onSelect: (selection: PipelineSelection) => void;
  derivedChannels: { topic: string; version: string }[];
}) {
  const gate = graph.quarantine_gate;
  const isQuarantineGated = gate ? gate.to_stages.includes(stage.stage) : false;
  const isQuarantineSource = gate ? gate.from_stage === stage.stage : false;
  const tierOneSteps = stage.user_steps.filter((step) => step.tier === 1);
  const tierTwoSteps = stage.user_steps.filter((step) => step.tier === 2);
  // What process_batch actually runs, drawn INSIDE it. Engine work first (the
  // transform, the catalog append), then the user's steps in the engine's own
  // cheap-first order -- the same order and the same source as the cards below.
  const batchContents: DagGraphNodeContent[] = graph.steps_known
    ? [
        ...stage.engine_steps.map((step) => ({
          id: `engine:${step.name}`,
          label: step.name,
          summary: step.summary,
          kind: "engine" as const,
          badge: "engine",
        })),
        ...stage.user_steps.map((step) => ({
          id: `user:${step.name}`,
          label: step.name,
          summary: `${step.kind}${step.critical ? ", critical" : ""} — runs in tier ${step.tier}`,
          kind: "user" as const,
          badge: step.critical ? "critical" : `t${step.tier}`,
        })),
      ]
    : [];
  const subDagNodes = topologyNodes(stage.dag).map((node) =>
    node.mapped
      ? {
          ...node,
          badge: "×N",
          summary: `${node.summary} — one mapped instance per planned batch`,
          contents: batchContents,
          contentsNote: batchContents.length > 1 ? "one process; no edges between them" : null,
        }
      : node,
  );
  const selectedContentKey =
    selection?.kind === "user-step" && selection.stage === stage.stage
      ? contentKey(mappedTaskId(stage) ?? "", `user:${selection.name}`)
      : selection?.kind === "engine-step" && selection.stage === stage.stage
        ? contentKey(mappedTaskId(stage) ?? "", `engine:${selection.name}`)
        : null;

  return (
    <div className={isExpanded ? "stage-lane-card is-expanded" : "stage-lane-card"}>
      <button
        type="button"
        className="stage-lane-head"
        onClick={onToggle}
        aria-expanded={isExpanded}
      >
        <ChevronDown className={isExpanded ? "chevron is-open" : "chevron"} />
        <span className="lane-stage">{stage.stage}</span>
        <span className="lane-title">{stage.title}</span>
        <span className="lane-description">{stage.description}</span>
        <span className="toolbar-spacer" />
        {isQuarantineSource ? <span className="chip chip-warn">sets quarantine</span> : null}
        {isQuarantineGated ? <span className="chip chip-warn">quarantine-gated</span> : null}
        <span className="lane-step-count">
          {graph.steps_known
            ? `${stage.user_steps.length} step${stage.user_steps.length === 1 ? "" : "s"}`
            : "steps unknown"}
        </span>
      </button>
      {isExpanded ? (
        <div className="stage-lane-body">
          <div className="stage-lane-meta">
            <span className="stage-meta-item">
              <span className="meta-label">sub-dag</span>
              <code className="cell-mono">{stage.dag.dag_id}</code>
            </span>
            <span className="stage-meta-item">
              <span className="meta-label">enabled by profiles</span>
              <span className="profile-chips">
                {stage.enabling_profiles.length === 0 ? (
                  <span className="empty-note">none</span>
                ) : (
                  stage.enabling_profiles.map((profile) => (
                    <span key={profile} className="chip chip-muted">
                      {profile}
                    </span>
                  ))
                )}
              </span>
            </span>
          </div>
          <DagGraph
            nodes={subDagNodes}
            edges={topologyEdges(stage.dag)}
            label={`${stage.stage} sub-DAG`}
            selectedNodeId={
              selection?.kind === "stage-task" && selection.stage === stage.stage
                ? selection.taskId
                : null
            }
            onSelectNode={(taskId) => onSelect({ kind: "stage-task", stage: stage.stage, taskId })}
            selectedContentKey={selectedContentKey}
            onSelectContent={(_taskId, contentId) => {
              const [contentKind, ...rest] = contentId.split(":");
              const name = rest.join(":");
              onSelect(
                contentKind === "engine"
                  ? { kind: "engine-step", stage: stage.stage, name }
                  : { kind: "user-step", stage: stage.stage, name },
              );
            }}
          />
          <div className="inside-panel">
            <div className="inside-panel-head">
              <span className="inside-panel-title">
                inside <code>process_batch</code>
              </span>
              <span className="inside-panel-note">
                one mapped task instance per batch runs those steps, per episode
              </span>
            </div>
            {/* The steps themselves are drawn inside the node above -- that is
                where containment belongs. What is left here is what a badge
                cannot say: what the tiers mean, and the channels the transform
                writes, which are outputs rather than steps. */}
            {!graph.steps_known ? (
              <p className="empty-note">
                No <code>--pipeline</code> configured, so the registered steps in this stage are
                unknown. The orchestration above is the same either way.
              </p>
            ) : stage.user_steps.length === 0 && stage.engine_steps.length === 0 ? (
              <p className="empty-note">Nothing registered in this stage.</p>
            ) : (
              <>
                <dl className="tier-legend">
                  {stage.engine_steps.length > 0 ? (
                    <div className="tier-legend-entry">
                      <dt>
                        <span className="tier-badge is-engine">engine</span>
                      </dt>
                      <dd>work the engine always does in this stage</dd>
                    </div>
                  ) : null}
                  {tierOneSteps.length > 0 ? (
                    <div className="tier-legend-entry">
                      <dt>
                        <span className="tier-badge">t1</span>
                      </dt>
                      <dd>no requires, no endpoint alias — the engine runs these first</dd>
                    </div>
                  ) : null}
                  {tierTwoSteps.length > 0 ? (
                    <div className="tier-legend-entry">
                      <dt>
                        <span className="tier-badge">t2</span>
                      </dt>
                      <dd>declares requires or an endpoint alias — runs after tier 1</dd>
                    </div>
                  ) : null}
                </dl>
                {stage.user_steps.length > 1 ? (
                  <p className="tier-note">
                    Only the tier boundary is a real ordering. Within a tier the steps have no
                    ordering between them: they run in one process, in the order shown, and nothing
                    passes a value to anything else.
                  </p>
                ) : null}
                {derivedChannels.length > 0 ? (
                  <div className="tier-group">
                    <div className="tier-head">
                      <span className="tier-badge is-engine">derived channels</span>
                      <span className="tier-caption">written by the canonical transform</span>
                    </div>
                    <div className="step-card-row">
                      {derivedChannels.map((channel) => (
                        <span key={channel.topic} className="step-card step-card-engine">
                          <span className="step-card-top">
                            <span className="step-card-name" title={channel.topic}>
                              {channel.topic}
                            </span>
                          </span>
                          <span className="step-card-meta">
                            <span className="step-card-kind">derived channel</span>
                            <VersionChip version={channel.version} />
                          </span>
                        </span>
                      ))}
                    </div>
                  </div>
                ) : null}
              </>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function SelectionDetails({
  graph,
  selection,
  onExpandStage,
}: {
  graph: PipelineGraphResponse;
  selection: PipelineSelection | null;
  onExpandStage: (stage: string) => void;
}) {
  if (selection === null) {
    return (
      <DetailsPanel title="How to read this graph" kicker="legend">
        <ul className="legend-list">
          <li>
            <span className="legend-swatch is-plain" aria-hidden="true" />
            <span>A task in the generated DAG. Solid connectors are real dependencies.</span>
          </li>
          <li>
            <span className="legend-swatch is-mapped" aria-hidden="true" />
            <span>
              A stacked card is dynamically mapped: <strong>×N</strong> instances, one per planned
              batch.
            </span>
          </li>
          <li>
            <span className="legend-chip">waits</span>
            <span>
              A deferred task. It released its worker slot and is waiting — not stalled, not
              running.
            </span>
          </li>
          <li>
            <span className="legend-swatch is-gate" aria-hidden="true" />
            <span>
              A dashed connector is a conditional gate, not a task dependency: the quarantine
              verdict.
            </span>
          </li>
        </ul>
        <p className="details-hint">
          Select any node or step to see what it does. The steps inside <code>process_batch</code>{" "}
          have no edges between them — that is the model, not a missing feature.
        </p>
      </DetailsPanel>
    );
  }

  if (selection.kind === "quarantine") {
    const gate = graph.quarantine_gate;
    if (!gate) return <DetailsPanel title="Quarantine gate">{null}</DetailsPanel>;
    return (
      <DetailsPanel title="Quarantine gate" kicker="conditional edge">
        <p className="details-prose">{gate.explanation}</p>
        <DetailRow label="decided in">{gate.from_stage}</DetailRow>
        <DetailRow label="skips">{gate.to_stages.join(", ")}</DetailRow>
        <DetailBlock label="critical checks">
          {gate.critical_step_names.length === 0 ? (
            <p className="empty-note">No critical checks are registered, so nothing quarantines.</p>
          ) : (
            <ul className="detail-list">
              {gate.critical_step_names.map((name) => (
                <li key={name} className="cell-mono">
                  {name}
                </li>
              ))}
            </ul>
          )}
        </DetailBlock>
      </DetailsPanel>
    );
  }

  if (selection.kind === "master-task") {
    const task = graph.master.tasks.find((entry) => entry.task_id === selection.taskId);
    const stage = stageOfMasterTask(graph, selection.taskId);
    return (
      <DetailsPanel title={selection.taskId} kicker="master dag task">
        <p className="details-prose">{task?.summary ?? "Unknown task."}</p>
        <DetailRow label="dag">
          <code className="cell-mono">{graph.master.dag_id}</code>
        </DetailRow>
        {task?.deferred ? (
          <DetailRow label="waiting">
            defers its worker slot while the sub-DAG runs (Airflow calls this deferred)
          </DetailRow>
        ) : null}
        {stage ? (
          <>
            <DetailRow label="stage">{stage.stage}</DetailRow>
            <DetailRow label="enabled by">
              {stage.enabling_profiles.length > 0 ? stage.enabling_profiles.join(", ") : "—"}
            </DetailRow>
            <DetailBlock label="sub-dag">
              <code className="cell-mono">{stage.dag.dag_id}</code>
              <button
                type="button"
                className="btn btn-tiny"
                onClick={() => onExpandStage(stage.stage)}
              >
                Expand {stage.stage}
              </button>
            </DetailBlock>
          </>
        ) : null}
      </DetailsPanel>
    );
  }

  if (selection.kind === "stage-task") {
    const stage = graph.stages.find((entry) => entry.stage === selection.stage);
    const task = stage?.dag.tasks.find((entry) => entry.task_id === selection.taskId);
    return (
      <DetailsPanel title={selection.taskId} kicker={`${selection.stage} sub-dag task`}>
        <p className="details-prose">{task?.summary ?? "Unknown task."}</p>
        <DetailRow label="dag">
          <code className="cell-mono">{stage?.dag.dag_id ?? "—"}</code>
        </DetailRow>
        {task?.mapped ? (
          <DetailRow label="fan-out">
            dynamically mapped — one instance per planned batch, each running every step below
          </DetailRow>
        ) : null}
        {stage ? (
          <DetailRow label="triggered by">
            <code className="cell-mono">{stage.trigger_task_id}</code>
          </DetailRow>
        ) : null}
      </DetailsPanel>
    );
  }

  if (selection.kind === "engine-step") {
    const stage = graph.stages.find((entry) => entry.stage === selection.stage);
    const step = stage?.engine_steps.find((entry) => entry.name === selection.name);
    return (
      <DetailsPanel title={selection.name} kicker="engine step">
        <p className="details-prose">{step?.summary ?? "Engine-owned work."}</p>
        <DetailRow label="stage">{selection.stage}</DetailRow>
        <DetailRow label="registered">no — the engine always runs it</DetailRow>
      </DetailsPanel>
    );
  }

  const stage = graph.stages.find((entry) => entry.stage === selection.stage);
  const step = stage?.user_steps.find((entry) => entry.name === selection.name);
  if (!step) return <DetailsPanel title={selection.name}>{null}</DetailsPanel>;
  return (
    <DetailsPanel title={step.name} kicker={`${step.kind} · ${selection.stage}`}>
      <DetailRow label="version">
        <VersionChip version={step.version} />
      </DetailRow>
      <DetailRow label="critical">
        {step.critical ? (
          <span className="chip chip-err">yes — a failure quarantines the episode</span>
        ) : (
          "no"
        )}
      </DetailRow>
      <DetailRow label="endpoint">{step.uses ?? "—"}</DetailRow>
      <DetailBlock label="requires">
        {step.requires.length === 0 ? (
          <p className="empty-note">nothing — it reads the canonical episode only</p>
        ) : (
          <ul className="detail-list">
            {step.requires.map((requirement) => (
              <li key={requirement} className="cell-mono">
                {requirement}
              </li>
            ))}
          </ul>
        )}
      </DetailBlock>
      <p className="details-hint">
        Tier {step.tier}
        {step.tier === 2
          ? " — it declares requires or an endpoint alias, so the engine runs it after the cheap steps."
          : " — no requires and no endpoint, so the engine runs it first."}{" "}
        Within a tier there is no ordering.
      </p>
    </DetailsPanel>
  );
}

function PipelineGraphView({
  graph,
  pipeline,
}: {
  graph: PipelineGraphResponse;
  pipeline: PipelineResponse | undefined;
}) {
  const [selection, setSelection] = useState<PipelineSelection | null>(null);
  const [expandedStagesOverride, setExpandedStagesOverride] = useState<ReadonlySet<string> | null>(
    null,
  );

  // Open the first stage that actually has registered steps, so the nested
  // layer is visible without hunting; the user's own toggles take over after.
  const defaultExpandedStages = useMemo(() => {
    const firstWithSteps = graph.stages.find((stage) => stage.user_steps.length > 0);
    const chosen = firstWithSteps ?? graph.stages[0];
    return new Set<string>(chosen ? [chosen.stage] : []);
  }, [graph]);
  const expandedStages = expandedStagesOverride ?? defaultExpandedStages;

  const toggleStage = (stageName: string) => {
    setExpandedStagesOverride(() => {
      const next = new Set(expandedStages);
      if (next.has(stageName)) next.delete(stageName);
      else next.add(stageName);
      return next;
    });
  };
  const expandStage = (stageName: string) => {
    setExpandedStagesOverride(() => new Set(expandedStages).add(stageName));
  };

  const masterNodes = topologyNodes(graph.master);
  const masterEdges = [
    ...topologyEdges(graph.master),
    ...quarantineEdges(graph, graph.quarantine_gate),
  ];
  const expandedTriggerIds = graph.stages
    .filter((stage) => expandedStages.has(stage.stage))
    .map((stage) => stage.trigger_task_id);

  const derivedChannelsByStage = (stageName: string) =>
    stageName === "sync" ? (pipeline?.manifest.derived_channels ?? []) : [];

  return (
    <>
      {!graph.dag_ids_known ? (
        <div className="notice-banner" role="status">
          <strong>No runtime addressed.</strong> The task graph below is what a bundle for this
          pipeline renders; the DAG ids are display-only until <code>hflow up</code> (or an{" "}
          <code>HFLOW_AIRFLOW_*</code> environment) gives the UI a real bundle.
        </div>
      ) : null}
      {!graph.steps_known ? (
        <div className="notice-banner" role="status">
          <strong>No pipeline configured.</strong> Orchestration is exact, but the steps inside{" "}
          <code>process_batch</code> are unknown. Restart with{" "}
          <code>hflow serve --pipeline path/to/pipeline.py</code> to see them.
        </div>
      ) : null}

      <div className="graph-layout">
        <div className="graph-column">
          <section className="section">
            <h2 className="section-title">
              Orchestration · <code className="cell-mono">{graph.master.dag_id}</code>
            </h2>
            <p className="section-note">
              The master DAG validates the conf, then walks the stage chain: each stage is gated on
              the run profile and triggered as its own sub-DAG, which the master waits for.
            </p>
            <DagGraph
              nodes={masterNodes}
              edges={masterEdges}
              label="Master ingest DAG"
              selectedNodeId={selection?.kind === "master-task" ? selection.taskId : null}
              expandedNodeIds={expandedTriggerIds}
              onSelectNode={(taskId) => {
                setSelection({ kind: "master-task", taskId });
                const stage = stageOfMasterTask(graph, taskId);
                if (stage) expandStage(stage.stage);
              }}
            />
            {graph.quarantine_gate ? (
              <button
                type="button"
                className={
                  selection?.kind === "quarantine" ? "gate-callout is-selected" : "gate-callout"
                }
                onClick={() => setSelection({ kind: "quarantine" })}
              >
                <span className="gate-callout-title">quarantine gate</span>
                <span className="gate-callout-text">{graph.quarantine_gate.explanation}</span>
              </button>
            ) : null}
          </section>

          <section className="section">
            <h2 className="section-title">Stages</h2>
            <p className="section-note">
              Expand a stage for its sub-DAG and the steps that run inside its{" "}
              <code>process_batch</code>.
            </p>
            <div className="stage-lane-list">
              {graph.stages.map((stage) => (
                <StageLane
                  key={stage.stage}
                  stage={stage}
                  graph={graph}
                  isExpanded={expandedStages.has(stage.stage)}
                  onToggle={() => toggleStage(stage.stage)}
                  selection={selection}
                  onSelect={setSelection}
                  derivedChannels={derivedChannelsByStage(stage.stage)}
                />
              ))}
            </div>
          </section>
        </div>
        <div className="details-column">
          <SelectionDetails graph={graph} selection={selection} onExpandStage={expandStage} />
        </div>
      </div>
    </>
  );
}

function ObservedVersionsSection({ pipeline }: { pipeline: PipelineResponse }) {
  return (
    <section className="section">
      <h2 className="section-title">Observed versions</h2>
      {pipeline.observed.length === 0 ? (
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
              {pipeline.observed.map((entry) => (
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
  );
}

export function PipelinePage() {
  useEffect(() => {
    document.title = "Pipeline · HFlow";
    return () => {
      document.title = "HFlow";
    };
  }, []);

  const graphQuery = useQuery({
    queryKey: ["pipeline-graph"],
    queryFn: fetchPipelineGraph,
    retry: (failureCount, error) =>
      !(error instanceof ApiError && (error.status === 404 || error.status === 409)) &&
      failureCount < 1,
  });
  const pipelineQuery = useQuery({
    queryKey: ["pipeline"],
    queryFn: fetchPipeline,
    // 409 means "no --pipeline configured" — a stable state, not a transient failure.
    retry: (failureCount, error) =>
      !(error instanceof ApiError && error.status === 409) && failureCount < 1,
  });

  const manifest = pipelineQuery.data?.manifest;
  const stale = pipelineQuery.data?.stale;
  const pipelineTitle = manifest?.pipeline_name ?? graphQuery.data?.master.dag_id ?? "Pipeline";

  return (
    <div className="pipeline-page">
      <div className="page-title-row">
        <h1 className="page-title">{pipelineTitle}</h1>
        {manifest ? <VersionChip version={manifest.pipeline_version} /> : null}
      </div>
      {manifest ? (
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
      ) : null}

      {stale && stale.count > 0 ? (
        <div className="stale-banner" role="status">
          <strong>
            {stale.count} episode{stale.count === 1 ? " is" : "s are"} stale
          </strong>{" "}
          against pipeline version <code className="cell-mono">{stale.pipeline_version}</code> —
          last processed by an older pipeline. The pipeline version is stamped by the canonical
          transform, so refreshing them means a re-run that includes the <code>sync</code> stage:
          the default <code>full</code> profile. (<code>metadata_backfill</code> and{" "}
          <code>relabel</code> re-run later stages only, and leave this version untouched.)
        </div>
      ) : null}

      {graphQuery.isPending ? (
        <LoadingPanel label="Loading the pipeline graph…" />
      ) : graphQuery.isError ? (
        graphQuery.error instanceof ApiError && graphQuery.error.status === 404 ? (
          <EmptyPanel
            title="This server does not serve the pipeline graph."
            hint="Upgrade hflow-server to see the ingest DAG and its stage sub-DAGs here."
          />
        ) : (
          <ErrorPanel
            error={graphQuery.error}
            onRetry={() => {
              void graphQuery.refetch();
            }}
          />
        )
      ) : (
        <PipelineGraphView graph={graphQuery.data} pipeline={pipelineQuery.data} />
      )}

      {pipelineQuery.data ? <ObservedVersionsSection pipeline={pipelineQuery.data} /> : null}
      {pipelineQuery.isError &&
      pipelineQuery.error instanceof ApiError &&
      pipelineQuery.error.status === 409 ? (
        <section className="section">
          <h2 className="section-title">Registered steps</h2>
          <EmptyPanel title="No pipeline manifest." hint={pipelineQuery.error.detail}>
            <p className="state-detail">
              Start the server with <code>hflow serve --pipeline path/to/pipeline.py</code> (append{" "}
              <code>:app</code> if the App object is not named <code>app</code>) to see registered
              steps, versions and staleness here. The file is imported once at startup, in your own
              environment.
            </p>
          </EmptyPanel>
        </section>
      ) : null}
    </div>
  );
}
