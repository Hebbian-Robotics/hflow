// The whole canvas, as a pure function of one focus and the payloads it needs.
//
// Nothing here fetches, lays out, or renders: it turns server payloads into
// nodes and edges, which is the only part of this UI with real decisions in it
// (which drill-down a node offers, which edges are honest to draw, what a
// missing run means). Keeping it pure is what makes those decisions testable
// without a browser -- see buildGraph.test.ts.
//
// ONE RULE runs through every level: an edge means a real dependency. The
// server is explicit that the pipeline's own steps have no dependency edges on
// each other (hflow_server._graph's module docstring), so this never draws
// arrows between them. Where an ordering does exist -- the stage chain, the
// tier boundary, the quarantine gate -- it is drawn, and everything else is
// grouped instead.

import type {
  EpisodeCheckRun,
  EpisodeDossier,
  EpisodePage,
  PipelineGate,
  PipelineGraph,
  PipelineGraphStage,
  PipelineUserStep,
  RunGraph,
  RunGraphStage,
  RunTaskInstance,
  Stage,
} from "../api";
import { airflowStateTone, checkStatusTone, type Tone } from "../tones";
import type { CanvasFocus } from "./focus";
import { EPISODES_FOCUS, shortEpisodeId } from "./focus";

// These are type ALIASES, not interfaces, and that is load-bearing: React
// Flow's Node<T> requires the node data to satisfy Record<string, unknown>, and
// TypeScript grants that implicit index signature to an object type alias but
// never to an interface.

/** A label/value pair for the inspector panel. */
export type DetailLine = {
  readonly label: string;
  readonly value: string;
};

/**
 * What a node draws.
 *
 * - `task` an orchestrator task instance, identified by its task id
 * - `item` a piece of data or work: an episode, a check, a pipeline step
 * - `note` structural furniture: an anchor, or a barrier between groups
 */
export type NodeShape = "task" | "item" | "note";

export type CanvasNodeData = {
  readonly title: string;
  readonly subtitle: string | null;
  readonly tone: Tone;
  readonly shape: NodeShape;
  /** Short pills on the node itself: a retry count, a duration, "critical". */
  readonly badges: readonly string[];
  /** The inspector's content for this node. */
  readonly detail: readonly DetailLine[];
  /** Where clicking this node goes. Null makes it a leaf. */
  readonly drillTo: CanvasFocus | null;
};

export type CanvasNode = {
  readonly id: string;
  readonly data: CanvasNodeData;
};

export type CanvasEdge = {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  /** Dashed means "a real relationship that is not an execution dependency". */
  readonly dashed: boolean;
  readonly label: string | null;
};

export type CanvasGraph = {
  readonly nodes: readonly CanvasNode[];
  readonly edges: readonly CanvasEdge[];
  /** Why the canvas is empty. Set only when there are no nodes at all. */
  readonly emptyMessage: string | null;
  /** Caveats about what IS drawn, shown above the canvas. */
  readonly notices: readonly string[];
};

export type CanvasInput = {
  readonly focus: CanvasFocus;
  readonly pipeline: PipelineGraph;
  /** The selected master run's live state, or null when none is selected. */
  readonly run: RunGraph | null;
  /** Episodes recorded by the selected run's stages. Null when not loaded. */
  readonly episodes: EpisodePage | null;
  /** The focused episode's dossier. Null outside the `episode` level. */
  readonly dossier: EpisodeDossier | null;
};

// Synthetic node ids are prefixed so they can never collide with an Airflow
// task id: Airflow restricts task ids to alphanumerics, dash, dot and
// underscore, so "~" is unavailable to them and available to us.
const SYNTHETIC = "~";

// Above this many mapped instances the fan-out is drawn as one stacked node
// carrying the counts instead of one node each. Twelve is where a fan stops
// being readable as individual boxes and starts being a wall; the stacked node
// loses no information, because the server already serves the complete state
// split in `mapped_summary.by_state`.
const MAX_DRAWN_FAN_NODES = 12;

/** How many of a check's measurements the inspector lists before eliding. */
const MAX_LISTED_MEASUREMENTS = 6;

const EMPTY_GRAPH_NOTICES: readonly string[] = [];

export function buildGraph(input: CanvasInput): CanvasGraph {
  switch (input.focus.level) {
    case "run":
      return runLevel(input);
    case "stage":
      return stageLevel(input, input.focus.stage);
    case "steps":
      return stepsLevel(input, input.focus.stage);
    case "episodes":
      return episodesLevel(input);
    case "episode":
      return episodeLevel(input, input.focus.episodeId);
  }
}

// --- level 1: the master run ------------------------------------------------

// The master task that is not about any one stage: it reads the trigger conf
// and publishes which stages this profile enables. Identified by elimination
// (every other master task belongs to a stage), so no task id is restated here.

/**
 * The run as a chain of stages: resolve the profile, then each stage in turn.
 *
 * ONE SUMMARY IS MADE HERE, deliberately. The master DAG spends two tasks per
 * stage -- ``enabled_<stage>`` decides whether the profile runs it, then
 * ``trigger_<stage>`` fires the sub-DAG and defers until it finishes -- and
 * this level draws them as one node per stage. Both task ids, both states and
 * both durations are on that node, so nothing is hidden; what is gained is a
 * five-node chain a reader takes in at once instead of a nine-rank ribbon that
 * only fits on screen at a zoom nobody can read. The full task-by-task picture
 * of a stage is one drill-down away.
 */
function runLevel({ pipeline, run, episodes }: CanvasInput): CanvasGraph {
  const instanceByTaskId = new Map<string, RunTaskInstance>();
  for (const task of run?.master.tasks ?? []) {
    if (task.task_id !== null) instanceByTaskId.set(task.task_id, task);
  }
  const stageTaskIds = new Set(
    pipeline.stages.flatMap((stage) => [stage.gate_task_id, stage.trigger_task_id]),
  );

  const nodes: CanvasNode[] = pipeline.master.tasks
    .filter((task) => !stageTaskIds.has(task.task_id))
    .map((task): CanvasNode => {
      const instance = instanceByTaskId.get(task.task_id) ?? null;
      return {
        id: task.task_id,
        data: {
          title: task.task_id,
          subtitle: task.summary,
          tone: airflowStateTone(instance?.state),
          shape: "task",
          badges: taskInstanceBadges(instance),
          detail: taskInstanceDetail(instance),
          drillTo: null,
        },
      };
    });
  const stageNodeIds = new Map<Stage, string>();
  for (const stagePipeline of pipeline.stages) {
    const node = stageChainNode(stagePipeline, instanceByTaskId);
    stageNodeIds.set(stagePipeline.stage, node.id);
    nodes.push(node);
  }

  // Every master edge, rewritten onto the merged stage nodes. An edge between
  // one stage's two tasks collapses into a self-edge and is dropped; the rest
  // keep the chain exactly as the master declares it.
  const nodeIdForTaskId = (taskId: string): string => {
    for (const stagePipeline of pipeline.stages) {
      if (taskId === stagePipeline.gate_task_id || taskId === stagePipeline.trigger_task_id) {
        return stageNodeIds.get(stagePipeline.stage) ?? taskId;
      }
    }
    return taskId;
  };
  const rewritten: CanvasEdge[] = [];
  const seenEdges = new Set<string>();
  for (const [source, target] of pipeline.master.edges) {
    const from = nodeIdForTaskId(source);
    const to = nodeIdForTaskId(target);
    const id = `${from}->${to}`;
    if (from === to || seenEdges.has(id)) continue;
    seenEdges.add(id);
    rewritten.push({ id, source: from, target: to, dashed: false, label: null });
  }
  const edges = withoutRedundantEdges(rewritten);

  // The data branch, hung off the end of the stage chain. Dashed, because the
  // rows are written inside the stage sub-DAGs rather than by any master task,
  // so this is a real relationship and not an Airflow dependency.
  if (run !== null) {
    const recordedTotal = episodes?.total ?? null;
    nodes.push({
      id: `${SYNTHETIC}episodes`,
      data: {
        title: "episodes recorded",
        subtitle: "catalog rows stamped with one of this run's stage run ids",
        tone: recordedTotal === 0 ? "muted" : "info",
        shape: "item",
        badges: recordedTotal === null ? [] : [`${recordedTotal}`],
        detail: [
          { label: "run", value: run.master.dag_run_id },
          ...(recordedTotal === null ? [] : [{ label: "rows", value: String(recordedTotal) }]),
          {
            label: "how this is counted",
            value:
              "an episode's row is the LATEST append, so a later run re-ingesting " +
              "an episode takes it out of this list",
          },
        ],
        drillTo: EPISODES_FOCUS,
      },
    });
    for (const sink of sinkNodeIds(
      nodes.map((node) => node.id).filter((nodeId) => nodeId !== `${SYNTHETIC}episodes`),
      edges.map((edge) => [edge.source, edge.target] as const),
    )) {
      edges.push({
        id: `${sink}->episodes`,
        source: sink,
        target: `${SYNTHETIC}episodes`,
        dashed: true,
        label: "recorded",
      });
    }
  }

  return {
    nodes,
    edges,
    emptyMessage: null,
    notices: run === null ? ["No run selected: this is the DAG's shape, not a run."] : [],
  };
}

/**
 * Drop every edge whose ordering another path already states.
 *
 * The master declares both a chain and a shortcut: the profile task feeds all
 * four stage gates, AND each stage's trigger feeds the next stage's gate. Every
 * shortcut is therefore implied by the chain, and drawing both turns a
 * five-node line into a tangle.
 *
 * This never adds an ordering and never removes one -- a dropped edge's
 * dependency still holds along the path that kept it -- so the picture stays
 * true while saying it once instead of twice. It is the only place this file
 * removes a real edge, and it is safe exactly because redundancy is the test.
 */
function withoutRedundantEdges(edges: readonly CanvasEdge[]): CanvasEdge[] {
  const successors = new Map<string, string[]>();
  for (const edge of edges) {
    const existing = successors.get(edge.source);
    if (existing === undefined) successors.set(edge.source, [edge.target]);
    else existing.push(edge.target);
  }

  function reachableAvoidingDirectHop(from: string, to: string): boolean {
    // Breadth-first from `from`'s successors, never using the from->to hop
    // itself: any OTHER route proves the direct edge redundant.
    const queue = (successors.get(from) ?? []).filter((next) => next !== to);
    const seen = new Set(queue);
    while (queue.length > 0) {
      const current = queue.shift() as string;
      if (current === to) return true;
      for (const next of successors.get(current) ?? []) {
        if (!seen.has(next)) {
          seen.add(next);
          queue.push(next);
        }
      }
    }
    return false;
  }

  return edges.filter((edge) => !reachableAvoidingDirectHop(edge.source, edge.target));
}

/** One stage as the run level draws it: its gate and its trigger, merged. */
function stageChainNode(
  stagePipeline: PipelineGraphStage,
  instanceByTaskId: Map<string, RunTaskInstance>,
): CanvasNode {
  const gate = instanceByTaskId.get(stagePipeline.gate_task_id) ?? null;
  const trigger = instanceByTaskId.get(stagePipeline.trigger_task_id) ?? null;
  // A gate that skipped means the profile did not enable this stage, and then
  // the trigger never ran -- so the gate is the state worth showing. Otherwise
  // the trigger is, because it is the task that waits for the stage's work.
  const skippedByProfile = gate?.state?.toLowerCase() === "skipped";
  const governing = skippedByProfile ? gate : (trigger ?? gate);
  return {
    id: stagePipeline.stage,
    data: {
      title: stagePipeline.title,
      subtitle: stagePipeline.description,
      tone: airflowStateTone(governing?.state),
      shape: "task",
      badges: [
        stagePipeline.stage,
        ...(skippedByProfile ? ["skipped by profile"] : []),
        ...taskInstanceBadges(skippedByProfile ? gate : trigger),
        ...(stagePipeline.enabling_profiles.length === 1
          ? // Only worth a pill when the stage is NOT universal: "full" runs
            // everything, so saying so on all four would be noise.
            [`only ${stagePipeline.enabling_profiles[0]}`]
          : []),
      ].filter((badge): badge is string => badge !== undefined),
      detail: [
        { label: "stage", value: stagePipeline.stage },
        { label: "enabled by profiles", value: stagePipeline.enabling_profiles.join(", ") },
        { label: "sub-dag", value: stagePipeline.dag.dag_id },
        // Both master tasks named, so the merge above hides no task id.
        {
          label: stagePipeline.gate_task_id,
          value: gate === null ? "no instance for this run" : (gate.state ?? "not scheduled yet"),
        },
        {
          label: stagePipeline.trigger_task_id,
          value:
            trigger === null ? "no instance for this run" : (trigger.state ?? "not scheduled yet"),
        },
        ...taskInstanceDetail(trigger),
      ],
      drillTo: { level: "stage", stage: stagePipeline.stage },
    },
  };
}

// --- level 2: one stage's sub-DAG -------------------------------------------

/**
 * One stage: plan the batches, fan `process_batch` out over them, close on a
 * budget gate. The fan is the only mapped task, and it is where the pipeline's
 * own steps run -- so it is the node that drills further in.
 */
function stageLevel(input: CanvasInput, stage: Stage): CanvasGraph {
  const stagePipeline = findStage(input.pipeline, stage);
  if (stagePipeline === null) return unknownStage(stage);
  const runStage = findRunStage(input.run, stage);

  const mappedTaskId = stagePipeline.dag.tasks.find((task) => task.mapped)?.task_id ?? null;
  const instancesByTaskId = new Map<string, RunTaskInstance[]>();
  for (const task of runStage?.tasks ?? []) {
    if (task.task_id === null) continue;
    const existing = instancesByTaskId.get(task.task_id);
    if (existing === undefined) instancesByTaskId.set(task.task_id, [task]);
    else existing.push(task);
  }

  const nodes: CanvasNode[] = [];
  // Every id the mapped task was expanded into, so the edges that pointed at
  // the mapped task can be rewired onto all of them.
  const fanNodeIds: string[] = [];

  for (const task of stagePipeline.dag.tasks) {
    const instances = instancesByTaskId.get(task.task_id) ?? [];
    if (task.task_id !== mappedTaskId) {
      const instance = instances[0] ?? null;
      nodes.push({
        id: task.task_id,
        data: {
          title: task.task_id,
          subtitle: task.summary,
          tone: airflowStateTone(instance?.state),
          shape: "task",
          badges: taskInstanceBadges(instance),
          detail: taskInstanceDetail(instance),
          drillTo: null,
        },
      });
      continue;
    }
    const fanNodes = fanOutNodes(task.task_id, task.summary, instances, runStage, stage);
    nodes.push(...fanNodes);
    fanNodeIds.push(...fanNodes.map((node) => node.id));
  }

  const edges: CanvasEdge[] = [];
  for (const [source, target] of stagePipeline.dag.edges) {
    const sources = source === mappedTaskId ? fanNodeIds : [source];
    const targets = target === mappedTaskId ? fanNodeIds : [target];
    for (const from of sources) {
      for (const to of targets) {
        edges.push({ id: `${from}->${to}`, source: from, target: to, dashed: false, label: null });
      }
    }
  }

  return {
    nodes,
    edges,
    emptyMessage: null,
    notices: [
      ...(input.run === null ? ["No run selected: this is the DAG's shape, not a run."] : []),
      ...(input.run !== null && runStage?.dag_run_id == null
        ? [`This stage did not run for the selected master run.`]
        : []),
      ...(runStage?.match === "heuristic"
        ? [
            "Airflow stores no link from a stage run back to the master run that " +
              "triggered it, so this stage run was matched by time window and could " +
              "belong to an overlapping master run.",
          ]
        : []),
    ],
  };
}

/**
 * The mapped task, expanded.
 *
 * One node per mapped instance while the fan is small enough to read, and one
 * stacked node carrying the state split once it is not. Before Airflow expands
 * the fan it reports a single instance at map index -1, which is drawn as
 * itself: "one unexpanded instance" is the truth at that moment.
 */
function fanOutNodes(
  taskId: string,
  summary: string,
  instances: readonly RunTaskInstance[],
  runStage: RunGraphStage | null,
  stage: Stage,
): CanvasNode[] {
  const drillTo: CanvasFocus = { level: "steps", stage };
  const summarized = runStage?.mapped_summary ?? null;

  if (instances.length === 0 || instances.length > MAX_DRAWN_FAN_NODES) {
    const total = summarized?.total ?? instances.length;
    return [
      {
        id: taskId,
        data: {
          title: taskId,
          subtitle: summary,
          tone: stackedFanTone(summarized?.by_state),
          shape: "task",
          badges: [
            ...(total > 0 ? [`x${total}`] : ["fans out per batch"]),
            ...Object.entries(summarized?.by_state ?? {}).map(
              ([state, count]) => `${count} ${state}`,
            ),
          ],
          detail: [
            { label: "mapped instances", value: total > 0 ? String(total) : "not planned yet" },
            ...Object.entries(summarized?.by_state ?? {}).map(([state, count]) => ({
              label: state,
              value: String(count),
            })),
          ],
          drillTo,
        },
      },
    ];
  }

  return instances.map((instance) => ({
    // The map index is part of the id: two instances of one mapped task differ
    // only by it, so leaving it out would collapse the whole fan into one node.
    id: `${taskId}${SYNTHETIC}${instance.map_index}`,
    data: {
      title: instance.map_index < 0 ? taskId : `${taskId} [${instance.map_index}]`,
      subtitle: instance.map_index < 0 ? summary : "one batch of episodes",
      tone: airflowStateTone(instance.state),
      shape: "task",
      badges: taskInstanceBadges(instance),
      detail: taskInstanceDetail(instance),
      drillTo,
    },
  }));
}

/**
 * One tone for a whole fan: the worst thing any instance is saying.
 *
 * Ordered worst-first so a single failure in a hundred successes still colours
 * the stacked node -- the opposite (a majority vote) would hide exactly the
 * instance somebody opened the page to find.
 */
function stackedFanTone(byState: Record<string, number> | undefined): Tone {
  const tones = new Set(
    Object.entries(byState ?? {})
      .filter(([, count]) => count > 0)
      .map(([state]) => airflowStateTone(state)),
  );
  for (const candidate of ["err", "warn", "run", "ok", "info"] as const) {
    if (tones.has(candidate)) return candidate;
  }
  return "muted";
}

/** The nodes nothing depends on: where a DAG's work ends. */
function sinkNodeIds(
  nodeIds: readonly string[],
  edges: readonly (readonly [string, string])[],
): string[] {
  const withOutgoing = new Set(edges.map(([source]) => source));
  return nodeIds.filter((nodeId) => !withOutgoing.has(nodeId));
}

// --- level 3: the steps inside one batch ------------------------------------

/**
 * What one `process_batch` does to each episode in its batch.
 *
 * Drawn as groups separated by the boundaries that are real, never as a chain:
 * the engine runs a stage's registered steps in tier order, so every tier-2
 * step runs after every tier-1 step, but steps WITHIN a tier have no ordering
 * and no dependency on each other. So a tier is a column, and only the
 * boundaries get arrows.
 */
function stepsLevel(input: CanvasInput, stage: Stage): CanvasGraph {
  const stagePipeline = findStage(input.pipeline, stage);
  if (stagePipeline === null) return unknownStage(stage);
  if (!input.pipeline.steps_known) {
    return {
      nodes: [],
      edges: [],
      emptyMessage:
        "This server was started without --pipeline, so what runs inside a batch " +
        "is unknown to it. Restart `hflow serve` with --pipeline path/to/pipeline.py.",
      notices: EMPTY_GRAPH_NOTICES,
    };
  }

  const columns: CanvasNode[][] = [];
  const anchor: CanvasNode = {
    id: `${SYNTHETIC}batch`,
    data: {
      title: "one episode",
      subtitle: `everything below runs per episode, inside this stage's process_batch`,
      tone: "muted",
      shape: "note",
      badges: [],
      detail: [{ label: "stage", value: `${stagePipeline.title} (${stage})` }],
      drillTo: null,
    },
  };
  columns.push([anchor]);

  // Labels and media are the RECEIVING side of the quarantine gate: for them it
  // is an entry condition, so it is drawn before their steps. Meta is the
  // deciding side, so its gate comes after the checks, below.
  const gate = input.pipeline.quarantine_gate;
  if (gate?.to_stages.includes(stage)) {
    columns.push([quarantineGateNode(gate, "entry")]);
  }

  const engineNodes = stagePipeline.engine_steps.map(
    (step): CanvasNode => ({
      id: `${SYNTHETIC}engine:${step.name}`,
      data: {
        title: step.name,
        subtitle: step.summary,
        tone: "info",
        shape: "item",
        badges: ["engine"],
        detail: [{ label: "owned by", value: "the engine, not a registration" }],
        drillTo: null,
      },
    }),
  );

  // Meta is the one stage with both, and there the engine's work is the catalog
  // append, which records what the checks decided -- so it runs last. Every
  // other stage's engine step is the only thing in it, and reads first.
  const engineStepsRunLast = stage === "meta";
  if (engineNodes.length > 0 && !engineStepsRunLast) columns.push(engineNodes);

  for (const tier of [1, 2] as const) {
    const inTier = stagePipeline.user_steps.filter((step) => step.tier === tier);
    if (inTier.length === 0) continue;
    if (tier === 2) {
      columns.push([
        {
          id: `${SYNTHETIC}tier-barrier`,
          data: {
            title: "tier 1 complete",
            subtitle: "steps declaring requires or uses run in the second tier",
            tone: "muted",
            shape: "note",
            badges: [],
            detail: [
              {
                label: "why",
                value:
                  "the engine sorts a stage's steps by tier and runs them in that " +
                  "order, so every tier-2 step runs after every tier-1 step",
              },
            ],
            drillTo: null,
          },
        },
      ]);
    }
    columns.push(inTier.map((step) => userStepNode(step, tier)));
  }

  if (gate !== null && gate.from_stage === stage) {
    columns.push([quarantineGateNode(gate, "decision")]);
  }
  if (engineNodes.length > 0 && engineStepsRunLast) columns.push(engineNodes);

  const nonEmpty = columns.filter((column) => column.length > 0);
  const userStepCount = stagePipeline.user_steps.length;
  return {
    nodes: nonEmpty.flat(),
    edges: chainColumns(nonEmpty),
    emptyMessage: null,
    notices: [
      ...(userStepCount === 0
        ? [`This pipeline registers no steps in the ${stage} stage; the work here is the engine's.`]
        : []),
      "Steps within one column have no ordering and no dependency on each other. " +
        "Only the boundaries between columns are real.",
    ],
  };
}

function userStepNode(step: PipelineUserStep, tier: 1 | 2): CanvasNode {
  const gateText = step.gate == null ? null : gateSummary(step.gate);
  return {
    id: `${SYNTHETIC}step:${step.name}`,
    data: {
      title: step.name,
      subtitle: gateText,
      tone: step.critical ? "warn" : "info",
      shape: "item",
      badges: [
        step.kind,
        ...(step.critical ? ["critical"] : []),
        ...(tier === 2 ? ["tier 2"] : []),
      ],
      detail: [
        { label: "kind", value: step.kind },
        { label: "version", value: step.version },
        { label: "critical", value: step.critical ? "yes: a False verdict quarantines" : "no" },
        ...(gateText === null ? [] : [{ label: "accepts when", value: gateText }]),
        ...(step.requires.length > 0
          ? [{ label: "requires", value: step.requires.join(", ") }]
          : []),
        ...(step.uses == null ? [] : [{ label: "uses", value: step.uses }]),
      ],
      drillTo: null,
    },
  };
}

function quarantineGateNode(
  gate: NonNullable<PipelineGraph["quarantine_gate"]>,
  side: "entry" | "decision",
): CanvasNode {
  const hasCriticalSteps = gate.critical_step_names.length > 0;
  return {
    id: `${SYNTHETIC}quarantine:${side}`,
    data: {
      title: "quarantine gate",
      subtitle:
        side === "entry"
          ? "a quarantined episode records every step below as skipped"
          : "a critical check's False verdict quarantines the episode",
      tone: hasCriticalSteps ? "warn" : "muted",
      shape: "note",
      badges: hasCriticalSteps ? [`${gate.critical_step_names.length} critical`] : ["no gate"],
      detail: [
        { label: "explanation", value: gate.explanation },
        {
          label: "critical steps",
          value: hasCriticalSteps ? gate.critical_step_names.join(", ") : "none registered",
        },
        { label: "affects stages", value: gate.to_stages.join(", ") },
      ],
      drillTo: null,
    },
  };
}

/** One threshold set as a single readable line. */
export function gateSummary(gate: PipelineGate): string {
  return gate.accept_when
    .map((threshold) => {
      const operator = threshold.comparison === "at_most" ? "<=" : ">=";
      const scope = threshold.across === "any_key" ? "any key" : "every key";
      return `${threshold.key_pattern} ${operator} ${threshold.value} (${scope})`;
    })
    .join(" and ");
}

/**
 * Wire consecutive columns together.
 *
 * Every boundary this crosses has a single-node column on at least one side (a
 * barrier, a gate, the anchor), so the edge count stays linear in the nodes
 * rather than multiplying two columns together.
 */
function chainColumns(columns: readonly CanvasNode[][]): CanvasEdge[] {
  const edges: CanvasEdge[] = [];
  for (let index = 0; index + 1 < columns.length; index += 1) {
    for (const from of columns[index] ?? []) {
      for (const to of columns[index + 1] ?? []) {
        edges.push({
          id: `${from.id}->${to.id}`,
          source: from.id,
          target: to.id,
          dashed: false,
          label: null,
        });
      }
    }
  }
  return edges;
}

// --- level 4: the episodes this run recorded --------------------------------

/**
 * The episodes whose current catalog row came out of this run.
 *
 * Two things this deliberately does NOT claim. Which BATCH produced which
 * episode is not drawable: Airflow reports the mapped instances, the catalog
 * records the run, and nothing ties an episode to a map index. And which STAGE
 * recorded it is not a useful split either, because the catalog keeps one row
 * per episode and the last stage to append wins -- so the fan here is from the
 * master run, over the union of its stage runs.
 */
function episodesLevel(input: CanvasInput): CanvasGraph {
  const run = input.run;
  if (run === null) {
    return {
      nodes: [],
      edges: [],
      emptyMessage: "No run is selected, so there are no recorded episodes to show.",
      notices: EMPTY_GRAPH_NOTICES,
    };
  }
  const rows = input.episodes?.rows ?? [];
  if (input.episodes !== null && rows.length === 0) {
    return {
      nodes: [],
      edges: [],
      emptyMessage:
        "No episode's current catalog row came out of this run. A run that failed " +
        "before process_batch appends nothing, and an episode a LATER run " +
        "re-ingested now belongs to that run instead.",
      notices: EMPTY_GRAPH_NOTICES,
    };
  }

  const anchorId = `${SYNTHETIC}run`;
  const nodes: CanvasNode[] = [
    {
      id: anchorId,
      data: {
        title: run.master.dag_run_id,
        subtitle: "ingest run",
        tone: airflowStateTone(run.master.state),
        shape: "note",
        badges: [`${input.episodes?.total ?? rows.length} episodes`],
        detail: [
          { label: "state", value: run.master.state ?? "unknown" },
          {
            label: "stage runs",
            value:
              stageRunIds(run).join(", ") ||
              "none matched, so nothing could be attributed to this run",
          },
        ],
        drillTo: null,
      },
    },
  ];
  const edges: CanvasEdge[] = [];

  for (const row of rows) {
    const episodeId = textField(row, "episode_id");
    if (episodeId === null) continue;
    const status = textField(row, "status");
    nodes.push({
      id: `${SYNTHETIC}episode:${episodeId}`,
      data: {
        title: shortEpisodeId(episodeId),
        subtitle: textField(row, "task"),
        tone: status === "quarantined" ? "warn" : "ok",
        shape: "item",
        badges: [...(status === null ? [] : [status])],
        detail: [
          { label: "episode_id", value: episodeId },
          ...[
            "task",
            "operator",
            "embodiment",
            "success",
            "recorded_at",
            "pipeline_version",
            "orchestrator_run_id",
          ]
            .map((column) => ({ label: column, value: textField(row, column) }))
            .filter((line): line is DetailLine => line.value !== null),
        ],
        drillTo: { level: "episode", episodeId },
      },
    });
    edges.push({
      id: `${anchorId}->${episodeId}`,
      source: anchorId,
      target: `${SYNTHETIC}episode:${episodeId}`,
      dashed: true,
      label: null,
    });
  }

  const total = input.episodes?.total ?? 0;
  return {
    nodes,
    edges,
    emptyMessage: null,
    notices:
      total > rows.length
        ? [`Showing the first ${rows.length} of ${total} episodes this run recorded.`]
        : EMPTY_GRAPH_NOTICES,
  };
}

/**
 * Every stage run id this master run was matched to.
 *
 * The set the episode query filters on, exported so App.tsx asks for exactly
 * what this level draws rather than deriving the same list a second way.
 */
export function stageRunIds(run: RunGraph | null): string[] {
  return (run?.stages ?? [])
    .map((stage) => stage.dag_run_id)
    .filter((runId): runId is string => runId !== null);
}

// --- level 5: one episode's checks ------------------------------------------

/**
 * Every check recorded for one episode, with its verdict and its gate.
 *
 * The tiers come from the live pipeline and the verdicts from the catalog, so a
 * check recorded by an earlier version of the pipeline can appear with no tier.
 * That is shown rather than hidden: it is the honest signal that the recorded
 * evidence and the current code have diverged.
 */
function episodeLevel(input: CanvasInput, episodeId: string): CanvasGraph {
  const dossier = input.dossier;
  if (dossier === null) {
    return {
      nodes: [],
      edges: [],
      emptyMessage: `Loading episode ${shortEpisodeId(episodeId)}...`,
      notices: EMPTY_GRAPH_NOTICES,
    };
  }

  const stepsByName = new Map<string, PipelineUserStep>();
  for (const stagePipeline of input.pipeline.stages) {
    for (const step of stagePipeline.user_steps) stepsByName.set(step.name, step);
  }
  const measurementsByCheck = new Map<string, string[]>();
  for (const measurement of dossier.measurements) {
    if (measurement.check_name === null || measurement.key === null) continue;
    const rendered = `${measurement.key} = ${measurementValueText(measurement)}`;
    const existing = measurementsByCheck.get(measurement.check_name);
    if (existing === undefined) measurementsByCheck.set(measurement.check_name, [rendered]);
    else existing.push(rendered);
  }

  const anchorId = `${SYNTHETIC}episode`;
  const quarantineTags = dossier.episode.quarantine_tags;
  const nodes: CanvasNode[] = [
    {
      id: anchorId,
      data: {
        title: shortEpisodeId(episodeId),
        subtitle: dossier.episode.status === "quarantined" ? "quarantined" : "ok",
        tone: dossier.episode.status === "quarantined" ? "warn" : "ok",
        shape: "note",
        badges: [`${dossier.check_runs.length} checks`],
        detail: [
          { label: "episode_id", value: episodeId },
          { label: "status", value: dossier.episode.status },
          ...(quarantineTags.length > 0
            ? [{ label: "quarantine tags", value: quarantineTags.join(", ") }]
            : []),
          ...["task", "operator", "embodiment", "pipeline_version", "orchestrator_run_id"]
            .map((column) => ({ label: column, value: textField(dossier.episode, column) }))
            .filter((line): line is DetailLine => line.value !== null),
        ],
        drillTo: null,
      },
    },
  ];

  // Grouped by the tier the CURRENT pipeline puts each check in; a recorded
  // check the pipeline no longer registers has no tier and gets its own group.
  const groups: { readonly label: string; readonly runs: EpisodeCheckRun[] }[] = [
    { label: "tier 1", runs: [] },
    { label: "tier 2", runs: [] },
    { label: "not registered", runs: [] },
  ];
  for (const checkRun of dossier.check_runs) {
    const step = checkRun.check_name === null ? undefined : stepsByName.get(checkRun.check_name);
    const groupIndex = step === undefined ? 2 : step.tier - 1;
    groups[groupIndex]?.runs.push(checkRun);
  }

  const edges: CanvasEdge[] = [];
  for (const group of groups) {
    for (const checkRun of group.runs) {
      const name = checkRun.check_name ?? "(unnamed check)";
      const step = stepsByName.get(name);
      const gateText = step?.gate == null ? null : gateSummary(step.gate);
      const nodeId = `${SYNTHETIC}check:${group.label}:${name}`;
      nodes.push({
        id: nodeId,
        data: {
          title: name,
          subtitle: gateText ?? checkRun.error ?? group.label,
          tone: checkStatusTone(checkRun.status),
          shape: "item",
          badges: [
            checkRun.status ?? "unknown",
            ...(checkRun.critical === true ? ["critical"] : []),
            ...(step === undefined ? ["not registered"] : []),
            ...durationBadge(checkRun.duration_s),
          ],
          detail: [
            { label: "status", value: checkRun.status ?? "unknown" },
            { label: "recorded version", value: checkRun.check_version ?? "unknown" },
            ...(step === undefined
              ? [
                  {
                    label: "current pipeline",
                    value: "does not register a step by this name any more",
                  },
                ]
              : [{ label: "current version", value: step.version }]),
            ...(gateText === null ? [] : [{ label: "accepts when", value: gateText }]),
            ...(checkRun.error === null ? [] : [{ label: "error", value: checkRun.error }]),
            ...measurementDetail(measurementsByCheck.get(name) ?? []),
          ],
          drillTo: null,
        },
      });
      edges.push({
        id: `${anchorId}->${nodeId}`,
        source: anchorId,
        target: nodeId,
        dashed: true,
        label: null,
      });
    }
  }

  const staleCheckCount = groups[2]?.runs.length ?? 0;
  return {
    nodes,
    edges,
    emptyMessage: null,
    notices: [
      ...(dossier.check_runs.length === 0 ? ["No checks were recorded for this episode."] : []),
      ...(staleCheckCount > 0
        ? [
            staleCheckCount === 1
              ? "1 recorded check no longer exists in the current pipeline."
              : `${staleCheckCount} recorded checks no longer exist in the current pipeline.`,
          ]
        : []),
    ],
  };
}

function measurementValueText(measurement: {
  value_double: number | null;
  value_text: string | null;
  value_bool: boolean | null;
}): string {
  if (measurement.value_double !== null) return String(measurement.value_double);
  if (measurement.value_bool !== null) return String(measurement.value_bool);
  return measurement.value_text ?? "null";
}

function measurementDetail(rendered: readonly string[]): DetailLine[] {
  if (rendered.length === 0) return [];
  const shown = rendered.slice(0, MAX_LISTED_MEASUREMENTS);
  const elided = rendered.length - shown.length;
  return [
    {
      label: "measurements",
      value: elided > 0 ? `${shown.join(", ")}, and ${elided} more` : shown.join(", "),
    },
  ];
}

// --- shared helpers ---------------------------------------------------------

function findStage(pipeline: PipelineGraph, stage: Stage): PipelineGraphStage | null {
  return pipeline.stages.find((candidate) => candidate.stage === stage) ?? null;
}

function findRunStage(run: RunGraph | null, stage: Stage): RunGraphStage | null {
  return run?.stages.find((candidate) => candidate.stage === stage) ?? null;
}

function unknownStage(stage: Stage): CanvasGraph {
  return {
    nodes: [],
    edges: [],
    emptyMessage: `This server's topology has no ${stage} stage.`,
    notices: EMPTY_GRAPH_NOTICES,
  };
}

function taskInstanceBadges(instance: RunTaskInstance | null): string[] {
  if (instance === null) return [];
  return [
    ...(instance.state === null ? [] : [instance.state]),
    // A first attempt is not worth a pill; a retry is exactly what someone
    // scanning the canvas is looking for.
    ...(instance.try_number !== null && instance.try_number > 1
      ? [`try ${instance.try_number}`]
      : []),
    ...durationBadge(instance.duration_s),
  ];
}

function taskInstanceDetail(instance: RunTaskInstance | null): DetailLine[] {
  if (instance === null) return [{ label: "state", value: "no instance for this run" }];
  return [
    { label: "state", value: instance.state ?? "not scheduled yet" },
    ...(instance.queued_at === null ? [] : [{ label: "queued", value: instance.queued_at }]),
    ...(instance.start_date === null ? [] : [{ label: "started", value: instance.start_date }]),
    ...(instance.end_date === null ? [] : [{ label: "ended", value: instance.end_date }]),
    ...(instance.duration_s === null
      ? []
      : [{ label: "duration", value: formatDuration(instance.duration_s) }]),
    ...(instance.try_number === null ? [] : [{ label: "try", value: String(instance.try_number) }]),
  ];
}

function durationBadge(seconds: number | null): string[] {
  return seconds === null ? [] : [formatDuration(seconds)];
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "unknown";
  if (seconds < 10) return `${seconds.toFixed(1)}s`;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

/** One column of a catalog row as display text, or null when absent or null. */
function textField(row: Record<string, unknown>, column: string): string | null {
  const value = row[column];
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return null;
}
