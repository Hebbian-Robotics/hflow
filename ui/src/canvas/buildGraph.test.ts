// buildGraph is where this UI's real decisions live, and it is pure, so it can
// be pinned without a browser. These tests are about the DECISIONS -- which
// edges are honest, what a missing run means, where a drill-down goes -- not
// about node counts for their own sake.

import { describe, expect, it } from "vitest";
import type {
  EpisodeDossier,
  EpisodePage,
  PipelineGraph,
  PipelineUserStep,
  RunGraph,
  RunTaskInstance,
  Stage,
} from "../api";
import { buildGraph, type CanvasGraph, stageRunIds } from "./buildGraph";
import type { CanvasFocus } from "./focus";

function dagTask(taskId: string, overrides: { mapped?: boolean; deferred?: boolean } = {}) {
  return {
    task_id: taskId,
    summary: `${taskId} summary`,
    mapped: overrides.mapped ?? false,
    deferred: overrides.deferred ?? false,
  };
}

function userStep(name: string, overrides: Partial<PipelineUserStep> = {}): PipelineUserStep {
  return {
    name,
    kind: "check",
    version: `v-${name}`,
    critical: false,
    requires: [],
    uses: null,
    gate: null,
    tier: 1,
    ...overrides,
  };
}

const STAGES: Stage[] = ["sync", "meta", "labels", "media"];

// The master's real task ids, spelled exactly as hflow.runtime mints them.
const RESOLVE_TASK_ID = "resolve_profile";

/**
 * The generated topology's real shape, small enough to read in one screen.
 *
 * The master's edges are all three families the renderer declares, because two
 * of them overlap and the overlap is what the run level has to resolve: the
 * profile task feeds EVERY stage gate, each gate feeds its own trigger, and
 * each trigger feeds the NEXT stage's gate. Leaving the third family out would
 * make a straight-line assertion pass for the wrong reason.
 */
function pipelineGraph(overrides: Partial<PipelineGraph> = {}): PipelineGraph {
  return {
    dag_ids_known: true,
    steps_known: true,
    master: {
      dag_id: "kitchen_ingest",
      tasks: [
        dagTask(RESOLVE_TASK_ID),
        ...STAGES.flatMap((stage) => [
          dagTask(`enabled_${stage}`),
          dagTask(`trigger_${stage}`, { deferred: true }),
        ]),
      ],
      edges: [
        ...STAGES.map((stage) => [RESOLVE_TASK_ID, `enabled_${stage}`] as [string, string]),
        ...STAGES.map((stage) => [`enabled_${stage}`, `trigger_${stage}`] as [string, string]),
        ...STAGES.slice(0, -1).map(
          (stage, index) =>
            [`trigger_${stage}`, `enabled_${STAGES[index + 1]}`] as [string, string],
        ),
      ],
    },
    stages: STAGES.map((stage) => ({
      stage,
      title: `${stage} title`,
      description: `${stage} description`,
      gate_task_id: `enabled_${stage}`,
      trigger_task_id: `trigger_${stage}`,
      enabling_profiles: stage === "meta" ? ["full", "metadata_backfill"] : ["full"],
      dag: {
        dag_id: `kitchen_ingest_${stage}`,
        tasks: [
          dagTask("plan_batches"),
          dagTask("process_batch", { mapped: true }),
          dagTask("error_budget_gate"),
        ],
        edges: [
          ["plan_batches", "process_batch"],
          ["process_batch", "error_budget_gate"],
        ],
      },
      engine_steps: stage === "meta" ? [{ name: "catalog registration", summary: "append" }] : [],
      user_steps: stage === "meta" ? [userStep("cheap_check")] : [],
    })),
    quarantine_gate: {
      from_stage: "meta",
      to_stages: ["labels", "media"],
      critical_step_names: ["cheap_check"],
      explanation: "a False verdict from a critical check quarantines the episode",
    },
    ...overrides,
  };
}

function taskInstance(
  taskId: string,
  state: string | null,
  mapIndex = -1,
  overrides: Partial<RunTaskInstance> = {},
): RunTaskInstance {
  return {
    task_id: taskId,
    state,
    start_date: "2026-08-23T10:00:00Z",
    end_date: "2026-08-23T10:00:04Z",
    queued_at: null,
    try_number: 1,
    map_index: mapIndex,
    duration_s: 4,
    ...overrides,
  };
}

function runGraph(
  options: { metaState?: string | null; metaRunId?: string | null; mappedStates?: string[] } = {},
): RunGraph {
  const mappedStates = options.mappedStates ?? ["success", "success"];
  const metaRunId = options.metaRunId === undefined ? "meta_run_1" : options.metaRunId;
  return {
    master: {
      dag_run_id: "master_run_1",
      state: "success",
      tasks: [
        taskInstance(RESOLVE_TASK_ID, "success"),
        ...STAGES.flatMap((stage) => [
          taskInstance(`enabled_${stage}`, "success"),
          taskInstance(`trigger_${stage}`, "success"),
        ]),
      ],
    },
    stages: STAGES.map((stage) => {
      const isMeta = stage === "meta";
      const byState: Record<string, number> = {};
      for (const state of mappedStates) byState[state] = (byState[state] ?? 0) + 1;
      return {
        stage,
        dag_id: `kitchen_ingest_${stage}`,
        dag_run_id: isMeta ? metaRunId : null,
        state: isMeta ? (options.metaState ?? "success") : null,
        match: isMeta && metaRunId !== null ? ("heuristic" as const) : null,
        tasks: isMeta
          ? [
              taskInstance("plan_batches", "success"),
              ...mappedStates.map((state, index) => taskInstance("process_batch", state, index)),
              taskInstance("error_budget_gate", "success"),
            ]
          : [],
        mapped_summary: isMeta
          ? { task_id: "process_batch", total: mappedStates.length, by_state: byState }
          : null,
      };
    }),
  };
}

function episodePage(rows: Record<string, unknown>[], total = rows.length): EpisodePage {
  return { rows, total, columns: [], sql: "SELECT 1" };
}

function build(
  focus: CanvasFocus,
  input: {
    pipeline?: PipelineGraph;
    run?: RunGraph | null;
    episodes?: EpisodePage | null;
    dossier?: EpisodeDossier | null;
  } = {},
): CanvasGraph {
  return buildGraph({
    focus,
    pipeline: input.pipeline ?? pipelineGraph(),
    run: input.run ?? null,
    episodes: input.episodes ?? null,
    dossier: input.dossier ?? null,
  });
}

function nodeIds(graph: CanvasGraph): string[] {
  return graph.nodes.map((node) => node.id);
}

function edgePairs(graph: CanvasGraph): string[] {
  return graph.edges.map((edge) => `${edge.source}->${edge.target}`);
}

describe("the run level", () => {
  it("draws one node per stage, merging that stage's gate and trigger", () => {
    const graph = build({ level: "run" });
    expect(nodeIds(graph)).toEqual([RESOLVE_TASK_ID, ...STAGES]);
    // Neither master task id survives as a node of its own...
    for (const stage of STAGES) {
      expect(nodeIds(graph)).not.toContain(`enabled_${stage}`);
      expect(nodeIds(graph)).not.toContain(`trigger_${stage}`);
    }
    // ...but both are still named on the merged node, so nothing is hidden.
    const meta = graph.nodes.find((node) => node.id === "meta");
    expect(meta?.data.detail.map((line) => line.label)).toEqual(
      expect.arrayContaining(["enabled_meta", "trigger_meta"]),
    );
  });

  it("rewrites the master's chain onto the merged nodes", () => {
    const graph = build({ level: "run" });
    expect(edgePairs(graph)).toContain(`${RESOLVE_TASK_ID}->sync`);
    expect(edgePairs(graph)).toContain("sync->meta");
    // The gate-to-trigger edge inside one stage would be a self-edge, so it is
    // dropped rather than drawn as a loop.
    expect(edgePairs(graph)).not.toContain("meta->meta");
  });

  it("states each ordering once, dropping the shortcuts the chain implies", () => {
    // The master feeds every stage gate from the profile task AND chains the
    // stages, so resolve->meta is already implied by resolve->sync->meta.
    const graph = build({ level: "run" });
    expect(edgePairs(graph)).not.toContain(`${RESOLVE_TASK_ID}->meta`);
    expect(edgePairs(graph)).not.toContain(`${RESOLVE_TASK_ID}->media`);
    // The ordering survives, just stated along the chain: a straight line.
    expect(edgePairs(graph).sort()).toEqual([
      "labels->media",
      "meta->labels",
      `${RESOLVE_TASK_ID}->sync`,
      "sync->meta",
    ]);
  });

  it("drills into a stage from its merged node", () => {
    const graph = build({ level: "run" });
    const node = graph.nodes.find((candidate) => candidate.id === "labels");
    expect(node?.data.drillTo).toEqual({ level: "stage", stage: "labels" });
  });

  it("offers no drill-down from a task that is not about one stage", () => {
    const graph = build({ level: "run" });
    const node = graph.nodes.find((candidate) => candidate.id === RESOLVE_TASK_ID);
    expect(node?.data.drillTo).toBeNull();
  });

  it("shows a gate's skip rather than the trigger it prevented", () => {
    const skipped = runGraph();
    const withSkippedSync: RunGraph = {
      ...skipped,
      master: {
        ...skipped.master,
        tasks: skipped.master.tasks.map((task) =>
          task.task_id === "enabled_sync" ? { ...task, state: "skipped" } : task,
        ),
      },
    };
    const node = build({ level: "run" }, { run: withSkippedSync }).nodes.find(
      (candidate) => candidate.id === "sync",
    );
    expect(node?.data.badges).toContain("skipped by profile");
    // Muted, not green: the stage did not succeed, it did not happen.
    expect(node?.data.tone).toBe("muted");
  });

  it("says so when the shape is all it is showing", () => {
    expect(build({ level: "run" }).notices.join(" ")).toContain("No run selected");
    expect(build({ level: "run" }, { run: runGraph() }).notices.join(" ")).not.toContain(
      "No run selected",
    );
  });

  it("colours a task from the selected run's instance, not from the topology", () => {
    const failed = runGraph();
    const withFailure: RunGraph = {
      ...failed,
      master: {
        ...failed.master,
        tasks: failed.master.tasks.map((task) =>
          task.task_id === "trigger_meta" ? { ...task, state: "failed" } : task,
        ),
      },
    };
    const graph = build({ level: "run" }, { run: withFailure });
    const node = graph.nodes.find((candidate) => candidate.id === "meta");
    expect(node?.data.tone).toBe("err");
  });

  it("hangs the recorded episodes off the end of the stage chain, dashed", () => {
    const graph = build({ level: "run" }, { run: runGraph(), episodes: episodePage([], 7) });
    const node = graph.nodes.find((candidate) => candidate.id === "~episodes");
    expect(node?.data.badges).toEqual(["7"]);
    expect(node?.data.drillTo).toEqual({ level: "episodes" });
    const incoming = graph.edges.filter((edge) => edge.target === "~episodes");
    // Off the chain's one sink, and dashed: the rows are appended inside the
    // stage sub-DAGs, so nothing on this level really precedes them.
    expect(incoming.map((edge) => edge.source)).toEqual(["media"]);
    expect(incoming[0]?.dashed).toBe(true);
  });

  it("offers no episodes branch without a run to attribute them to", () => {
    expect(nodeIds(build({ level: "run" }))).not.toContain("~episodes");
  });
});

describe("the stage level", () => {
  it("expands the mapped task into one node per instance, rewiring both sides", () => {
    const graph = build({ level: "stage", stage: "meta" }, { run: runGraph() });
    expect(nodeIds(graph)).toContain("process_batch~0");
    expect(nodeIds(graph)).toContain("process_batch~1");
    expect(nodeIds(graph)).not.toContain("process_batch");
    // The fan is a real fan-out AND a real join: the plan feeds every instance,
    // and the budget gate waits for all of them.
    expect(edgePairs(graph)).toContain("plan_batches->process_batch~0");
    expect(edgePairs(graph)).toContain("plan_batches->process_batch~1");
    expect(edgePairs(graph)).toContain("process_batch~0->error_budget_gate");
    expect(edgePairs(graph)).toContain("process_batch~1->error_budget_gate");
  });

  it("stacks the fan instead of drawing it once it stops being readable", () => {
    const wide = runGraph({ mappedStates: Array.from({ length: 40 }, () => "success") });
    const graph = build({ level: "stage", stage: "meta" }, { run: wide });
    expect(nodeIds(graph)).toContain("process_batch");
    expect(nodeIds(graph)).not.toContain("process_batch~0");
    const stacked = graph.nodes.find((node) => node.id === "process_batch");
    expect(stacked?.data.badges).toContain("x40");
    expect(stacked?.data.badges).toContain("40 success");
  });

  it("takes the worst state in a stacked fan, so one failure in many is visible", () => {
    const mostlyFine = runGraph({
      mappedStates: [...Array.from({ length: 39 }, () => "success"), "failed"],
    });
    const graph = build({ level: "stage", stage: "meta" }, { run: mostlyFine });
    expect(graph.nodes.find((node) => node.id === "process_batch")?.data.tone).toBe("err");
  });

  it("leaves the mapped task unexpanded when no run is selected", () => {
    const graph = build({ level: "stage", stage: "meta" });
    const node = graph.nodes.find((candidate) => candidate.id === "process_batch");
    expect(node?.data.badges).toContain("fans out per batch");
    expect(node?.data.drillTo).toEqual({ level: "steps", stage: "meta" });
  });

  it("never offers an episodes branch per stage", () => {
    // Deliberate: the catalog keeps one row per episode, so a stage superseded
    // by a later stage of the same ingest would honestly answer "0 episodes".
    // The branch lives on the run instead.
    for (const stage of STAGES) {
      expect(nodeIds(build({ level: "stage", stage }, { run: runGraph() }))).not.toContain(
        "~episodes",
      );
    }
  });

  it("says when the selected run never reached this stage", () => {
    const graph = build({ level: "stage", stage: "labels" }, { run: runGraph() });
    expect(graph.notices.join(" ")).toContain("did not run");
  });

  it("carries the heuristic attribution caveat only when a stage run was matched", () => {
    const matched = build({ level: "stage", stage: "meta" }, { run: runGraph() });
    expect(matched.notices.join(" ")).toContain("matched by time window");
    const unmatched = build({ level: "stage", stage: "sync" }, { run: runGraph() });
    expect(unmatched.notices.join(" ")).not.toContain("matched by time window");
  });
});

describe("the steps level", () => {
  it("never draws an edge between two steps of the same tier", () => {
    const pipeline = pipelineGraph();
    const meta = pipeline.stages.find((stage) => stage.stage === "meta");
    if (meta === undefined) throw new Error("fixture has no meta stage");
    const withThreeChecks: PipelineGraph = {
      ...pipeline,
      stages: pipeline.stages.map((stage) =>
        stage.stage === "meta"
          ? {
              ...stage,
              user_steps: [userStep("check_a"), userStep("check_b"), userStep("check_c")],
            }
          : stage,
      ),
    };
    const graph = build({ level: "steps", stage: "meta" }, { pipeline: withThreeChecks });
    const stepIds = new Set(["~step:check_a", "~step:check_b", "~step:check_c"]);
    for (const edge of graph.edges) {
      expect(stepIds.has(edge.source) && stepIds.has(edge.target)).toBe(false);
    }
    expect(graph.notices.join(" ")).toContain("no ordering");
  });

  it("separates the tiers with a barrier, because that ordering is real", () => {
    const pipeline = pipelineGraph();
    const withBothTiers: PipelineGraph = {
      ...pipeline,
      stages: pipeline.stages.map((stage) =>
        stage.stage === "meta"
          ? {
              ...stage,
              user_steps: [
                userStep("cheap", { tier: 1 }),
                userStep("expensive", { tier: 2, requires: ["cheap"] }),
              ],
            }
          : stage,
      ),
    };
    const graph = build({ level: "steps", stage: "meta" }, { pipeline: withBothTiers });
    expect(edgePairs(graph)).toContain("~step:cheap->~tier-barrier");
    expect(edgePairs(graph)).toContain("~tier-barrier->~step:expensive");
  });

  it("omits the tier barrier when only one tier has steps", () => {
    const graph = build({ level: "steps", stage: "meta" });
    expect(nodeIds(graph)).not.toContain("~tier-barrier");
  });

  it("puts meta's catalog append after the checks and the gate", () => {
    const graph = build({ level: "steps", stage: "meta" });
    const ids = nodeIds(graph);
    expect(ids.indexOf("~step:cheap_check")).toBeLessThan(ids.indexOf("~quarantine:decision"));
    expect(ids.indexOf("~quarantine:decision")).toBeLessThan(
      ids.indexOf("~engine:catalog registration"),
    );
  });

  it("draws the quarantine gate as an ENTRY condition on the receiving stages", () => {
    const labels = build({ level: "steps", stage: "labels" });
    expect(nodeIds(labels)).toContain("~quarantine:entry");
    expect(nodeIds(labels)).not.toContain("~quarantine:decision");
    const meta = build({ level: "steps", stage: "meta" });
    expect(nodeIds(meta)).toContain("~quarantine:decision");
    expect(nodeIds(meta)).not.toContain("~quarantine:entry");
  });

  it("refuses to guess when the server has no pipeline imported", () => {
    const graph = build(
      { level: "steps", stage: "meta" },
      { pipeline: pipelineGraph({ steps_known: false }) },
    );
    expect(graph.nodes).toHaveLength(0);
    expect(graph.emptyMessage).toContain("--pipeline");
  });

  it("carries a step's gate onto the node rather than just the critical flag", () => {
    const pipeline = pipelineGraph();
    const gated: PipelineGraph = {
      ...pipeline,
      stages: pipeline.stages.map((stage) =>
        stage.stage === "meta"
          ? {
              ...stage,
              user_steps: [
                userStep("blur", {
                  critical: true,
                  gate: {
                    accept_when: [
                      {
                        key_pattern: "blur_fraction",
                        comparison: "at_most",
                        value: 0.3,
                        across: "every_key",
                      },
                    ],
                  },
                }),
              ],
            }
          : stage,
      ),
    };
    const graph = build({ level: "steps", stage: "meta" }, { pipeline: gated });
    const node = graph.nodes.find((candidate) => candidate.id === "~step:blur");
    expect(node?.data.subtitle).toBe("blur_fraction <= 0.3 (every key)");
    expect(node?.data.badges).toContain("critical");
  });
});

describe("the episodes level", () => {
  it("fans the recorded episodes off the master run and drills into each", () => {
    const graph = build(
      { level: "episodes" },
      {
        run: runGraph(),
        episodes: episodePage([
          { episode_id: "abcdef0123456789", status: "ok", task: "pour" },
          { episode_id: "fedcba9876543210", status: "quarantined", task: "pour" },
        ]),
      },
    );
    expect(nodeIds(graph)).toContain("~episode:abcdef0123456789");
    expect(edgePairs(graph)).toContain("~run->~episode:abcdef0123456789");
    const quarantined = graph.nodes.find((node) => node.id === "~episode:fedcba9876543210");
    expect(quarantined?.data.tone).toBe("warn");
    expect(quarantined?.data.drillTo).toEqual({
      level: "episode",
      episodeId: "fedcba9876543210",
    });
  });

  it("says how many episodes it is NOT showing rather than silently truncating", () => {
    const graph = build(
      { level: "episodes" },
      { run: runGraph(), episodes: episodePage([{ episode_id: "abc", status: "ok" }], 900) },
    );
    expect(graph.notices.join(" ")).toContain("first 1 of 900");
  });

  it("explains an empty result, naming re-ingest as one reason for it", () => {
    const graph = build({ level: "episodes" }, { run: runGraph(), episodes: episodePage([]) });
    expect(graph.nodes).toHaveLength(0);
    expect(graph.emptyMessage).toContain("re-ingested");
  });
});

describe("the episode query's scope", () => {
  it("is every matched stage run of the selected master run", () => {
    // Deliberately the union: the catalog keeps one row per episode, so asking
    // with a single stage's id answers 0 for every stage the same ingest later
    // superseded.
    expect(stageRunIds(runGraph())).toEqual(["meta_run_1"]);
    expect(stageRunIds(null)).toEqual([]);
  });
});

describe("the episode level", () => {
  function dossier(overrides: Partial<EpisodeDossier> = {}): EpisodeDossier {
    return {
      episode: { status: "ok", quarantine_tags: [], episode_id: "abc", task: "pour" },
      check_runs: [
        {
          check_name: "cheap_check",
          check_version: "v-cheap_check",
          critical: false,
          status: "passed",
          duration_s: 0.2,
          error: null,
          recorded_at: "2026-08-23T10:00:00Z",
          run_fingerprint: "fp",
        },
      ],
      measurements: [],
      intervals: [],
      tags: [],
      history: [],
      media: [],
      canonical_url: null,
      ...overrides,
    };
  }

  it("reads a check's verdict from the catalog and its gate from the live pipeline", () => {
    const pipeline = pipelineGraph();
    const gated: PipelineGraph = {
      ...pipeline,
      stages: pipeline.stages.map((stage) =>
        stage.stage === "meta"
          ? {
              ...stage,
              user_steps: [
                userStep("cheap_check", {
                  gate: {
                    accept_when: [
                      {
                        key_pattern: "gap_*",
                        comparison: "at_least",
                        value: 2,
                        across: "any_key",
                      },
                    ],
                  },
                }),
              ],
            }
          : stage,
      ),
    };
    const graph = build(
      { level: "episode", episodeId: "abc" },
      { pipeline: gated, dossier: dossier() },
    );
    const node = graph.nodes.find((candidate) => candidate.id === "~check:tier 1:cheap_check");
    expect(node?.data.tone).toBe("ok");
    expect(node?.data.subtitle).toBe("gap_* >= 2 (any key)");
  });

  it("shows a recorded check the pipeline no longer registers instead of dropping it", () => {
    const graph = build(
      { level: "episode", episodeId: "abc" },
      {
        dossier: dossier({
          check_runs: [
            {
              check_name: "deleted_check",
              check_version: "old",
              critical: false,
              status: "passed",
              duration_s: 1,
              error: null,
              recorded_at: null,
              run_fingerprint: null,
            },
          ],
        }),
      },
    );
    const node = graph.nodes.find(
      (candidate) => candidate.id === "~check:not registered:deleted_check",
    );
    expect(node?.data.badges).toContain("not registered");
    expect(graph.notices.join(" ")).toContain(
      "1 recorded check no longer exists in the current pipeline",
    );
  });

  it("keeps a measured check distinct from a passed one", () => {
    const graph = build(
      { level: "episode", episodeId: "abc" },
      {
        dossier: dossier({
          check_runs: [
            {
              check_name: "cheap_check",
              check_version: "v",
              critical: false,
              status: "measured",
              duration_s: 1,
              error: null,
              recorded_at: null,
              run_fingerprint: null,
            },
          ],
        }),
      },
    );
    expect(graph.nodes.find((node) => node.id === "~check:tier 1:cheap_check")?.data.tone).toBe(
      "info",
    );
  });

  it("lists a check's own measurements, elided past the display limit", () => {
    const graph = build(
      { level: "episode", episodeId: "abc" },
      {
        dossier: dossier({
          measurements: Array.from({ length: 9 }, (_unused, index) => ({
            key: `key_${index}`,
            value_double: index,
            value_text: null,
            value_bool: null,
            check_name: "cheap_check",
            check_version: "v",
            recorded_at: null,
          })),
        }),
      },
    );
    const node = graph.nodes.find((candidate) => candidate.id === "~check:tier 1:cheap_check");
    const measurements = node?.data.detail.find((line) => line.label === "measurements");
    expect(measurements?.value).toContain("key_0 = 0");
    expect(measurements?.value).toContain("and 3 more");
  });
});
