// The whole screen: a breadcrumb, one canvas, one inspector.
//
// Everything the canvas draws comes from buildGraph, and everything it knows
// comes from the hooks in api.ts, so this file only decides what is on screen
// and what a click does.

import {
  Background,
  Controls,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  useStore,
} from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type RuntimeRunSummary,
  useEpisodeDossier,
  usePipelineGraph,
  useRunEpisodes,
  useRunGraph,
  useRuntimeRuns,
  useRuntimeStatus,
  useWorkspaceConfig,
} from "./api";
import {
  buildGraph,
  type CanvasGraph,
  type CanvasNodeData,
  stageRunIds,
} from "./canvas/buildGraph";
import { CANVAS_NODE_TYPES, type FlowNode } from "./canvas/CanvasNodeView";
import {
  type Breadcrumb,
  breadcrumbs,
  type CanvasFocus,
  parentFocus,
  RUN_FOCUS,
} from "./canvas/focus";
import { graphBounds, layoutGraph } from "./canvas/layout";

/** What each level is, in one line, so the canvas explains itself. */
const LEVEL_CAPTIONS: Record<CanvasFocus["level"], string> = {
  run: "The ingest DAG. Each stage is gated by the run profile, then triggered in order.",
  stage:
    "Inside one stage: batches are planned, process_batch fans out over them, and a " +
    "budget gate closes the stage.",
  steps: "What one process_batch does to each episode in its batch.",
  episodes: "The episodes whose current catalog row came out of this run.",
  episode: "Every check recorded for this episode, with its verdict and the gate it was judged on.",
};

export function App() {
  return (
    <ReactFlowProvider>
      <Workspace />
    </ReactFlowProvider>
  );
}

function Workspace() {
  const [focus, setFocus] = useState<CanvasFocus>(RUN_FOCUS);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  const config = useWorkspaceConfig();
  const runtimeAddressed = config.data?.capabilities.runtime === true;
  const status = useRuntimeStatus();
  const runs = useRuntimeRuns(runtimeAddressed);
  const pipelineGraph = usePipelineGraph();
  const runGraph = useRunGraph(selectedRunId);

  // The newest run, once, so the canvas opens on something real instead of on
  // an empty picker. Later refetches must not yank the selection off whatever
  // the user chose, so this only fires while nothing is selected.
  const newestRunId = runs.data?.runs[0]?.dag_run_id ?? null;
  useEffect(() => {
    if (selectedRunId === null && newestRunId !== null) setSelectedRunId(newestRunId);
  }, [selectedRunId, newestRunId]);

  // Every stage run of the selected master run, which is the filter the
  // episodes branch is scoped by (see stageRunIds and useRunEpisodes for why it
  // is the union and not one stage).
  const episodes = useRunEpisodes(stageRunIds(runGraph.data ?? null));
  const dossier = useEpisodeDossier(focus.level === "episode" ? focus.episodeId : null);

  const graph = useMemo(() => {
    if (pipelineGraph.data === undefined) return null;
    return buildGraph({
      focus,
      pipeline: pipelineGraph.data,
      run: runGraph.data ?? null,
      episodes: episodes.data ?? null,
      dossier: dossier.data ?? null,
    });
  }, [focus, pipelineGraph.data, runGraph.data, episodes.data, dossier.data]);

  const navigate = useCallback((next: CanvasFocus) => {
    setFocus(next);
    setSelectedNodeId(null);
  }, []);

  // Escape walks back out, the way every drill-down does.
  const parent = parentFocus(focus);
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape" && parent !== null) navigate(parent);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [parent, navigate]);

  const selectedNode = graph?.nodes.find((node) => node.id === selectedNodeId)?.data ?? null;

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">HFlow</span>
        <RunPicker
          runs={runs.data?.runs ?? []}
          selectedRunId={selectedRunId}
          onSelect={(runId) => {
            setSelectedRunId(runId);
            navigate(RUN_FOCUS);
          }}
          disabled={!runtimeAddressed}
        />
        <span className="topbar-spacer" />
        <span className={`chip tone-${status.data?.available === true ? "ok" : "muted"}`}>
          {runtimeAddressed
            ? status.data?.available === true
              ? "runtime up"
              : "runtime down"
            : "no runtime"}
        </span>
        {config.data === undefined ? null : (
          <span className="topbar-meta" title={config.data.data_root}>
            {config.data.data_root}
          </span>
        )}
      </header>

      <div className="body">
        <main className="stage-area">
          <nav className="crumbs" aria-label="Breadcrumb">
            {breadcrumbs(focus).map((crumb, index, trail) => (
              <CrumbButton
                key={`${crumb.focus.level}:${crumb.label}`}
                crumb={crumb}
                isLast={index === trail.length - 1}
                onSelect={navigate}
              />
            ))}
          </nav>
          <p className="caption">{LEVEL_CAPTIONS[focus.level]}</p>
          {(graph?.notices ?? []).map((notice) => (
            <p className="notice" key={notice}>
              {notice}
            </p>
          ))}
          <CanvasSurface
            graph={graph}
            error={pipelineGraph.error ?? runGraph.error ?? null}
            selectedNodeId={selectedNodeId}
            onSelect={setSelectedNodeId}
            onDrill={navigate}
          />
        </main>
        <aside className="inspector">
          <Inspector node={selectedNode} onDrill={navigate} />
        </aside>
      </div>
    </div>
  );
}

function CrumbButton({
  crumb,
  isLast,
  onSelect,
}: {
  crumb: Breadcrumb;
  isLast: boolean;
  onSelect: (focus: CanvasFocus) => void;
}) {
  return (
    <>
      <button
        type="button"
        className="crumb"
        aria-current={isLast ? "page" : undefined}
        disabled={isLast}
        onClick={() => onSelect(crumb.focus)}
      >
        {crumb.label}
      </button>
      {isLast ? null : <span className="crumb-sep">/</span>}
    </>
  );
}

function RunPicker({
  runs,
  selectedRunId,
  onSelect,
  disabled,
}: {
  runs: readonly RuntimeRunSummary[];
  selectedRunId: string | null;
  onSelect: (runId: string) => void;
  disabled: boolean;
}) {
  if (disabled) {
    return (
      <span className="topbar-meta">no runtime addressed &mdash; showing the DAG's shape</span>
    );
  }
  if (runs.length === 0) {
    return <span className="topbar-meta">no runs yet</span>;
  }
  return (
    <label className="run-picker">
      <span className="run-picker-label">run</span>
      <select
        value={selectedRunId ?? ""}
        onChange={(event) => onSelect(event.target.value)}
        // Every option's own tone would need a styled listbox; the selected
        // run's state is on the canvas, so the picker stays a plain select.
      >
        {runs.map((run) => (
          <option key={run.dag_run_id ?? ""} value={run.dag_run_id ?? ""}>
            {run.dag_run_id} ({run.state ?? "unknown"})
          </option>
        ))}
      </select>
    </label>
  );
}

function CanvasSurface({
  graph,
  error,
  selectedNodeId,
  onSelect,
  onDrill,
}: {
  graph: CanvasGraph | null;
  error: Error | null;
  selectedNodeId: string | null;
  onSelect: (nodeId: string) => void;
  onDrill: (focus: CanvasFocus) => void;
}) {
  if (error !== null) {
    return (
      <div className="surface surface-message">
        <p className="error-title">Could not draw this.</p>
        <p className="error-detail">{error.message}</p>
      </div>
    );
  }
  if (graph === null) {
    return <div className="surface surface-message">Loading the topology...</div>;
  }
  if (graph.nodes.length === 0) {
    return (
      <div className="surface surface-message">{graph.emptyMessage ?? "Nothing to draw."}</div>
    );
  }
  return (
    <div className="surface">
      <CanvasFlow
        graph={graph}
        selectedNodeId={selectedNodeId}
        onSelect={onSelect}
        onDrill={onDrill}
      />
    </div>
  );
}

/**
 * The flow itself, split out so it MOUNTS WITH ReactFlow.
 *
 * The framing effect below has to run against a mounted flow, and the
 * message states above unmount it -- so an effect living beside them would
 * fire while there is no canvas to frame.
 */
function CanvasFlow({
  graph,
  selectedNodeId,
  onSelect,
  onDrill,
}: {
  graph: CanvasGraph;
  selectedNodeId: string | null;
  onSelect: (nodeId: string) => void;
  onDrill: (focus: CanvasFocus) => void;
}) {
  const { fitBounds } = useReactFlow();
  // The flow's own measured container. React Flow fills it from a
  // ResizeObserver, which fires AFTER the render that changed the layout, so
  // framing without watching this uses the previous level's canvas height --
  // and clips the new graph by however much the notices above it grew.
  const flowSize = useStore((state) => `${Math.round(state.width)}x${Math.round(state.height)}`);
  const positioned = useMemo(() => layoutGraph(graph.nodes, graph.edges), [graph]);
  const flowNodes = useMemo<FlowNode[]>(
    () =>
      positioned.map((node) => ({
        id: node.id,
        type: "canvas" as const,
        position: { x: node.position.x, y: node.position.y },
        data: node.data,
        selected: node.id === selectedNodeId,
        draggable: false,
        // The box goes on `style`, not on the node's own width/height fields.
        // Setting those tells React Flow the size is already known, which
        // leaves `measured` unset -- and useNodesInitialized never turns true,
        // so nothing ever frames the graph. Styling it lets React Flow measure
        // the wrapper it just sized, which is the same number either way.
        style: { width: node.width, height: node.height },
      })),
    [positioned, selectedNodeId],
  );
  const flowEdges = useMemo(() => {
    // React Flow warns about an edge naming a node that is not on the canvas,
    // so a rewired level's leftover edge is dropped rather than handed over.
    const drawnNodeIds = new Set(positioned.map((node) => node.id));
    return graph.edges
      .filter((edge) => drawnNodeIds.has(edge.source) && drawnNodeIds.has(edge.target))
      .map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: "smoothstep" as const,
        label: edge.label ?? undefined,
        className: edge.dashed ? "edge-dashed" : undefined,
      }));
  }, [graph, positioned]);

  // Re-frame whenever the graph's own BOX changes: a new level, or a fan that
  // just expanded, has nothing to do with the previous viewport. Deliberately
  // not fitView: that one needs every node measured in the DOM first, so on a
  // first paint it silently frames nothing. The box is already known here.
  //
  // Keyed on the box and not on the graph object, because the 4s poll rebuilds
  // an identical graph and re-framing on every tick would pan under the reader.
  const bounds = useMemo(() => graphBounds(positioned), [positioned]);
  const lastFramedBox = useRef("");
  useEffect(() => {
    if (bounds === null) return;
    const box = `${bounds.x}:${bounds.y}:${bounds.width}:${bounds.height}@${flowSize}`;
    if (box === lastFramedBox.current) return;
    lastFramedBox.current = box;
    // One frame later: the flow measures its own container on mount, and
    // framing against a zero-sized container would land nowhere.
    const framed = requestAnimationFrame(() => fitBounds(bounds, { padding: 0.15, duration: 200 }));
    return () => cancelAnimationFrame(framed);
  }, [bounds, flowSize, fitBounds]);

  return (
    <ReactFlow
      nodes={flowNodes}
      edges={flowEdges}
      nodeTypes={CANVAS_NODE_TYPES}
      nodesDraggable={false}
      nodesConnectable={false}
      edgesFocusable={false}
      // Positions are computed, so there is nothing to persist and nothing to
      // undo; selecting is the only node interaction this canvas has.
      onNodeClick={(_event, node) => onSelect(node.id)}
      onNodeDoubleClick={(_event, node) => {
        const drillTo = (node.data as CanvasNodeData).drillTo;
        if (drillTo !== null) onDrill(drillTo);
      }}
      proOptions={{ hideAttribution: false }}
      minZoom={0.15}
      maxZoom={1.6}
    >
      <Background gap={18} size={1} />
      <Controls showInteractive={false} />
    </ReactFlow>
  );
}

function Inspector({
  node,
  onDrill,
}: {
  node: CanvasNodeData | null;
  onDrill: (focus: CanvasFocus) => void;
}) {
  if (node === null) {
    return (
      <div className="inspector-empty">
        <p>Select a node to see what it is.</p>
        <p className="inspector-hint">
          A node with a &rsaquo; has more inside it: open it from here, or double-click it. Escape
          goes back out.
        </p>
      </div>
    );
  }
  return (
    <>
      {/* Tone on the heading, badges below it. The node's SHAPE was here once
          and it told the reader nothing: "task" and "item" are this canvas's
          own vocabulary, not facts about what they selected. */}
      <h2 className={`inspector-title tone-${node.tone}`}>{node.title}</h2>
      {node.badges.length === 0 ? null : (
        <div className="node-badges">
          {node.badges.map((badge) => (
            <span className="badge" key={badge}>
              {badge}
            </span>
          ))}
        </div>
      )}
      {node.subtitle === null ? null : <p className="inspector-subtitle">{node.subtitle}</p>}
      <DrillButton drillTo={node.drillTo} onDrill={onDrill} />
      <dl className="inspector-detail">
        {node.detail.map((line) => (
          <div className="detail-line" key={`${line.label}:${line.value}`}>
            <dt>{line.label}</dt>
            <dd>{line.value}</dd>
          </div>
        ))}
      </dl>
    </>
  );
}

/** Its own component so the non-null focus is a narrowed value, not an assertion. */
function DrillButton({
  drillTo,
  onDrill,
}: {
  drillTo: CanvasFocus | null;
  onDrill: (focus: CanvasFocus) => void;
}) {
  if (drillTo === null) return null;
  return (
    <button type="button" className="drill-button" onClick={() => onDrill(drillTo)}>
      Open &rarr;
    </button>
  );
}
