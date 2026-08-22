// Hand-rolled left-to-right DAG layout: longest-path layering plus barycenter
// row ordering, then bezier routing. Pure functions, no dependency — the
// offline rule (and the "no captive UI" posture) means the workspace UI ships
// no graph or chart library, so the ~100 lines below are the layout engine.

export type DagEdgeKind = "dependency" | "gate";

/** The minimum a node must carry to be placed; DagGraph passes richer objects. */
export interface LayoutInputNode {
  id: string;
}

export interface LayoutInputEdge {
  from: string;
  to: string;
  /** "gate" edges are conditional (the quarantine gate), drawn distinctly. */
  kind?: DagEdgeKind;
  label?: string | null;
}

export interface PositionedNode {
  id: string;
  layer: number;
  row: number;
  x: number;
  y: number;
}

export interface RoutedEdge {
  key: string;
  path: string;
  kind: DagEdgeKind;
  label: string | null;
  labelX: number;
  labelY: number;
  /** Hover text on the connector itself. */
  title: string;
}

export interface DagLayout {
  nodes: Map<string, PositionedNode>;
  edges: RoutedEdge[];
  width: number;
  height: number;
}

export const NODE_WIDTH = 158;
/** Two lines of task id plus the state/badge row: the longest generated id
 * (quarantine_budget_gate) wraps rather than being cut off. */
export const NODE_HEIGHT = 58;

const COLUMN_GAP = 44;
const ROW_GAP = 18;
const CANVAS_PADDING = 14;
/** Room for the offset cards behind a mapped (fan-out) node. */
const STACK_ALLOWANCE = 10;
/** Control-point inset: small, so long connectors rise steeply and clear the
 * nodes they fly over instead of grazing their boxes. */
const ARC_CONTROL_INSET = 28;
const ARC_BASE_LIFT = 30;
const ARC_LIFT_PER_LAYER = 11;
const GATE_BASE_DIP = 32;
/** Long connectors stop climbing after this many layers: by then they already
 * clear every node they fly over, and more lift is only wasted whitespace. */
const MAX_ARC_LIFT_STEPS = 5;

/** Longest-path layering (Kahn): a node sits one layer right of its deepest
 * upstream. Nodes inside a cycle never dequeue and keep their relaxed layer —
 * a cycle is not a DAG, and rendering something beats rendering nothing. */
function assignLayers(nodeIds: readonly string[], edges: readonly LayoutInputEdge[]) {
  const layerByNodeId = new Map<string, number>(nodeIds.map((nodeId) => [nodeId, 0]));
  const downstreamByNodeId = new Map<string, string[]>(nodeIds.map((nodeId) => [nodeId, []]));
  const remainingUpstream = new Map<string, number>(nodeIds.map((nodeId) => [nodeId, 0]));

  for (const edge of edges) {
    // Edges naming a task this graph does not render are ignored, not fatal.
    if (!layerByNodeId.has(edge.from) || !layerByNodeId.has(edge.to)) continue;
    downstreamByNodeId.get(edge.from)?.push(edge.to);
    remainingUpstream.set(edge.to, (remainingUpstream.get(edge.to) ?? 0) + 1);
  }

  const queue = nodeIds.filter((nodeId) => remainingUpstream.get(nodeId) === 0);
  for (let cursor = 0; cursor < queue.length; cursor++) {
    const nodeId = queue[cursor] as string;
    const nodeLayer = layerByNodeId.get(nodeId) ?? 0;
    for (const downstreamId of downstreamByNodeId.get(nodeId) ?? []) {
      layerByNodeId.set(
        downstreamId,
        Math.max(layerByNodeId.get(downstreamId) ?? 0, nodeLayer + 1),
      );
      const remaining = (remainingUpstream.get(downstreamId) ?? 0) - 1;
      remainingUpstream.set(downstreamId, remaining);
      if (remaining === 0) queue.push(downstreamId);
    }
  }
  return layerByNodeId;
}

/** Two barycenter sweeps: pull each node towards the mean row of its
 * neighbours in the previous/next column, which straightens the chains and
 * keeps parallel branches from crossing. */
function orderRows(
  columns: string[][],
  upstreamByNodeId: Map<string, string[]>,
  downstreamByNodeId: Map<string, string[]>,
): Map<string, number> {
  const rowByNodeId = new Map<string, number>();
  const refreshRows = () => {
    for (const column of columns) {
      for (const [rowIndex, nodeId] of column.entries()) rowByNodeId.set(nodeId, rowIndex);
    }
  };
  refreshRows();

  const barycenterOf = (nodeId: string, neighbours: readonly string[]): number => {
    const rows = neighbours
      .map((neighbourId) => rowByNodeId.get(neighbourId))
      .filter((row): row is number => row !== undefined);
    if (rows.length === 0) return rowByNodeId.get(nodeId) ?? 0;
    return rows.reduce((total, row) => total + row, 0) / rows.length;
  };

  const sortColumn = (column: string[], neighboursOf: Map<string, string[]>) => {
    const keyed = column.map((nodeId, index) => ({
      nodeId,
      index,
      key: barycenterOf(nodeId, neighboursOf.get(nodeId) ?? []),
    }));
    keyed.sort((left, right) => left.key - right.key || left.index - right.index);
    column.splice(0, column.length, ...keyed.map((entry) => entry.nodeId));
  };

  for (let pass = 0; pass < 2; pass++) {
    for (let columnIndex = 1; columnIndex < columns.length; columnIndex++) {
      sortColumn(columns[columnIndex] as string[], upstreamByNodeId);
      refreshRows();
    }
    for (let columnIndex = columns.length - 2; columnIndex >= 0; columnIndex--) {
      sortColumn(columns[columnIndex] as string[], downstreamByNodeId);
      refreshRows();
    }
  }
  return rowByNodeId;
}

export function layoutDag(
  nodes: readonly LayoutInputNode[],
  edges: readonly LayoutInputEdge[],
): DagLayout {
  const nodeIds = nodes.map((node) => node.id);
  const layerByNodeId = assignLayers(nodeIds, edges);

  const upstreamByNodeId = new Map<string, string[]>(nodeIds.map((nodeId) => [nodeId, []]));
  const downstreamByNodeId = new Map<string, string[]>(nodeIds.map((nodeId) => [nodeId, []]));
  const drawableEdges = edges.filter(
    (edge) => layerByNodeId.has(edge.from) && layerByNodeId.has(edge.to),
  );
  for (const edge of drawableEdges) {
    downstreamByNodeId.get(edge.from)?.push(edge.to);
    upstreamByNodeId.get(edge.to)?.push(edge.from);
  }

  const columnCount = Math.max(1, ...nodeIds.map((nodeId) => (layerByNodeId.get(nodeId) ?? 0) + 1));
  const columns: string[][] = Array.from({ length: columnCount }, () => []);
  for (const nodeId of nodeIds) columns[layerByNodeId.get(nodeId) ?? 0]?.push(nodeId);
  const rowByNodeId = orderRows(columns, upstreamByNodeId, downstreamByNodeId);

  // Headroom for connectors that fly over a column, and legroom for the gate
  // connectors, which dip below the band so they never trace a real edge.
  const spanOf = (edge: LayoutInputEdge): number =>
    (layerByNodeId.get(edge.to) ?? 0) - (layerByNodeId.get(edge.from) ?? 0);
  const liftStepsOf = (span: number): number => Math.min(Math.abs(span) - 1, MAX_ARC_LIFT_STEPS);
  const liftOf = (span: number): number => ARC_BASE_LIFT + liftStepsOf(span) * ARC_LIFT_PER_LAYER;
  const dipOf = (span: number): number => GATE_BASE_DIP + liftStepsOf(span) * ARC_LIFT_PER_LAYER;
  const arcLifts = drawableEdges
    .filter((edge) => edge.kind !== "gate" && Math.abs(spanOf(edge)) > 1)
    .map((edge) => liftOf(spanOf(edge)));
  const gateDips = drawableEdges
    .filter((edge) => edge.kind === "gate")
    .map((edge) => dipOf(spanOf(edge)));
  const headroom = arcLifts.length > 0 ? Math.max(...arcLifts) : 0;
  const legroom = gateDips.length > 0 ? Math.max(...gateDips) + 14 : 0;

  const tallestColumn = Math.max(1, ...columns.map((column) => column.length));
  const bandHeight = tallestColumn * NODE_HEIGHT + (tallestColumn - 1) * ROW_GAP;
  const bandTop = CANVAS_PADDING + headroom;

  const positionedNodes = new Map<string, PositionedNode>();
  columns.forEach((column, columnIndex) => {
    const columnHeight = column.length * NODE_HEIGHT + (column.length - 1) * ROW_GAP;
    const columnTop = bandTop + (bandHeight - columnHeight) / 2;
    column.forEach((nodeId, rowIndex) => {
      positionedNodes.set(nodeId, {
        id: nodeId,
        layer: columnIndex,
        row: rowByNodeId.get(nodeId) ?? rowIndex,
        x: CANVAS_PADDING + columnIndex * (NODE_WIDTH + COLUMN_GAP),
        y: columnTop + rowIndex * (NODE_HEIGHT + ROW_GAP),
      });
    });
  });

  const routedEdges: RoutedEdge[] = [];
  for (const edge of drawableEdges) {
    const source = positionedNodes.get(edge.from);
    const target = positionedNodes.get(edge.to);
    if (!source || !target) continue;
    const kind: DagEdgeKind = edge.kind ?? "dependency";
    const startX = source.x + NODE_WIDTH;
    const startY = source.y + NODE_HEIGHT / 2;
    const endX = target.x;
    const endY = target.y + NODE_HEIGHT / 2;
    const span = target.layer - source.layer;

    let path: string;
    let labelX = (startX + endX) / 2;
    let labelY = (startY + endY) / 2 - 7;
    if (kind === "gate") {
      const apexY = bandTop + bandHeight + dipOf(span);
      path = `M${startX} ${startY} C${startX + ARC_CONTROL_INSET} ${apexY} ${endX - ARC_CONTROL_INSET} ${apexY} ${endX} ${endY}`;
      labelY = apexY - 6;
    } else if (Math.abs(span) > 1) {
      const apexY = bandTop - liftOf(span);
      path = `M${startX} ${startY} C${startX + ARC_CONTROL_INSET} ${apexY} ${endX - ARC_CONTROL_INSET} ${apexY} ${endX} ${endY}`;
      labelY = apexY + 14;
    } else {
      const bend = Math.max(22, (endX - startX) * 0.45);
      path = `M${startX} ${startY} C${startX + bend} ${startY} ${endX - bend} ${endY} ${endX} ${endY}`;
      labelX = (startX + endX) / 2;
    }
    routedEdges.push({
      key: `${edge.from}->${edge.to}:${kind}`,
      path,
      kind,
      label: edge.label ?? null,
      labelX,
      labelY,
      title: edge.label ? `${edge.from} → ${edge.to} (${edge.label})` : `${edge.from} → ${edge.to}`,
    });
  }

  return {
    nodes: positionedNodes,
    edges: routedEdges,
    width:
      CANVAS_PADDING * 2 +
      columnCount * NODE_WIDTH +
      (columnCount - 1) * COLUMN_GAP +
      STACK_ALLOWANCE,
    height: bandTop + bandHeight + legroom + CANVAS_PADDING + STACK_ALLOWANCE,
  };
}
