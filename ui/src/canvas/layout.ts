// Positions for a built graph. Left to right, one rank per dependency depth --
// the reading order every workflow builder uses, so nobody has to learn this
// canvas before they can follow it.
//
// dagre is given fixed node sizes rather than measured ones. Measuring would
// mean rendering, measuring, then laying out, which flashes the graph at the
// wrong positions on every focus change; instead the CSS clamps content to
// these boxes, so what dagre is told is always the truth.

import dagre from "@dagrejs/dagre";
import type { CanvasEdge, CanvasNode, NodeShape } from "./buildGraph";

export interface PositionedNode extends CanvasNode {
  readonly position: { readonly x: number; readonly y: number };
  readonly width: number;
  readonly height: number;
}

// Boxes and gaps are deliberately tight. A level is framed by fitting its whole
// graph on screen, so every pixel of node width and rank gap is paid for in
// zoom: at the widths a comfortable card would want, a six-rank chain fits only
// at a zoom where none of its text can be read.
const NODE_SIZE: Record<NodeShape, { readonly width: number; readonly height: number }> = {
  task: { width: 196, height: 110 },
  item: { width: 196, height: 110 },
  // Furniture is smaller on purpose: a barrier or an anchor should read as a
  // marker between the real nodes, not as one of them.
  note: { width: 168, height: 92 },
};

/**
 * One node's box, as a FRESH object every call.
 *
 * The copy is not defensive style, it is required. dagre stores the object it
 * is handed as the node's label and writes the computed x/y onto that same
 * object, so handing every task node one shared literal makes all of them
 * share one position -- which lays the whole graph on top of itself.
 */
export function nodeSize(shape: NodeShape): { width: number; height: number } {
  return { ...NODE_SIZE[shape] };
}

export type GraphBounds = {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
};

/**
 * The box the laid-out graph occupies, or null for an empty graph.
 *
 * Computed here rather than asked of React Flow, because the positions are
 * already known: framing the canvas off this needs no node to have been
 * measured in the DOM first, so it works on the first paint of a level.
 */
export function graphBounds(nodes: readonly PositionedNode[]): GraphBounds | null {
  if (nodes.length === 0) return null;
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const node of nodes) {
    minX = Math.min(minX, node.position.x);
    minY = Math.min(minY, node.position.y);
    maxX = Math.max(maxX, node.position.x + node.width);
    maxY = Math.max(maxY, node.position.y + node.height);
  }
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

export function layoutGraph(
  nodes: readonly CanvasNode[],
  edges: readonly CanvasEdge[],
): PositionedNode[] {
  const graph = new dagre.graphlib.Graph();
  graph.setGraph({ rankdir: "LR", nodesep: 20, ranksep: 68, marginx: 16, marginy: 16 });
  graph.setDefaultEdgeLabel(() => ({}));

  for (const node of nodes) {
    graph.setNode(node.id, nodeSize(node.data.shape));
  }
  for (const edge of edges) {
    // An edge naming a node this level did not draw would make dagre invent an
    // empty one, so it is dropped instead. Levels that rewire a mapped task
    // into a fan are exactly where such an edge could slip through.
    if (graph.hasNode(edge.source) && graph.hasNode(edge.target)) {
      graph.setEdge(edge.source, edge.target);
    }
  }
  dagre.layout(graph);

  return nodes.map((node) => {
    const { width, height } = nodeSize(node.data.shape);
    const laid = graph.node(node.id);
    return {
      ...node,
      width,
      height,
      // dagre centres a node on its position; React Flow anchors at the corner.
      position: { x: (laid?.x ?? 0) - width / 2, y: (laid?.y ?? 0) - height / 2 },
    };
  });
}
