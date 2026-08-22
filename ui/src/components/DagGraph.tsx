import { useId, useMemo } from "react";
import {
  type DagEdgeKind,
  type LayoutInputEdge,
  layoutDag,
  NODE_HEIGHT,
  NODE_WIDTH,
} from "../dagLayout";
import { runStateTone } from "../runState";

// The graph renderer: SVG connectors under a layer of real <button> nodes, so
// nodes are focusable, Enter/Space selects, and text gets browser typography
// (ellipsis, hover titles) instead of hand-measured SVG glyphs.

export interface DagGraphNode {
  /** Task id — the identity the details panel and the API speak. */
  id: string;
  label: string;
  /** Hover text: what this task actually does. */
  summary: string;
  /** Dynamically mapped: rendered as a stacked card with a fan-out badge. */
  mapped?: boolean;
  /** Defers while waiting; rendered as "waits", never as a stalled spinner. */
  deferred?: boolean;
  /** Live task state; absent on the static topology view. */
  state?: string | null;
  /** Short readout at the node's foot: "×N", "12/15 ok", a duration. */
  badge?: string | null;
}

export type DagGraphEdge = LayoutInputEdge;

interface DagGraphProps {
  nodes: readonly DagGraphNode[];
  edges: readonly DagGraphEdge[];
  /** Accessible name for the graph region. */
  label: string;
  selectedNodeId: string | null;
  onSelectNode: (nodeId: string) => void;
  /** Nodes that expand something below the graph (stage triggers). */
  expandedNodeIds?: readonly string[];
}

/** Past this width the canvas overflows any realistic pane, so the caption
 * tells the reader the graph continues past the edge. */
const WIDE_GRAPH_PX = 900;

function arrowMarkerId(baseId: string, kind: DagEdgeKind): string {
  return `${baseId}-arrow-${kind}`;
}

export function DagGraph({
  nodes,
  edges,
  label,
  selectedNodeId,
  onSelectNode,
  expandedNodeIds = [],
}: DagGraphProps) {
  // useId strings contain colons, which url(#…) will not take.
  const baseId = useId().replace(/:/g, "");
  const layout = useMemo(() => layoutDag(nodes, edges), [nodes, edges]);

  return (
    <figure className="dag-figure" aria-label={label}>
      <div className="dag-scroll">
        <div
          className="dag-canvas"
          style={{ width: `${layout.width}px`, height: `${layout.height}px` }}
        >
          <svg
            className="dag-edge-layer"
            width={layout.width}
            height={layout.height}
            aria-hidden="true"
            focusable="false"
          >
            <defs>
              {(["dependency", "gate"] as const).map((kind) => (
                <marker
                  key={kind}
                  id={arrowMarkerId(baseId, kind)}
                  viewBox="0 0 8 8"
                  refX="7.2"
                  refY="4"
                  markerWidth="7"
                  markerHeight="7"
                  orient="auto-start-reverse"
                >
                  <path d="M0.4 0.6 L7.6 4 L0.4 7.4 z" className={`dag-arrow is-${kind}`} />
                </marker>
              ))}
            </defs>
            {layout.edges.map((edge) => (
              <path
                key={edge.key}
                d={edge.path}
                className={`dag-edge is-${edge.kind}`}
                markerEnd={`url(#${arrowMarkerId(baseId, edge.kind)})`}
              >
                <title>{edge.title}</title>
              </path>
            ))}
            {layout.edges
              .filter((edge) => edge.label !== null)
              .map((edge) => (
                <text
                  key={`${edge.key}:label`}
                  className={`dag-edge-label is-${edge.kind}`}
                  x={edge.labelX}
                  y={edge.labelY}
                  textAnchor="middle"
                >
                  {edge.label}
                </text>
              ))}
          </svg>
          {nodes.map((node) => {
            const position = layout.nodes.get(node.id);
            if (!position) return null;
            const tone = node.state ? runStateTone(node.state) : null;
            const isSelected = node.id === selectedNodeId;
            const isExpanded = expandedNodeIds.includes(node.id);
            const badgeText = node.badge ?? (node.mapped ? "×N" : null);
            const nodeClasses = [
              "dag-node",
              tone ? `is-${tone}` : "",
              isSelected ? "is-selected" : "",
              isExpanded ? "is-expanded" : "",
            ]
              .filter(Boolean)
              .join(" ");
            return (
              <div
                key={node.id}
                className="dag-node-slot"
                style={{
                  left: `${position.x}px`,
                  top: `${position.y}px`,
                  width: `${NODE_WIDTH}px`,
                  height: `${NODE_HEIGHT}px`,
                }}
              >
                {node.mapped ? (
                  <>
                    <span className="dag-node-stack is-back" aria-hidden="true" />
                    <span className="dag-node-stack is-mid" aria-hidden="true" />
                  </>
                ) : null}
                <button
                  type="button"
                  className={nodeClasses}
                  onClick={() => onSelectNode(node.id)}
                  aria-current={isSelected ? "true" : undefined}
                  aria-expanded={isExpanded ? true : undefined}
                  title={node.summary}
                >
                  <span className="dag-node-id">{node.label}</span>
                  <span className="dag-node-meta">
                    {node.state ? (
                      <span className={`dag-node-state is-${tone}`}>{node.state}</span>
                    ) : null}
                    {node.deferred ? (
                      <span className="dag-node-waits" title="defers its worker slot and waits">
                        waits
                      </span>
                    ) : null}
                    {badgeText ? <span className="dag-node-badge">{badgeText}</span> : null}
                  </span>
                </button>
              </div>
            );
          })}
        </div>
      </div>
      <figcaption className="dag-hint">
        Click a node (or focus it and press Enter) for details.
        {edges.some((edge) => edge.kind === "gate")
          ? " Dashed connectors are conditional gates, not task dependencies."
          : null}
        {/* A long chain is wider than any pane; say so rather than letting the
            reader assume the graph ends at the pane's edge. */}
        {layout.width > WIDE_GRAPH_PX ? " Scroll sideways to follow the whole chain." : null}
      </figcaption>
    </figure>
  );
}
