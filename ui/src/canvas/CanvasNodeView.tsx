// The one node renderer. Every level draws the same box, differing only in its
// tone, its shape and whether it offers a drill-down, so there is nothing per
// level to keep consistent.

import { Handle, type Node, type NodeProps, Position } from "@xyflow/react";
import type { CanvasNodeData } from "./buildGraph";

export type FlowNode = Node<CanvasNodeData, "canvas">;

export function CanvasNodeView({ data, selected }: NodeProps<FlowNode>) {
  const drillable = data.drillTo !== null;
  return (
    <div
      className={`node node-${data.shape} tone-${data.tone}`}
      data-selected={selected ? "true" : undefined}
      data-drillable={drillable ? "true" : undefined}
    >
      {/* Left in, right out: the layout is left-to-right, so the handles have
          to match or every edge would leave from the wrong side. */}
      <Handle type="target" position={Position.Left} isConnectable={false} />
      <div className="node-title">
        <span className="node-title-text">{data.title}</span>
        {/* Decorative, so hidden from assistive tech: what it announces is
            already on the inspector's own "Open" button, and a bare chevron
            read aloud on every second node would be noise. */}
        {drillable ? (
          <span className="node-drill" aria-hidden="true">
            &rsaquo;
          </span>
        ) : null}
      </div>
      {data.subtitle === null ? null : <div className="node-subtitle">{data.subtitle}</div>}
      {data.badges.length === 0 ? null : (
        <div className="node-badges">
          {data.badges.map((badge) => (
            <span className="badge" key={badge}>
              {badge}
            </span>
          ))}
        </div>
      )}
      <Handle type="source" position={Position.Right} isConnectable={false} />
    </div>
  );
}

export const CANVAS_NODE_TYPES = { canvas: CanvasNodeView };
