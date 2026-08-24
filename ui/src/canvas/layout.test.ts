// The layout is a thin call into dagre, so these tests cover only what this
// module itself decides: that every node gets its OWN position, that the
// direction reads left to right, and that an edge naming a node this level did
// not draw cannot make dagre invent one.

import { describe, expect, it } from "vitest";
import type { CanvasEdge, CanvasNode, NodeShape } from "./buildGraph";
import { layoutGraph } from "./layout";

function node(id: string, shape: NodeShape = "task"): CanvasNode {
  return {
    id,
    data: {
      title: id,
      subtitle: null,
      tone: "muted",
      shape,
      badges: [],
      detail: [],
      drillTo: null,
    },
  };
}

function edge(source: string, target: string): CanvasEdge {
  return { id: `${source}->${target}`, source, target, dashed: false, label: null };
}

describe("layoutGraph", () => {
  it("gives every node its own position", () => {
    // The regression this exists for: dagre writes the computed x/y onto the
    // very object it was handed as the node's label, so one shared size
    // literal per shape put every node of that shape at one point and laid the
    // whole graph on top of itself.
    const chain = ["a", "b", "c", "d"].map((id) => node(id));
    const positioned = layoutGraph(chain, [edge("a", "b"), edge("b", "c"), edge("c", "d")]);
    const distinct = new Set(positioned.map((laid) => `${laid.position.x},${laid.position.y}`));
    expect(distinct.size).toBe(chain.length);
  });

  it("lays a chain out left to right, in dependency order", () => {
    const positioned = layoutGraph(
      [node("first"), node("second"), node("third")],
      [edge("first", "second"), edge("second", "third")],
    );
    const xByNode = new Map(positioned.map((laid) => [laid.id, laid.position.x]));
    const first = xByNode.get("first") ?? 0;
    const second = xByNode.get("second") ?? 0;
    const third = xByNode.get("third") ?? 0;
    expect(first).toBeLessThan(second);
    expect(second).toBeLessThan(third);
  });

  it("spreads a fan across one rank without stacking it", () => {
    const fan = [node("plan"), node("batch0"), node("batch1"), node("batch2"), node("gate")];
    const positioned = layoutGraph(fan, [
      edge("plan", "batch0"),
      edge("plan", "batch1"),
      edge("plan", "batch2"),
      edge("batch0", "gate"),
      edge("batch1", "gate"),
      edge("batch2", "gate"),
    ]);
    const batches = positioned.filter((laid) => laid.id.startsWith("batch"));
    // Same rank, so one x; different rows, so three distinct y values.
    expect(new Set(batches.map((laid) => laid.position.x)).size).toBe(1);
    expect(new Set(batches.map((laid) => laid.position.y)).size).toBe(3);
  });

  it("drops an edge naming a node that was not drawn", () => {
    // Otherwise dagre invents an empty node for the missing endpoint and the
    // canvas grows a blank box nothing explains.
    const positioned = layoutGraph([node("only")], [edge("only", "vanished")]);
    expect(positioned.map((laid) => laid.id)).toEqual(["only"]);
  });

  it("carries each shape's own box size through", () => {
    const positioned = layoutGraph([node("task"), node("note", "note")], []);
    const sizeById = new Map(positioned.map((laid) => [laid.id, laid.width]));
    expect(sizeById.get("note")).toBeLessThan(sizeById.get("task") ?? 0);
  });
});
