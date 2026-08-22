import {
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
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

/**
 * One thing that runs INSIDE a task, drawn within its node.
 *
 * These are not graph nodes and never get connectors: the steps inside a
 * `process_batch` have no dependency edges on each other, so the only honest
 * drawing is containment. Nesting them says "this task runs these, in this
 * order, in one process" without inventing arrows.
 */
export interface DagGraphNodeContent {
  /** Selection identity, unique within the node. */
  id: string;
  label: string;
  /** Hover text: what this step does. */
  summary: string;
  /** Engine-owned work versus a user registration; drawn differently. */
  kind: "engine" | "user";
  /** A short trailing mark on the row: a tier number, "critical", a duration. */
  badge?: string | null;
}

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
  /** What runs inside this task, drawn nested within the node. */
  contents?: readonly DagGraphNodeContent[];
  /** One line under the contents: what their arrangement does and does not mean. */
  contentsNote?: string | null;
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
  /** Selected content row, as "<nodeId>/<contentId>"; null when none is. */
  selectedContentKey?: string | null;
  onSelectContent?: (nodeId: string, contentId: string) => void;
}

/** Past this width the canvas overflows any realistic pane, so the caption
 * tells the reader the graph continues past the edge. */
const WIDE_GRAPH_PX = 900;

// A container node's height is computed here rather than measured, because the
// layout needs it before anything renders. These must track the CSS in
// index.css (.dag-node-content-row and friends); the graph is one component
// with one stylesheet, so the coupling is local and visible.
const CONTENT_ROW_HEIGHT = 20;
const CONTENT_ROW_GAP = 3;
const CONTENTS_PADDING = 8;
/** Two wrapped lines at the note's font size, plus its padding. Keep
 * `contentsNote` short enough to fit in two: the node clips rather than grows. */
const CONTENTS_NOTE_HEIGHT = 38;

export function contentAwareNodeHeight(node: DagGraphNode): number {
  const contents = node.contents ?? [];
  if (contents.length === 0) return NODE_HEIGHT;
  const rows = contents.length * CONTENT_ROW_HEIGHT + (contents.length - 1) * CONTENT_ROW_GAP;
  return NODE_HEIGHT + CONTENTS_PADDING + rows + (node.contentsNote ? CONTENTS_NOTE_HEIGHT : 0);
}

export function contentKey(nodeId: string, contentId: string): string {
  return `${nodeId}/${contentId}`;
}

function arrowMarkerId(baseId: string, kind: DagEdgeKind): string {
  return `${baseId}-arrow-${kind}`;
}

// ---- grab-to-pan -----------------------------------------------------------
//
// A long chain is wider than any pane, and reaching for a scrollbar (or
// knowing that shift+wheel scrolls sideways) is not something a reader should
// have to discover. So the empty space between the nodes is a handle: press it
// and drag, and the view follows the pointer.
//
// This is ~60 lines of pointer arithmetic on the scroll offset, deliberately
// not a pan/zoom library — the repo already turned down a graph library at
// these sizes, and the whole of the behaviour is below.
//
// Motion: the offset tracks the pointer 1:1 and stops when the pointer stops.
// There is no inertia, no fling, no easing — nothing that keeps moving after
// the gesture, so there is nothing for prefers-reduced-motion to switch off.
// `.dag-scroll` pins `scroll-behavior: auto` so a later global smooth-scroll
// cannot turn a drag into an animation behind our back.

/**
 * How far the pointer must travel before a press counts as a pan. Under it the
 * press is still a click, so a reader who wobbles a pixel while pressing gets
 * what they aimed at instead of a one-pixel pan and a swallowed click.
 */
const PAN_THRESHOLD_PX = 4;

/**
 * Anything the reader can operate. A press landing on one of these belongs to
 * that control — nodes keep their click, their selection and their focus, and
 * only the graph's empty space pans.
 */
const INTERACTIVE_SELECTOR = "a, button, input, select, textarea, [contenteditable], [tabindex]";

interface PanGesture {
  pointerId: number;
  originClientX: number;
  originClientY: number;
  originScrollLeft: number;
  originScrollTop: number;
  /** True once the pointer cleared PAN_THRESHOLD_PX: from there it is a pan. */
  isPan: boolean;
}

function useDragToPan() {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const gestureRef = useRef<PanGesture | null>(null);
  const suppressNextClickRef = useRef(false);
  const [isPanning, setIsPanning] = useState(false);
  const [canPan, setCanPan] = useState(false);

  // Only wear the grab cursor where there is somewhere to grab towards: on a
  // graph that already fits, "grab" would be a promise the pane cannot keep.
  useEffect(() => {
    const scroll = scrollRef.current;
    if (scroll === null) return;
    const measure = () => {
      setCanPan(
        scroll.scrollWidth - scroll.clientWidth > 1 ||
          scroll.scrollHeight - scroll.clientHeight > 1,
      );
    };
    measure();
    if (typeof ResizeObserver !== "function") return;
    // Both boxes: the pane resizes with the window, the canvas resizes when a
    // run adds tasks to the graph.
    const observer = new ResizeObserver(measure);
    observer.observe(scroll);
    const canvas = scroll.firstElementChild;
    if (canvas !== null) observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  const handlePointerDown = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    // Clear here rather than only after a click: a gesture can end without one
    // (released off-window), and a stale flag would eat a later real click.
    suppressNextClickRef.current = false;
    const scroll = scrollRef.current;
    if (scroll === null) return;
    // Touch already drags this container natively, with the momentum and the
    // rubber-banding the platform tunes. Taking that gesture over would mean
    // re-implementing it worse and fighting the browser for the first frames
    // before it sends pointercancel. Mouse, trackpad and pen have no such
    // default, and are what this is here for.
    if (event.pointerType === "touch") return;
    if (!event.isPrimary || event.button !== 0) return;
    const target = event.target;
    if (target instanceof Element) {
      const control = target.closest(INTERACTIVE_SELECTOR);
      if (control !== null && scroll.contains(control)) return;
    }
    gestureRef.current = {
      pointerId: event.pointerId,
      originClientX: event.clientX,
      originClientY: event.clientY,
      originScrollLeft: scroll.scrollLeft,
      originScrollTop: scroll.scrollTop,
      isPan: false,
    };
    // Capture from the press, not from the threshold: a quick drag can leave
    // the pane between two frames, and uncaptured moves would go to whatever
    // is underneath. Only background presses reach this line, so no node's
    // click or focus is ever retargeted by it.
    scroll.setPointerCapture(event.pointerId);
  }, []);

  const handlePointerMove = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const gesture = gestureRef.current;
    const scroll = scrollRef.current;
    if (gesture === null || scroll === null || event.pointerId !== gesture.pointerId) return;
    const deltaX = event.clientX - gesture.originClientX;
    const deltaY = event.clientY - gesture.originClientY;
    if (!gesture.isPan) {
      if (Math.hypot(deltaX, deltaY) < PAN_THRESHOLD_PX) return;
      gesture.isPan = true;
      setIsPanning(true);
    }
    // Both axes, always: the offsets clamp themselves, so an axis that does
    // not overflow simply does not move.
    scroll.scrollLeft = gesture.originScrollLeft - deltaX;
    scroll.scrollTop = gesture.originScrollTop - deltaY;
  }, []);

  const handlePointerEnd = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const gesture = gestureRef.current;
    if (gesture === null || event.pointerId !== gesture.pointerId) return;
    gestureRef.current = null;
    const scroll = scrollRef.current;
    if (scroll?.hasPointerCapture(event.pointerId)) {
      scroll.releasePointerCapture(event.pointerId);
    }
    setIsPanning(false);
    // A drag that ends is not a click. Nothing under the graph listens for a
    // background click today; swallow it anyway, so adding one later does not
    // quietly start firing it at the end of every pan.
    if (gesture.isPan) suppressNextClickRef.current = true;
  }, []);

  const handleClickCapture = useCallback((event: ReactMouseEvent<HTMLDivElement>) => {
    if (!suppressNextClickRef.current) return;
    suppressNextClickRef.current = false;
    event.preventDefault();
    event.stopPropagation();
  }, []);

  return {
    scrollRef,
    canPan,
    isPanning,
    panHandlers: {
      onPointerDown: handlePointerDown,
      onPointerMove: handlePointerMove,
      onPointerUp: handlePointerEnd,
      onPointerCancel: handlePointerEnd,
      onClickCapture: handleClickCapture,
    },
  };
}

export function DagGraph({
  nodes,
  edges,
  label,
  selectedNodeId,
  onSelectNode,
  expandedNodeIds = [],
  selectedContentKey = null,
  onSelectContent,
}: DagGraphProps) {
  // useId strings contain colons, which url(#…) will not take.
  const baseId = useId().replace(/:/g, "");
  const layoutNodes = useMemo(
    () => nodes.map((node) => ({ id: node.id, height: contentAwareNodeHeight(node) })),
    [nodes],
  );
  const layout = useMemo(() => layoutDag(layoutNodes, edges), [layoutNodes, edges]);
  const { scrollRef, canPan, isPanning, panHandlers } = useDragToPan();
  const scrollClasses = ["dag-scroll", canPan ? "is-pannable" : "", isPanning ? "is-panning" : ""]
    .filter(Boolean)
    .join(" ");

  return (
    <figure className="dag-figure" aria-label={label}>
      {/* The pan handlers add nothing a keyboard user needs and so the container
          stays a plain scroller with no role: the nodes inside are real buttons
          that focus and scroll themselves into view, and the wheel, the
          scrollbar and shift+wheel all still work. This is a second way to
          reach the same scroll offset, not a control. */}
      <div className={scrollClasses} ref={scrollRef} {...panHandlers}>
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
            const contents = node.contents ?? [];
            const hasContents = contents.length > 0;
            const nodeClasses = [
              "dag-node",
              hasContents ? "has-contents" : "",
              tone ? `is-${tone}` : "",
              isSelected ? "is-selected" : "",
              isExpanded ? "is-expanded" : "",
            ]
              .filter(Boolean)
              .join(" ");
            // A node with contents holds real buttons, so it cannot itself be
            // one: it becomes a container whose HEAD is the button that selects
            // the task. Without contents it stays a single button, as before.
            const headContent = (
              <>
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
              </>
            );
            return (
              <div
                key={node.id}
                className="dag-node-slot"
                style={{
                  left: `${position.x}px`,
                  top: `${position.y}px`,
                  width: `${NODE_WIDTH}px`,
                  height: `${position.height}px`,
                }}
              >
                {node.mapped ? (
                  <>
                    <span className="dag-node-stack is-back" aria-hidden="true" />
                    <span className="dag-node-stack is-mid" aria-hidden="true" />
                  </>
                ) : null}
                {hasContents ? (
                  <div className={nodeClasses}>
                    <button
                      type="button"
                      className="dag-node-head"
                      onClick={() => onSelectNode(node.id)}
                      aria-current={isSelected ? "true" : undefined}
                      aria-expanded={isExpanded ? true : undefined}
                      title={node.summary}
                    >
                      {headContent}
                    </button>
                    <div className="dag-node-contents">
                      {contents.map((content) => {
                        const key = contentKey(node.id, content.id);
                        return (
                          <button
                            key={content.id}
                            type="button"
                            className={[
                              "dag-node-content-row",
                              `is-${content.kind}`,
                              key === selectedContentKey ? "is-selected" : "",
                            ]
                              .filter(Boolean)
                              .join(" ")}
                            title={content.summary}
                            aria-current={key === selectedContentKey ? "true" : undefined}
                            onClick={() => onSelectContent?.(node.id, content.id)}
                          >
                            <span className="dag-node-content-name">{content.label}</span>
                            {content.badge ? (
                              <span className="dag-node-content-badge">{content.badge}</span>
                            ) : null}
                          </button>
                        );
                      })}
                    </div>
                    {node.contentsNote ? (
                      <p className="dag-node-contents-note">{node.contentsNote}</p>
                    ) : null}
                  </div>
                ) : (
                  <button
                    type="button"
                    className={nodeClasses}
                    onClick={() => onSelectNode(node.id)}
                    aria-current={isSelected ? "true" : undefined}
                    aria-expanded={isExpanded ? true : undefined}
                    title={node.summary}
                  >
                    {headContent}
                  </button>
                )}
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
            reader assume the graph ends at the pane's edge — and name the
            gesture, since a drag handle with no scrollbar in sight is not
            something anyone goes looking for. */}
        {layout.width > WIDE_GRAPH_PX
          ? " Drag the background, or scroll sideways, to follow the whole chain."
          : null}
      </figcaption>
    </figure>
  );
}
