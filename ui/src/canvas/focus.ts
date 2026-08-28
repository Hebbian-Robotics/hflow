// Where the canvas is pointed. One value describes the whole screen, so
// navigation is "replace the focus", never "mutate five pieces of state", and
// the breadcrumb is derived from the focus rather than tracked beside it.

import type { Stage } from "../api";

/**
 * The five things the canvas can be drawing.
 *
 * Each level names its own parent, which is what lets the breadcrumb be a pure
 * function of the focus. They are levels of a DRILL-DOWN, not layers drawn at
 * once: entering one replaces the canvas.
 *
 * Two branches leave the run, because a run has two things worth following:
 *
 *   run -> stage -> steps        the orchestration, and the code inside it
 *   run -> episodes -> episode   the data it produced
 *
 * The data branch hangs off the RUN and not off a stage, and that is not a
 * simplification. The catalog's `episodes` view is one row per episode (latest
 * append wins), so a full ingest leaves every episode's current row stamped
 * with the LAST stage that recorded. Scoping this branch to one stage would
 * therefore answer "0 episodes" for three stages out of four.
 */
export type CanvasFocus =
  | { readonly level: "run" }
  | { readonly level: "stage"; readonly stage: Stage }
  | { readonly level: "steps"; readonly stage: Stage }
  | { readonly level: "episodes" }
  | { readonly level: "episode"; readonly episodeId: string };

export const RUN_FOCUS: CanvasFocus = { level: "run" };
export const EPISODES_FOCUS: CanvasFocus = { level: "episodes" };

export type Breadcrumb = {
  readonly label: string;
  readonly focus: CanvasFocus;
};

/**
 * How far to shorten an episode id for a label. Long enough to stay unique
 * across a run's episodes, short enough to fit a crumb.
 */
const EPISODE_ID_LABEL_LENGTH = 12;

export function shortEpisodeId(episodeId: string): string {
  return episodeId.slice(0, EPISODE_ID_LABEL_LENGTH);
}

/** The trail from the run down to this focus, this focus included. */
export function breadcrumbs(focus: CanvasFocus): Breadcrumb[] {
  const trail: Breadcrumb[] = [{ label: "Run", focus: RUN_FOCUS }];
  switch (focus.level) {
    case "run":
      return trail;
    case "stage":
      trail.push({ label: focus.stage, focus });
      return trail;
    case "steps":
      trail.push({ label: focus.stage, focus: { level: "stage", stage: focus.stage } });
      trail.push({ label: "steps", focus });
      return trail;
    case "episodes":
      trail.push({ label: "episodes", focus });
      return trail;
    case "episode":
      trail.push({ label: "episodes", focus: EPISODES_FOCUS });
      trail.push({ label: shortEpisodeId(focus.episodeId), focus });
      return trail;
  }
}

/** The stage a focus is scoped to, or null on the levels that are run-scoped. */
export function focusedStage(focus: CanvasFocus): Stage | null {
  return focus.level === "stage" || focus.level === "steps" ? focus.stage : null;
}

/** The level this one drills back out to, or null at the top. */
export function parentFocus(focus: CanvasFocus): CanvasFocus | null {
  const trail = breadcrumbs(focus);
  return trail.length > 1 ? (trail[trail.length - 2]?.focus ?? null) : null;
}
