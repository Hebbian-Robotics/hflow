import { useId } from "react";

// The company mark — the one glyph in the app that does not come from
// lucide-react, because this is an identity, not an icon from a set.
//
// SOURCE OF TRUTH for the artwork: docs/assets/hebbian-logo-on-black.svg.
// The five path strings below are copied verbatim from that file, and the
// favicon in index.html carries the same five. Re-cut the logo there and all
// three change together.
//
// Two deliberate departures from that file, both so the mark works as an app
// glyph rather than as a 1024px artboard:
//
//   * the black background rect is dropped and the paths take `currentColor`,
//     so the mark is accent-on-surface in light and in dark instead of a black
//     tile punched into the rail;
//   * the viewBox is the glyph's own bounding box, not the artboard's, so the
//     mark fills its box instead of floating small in a mostly-empty square.
//     The artboard draws the group at translate(274.13 212) scale(4.347826);
//     dropping that is safe, since a uniform scale plus a translate is exactly
//     what a viewBox change undoes. In the paths' own units the bounds solve
//     to x 1.800..107.811, y -0.225..137.380 — from the cubic-Bézier extrema,
//     not from sampling or from eyeballing the artboard — and the box below is
//     those bounds rounded outward to 2dp.
//
// The glyph is TALLER THAN IT IS WIDE (0.77 : 1), so it cannot take the square
// `.lucide` sizing the rest of the icons use; `.brand-mark` in styles.css sizes
// it by height and lets the width follow from this aspect ratio.

/** Verbatim from docs/assets/hebbian-logo-on-black.svg — see the note above. */
const GLYPH_PATHS: readonly string[] = [
  "m24.3 68v-61c-0.2-3.5-3.1-6.5-6.3-7-1.9-0.3-6.3-0.3-8.6 0-2.5 0.2-4 1-5.4 2.6-1.8 2.1-2.2 3.6-2.2 7.4v114.2 2.4c0.1 4.1 0.2 5.2 1.4 7.1s3 3.3 5.3 3.6 6.9-0.3 9.2-0.9c2.7-0.6 5.5-2.7 6.3-5.9 0.5-2.4 0.3-0.9 0.3-62.5z",
  "m99.9 1.7c-1.7-0.1-5.3-0.1-7.5 0.2-3.1 0.5-5.9 3.1-6.3 6.3-0.2 1.7-1.2 59.2-1.2 59.8l0.1 58.9c0.1 2.6 0.1 4 1.3 5.9 1.3 2 3.2 3.3 5.6 3.7 2.3 0.3 6.5 0.4 9.1 0.1 2.8-0.4 5.7-2.5 6.5-5.6 0.4-1.6 0.3-4.4 0.3-6.5v-112.1c0-3.4-0.1-5.4-1.3-7-1.2-2-3.6-3.5-6.6-3.7z",
  "m44 16.2c-0.6 1.6-0.6 3.5-0.6 6.8v15c0.1 4.4 0.1 6 1.6 8s3.1 3.4 5.5 3.7c1.8 0.3 6.4 0.4 8.5 0.1 3.3-0.4 6.2-3 6.7-6.2 0.2-1.5 0.2-2.4 0.2-6.1l-0.1-16.5c0-3.6-0.4-5.1-2.2-7.1-1.1-1.2-2.8-2.4-5.1-2.7-1.4-0.2-5.5-0.2-7.5 0-3.1 0.2-6 2.3-7 5z",
  "m68.8 58.5c-1.7-0.3-3.8-0.4-6.2-0.4h-17.7c-1.3 0-3 0.1-4.1 0.3-2.8 1-5.2 3.6-5.5 6.7-0.2 1.7-0.2 5.9 0.1 8.4 0.5 2.9 2.4 4.9 5.4 6.2 1.4 0.2 3.2 0.4 4.2 0.4h19.6c1.7 0 2.3 0 4.5-0.3 2.7-0.8 5-3.3 5.5-6.3 0.2-1.9 0.1-6.9-0.4-9.5-0.8-3-3.2-4.8-5.4-5.5z",
  "m59.4 88c-1.8-0.3-6.9-0.4-9.1 0-2.9 0.6-5.4 2.8-6.2 5.8-0.4 1.8-0.3 4.3-0.3 8.2v11.7c0 4.9 0 7.7 1.3 9.5 1.2 1.9 2.8 3.4 5.4 3.8 1.6 0.3 6.7 0.3 8.4 0 2.6-0.3 4.8-1.9 5.8-3.9 1.2-1.7 1.1-4.5 1.1-8.5v-15.2c0-2.5 0-4.8-0.5-6.1-0.9-2.8-3.4-4.9-5.9-5.3z",
];

/** The glyph's own bounds, rounded outward — see the note above. */
const GLYPH_VIEW_BOX = "1.79 -0.23 106.03 137.62";

export function BrandMark({
  className = "brand-mark",
  title = "Hebbian Robotics",
}: {
  className?: string;
  /**
   * Accessible name for the mark. This is a company logo, not decoration, so
   * it is always named — the default is who the logo belongs to.
   */
  title?: string;
}) {
  // useId, so one <title> node is both the accessible name and the hover
  // tooltip, rather than an aria-label repeating it a second time.
  const titleId = useId();
  return (
    <svg
      className={className}
      viewBox={GLYPH_VIEW_BOX}
      fill="currentColor"
      role="img"
      aria-labelledby={titleId}
      focusable="false"
    >
      <title id={titleId}>{title}</title>
      {GLYPH_PATHS.map((path) => (
        <path key={path} d={path} />
      ))}
    </svg>
  );
}
