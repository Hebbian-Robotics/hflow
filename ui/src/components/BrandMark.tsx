// The one glyph that stays hand-drawn: this is HFlow's identity, not an icon
// from a set, and it must stay byte-identical to the favicon in index.html.
// Every other glyph in the app comes from lucide-react.
//
// Sized by the shared `.lucide` scale in styles.css (see "icon scale" there),
// which this claims by carrying the same class.

export function BrandMark({ className }: { className?: string }) {
  return (
    <svg
      className={className === undefined ? "lucide" : `lucide ${className}`}
      viewBox="0 0 16 16"
      fill="none"
      aria-hidden="true"
      focusable="false"
    >
      <path
        d="M1.5 4.5h8M1.5 8h13M1.5 11.5h5"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <circle cx="12.5" cy="4.5" r="1.7" fill="currentColor" />
      <circle cx="9.5" cy="11.5" r="1.7" fill="currentColor" />
    </svg>
  );
}
