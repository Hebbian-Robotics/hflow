import type { ReactNode } from "react";

// The always-present second pane. Selecting a graph node fills it; selecting
// nothing shows the hint. It never disappears when the view changes, so the
// reader's eye keeps one home for "what am I looking at".

export function DetailsPanel({
  title,
  kicker,
  children,
}: {
  title: string;
  /** Tiny uppercase line above the title: what kind of thing this is. */
  kicker?: string;
  children: ReactNode;
}) {
  return (
    <aside className="details-panel" aria-label="Details">
      <div className="details-panel-head">
        {kicker ? <span className="details-kicker">{kicker}</span> : null}
        <h3 className="details-title" title={title}>
          {title}
        </h3>
      </div>
      <div className="details-panel-body">{children}</div>
    </aside>
  );
}

export function DetailRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="detail-row">
      <span className="detail-label">{label}</span>
      <span className="detail-value">{children}</span>
    </div>
  );
}

/** A block that needs the full width of the panel (lists, prose, buttons). */
export function DetailBlock({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="detail-block">
      <span className="detail-label">{label}</span>
      <div className="detail-block-body">{children}</div>
    </div>
  );
}
