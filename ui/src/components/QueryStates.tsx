import type { ReactNode } from "react";
import { describeApiError } from "../api";

export function LoadingPanel({ label }: { label: string }) {
  return (
    <div className="state-panel" role="status">
      <span className="spinner" aria-hidden="true" />
      <p className="state-title">{label}</p>
    </div>
  );
}

export function ErrorPanel({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  return (
    <div className="state-panel state-panel-error" role="alert">
      <p className="state-title">Request failed</p>
      <p className="state-detail">{describeApiError(error)}</p>
      {onRetry ? (
        <button type="button" className="btn" onClick={onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  );
}

export function EmptyPanel({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children?: ReactNode;
}) {
  return (
    <div className="state-panel">
      <p className="state-title">{title}</p>
      {hint ? <p className="state-detail">{hint}</p> : null}
      {children}
    </div>
  );
}
