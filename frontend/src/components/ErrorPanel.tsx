import { describeError, type ErrorContext } from "../lib/errors/describe";

interface Props {
  error: unknown;
  context?: ErrorContext | undefined;
  onRetry?: (() => void) | undefined;
}

/** An API failure as an explanation, a corrective action, the error code and the request ID. */
export function ErrorPanel({ error, context, onRetry }: Props) {
  const v = describeError(error, context);
  return (
    <div className="notice notice-error" role="alert">
      <p className="notice-title">
        <span aria-hidden="true">⚠ </span>
        {v.title}
      </p>
      <p>{v.explanation}</p>
      {v.action ? <p className="notice-action">{v.action}</p> : null}
      <p className="notice-meta">
        Code <code>{v.code}</code>
        {v.requestId ? (
          <>
            {" · "}Request <code>{v.requestId}</code>
          </>
        ) : null}
      </p>
      {onRetry ? (
        <button type="button" className="btn btn-small" onClick={onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  );
}
