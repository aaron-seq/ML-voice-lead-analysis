import { ApiError } from '@/lib/api';

/**
 * Error display that distinguishes "you typed something wrong" from "the
 * system broke", and surfaces the request id so a user can quote it in a bug
 * report that is actually traceable in the server logs.
 */
export function ErrorNotice({ error }: { error: ApiError | Error }) {
  const api = error instanceof ApiError ? error : null;
  const critical = !api || !api.isUserError;

  return (
    <div
      className={`notice ${critical ? 'notice-critical' : 'notice-warning'}`}
      role="alert"
    >
      <strong>{critical ? 'Something went wrong' : 'Check your input'}</strong>
      <p>{error.message}</p>
      {api?.requestId && (
        <p className="muted small">Request ID: {api.requestId}</p>
      )}
    </div>
  );
}
