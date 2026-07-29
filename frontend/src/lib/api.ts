/**
 * Typed API client.
 *
 * One place that knows the wire format. Errors are normalised into `ApiError`
 * so components can render `error.message` without unwrapping a fetch failure,
 * an HTTP status and a JSON error body separately.
 */

import type {
  CallAnalysis,
  CallDetail,
  CallFilters,
  CallSummaryRow,
  Paginated,
  PipelineAnalytics,
  Transcript,
} from './types';

const BASE_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
).replace(/\/$/, '');

const API_PREFIX = '/api/v1';

/** Set when the deployment has AUTH_ENABLED=true. */
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? '';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string = 'error',
    readonly context: Record<string, unknown> = {},
    readonly requestId: string | null = null,
  ) {
    super(message);
    this.name = 'ApiError';
  }

  /** True for errors the user can fix by changing their input. */
  get isUserError(): boolean {
    return this.status >= 400 && this.status < 500;
  }
}

interface RequestOptions extends RequestInit {
  /** Abort the request after this many milliseconds. */
  timeoutMs?: number;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { timeoutMs = 30_000, headers, ...rest } = options;

  // Without a timeout a hung backend leaves the UI spinning forever.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  const requestHeaders: Record<string, string> = { ...(headers as object) };
  if (API_KEY) requestHeaders['X-API-Key'] = API_KEY;

  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...rest,
      headers: requestHeaders,
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timer);
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiError('The request timed out.', 0, 'timeout');
    }
    throw new ApiError(
      `Could not reach the API at ${BASE_URL}. Is the backend running?`,
      0,
      'network_error',
    );
  }
  clearTimeout(timer);

  if (response.status === 204) return undefined as T;

  const requestId = response.headers.get('X-Request-ID');

  if (!response.ok) {
    let code = `http_${response.status}`;
    let message = response.statusText || 'Request failed';
    let context: Record<string, unknown> = {};
    try {
      const body = await response.json();
      code = body.code ?? code;
      message = body.message ?? message;
      context = body.context ?? {};
    } catch {
      // Non-JSON error body (a proxy error page, say). Keep the status text.
    }
    throw new ApiError(message, response.status, code, context, requestId);
  }

  return (await response.json()) as T;
}

function buildQuery(filters: CallFilters): string {
  const params = new URLSearchParams();
  if (filters.search) params.set('search', filters.search);
  // Repeated key, not a comma list — matches FastAPI's list query parsing.
  filters.band?.forEach((b) => params.append('band', b));
  if (filters.owner) params.set('owner', filters.owner);
  if (filters.min_score !== undefined) {
    params.set('min_score', String(filters.min_score));
  }
  if (filters.sort_by) params.set('sort_by', filters.sort_by);
  if (filters.sort_desc !== undefined) {
    params.set('sort_desc', String(filters.sort_desc));
  }
  params.set('page', String(filters.page ?? 1));
  params.set('page_size', String(filters.page_size ?? 20));
  return params.toString();
}

export const api = {
  listCalls(filters: CallFilters = {}): Promise<Paginated<CallSummaryRow>> {
    return request(`${API_PREFIX}/calls?${buildQuery(filters)}`);
  },

  getCall(id: string, includeTranscript = false): Promise<CallDetail> {
    return request(
      `${API_PREFIX}/calls/${encodeURIComponent(id)}` +
        `?include_transcript=${includeTranscript}`,
    );
  },

  createCall(payload: {
    title: string;
    transcript: string;
    owner?: string;
    tags?: string[];
  }): Promise<CallDetail> {
    return request(`${API_PREFIX}/calls`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      timeoutMs: 60_000,
    });
  },

  uploadCall(file: File, title?: string, owner?: string): Promise<CallDetail> {
    const form = new FormData();
    form.append('file', file);
    if (title) form.append('title', title);
    if (owner) form.append('owner', owner);
    // Content-Type is omitted deliberately: the browser must set the
    // multipart boundary itself.
    return request(`${API_PREFIX}/calls/upload`, {
      method: 'POST',
      body: form,
      timeoutMs: 120_000,
    });
  },

  analyseOnly(transcript: string): Promise<{
    analysis: CallAnalysis;
    transcript: Transcript;
  }> {
    return request(`${API_PREFIX}/calls/analyse`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transcript }),
      timeoutMs: 60_000,
    });
  },

  reanalyse(id: string): Promise<CallDetail> {
    return request(`${API_PREFIX}/calls/${encodeURIComponent(id)}/reanalyse`, {
      method: 'POST',
      timeoutMs: 60_000,
    });
  },

  deleteCall(id: string): Promise<void> {
    return request(`${API_PREFIX}/calls/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
  },

  analytics(days = 30): Promise<PipelineAnalytics> {
    return request(`${API_PREFIX}/analytics/pipeline?days=${days}`);
  },

  exportCsvUrl(filters: CallFilters = {}): string {
    const params = new URLSearchParams();
    filters.band?.forEach((b) => params.append('band', b));
    if (filters.owner) params.set('owner', filters.owner);
    return `${BASE_URL}${API_PREFIX}/export/calls.csv?${params.toString()}`;
  },
};
