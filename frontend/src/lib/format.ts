/** Presentation helpers shared across components. */

import type { ActionPriority, CallOutcomeLabel, LeadBand } from './types';

/**
 * Lead bands are an ordinal tier, so they use one hue stepped light-to-dark
 * rather than four unrelated colours. Both ramps were checked with the
 * data-viz palette validator (monotone lightness, adjacent ΔL ≥ 0.06, light
 * end clears the surface). The dark ramp is a separate selection, not an
 * inversion: on a dark surface the *brighter* step reads as most prominent.
 *
 * Colour never carries the meaning alone — every band is rendered with its
 * label beside the swatch.
 */
export const BAND_COLOR: Record<LeadBand, { light: string; dark: string }> = {
  hot: { light: '#1c5cab', dark: '#cde2fb' },
  warm: { light: '#2a78d6', dark: '#9ec5f4' },
  cold: { light: '#5598e7', dark: '#5598e7' },
  disqualified: { light: '#86b6ef', dark: '#256abf' },
};

export const BAND_LABEL: Record<LeadBand, string> = {
  hot: 'Hot',
  warm: 'Warm',
  cold: 'Cold',
  disqualified: 'Disqualified',
};

export const OUTCOME_LABEL: Record<CallOutcomeLabel, string> = {
  meeting_booked: 'Meeting booked',
  follow_up_agreed: 'Follow-up agreed',
  information_requested: 'Information requested',
  not_interested: 'Not interested',
  not_a_fit: 'Not a fit',
  gatekeeper_only: 'Gatekeeper only',
  no_clear_outcome: 'No clear outcome',
};

export const PRIORITY_LABEL: Record<ActionPriority, string> = {
  p0: 'Today',
  p1: 'This week',
  p2: 'Backlog',
};

export function titleise(value: string): string {
  return value
    .replace(/_/g, ' ')
    .replace(/^\w/, (c) => c.toUpperCase());
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null) return '—';
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  if (mins === 0) return `${secs}s`;
  return `${mins}m ${String(secs).padStart(2, '0')}s`;
}

export function formatTimestamp(seconds: number | null | undefined): string {
  if (seconds == null) return '';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${String(secs).padStart(2, '0')}`;
}

export function formatRelativeDate(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return '—';

  const diffMs = Date.now() - then;
  const minutes = Math.round(diffMs / 60_000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;

  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;

  const days = Math.round(hours / 24);
  if (days < 30) return `${days}d ago`;

  return new Date(iso).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

export function formatPercent(value: number | null | undefined): string {
  if (value == null) return '—';
  return `${Math.round(value * 100)}%`;
}

/** Sentiment as a word, so the number is never the only cue. */
export function sentimentLabel(score: number | null | undefined): string {
  if (score == null) return 'Unknown';
  if (score > 0.25) return 'Positive';
  if (score < -0.25) return 'Negative';
  return 'Neutral';
}

export function classNames(...values: (string | false | null | undefined)[]): string {
  return values.filter(Boolean).join(' ');
}
