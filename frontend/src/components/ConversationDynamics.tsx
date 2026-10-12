'use client';

/**
 * Timing-derived conversation dynamics.
 *
 * Renders against `available` rather than truthiness. Zero interruptions is a
 * real, useful measurement and must not be displayed as missing data, which is
 * what a `{dynamics.interruptions_by_rep && ...}` guard would do.
 *
 * When timings are absent the panel says so and names the fix, because the
 * reason is actionable by the reader: it is a property of their transcript
 * export, not of this call.
 */

import { formatPercent } from '@/lib/format';
import type { ConversationDynamics as Dynamics } from '@/lib/types';

function seconds(value: number | null): string {
  if (value === null) return '—';
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  return `${minutes}m ${Math.round(value % 60)}s`;
}

/** Prospect's share of talk time across the three thirds of the call. */
function TrajectoryBar({ shares }: { shares: number[] }) {
  const labels = ['Opening', 'Middle', 'Close'];
  return (
    <div className="trajectory">
      {shares.map((share, index) => (
        <div className="trajectory-step" key={labels[index]}>
          <div
            className="trajectory-track"
            role="img"
            aria-label={`${labels[index]}: prospect spoke ${formatPercent(share)} of the time`}
          >
            <div
              className="trajectory-fill"
              style={{ height: `${Math.round(share * 100)}%` }}
            />
          </div>
          <span className="trajectory-value">{formatPercent(share)}</span>
          <span className="trajectory-label">{labels[index]}</span>
        </div>
      ))}
    </div>
  );
}

export function ConversationDynamics({ dynamics }: { dynamics: Dynamics }) {
  if (!dynamics.available) {
    return (
      <section className="panel" aria-labelledby="dynamics-heading">
        <header className="panel-header">
          <h2 id="dynamics-heading">Conversation dynamics</h2>
        </header>
        <p className="muted small">
          {dynamics.unavailable_reason ??
            'This transcript carries no segment timings.'}
        </p>
      </section>
    );
  }

  return (
    <section className="panel" aria-labelledby="dynamics-heading">
      <header className="panel-header">
        <h2 id="dynamics-heading">Conversation dynamics</h2>
        <span className="muted small">measured from timings</span>
      </header>

      {dynamics.observations.length > 0 && (
        <ul className="rationale">
          {dynamics.observations.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}

      <dl className="stat-grid">
        <div>
          <dt>Talk time (rep)</dt>
          <dd>
            {formatPercent(dynamics.rep_talk_time_ratio)}
            <span className="muted"> {seconds(dynamics.rep_talk_seconds)}</span>
          </dd>
        </div>
        <div>
          <dt>Longest rep stretch</dt>
          <dd>{seconds(dynamics.longest_rep_monologue_s)}</dd>
        </div>
        <div>
          <dt>Interruptions</dt>
          <dd>
            {dynamics.interruptions_by_rep} by rep
            <span className="muted">
              {' / '}
              {dynamics.interruptions_by_prospect} by prospect
            </span>
          </dd>
        </div>
        <div>
          <dt>Pause before replying</dt>
          <dd>
            {seconds(dynamics.rep_response_latency_s)} rep
            <span className="muted">
              {' / '}
              {seconds(dynamics.prospect_response_latency_s)} prospect
            </span>
          </dd>
        </div>
        <div>
          <dt>Speech rate</dt>
          <dd>
            {dynamics.rep_speech_rate_wpm === null
              ? '—'
              : `${Math.round(dynamics.rep_speech_rate_wpm)} wpm`}
            <span className="muted">
              {' / '}
              {dynamics.prospect_speech_rate_wpm === null
                ? '—'
                : `${Math.round(dynamics.prospect_speech_rate_wpm)} wpm`}
            </span>
          </dd>
        </div>
        <div>
          <dt>Silence</dt>
          <dd>{formatPercent(dynamics.dead_air_ratio)}</dd>
        </div>
      </dl>

      {dynamics.prospect_share_by_third && (
        <>
          <h3 className="chart-title">Prospect&rsquo;s share of the conversation</h3>
          <TrajectoryBar shares={dynamics.prospect_share_by_third} />
        </>
      )}
    </section>
  );
}
