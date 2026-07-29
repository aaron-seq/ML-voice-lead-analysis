'use client';

/**
 * Full call view.
 *
 * Information hierarchy, top to bottom: what to do next, why the score is what
 * it is, what was said. A rep opening this before a follow-up call should get
 * their answer in the first screenful.
 */

import { useEffect, useState } from 'react';

import { ApiError, api } from '@/lib/api';
import {
  OUTCOME_LABEL,
  PRIORITY_LABEL,
  classNames,
  formatDuration,
  formatPercent,
  formatTimestamp,
  sentimentLabel,
  titleise,
} from '@/lib/format';
import type { CallDetail as CallDetailType, Evidence } from '@/lib/types';

import { ScoreBreakdown } from './ScoreBreakdown';
import { SentimentTimeline } from './SentimentTimeline';
import { BandBadge } from './ui/BandBadge';
import { ErrorNotice } from './ui/ErrorNotice';
import { Spinner } from './ui/Spinner';

/**
 * Map each score component to the signals that fed it, mirroring
 * `_COMPONENT_SIGNALS` in backend/app/analysis/scoring.py, so the UI can show
 * the transcript quotes behind a component's points.
 */
const COMPONENT_SIGNALS: Record<string, string[]> = {
  need: ['need', 'technical_fit'],
  next_step: ['next_step'],
  budget: ['budget'],
  authority: ['authority'],
  timeline: ['timeline', 'urgency'],
};

function buildEvidenceMap(call: CallDetailType): Record<string, Evidence[]> {
  const map: Record<string, Evidence[]> = {};
  if (!call.analysis) return map;
  for (const [component, types] of Object.entries(COMPONENT_SIGNALS)) {
    const evidence = call.analysis.signals
      .filter((s) => types.includes(s.type))
      .flatMap((s) => s.evidence)
      .slice(0, 3);
    if (evidence.length) map[component] = evidence;
  }
  return map;
}

export function CallDetailView({
  callId,
  onDeleted,
}: {
  callId: string;
  onDeleted?: () => void;
}) {
  const [call, setCall] = useState<CallDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);
  const [showTranscript, setShowTranscript] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    api
      .getCall(callId, true)
      .then((data) => {
        // Guard against a stale response overwriting a newer selection.
        if (!cancelled) setCall(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err as ApiError);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [callId]);

  async function handleReanalyse() {
    setBusy(true);
    try {
      await api.reanalyse(callId);
      setCall(await api.getCall(callId, true));
      setError(null);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete() {
    if (!window.confirm('Delete this call and its transcript permanently?')) {
      return;
    }
    setBusy(true);
    try {
      await api.deleteCall(callId);
      onDeleted?.();
    } catch (err) {
      setError(err as ApiError);
      setBusy(false);
    }
  }

  if (loading) return <Spinner label="Loading call…" />;
  if (error && !call) return <ErrorNotice error={error} />;
  if (!call) return null;

  const { analysis } = call;

  if (call.status === 'failed') {
    return (
      <div className="detail">
        <h1>{call.title}</h1>
        <div className="notice notice-critical" role="alert">
          <strong>Analysis failed.</strong> {call.error ?? 'Unknown error.'}
          <p>
            The original transcript is still stored. Fix the source and
            re-analyse, or delete the call.
          </p>
          <button type="button" onClick={handleReanalyse} disabled={busy}>
            Retry analysis
          </button>
        </div>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="detail">
        <h1>{call.title}</h1>
        <p className="muted">This call has not been analysed yet.</p>
      </div>
    );
  }

  const unaddressed = analysis.objections.filter((o) => !o.addressed);

  return (
    <article className="detail">
      {error && <ErrorNotice error={error} />}

      <header className="detail-header">
        <div>
          <h1>{call.title}</h1>
          <p className="detail-meta">
            <BandBadge band={analysis.lead_score.band} score={analysis.lead_score.score} />
            <span>{OUTCOME_LABEL[analysis.outcome.label]}</span>
            <span>{formatDuration(analysis.metadata.duration_s)}</span>
            <span>{analysis.metadata.word_count} words</span>
            {call.owner && <span>Owner: {call.owner}</span>}
          </p>
        </div>
        <div className="detail-actions">
          <button type="button" onClick={handleReanalyse} disabled={busy}>
            {busy ? 'Working…' : 'Re-analyse'}
          </button>
          <button
            type="button"
            className="danger"
            onClick={handleDelete}
            disabled={busy}
          >
            Delete
          </button>
        </div>
      </header>

      {analysis.metadata.warnings.length > 0 && (
        <div className="notice notice-warning" role="status">
          <strong>Caveats</strong>
          <ul>
            {analysis.metadata.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      <p className="headline">{analysis.summary.headline}</p>

      {/* Actions come first: this is what the user is here to find out. */}
      <section className="panel" aria-labelledby="actions-heading">
        <header className="panel-header">
          <h2 id="actions-heading">Do next</h2>
        </header>
        <ol className="action-list">
          {analysis.next_actions.map((action) => (
            <li key={action.action} className={`action prio-${action.priority}`}>
              <div className="action-head">
                <span className={`prio-tag prio-${action.priority}`}>
                  {PRIORITY_LABEL[action.priority]}
                </span>
                <span className="action-title">{action.action}</span>
                <span className="action-due">
                  within {action.due_within_hours}h
                </span>
              </div>
              <p className="action-why">{action.rationale}</p>
            </li>
          ))}
        </ol>
      </section>

      {unaddressed.length > 0 && (
        <section className="panel" aria-labelledby="objections-heading">
          <header className="panel-header">
            <h2 id="objections-heading">Unaddressed objections</h2>
            <p className="panel-sub">
              Raised by the prospect with no substantive response on the call.
            </p>
          </header>
          {unaddressed.map((objection) => (
            <div key={objection.category} className="objection">
              <div className="objection-head">
                <span className="status-icon status-critical" aria-hidden="true">
                  !
                </span>
                <strong>{titleise(objection.category)}</strong>
                <span className="muted">
                  severity {formatPercent(objection.severity)}
                </span>
              </div>
              {objection.evidence[0] && (
                <blockquote className="quote">
                  &ldquo;{objection.evidence[0].quote}&rdquo;
                </blockquote>
              )}
              <p className="playbook">{objection.suggested_response}</p>
            </div>
          ))}
        </section>
      )}

      <ScoreBreakdown
        leadScore={analysis.lead_score}
        evidenceByComponent={buildEvidenceMap(call)}
      />

      <section className="panel" aria-labelledby="conversation-heading">
        <header className="panel-header">
          <h2 id="conversation-heading">Conversation</h2>
        </header>
        <dl className="stat-grid">
          <div>
            <dt>Prospect sentiment</dt>
            <dd>
              {sentimentLabel(analysis.sentiment.prospect ?? analysis.sentiment.overall)}
              <span className="muted">
                {' '}
                ({(analysis.sentiment.prospect ?? analysis.sentiment.overall).toFixed(2)})
              </span>
            </dd>
          </div>
          <div>
            <dt>Trend</dt>
            <dd>
              {analysis.sentiment.trend > 0.1
                ? 'Warming'
                : analysis.sentiment.trend < -0.1
                  ? 'Cooling'
                  : 'Flat'}
            </dd>
          </div>
          <div>
            <dt>Talk ratio (rep)</dt>
            <dd>{formatPercent(analysis.talk_metrics.rep_talk_ratio)}</dd>
          </div>
          <div>
            <dt>Rep questions</dt>
            <dd>{analysis.talk_metrics.question_count_rep}</dd>
          </div>
        </dl>

        <SentimentTimeline points={analysis.sentiment.timeline} />
      </section>

      {analysis.summary.key_quotes.length > 0 && (
        <section className="panel" aria-labelledby="quotes-heading">
          <header className="panel-header">
            <h2 id="quotes-heading">Key moments</h2>
          </header>
          {analysis.summary.key_quotes.map((quote, i) => (
            <blockquote key={i} className="quote">
              {quote.start_s != null && (
                <span className="quote-time">{formatTimestamp(quote.start_s)}</span>
              )}
              <span className={classNames('quote-role', `role-${quote.role}`)}>
                {titleise(quote.role)}
              </span>
              <span className="quote-text">&ldquo;{quote.quote}&rdquo;</span>
            </blockquote>
          ))}
        </section>
      )}

      {call.transcript && (
        <section className="panel">
          <header className="panel-header">
            <h2>
              <button
                type="button"
                className="link-button"
                aria-expanded={showTranscript}
                onClick={() => setShowTranscript((v) => !v)}
              >
                {showTranscript ? 'Hide' : 'Show'} full transcript (
                {call.transcript.segments.length} turns)
              </button>
            </h2>
          </header>
          {showTranscript && (
            <ol className="transcript">
              {call.transcript.segments.map((segment) => (
                <li key={segment.index} className={`turn role-${segment.role}`}>
                  <span className="turn-speaker">{segment.speaker}</span>
                  <span className="turn-text">{segment.text}</span>
                </li>
              ))}
            </ol>
          )}
        </section>
      )}

      <footer className="detail-footer muted">
        Analysed with engine {analysis.metadata.engine_version} in{' '}
        {analysis.metadata.processing_ms.toFixed(0)}ms &middot; sentiment via{' '}
        {analysis.metadata.analyzers.sentiment}
      </footer>
    </article>
  );
}
