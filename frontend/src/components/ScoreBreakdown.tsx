'use client';

/**
 * The explainable lead score.
 *
 * This is the component that justifies the product. A number alone invites
 * "the AI said 73" and gets ignored the first time a rep disagrees with it.
 * Showing the weighted arithmetic, the penalties, and the transcript quote
 * behind each component turns the score into an argument the user can check.
 */

import { useState } from 'react';

import type { Evidence, LeadScore } from '@/lib/types';
import { BAND_LABEL, classNames, formatPercent, titleise } from '@/lib/format';

function EvidenceQuote({ evidence }: { evidence: Evidence }) {
  return (
    <blockquote className="quote">
      <span className={classNames('quote-role', `role-${evidence.role}`)}>
        {titleise(evidence.role)}
      </span>
      <span className="quote-text">&ldquo;{evidence.quote}&rdquo;</span>
      {evidence.matched_phrase && (
        <span className="quote-match">matched: {evidence.matched_phrase}</span>
      )}
    </blockquote>
  );
}

interface Props {
  leadScore: LeadScore;
  /** Evidence keyed by component name, so a row can show its own proof. */
  evidenceByComponent?: Record<string, Evidence[]>;
}

export function ScoreBreakdown({ leadScore, evidenceByComponent = {} }: Props) {
  const [expanded, setExpanded] = useState<string | null>(null);

  const maxWeight = Math.max(...leadScore.components.map((c) => c.weight), 1);

  return (
    <section className="panel" aria-labelledby="score-heading">
      <header className="panel-header">
        <h2 id="score-heading">Why this score</h2>
        <p className="panel-sub">
          {leadScore.score.toFixed(0)} of 100 &middot;{' '}
          {BAND_LABEL[leadScore.band]} &middot; confidence{' '}
          {formatPercent(leadScore.confidence)}
        </p>
      </header>

      {leadScore.confidence < 0.45 && (
        <p className="notice notice-warning" role="status">
          <strong>Low confidence.</strong> This score rests on limited evidence
          — a short transcript, few signals, or missing speaker labels. Treat it
          as indicative rather than decisive.
        </p>
      )}

      <table className="score-table">
        <caption className="sr-only">
          Lead score components, their weights and contributions
        </caption>
        <thead>
          <tr>
            <th scope="col">Component</th>
            <th scope="col" className="num">
              Points
            </th>
            <th scope="col">Contribution</th>
          </tr>
        </thead>
        <tbody>
          {leadScore.components.map((component) => {
            const evidence = evidenceByComponent[component.name] ?? [];
            const isOpen = expanded === component.name;
            return (
              <>
                <tr
                  key={component.name}
                  className={classNames(
                    'score-row',
                    evidence.length > 0 && 'has-evidence',
                  )}
                  onClick={() =>
                    evidence.length > 0 &&
                    setExpanded(isOpen ? null : component.name)
                  }
                >
                  <th scope="row">
                    {titleise(component.name)}
                    {evidence.length > 0 && (
                      <button
                        type="button"
                        className="evidence-toggle"
                        aria-expanded={isOpen}
                        aria-label={`Show evidence for ${component.name}`}
                      >
                        {isOpen ? '−' : '+'} {evidence.length} quote
                        {evidence.length === 1 ? '' : 's'}
                      </button>
                    )}
                  </th>
                  <td className="num">
                    {component.contribution.toFixed(1)}
                    <span className="muted"> / {component.weight.toFixed(0)}</span>
                  </td>
                  <td>
                    {/* Bar width is relative to the largest weight, so bars
                        compare contributions across components honestly. */}
                    <div className="bar-track">
                      <div
                        className="bar-fill"
                        style={{
                          width: `${(component.contribution / maxWeight) * 100}%`,
                        }}
                      />
                    </div>
                    <span className="bar-detail">{component.detail}</span>
                  </td>
                </tr>
                {isOpen &&
                  evidence.map((item, i) => (
                    <tr key={`${component.name}-ev-${i}`} className="evidence-row">
                      <td colSpan={3}>
                        <EvidenceQuote evidence={item} />
                      </td>
                    </tr>
                  ))}
              </>
            );
          })}

          {leadScore.penalties.map((penalty) => (
            <tr key={penalty.name} className="score-row penalty">
              <th scope="row">
                <span className="penalty-icon" aria-hidden="true">
                  ▼
                </span>{' '}
                {titleise(penalty.name)}
              </th>
              <td className="num">{penalty.contribution.toFixed(1)}</td>
              <td>
                <span className="bar-detail">{penalty.detail}</span>
              </td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <th scope="row">Total</th>
            <td className="num total">{leadScore.score.toFixed(1)}</td>
            <td />
          </tr>
        </tfoot>
      </table>

      {leadScore.rationale.length > 0 && (
        <div className="rationale">
          <h3>In short</h3>
          <ul>
            {leadScore.rationale.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
