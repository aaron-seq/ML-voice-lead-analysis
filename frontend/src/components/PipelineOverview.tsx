'use client';

/**
 * Portfolio view — the manager's screen.
 *
 * Every figure here comes from the API's SQL aggregation over real stored
 * calls. When there is no data it says so rather than showing plausible
 * numbers.
 */

import { useEffect, useState } from 'react';

import { ApiError, api } from '@/lib/api';
import { BAND_COLOR, BAND_LABEL, formatPercent, titleise } from '@/lib/format';
import type { LeadBand, PipelineAnalytics } from '@/lib/types';

import { EmptyState } from './ui/EmptyState';
import { ErrorNotice } from './ui/ErrorNotice';
import { Spinner } from './ui/Spinner';

const BANDS: LeadBand[] = ['hot', 'warm', 'cold', 'disqualified'];

export function PipelineOverview() {
  const [data, setData] = useState<PipelineAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    api
      .analytics(30)
      .then(setData)
      .catch((err) => setError(err as ApiError))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Spinner label="Loading analytics…" />;
  if (error) return <ErrorNotice error={error} />;
  if (!data) return null;

  if (data.total_calls === 0) {
    return (
      <EmptyState title="No calls analysed yet">
        <p className="muted">
          Analytics appear once you have ingested at least one call.
        </p>
      </EmptyState>
    );
  }

  const bandTotal = BANDS.reduce((sum, b) => sum + data.band_breakdown[b], 0);

  return (
    <div className="overview">
      <section className="tiles">
        <div className="tile">
          <span className="tile-label">Calls analysed</span>
          <span className="tile-value">{data.analysed_calls}</span>
          {data.failed_calls > 0 && (
            <span className="tile-sub status-critical">
              {data.failed_calls} failed
            </span>
          )}
        </div>
        <div className="tile">
          <span className="tile-label">Average lead score</span>
          <span className="tile-value">
            {data.avg_lead_score?.toFixed(1) ?? '—'}
          </span>
          <span className="tile-sub muted">out of 100</span>
        </div>
        <div className="tile">
          <span className="tile-label">Qualified rate</span>
          <span className="tile-value">
            {bandTotal
              ? formatPercent(
                  (data.band_breakdown.hot + data.band_breakdown.warm) / bandTotal,
                )
              : '—'}
          </span>
          <span className="tile-sub muted">hot or warm</span>
        </div>
        <div className="tile">
          <span className="tile-label">Meetings booked</span>
          <span className="tile-value">
            {data.outcome_breakdown.meeting_booked ?? 0}
          </span>
        </div>
      </section>

      <section className="panel">
        <header className="panel-header">
          <h2>Lead bands</h2>
        </header>
        {/* Ordinal tiers: one hue stepped by rank, with the label always
            present so colour is never the only cue. */}
        <ul className="band-bars">
          {BANDS.map((band) => {
            const count = data.band_breakdown[band];
            const pct = bandTotal ? (count / bandTotal) * 100 : 0;
            return (
              <li key={band}>
                <span className="band-bar-label">{BAND_LABEL[band]}</span>
                <span className="band-bar-track">
                  <span
                    className="band-bar-fill"
                    style={{
                      width: `${pct}%`,
                      ['--band-color' as string]: BAND_COLOR[band].light,
                    }}
                  />
                </span>
                <span className="band-bar-value">
                  {count} <span className="muted">({pct.toFixed(0)}%)</span>
                </span>
              </li>
            );
          })}
        </ul>
      </section>

      <div className="two-col">
        {data.top_objections.length > 0 && (
          <section className="panel">
            <header className="panel-header">
              <h2>Most common objections</h2>
              <p className="panel-sub">Last 30 days</p>
            </header>
            <table className="mini-table">
              <thead>
                <tr>
                  <th scope="col">Objection</th>
                  <th scope="col" className="num">
                    Calls
                  </th>
                  <th scope="col" className="num">
                    Unaddressed
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.top_objections.map((row) => (
                  <tr key={row.category}>
                    <th scope="row">{titleise(row.category)}</th>
                    <td className="num">{row.count}</td>
                    <td className="num">
                      {row.unaddressed > 0 ? (
                        <span className="status-critical">{row.unaddressed}</span>
                      ) : (
                        row.unaddressed
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        {data.top_risks.length > 0 && (
          <section className="panel">
            <header className="panel-header">
              <h2>Recurring deal risks</h2>
              <p className="panel-sub">
                Coaching signal: what the team consistently misses.
              </p>
            </header>
            <table className="mini-table">
              <thead>
                <tr>
                  <th scope="col">Risk</th>
                  <th scope="col" className="num">
                    Calls
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.top_risks.map((row) => (
                  <tr key={row.risk}>
                    <th scope="row">{titleise(row.risk)}</th>
                    <td className="num">{row.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}
      </div>

      <p className="muted small">
        <a href={api.exportCsvUrl()} download>
          Export all calls as CSV
        </a>
      </p>
    </div>
  );
}
