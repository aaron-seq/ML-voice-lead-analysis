'use client';

/** Filterable call list — the working queue. */

import { useCallback, useEffect, useState } from 'react';

import { ApiError, api } from '@/lib/api';
import {
  BAND_LABEL,
  OUTCOME_LABEL,
  classNames,
  formatRelativeDate,
  sentimentLabel,
} from '@/lib/format';
import type { CallFilters, CallSummaryRow, LeadBand, PageMeta } from '@/lib/types';

import { BandBadge } from './ui/BandBadge';
import { EmptyState } from './ui/EmptyState';
import { ErrorNotice } from './ui/ErrorNotice';
import { Spinner } from './ui/Spinner';

const BANDS: LeadBand[] = ['hot', 'warm', 'cold', 'disqualified'];

interface Props {
  selectedId: string | null;
  onSelect: (id: string) => void;
  /** Bumped by the parent after an upload to force a refetch. */
  refreshToken?: number;
}

export function CallList({ selectedId, onSelect, refreshToken = 0 }: Props) {
  const [rows, setRows] = useState<CallSummaryRow[]>([]);
  const [meta, setMeta] = useState<PageMeta | null>(null);
  const [filters, setFilters] = useState<CallFilters>({ page: 1, page_size: 20 });
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  // Debounce the search box so typing does not fire a request per keystroke.
  useEffect(() => {
    const timer = setTimeout(() => {
      setFilters((f) => ({ ...f, search: search || undefined, page: 1 }));
    }, 250);
    return () => clearTimeout(timer);
  }, [search]);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.listCalls(filters);
      setRows(data.items);
      setMeta(data.pagination);
      setError(null);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    void load();
  }, [load, refreshToken]);

  function toggleBand(band: LeadBand) {
    setFilters((f) => {
      const current = f.band ?? [];
      const next = current.includes(band)
        ? current.filter((b) => b !== band)
        : [...current, band];
      return { ...f, band: next.length ? next : undefined, page: 1 };
    });
  }

  return (
    <div className="call-list">
      <div className="filters">
        <input
          type="search"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search calls…"
          aria-label="Search calls by title"
          className="search-input"
        />
        <div className="band-filters" role="group" aria-label="Filter by lead band">
          {BANDS.map((band) => {
            const active = filters.band?.includes(band) ?? false;
            return (
              <button
                key={band}
                type="button"
                aria-pressed={active}
                className={classNames('chip', `chip-${band}`, active && 'chip-active')}
                onClick={() => toggleBand(band)}
              >
                {BAND_LABEL[band]}
              </button>
            );
          })}
        </div>
      </div>

      {error && <ErrorNotice error={error} />}
      {loading && rows.length === 0 && <Spinner label="Loading calls…" />}

      {!loading && rows.length === 0 && !error && (
        <EmptyState title="No calls yet">
          <p className="muted">
            Upload a transcript to see it scored, or run{' '}
            <code>python -m app.cli.main import ./transcripts</code>.
          </p>
        </EmptyState>
      )}

      <ul className="rows">
        {rows.map((row) => (
          <li key={row.id}>
            <button
              type="button"
              className={classNames('row', selectedId === row.id && 'row-selected')}
              onClick={() => onSelect(row.id)}
              aria-current={selectedId === row.id}
            >
              <span className="row-title">{row.title}</span>
              <span className="row-meta">
                {row.lead_band ? (
                  <BandBadge band={row.lead_band} score={row.lead_score ?? undefined} />
                ) : (
                  <span className={`status-tag status-${row.status}`}>
                    {row.status}
                  </span>
                )}
                {row.outcome && <span>{OUTCOME_LABEL[row.outcome]}</span>}
                <span>{sentimentLabel(row.sentiment)}</span>
                <span className="muted">{formatRelativeDate(row.created_at)}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>

      {meta && meta.total_pages > 1 && (
        <nav className="pager" aria-label="Pagination">
          <button
            type="button"
            disabled={!meta.has_previous}
            onClick={() => setFilters((f) => ({ ...f, page: (f.page ?? 1) - 1 }))}
          >
            Previous
          </button>
          <span className="muted">
            Page {meta.page} of {meta.total_pages} &middot; {meta.total_items} calls
          </span>
          <button
            type="button"
            disabled={!meta.has_next}
            onClick={() => setFilters((f) => ({ ...f, page: (f.page ?? 1) + 1 }))}
          >
            Next
          </button>
        </nav>
      )}
    </div>
  );
}
