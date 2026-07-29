'use client';

import { useState } from 'react';

import { CallDetailView } from '@/components/CallDetail';
import { CallList } from '@/components/CallList';
import { PipelineOverview } from '@/components/PipelineOverview';
import { UploadPanel } from '@/components/UploadPanel';
import { EmptyState } from '@/components/ui/EmptyState';
import { classNames } from '@/lib/format';

type View = 'calls' | 'overview' | 'upload';

export default function HomePage() {
  const [view, setView] = useState<View>('calls');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);

  function handleUploaded(id: string) {
    setSelectedId(id);
    setView('calls');
    // Force the list to refetch so the new call appears immediately.
    setRefreshToken((n) => n + 1);
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <h1>CallSense</h1>
            <p>Call intelligence for sales teams</p>
          </div>
        </div>
        <nav aria-label="Main">
          {(
            [
              ['calls', 'Calls'],
              ['overview', 'Pipeline'],
              ['upload', 'Add call'],
            ] as [View, string][]
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              className={classNames('nav-item', view === id && 'nav-active')}
              aria-current={view === id ? 'page' : undefined}
              onClick={() => setView(id)}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>

      <main className="app-main">
        {view === 'calls' && (
          <div className="split">
            <aside className="split-list" aria-label="Call list">
              <CallList
                selectedId={selectedId}
                onSelect={setSelectedId}
                refreshToken={refreshToken}
              />
            </aside>
            <section className="split-detail">
              {selectedId ? (
                <CallDetailView
                  key={selectedId}
                  callId={selectedId}
                  onDeleted={() => {
                    setSelectedId(null);
                    setRefreshToken((n) => n + 1);
                  }}
                />
              ) : (
                <EmptyState title="Select a call">
                  <p className="muted">
                    Pick a call to see its score breakdown, objections and
                    recommended next actions.
                  </p>
                </EmptyState>
              )}
            </section>
          </div>
        )}

        {view === 'overview' && <PipelineOverview />}
        {view === 'upload' && <UploadPanel onUploaded={handleUploaded} />}
      </main>
    </div>
  );
}
