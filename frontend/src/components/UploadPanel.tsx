'use client';

/** Transcript upload: drop a file or paste text. */

import { useRef, useState } from 'react';

import { ApiError, api } from '@/lib/api';
import { classNames } from '@/lib/format';

import { ErrorNotice } from './ui/ErrorNotice';

const ACCEPTED = '.txt,.json,.vtt,.srt,.md';

export function UploadPanel({ onUploaded }: { onUploaded: (id: string) => void }) {
  const [mode, setMode] = useState<'file' | 'paste'>('file');
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [title, setTitle] = useState('');
  const [text, setText] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  async function submit(fn: () => Promise<{ id: string }>) {
    setBusy(true);
    setError(null);
    try {
      const call = await fn();
      setTitle('');
      setText('');
      onUploaded(call.id);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setBusy(false);
    }
  }

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (file) void submit(() => api.uploadCall(file, title || undefined));
  };

  return (
    <section className="panel upload-panel" aria-labelledby="upload-heading">
      <header className="panel-header">
        <h2 id="upload-heading">Add a call</h2>
        <p className="panel-sub">
          Transcripts from AWS Transcribe, Whisper, WebVTT/SRT, or plain
          &ldquo;Speaker: text&rdquo; are all detected automatically.
        </p>
      </header>

      <div className="mode-switch" role="tablist" aria-label="Upload method">
        <button
          type="button"
          role="tab"
          aria-selected={mode === 'file'}
          className={classNames('chip', mode === 'file' && 'chip-active')}
          onClick={() => setMode('file')}
        >
          Upload file
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === 'paste'}
          className={classNames('chip', mode === 'paste' && 'chip-active')}
          onClick={() => setMode('paste')}
        >
          Paste text
        </button>
      </div>

      <label className="field">
        <span>Title (optional)</span>
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Acme Corp — discovery call"
          maxLength={200}
        />
      </label>

      {mode === 'file' ? (
        <div
          className={classNames('dropzone', dragging && 'dropzone-active')}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            handleFiles(e.dataTransfer.files);
          }}
        >
          <p>Drop a transcript here</p>
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            disabled={busy}
          >
            {busy ? 'Analysing…' : 'Choose a file'}
          </button>
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED}
            hidden
            onChange={(e) => handleFiles(e.target.files)}
          />
          <p className="muted small">Accepts {ACCEPTED}</p>
        </div>
      ) : (
        <>
          <label className="field">
            <span>Transcript</span>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={10}
              placeholder={'Rep: Hi, thanks for taking the time…\nProspect: Sure, what is this about?'}
            />
          </label>
          <button
            type="button"
            className="primary"
            disabled={busy || !text.trim()}
            onClick={() =>
              void submit(() =>
                api.createCall({
                  title: title.trim() || 'Untitled call',
                  transcript: text,
                }),
              )
            }
          >
            {busy ? 'Analysing…' : 'Analyse call'}
          </button>
        </>
      )}

      {error && <ErrorNotice error={error} />}
    </section>
  );
}
