'use client';

/**
 * Prospect sentiment across the call.
 *
 * A single-series line on a diverging scale: the zero line is the reference,
 * above is positive, below negative. One series, so no legend box — the
 * heading names it. Hover gives the exact value per point.
 *
 * Rendered as inline SVG rather than a chart library: it is one polyline, and
 * pulling in a charting runtime for it would cost more than it returns.
 */

import { useId, useState } from 'react';

import type { SentimentPoint } from '@/lib/types';

const WIDTH = 720;
const HEIGHT = 140;
const PAD_X = 8;
const PAD_Y = 12;

export function SentimentTimeline({ points }: { points: SentimentPoint[] }) {
  const clipId = useId();
  const [hover, setHover] = useState<SentimentPoint | null>(null);

  if (points.length < 2) {
    return (
      <p className="muted small">
        Not enough turns to plot a sentiment timeline.
      </p>
    );
  }

  const x = (i: number) =>
    PAD_X + (i / (points.length - 1)) * (WIDTH - PAD_X * 2);
  // score is on [-1, 1]; map to the plot box with 0 at the vertical centre.
  const y = (score: number) =>
    PAD_Y + ((1 - score) / 2) * (HEIGHT - PAD_Y * 2);

  const line = points.map((p, i) => `${x(i)},${y(p.score)}`).join(' ');
  const zeroY = y(0);
  const areaAbove = `${PAD_X},${zeroY} ${line} ${WIDTH - PAD_X},${zeroY}`;

  return (
    <figure className="chart">
      <figcaption className="chart-title">
        Sentiment across the call
        <span className="muted"> — above the line is positive</span>
      </figcaption>

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="sentiment-svg"
        role="img"
        aria-label={`Sentiment timeline across ${points.length} turns, ranging from ${Math.min(
          ...points.map((p) => p.score),
        ).toFixed(2)} to ${Math.max(...points.map((p) => p.score)).toFixed(2)}`}
        onMouseLeave={() => setHover(null)}
      >
        <defs>
          <clipPath id={`${clipId}-above`}>
            <rect x="0" y="0" width={WIDTH} height={zeroY} />
          </clipPath>
          <clipPath id={`${clipId}-below`}>
            <rect x="0" y={zeroY} width={WIDTH} height={HEIGHT - zeroY} />
          </clipPath>
        </defs>

        {/* Zero reference, recessive. */}
        <line
          x1={PAD_X}
          x2={WIDTH - PAD_X}
          y1={zeroY}
          y2={zeroY}
          className="chart-baseline"
        />

        {/* Diverging fills: positive above the line, negative below. */}
        <polygon
          points={areaAbove}
          className="sentiment-area positive"
          clipPath={`url(#${clipId}-above)`}
        />
        <polygon
          points={areaAbove}
          className="sentiment-area negative"
          clipPath={`url(#${clipId}-below)`}
        />

        <polyline points={line} className="sentiment-line" />

        {/* Invisible wide hit targets — the mark itself is too thin to hover. */}
        {points.map((point, i) => (
          <rect
            key={point.segment_index}
            x={x(i) - (WIDTH - PAD_X * 2) / (points.length - 1) / 2}
            y={0}
            width={(WIDTH - PAD_X * 2) / (points.length - 1)}
            height={HEIGHT}
            fill="transparent"
            onMouseEnter={() => setHover(point)}
          />
        ))}

        {hover && (
          <g pointerEvents="none">
            <line
              x1={x(points.indexOf(hover))}
              x2={x(points.indexOf(hover))}
              y1={PAD_Y}
              y2={HEIGHT - PAD_Y}
              className="chart-crosshair"
            />
            <circle
              cx={x(points.indexOf(hover))}
              cy={y(hover.score)}
              r={5}
              className="sentiment-dot"
            />
          </g>
        )}
      </svg>

      <p className="chart-readout" aria-live="polite">
        {hover
          ? `Turn ${hover.segment_index + 1}: ${hover.score >= 0 ? '+' : ''}${hover.score.toFixed(2)}`
          : `${points.length} turns`}
      </p>
    </figure>
  );
}
