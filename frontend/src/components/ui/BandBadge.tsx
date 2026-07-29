'use client';

import { BAND_LABEL } from '@/lib/format';
import type { LeadBand } from '@/lib/types';

/**
 * Lead band indicator.
 *
 * The swatch is decorative; the label is the meaning. Colour never carries
 * the distinction on its own, which keeps the badge readable under any form
 * of colour-vision deficiency and in forced-colors mode.
 */
export function BandBadge({ band, score }: { band: LeadBand; score?: number }) {
  return (
    <span className={`band-badge band-${band}`}>
      <span className="band-dot" aria-hidden="true" />
      {BAND_LABEL[band]}
      {score !== undefined && <span className="band-score">{score.toFixed(0)}</span>}
    </span>
  );
}
