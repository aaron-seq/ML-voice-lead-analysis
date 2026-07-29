import type { Metadata, Viewport } from 'next';

import './globals.css';

export const metadata: Metadata = {
  title: 'CallSense — Call intelligence for sales teams',
  description:
    'Turn recorded sales calls into qualified leads, explained scores and a '
    + 'prioritised follow-up list. Every score links back to the transcript.',
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  // Both schemes are designed, so let the OS pick.
  colorScheme: 'light dark',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
