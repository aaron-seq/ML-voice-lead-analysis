import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'ML Voice Lead Analysis | AI-Powered Sales Call Insights',
  description: 'Transform your sales conversations into actionable insights with advanced ML, real-time sentiment analysis, and intelligent lead scoring.',
  keywords: ['voice analysis', 'lead scoring', 'sentiment analysis', 'sales calls', 'AI', 'machine learning'],
  authors: [{ name: 'Aaron Sequeira' }],
  openGraph: {
    title: 'ML Voice Lead Analysis',
    description: 'AI-Powered Sales Call Analysis Platform',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
