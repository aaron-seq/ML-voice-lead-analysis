/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // The dashboard is a pure API client; no server-side data fetching, so it
  // can be deployed as a static export or behind any Node host.
  eslint: { dirs: ['src'] },
};

module.exports = nextConfig;
