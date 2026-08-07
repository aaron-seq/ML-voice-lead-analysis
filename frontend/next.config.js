/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // The dashboard is a pure API client; no server-side data fetching, so it
  // can be deployed as a static export or behind any Node host.
  //
  // No `eslint` key: Next 16 removed `next lint`, so the option is rejected as
  // unrecognised and only produced a warning on every build. Linting is now
  // `npm run lint` against eslint.config.mjs, which sets its own scope.
};

module.exports = nextConfig;
