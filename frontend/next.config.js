/** @type {import('next').NextConfig} */
const nextConfig = {
  // Build optimizations
  reactStrictMode: true,
  swcMinify: true,
  
  // Performance optimizations
  experimental: {
    optimizePackageImports: ['@heroicons/react', 'lucide-react'],
    turbo: {
      rules: {
        '*.svg': {
          loaders: ['@svgr/webpack'],
          as: '*.js'
        }
      }
    }
  },
  
  // Image optimization
  images: {
    domains: [
      'localhost',
      'ml-voice-analysis.vercel.app',
      'ml-voice-analysis.netlify.app',
      'ml-voice-analysis.railway.app'
    ],
    formats: ['image/webp', 'image/avif'],
    minimumCacheTTL: 60 * 60 * 24 * 30, // 30 days
  },
  
  // Environment configuration
  env: {
    CUSTOM_KEY: process.env.CUSTOM_KEY,
    APP_VERSION: process.env.npm_package_version || '3.1.0',
  },
  
  // Public runtime config
  publicRuntimeConfig: {
    API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    APP_NAME: process.env.NEXT_PUBLIC_APP_NAME || 'ML Voice Lead Analysis',
  },
  
  // Redirects for better UX
  async redirects() {
    return [
      {
        source: '/dashboard',
        destination: '/',
        permanent: true,
      },
      {
        source: '/docs',
        destination: '/api/v1/docs',
        permanent: false,
      },
    ]
  },
  
  // Headers for security and performance
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
        ],
      },
      {
        source: '/api/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'no-store, must-revalidate',
          },
        ],
      },
      {
        source: '/_next/static/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
    ]
  },
  
  // Webpack configuration
  webpack: (config, { buildId, dev, isServer, defaultLoaders, webpack }) => {
    // Bundle analyzer (only in development)
    if (process.env.ANALYZE === 'true') {
      const BundleAnalyzerPlugin = require('@next/bundle-analyzer')({
        enabled: process.env.ANALYZE === 'true',
      })
      config.plugins.push(new BundleAnalyzerPlugin())
    }
    
    // Optimize for production
    if (!dev && !isServer) {
      config.resolve.alias = {
        ...config.resolve.alias,
        // Reduce bundle size by aliasing React to the same instance
        'react': require.resolve('react'),
        'react-dom': require.resolve('react-dom'),
      }
    }
    
    // Handle SVG imports
    config.module.rules.push({
      test: /\.svg$/,
      use: ['@svgr/webpack'],
    })
    
    return config
  },
  
  // Output configuration for different deployment platforms
  output: process.env.NODE_ENV === 'production' ? 'standalone' : undefined,
  
  // Compiler options
  compiler: {
    // Remove console logs in production
    removeConsole: process.env.NODE_ENV === 'production',
    // Enable React Compiler (experimental)
    reactRemoveProperties: process.env.NODE_ENV === 'production',
  },
  
  // ESLint configuration
  eslint: {
    // Disable ESLint during builds (handled by CI/CD)
    ignoreDuringBuilds: true,
  },
  
  // TypeScript configuration
  typescript: {
    // Disable type checking during builds (handled by CI/CD)
    ignoreBuildErrors: true,
  },
  
  // Deployment-specific configurations
  ...(process.env.VERCEL && {
    // Vercel-specific optimizations
    experimental: {
      optimizeCss: true,
      scrollRestoration: true,
    },
  }),
  
  ...(process.env.NETLIFY && {
    // Netlify-specific configurations
    trailingSlash: true,
  }),
  
  // Development-specific configurations
  ...(process.env.NODE_ENV === 'development' && {
    // Enable development optimizations
    devIndicators: {
      buildActivity: true,
    },
    // Faster refresh
    experimental: {
      fastRefresh: true,
    },
  }),
}

// Export with conditional configurations for different environments
module.exports = nextConfig