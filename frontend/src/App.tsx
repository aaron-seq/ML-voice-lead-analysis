"use client"

import React, { useState, useEffect, useCallback } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { Toaster } from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  PhoneIcon, 
  ChartBarIcon, 
  UsersIcon,
  CogIcon,
  MagnifyingGlassIcon,
  BellIcon
} from '@heroicons/react/24/outline';

import { CallListPanel } from './components/CallListPanel';
import { CallDetailsPanel } from './components/CallDetailsPanel';
import { AnalyticsDashboard } from './components/AnalyticsDashboard';
import { NotificationCenter } from './components/NotificationCenter';
import { SettingsPanel } from './components/SettingsPanel';
import { SearchBar } from './components/SearchBar';
import { LoadingSpinner } from './components/ui/LoadingSpinner';
import { ErrorBoundary } from './components/ErrorBoundary';

// Create a React Query client with optimized settings
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: attemptIndex => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes
      refetchOnWindowFocus: false,
    },
  },
});

// Types
interface NavigationItem {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  component: React.ComponentType;
}

// Main application component
function VoiceLeadAnalysisDashboard() {
  const [activeView, setActiveView] = useState<string>('calls');
  const [selectedCallFile, setSelectedCallFile] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState<boolean>(false);

  // Navigation configuration
  const navigationItems: NavigationItem[] = [
    {
      id: 'calls',
      label: 'Call Analysis',
      icon: PhoneIcon,
      component: CallListPanel,
    },
    {
      id: 'analytics',
      label: 'Analytics',
      icon: ChartBarIcon,
      component: AnalyticsDashboard,
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: CogIcon,
      component: SettingsPanel,
    },
  ];

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey) {
        switch (event.key) {
          case 'k':
            event.preventDefault();
            // Focus search bar
            document.getElementById('search-input')?.focus();
            break;
          case '1':
            event.preventDefault();
            setActiveView('calls');
            break;
          case '2':
            event.preventDefault();
            setActiveView('analytics');
            break;
          case '3':
            event.preventDefault();
            setActiveView('settings');
            break;
        }
      }
      
      if (event.key === 'Escape') {
        setSelectedCallFile(null);
        setIsMobileMenuOpen(false);
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, []);

  // Handle call selection
  const handleCallSelection = useCallback((fileName: string) => {
    setSelectedCallFile(fileName);
    // Close mobile menu if open
    if (isMobileMenuOpen) {
      setIsMobileMenuOpen(false);
    }
  }, [isMobileMenuOpen]);

  // Handle navigation
  const handleNavigation = useCallback((viewId: string) => {
    setActiveView(viewId);
    setSelectedCallFile(null);
    setIsMobileMenuOpen(false);
  }, []);

  // Render main content based on active view
  const renderMainContent = () => {
    if (activeView === 'calls') {
      return (
        <div className="flex h-full">
          <div className="w-1/3 border-r border-gray-200">
            <CallListPanel
              searchQuery={searchQuery}
              selectedCall={selectedCallFile}
              onCallSelect={handleCallSelection}
            />
          </div>
          <div className="flex-1">
            <CallDetailsPanel selectedCallFile={selectedCallFile} />
          </div>
        </div>
      );
    }

    const ActiveComponent = navigationItems.find(item => item.id === activeView)?.component;
    return ActiveComponent ? <ActiveComponent /> : <div>View not found</div>;
  };

  return (
    <ErrorBoundary>
      <div className="flex h-screen bg-gray-50 text-gray-900">
        {/* Sidebar Navigation */}
        <aside className="hidden md:flex md:flex-shrink-0">
          <div className="flex flex-col w-64">
            <div className="flex flex-col flex-grow border-r border-gray-200 pt-5 pb-4 bg-white overflow-y-auto">
              {/* Logo/Brand */}
              <div className="flex items-center flex-shrink-0 px-4 mb-6">
                <PhoneIcon className="h-8 w-8 text-blue-600" />
                <h1 className="ml-3 text-xl font-bold text-gray-900">
                  Voice Lead AI
                </h1>
              </div>

              {/* Search Bar */}
              <div className="px-4 mb-6">
                <SearchBar
                  value={searchQuery}
                  onChange={setSearchQuery}
                  placeholder="Search calls..."
                />
              </div>

              {/* Navigation Menu */}
              <nav className="flex-1 px-2 space-y-1">
                {navigationItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = activeView === item.id;
                  
                  return (
                    <motion.button
                      key={item.id}
                      onClick={() => handleNavigation(item.id)}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className={`
                        group flex items-center px-2 py-2 text-sm font-medium rounded-md w-full
                        ${isActive
                          ? 'bg-blue-50 border-r-2 border-blue-600 text-blue-700'
                          : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                        }
                      `}
                    >
                      <Icon
                        className={`
                          mr-3 flex-shrink-0 h-6 w-6
                          ${isActive ? 'text-blue-600' : 'text-gray-400 group-hover:text-gray-500'}
                        `}
                      />
                      {item.label}
                    </motion.button>
                  );
                })}
              </nav>

              {/* Footer */}
              <div className="flex-shrink-0 px-4 py-3 border-t border-gray-200">
                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-500">
                    v3.0.0
                  </div>
                  <NotificationCenter />
                </div>
              </div>
            </div>
          </div>
        </aside>

        {/* Mobile menu overlay */}
        <AnimatePresence>
          {isMobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-40 md:hidden"
              onClick={() => setIsMobileMenuOpen(false)}
            >
              <div className="absolute inset-0 bg-gray-600 opacity-75" />
              <div className="relative flex-1 flex flex-col max-w-xs w-full bg-white">
                {/* Mobile navigation content */}
                <div className="flex-1 h-0 pt-5 pb-4 overflow-y-auto">
                  <div className="flex-shrink-0 flex items-center px-4">
                    <PhoneIcon className="h-8 w-8 text-blue-600" />
                    <h1 className="ml-3 text-xl font-bold text-gray-900">
                      Voice Lead AI
                    </h1>
                  </div>
                  <nav className="mt-5 px-2 space-y-1">
                    {navigationItems.map((item) => {
                      const Icon = item.icon;
                      const isActive = activeView === item.id;
                      
                      return (
                        <button
                          key={item.id}
                          onClick={() => handleNavigation(item.id)}
                          className={`
                            group flex items-center px-2 py-2 text-base font-medium rounded-md w-full
                            ${isActive
                              ? 'bg-blue-50 text-blue-700'
                              : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                            }
                          `}
                        >
                          <Icon className="mr-4 flex-shrink-0 h-6 w-6" />
                          {item.label}
                        </button>
                      );
                    })}
                  </nav>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Main content */}
        <div className="flex flex-col w-0 flex-1 overflow-hidden">
          {/* Top navigation bar for mobile */}
          <div className="md:hidden relative z-10 flex-shrink-0 flex h-16 bg-white shadow">
            <button
              className="px-4 border-r border-gray-200 text-gray-500 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500 md:hidden"
              onClick={() => setIsMobileMenuOpen(true)}
            >
              <span className="sr-only">Open sidebar</span>
              <svg
                className="h-6 w-6"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16M4 18h7"
                />
              </svg>
            </button>
          </div>

          {/* Page content */}
          <main className="flex-1 relative overflow-y-auto focus:outline-none">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeView}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.2 }}
                className="h-full"
              >
                {renderMainContent()}
              </motion.div>
            </AnimatePresence>
          </main>
        </div>

        {/* Toast notifications */}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#ffffff',
              color: '#374151',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
            },
          }}
        />
      </div>
    </ErrorBoundary>
  );
}

// Main App component with providers
export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <VoiceLeadAnalysisDashboard />
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}
