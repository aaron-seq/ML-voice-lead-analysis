'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Phone, BarChart3, Search, Filter, RefreshCw } from 'lucide-react';

import { CallListSidebar } from './CallListSidebar';
import { CallAnalysisView } from './CallAnalysisView';
import { DashboardHeader } from './DashboardHeader';
import { StatsOverview } from './StatsOverview';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';

interface DashboardState {
  selectedCallFile: string | null;
  searchQuery: string;
  filterCriteria: {
    leadScore: string;
    sentiment: string;
    dateRange: string;
  };
}

const VoiceLeadAnalysisDashboard: React.FC = () => {
  const [dashboardState, setDashboardState] = useState<DashboardState>({
    selectedCallFile: null,
    searchQuery: '',
    filterCriteria: {
      leadScore: 'all',
      sentiment: 'all',
      dateRange: 'all'
    }
  });

  const handleCallSelection = (fileName: string) => {
    setDashboardState(prev => ({
      ...prev,
      selectedCallFile: fileName
    }));
  };

  const handleSearch = (query: string) => {
    setDashboardState(prev => ({
      ...prev,
      searchQuery: query
    }));
  };

  const handleFilterChange = (filterType: string, value: string) => {
    setDashboardState(prev => ({
      ...prev,
      filterCriteria: {
        ...prev.filterCriteria,
        [filterType]: value
      }
    }));
  };

  return (
    <ErrorBoundary>
      <div className="flex h-screen bg-gray-50">
        {/* Sidebar */}
        <motion.aside
          initial={{ x: -300, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.3 }}
          className="w-80 bg-white border-r border-gray-200 shadow-sm flex flex-col"
        >
          <div className="p-6 border-b border-gray-100">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                <Phone className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Voice Analysis</h1>
                <p className="text-sm text-gray-500">Sales Call Insights</p>
              </div>
            </div>
          </div>

          <CallListSidebar
            selectedCall={dashboardState.selectedCallFile}
            onCallSelect={handleCallSelection}
            searchQuery={dashboardState.searchQuery}
            onSearch={handleSearch}
            filterCriteria={dashboardState.filterCriteria}
            onFilterChange={handleFilterChange}
          />
        </motion.aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          <DashboardHeader />
          
          <div className="flex-1 overflow-y-auto">
            <div className="max-w-7xl mx-auto px-6 py-8">
              {!dashboardState.selectedCallFile ? (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                >
                  <StatsOverview />
                  
                  <div className="mt-12 text-center">
                    <div className="w-24 h-24 mx-auto bg-gray-100 rounded-full flex items-center justify-center mb-6">
                      <BarChart3 className="w-10 h-10 text-gray-400" />
                    </div>
                    <h2 className="text-2xl font-semibold text-gray-900 mb-3">
                      Select a Call to Analyze
                    </h2>
                    <p className="text-gray-500 max-w-md mx-auto">
                      Choose a call from the sidebar to view detailed analysis including 
                      sentiment, lead scoring, and key insights.
                    </p>
                  </div>
                </motion.div>
              ) : (
                <AnimatePresence mode="wait">
                  <motion.div
                    key={dashboardState.selectedCallFile}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    transition={{ duration: 0.3 }}
                  >
                    <CallAnalysisView fileName={dashboardState.selectedCallFile} />
                  </motion.div>
                </AnimatePresence>
              )}
            </div>
          </div>
        </main>
      </div>
    </ErrorBoundary>
  );
};

export default VoiceLeadAnalysisDashboard;
