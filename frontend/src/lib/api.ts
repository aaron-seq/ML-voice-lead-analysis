import axios, { AxiosResponse } from 'axios';

// Types
export interface CallSummary {
  file_name: string;
  created_at?: string;
  file_size?: number;
  lead_score?: string;
}

export interface LeadScoreDetails {
  score: string;
  confidence: number;
  reasoning?: string;
}

export interface HighInterestMoment {
  keyword: string;
  context: string;
  timestamp?: number;
  sentiment_score?: number;
}

export interface CallAnalysisResult {
  file_name: string;
  transcript: string;
  sentiment_score: number;
  key_phrases: string[];
  discussion_topics: string[];
  high_interest_moments: HighInterestMoment[];
  lead_classification: LeadScoreDetails;
  call_duration?: number;
  participant_count?: number;
}

export interface CallListResponse {
  success: boolean;
  calls: CallSummary[];
  total_count: number;
  page: number;
  page_size: number;
  has_next: boolean;
  message?: string;
}

export interface ApiError {
  success: false;
  message: string;
  error_code?: number;
}

// Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_VERSION = 'v1';

// Create axios instance
const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/${API_VERSION}`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null;
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      if (typeof window !== 'undefined') {
        localStorage.removeItem('auth_token');
        // Redirect to login if needed
      }
    }
    return Promise.reject(error);
  }
);

// API Functions
export class VoiceAnalysisAPI {
  
  /**
   * Fetch paginated list of analyzed calls
   */
  static async getCallsList(params: {
    page?: number;
    page_size?: number;
  } = {}): Promise<CallListResponse> {
    try {
      const response: AxiosResponse<CallListResponse> = await apiClient.get('/calls', {
        params: {
          page: params.page || 1,
          page_size: params.page_size || 20,
        },
      });
      
      return response.data;
    } catch (error: any) {
      console.error('Failed to fetch calls list:', error);
      throw new Error(
        error.response?.data?.message || 'Failed to fetch calls list'
      );
    }
  }

  /**
   * Fetch detailed analysis for a specific call
   */
  static async getCallAnalysis(fileName: string): Promise<CallAnalysisResult> {
    try {
      const response: AxiosResponse<CallAnalysisResult> = await apiClient.get(
        `/calls/${encodeURIComponent(fileName)}`
      );
      
      return response.data;
    } catch (error: any) {
      console.error(`Failed to fetch analysis for ${fileName}:`, error);
      throw new Error(
        error.response?.data?.message || `Failed to fetch analysis for ${fileName}`
      );
    }
  }

  /**
   * Trigger re-analysis of a call
   */
  static async triggerReanalysis(fileName: string): Promise<{ success: boolean; message: string }> {
    try {
      const response = await apiClient.post(
        `/calls/${encodeURIComponent(fileName)}/reanalyze`
      );
      
      return response.data;
    } catch (error: any) {
      console.error(`Failed to trigger reanalysis for ${fileName}:`, error);
      throw new Error(
        error.response?.data?.message || `Failed to trigger reanalysis for ${fileName}`
      );
    }
  }

  /**
   * Health check endpoint
   */
  static async healthCheck(): Promise<{
    success: boolean;
    version: string;
    environment: string;
    aws_connection: boolean;
  }> {
    try {
      const response = await apiClient.get('/health');
      return response.data;
    } catch (error: any) {
      console.error('Health check failed:', error);
      throw new Error('Health check failed');
    }
  }

  /**
   * Search calls (if implemented on backend)
   */
  static async searchCalls(query: string, filters?: {
    lead_score?: string;
    sentiment?: string;
    date_range?: string;
  }): Promise<CallListResponse> {
    try {
      const response: AxiosResponse<CallListResponse> = await apiClient.get('/calls/search', {
        params: {
          q: query,
          ...filters,
        },
      });
      
      return response.data;
    } catch (error: any) {
      console.error('Search failed:', error);
      throw new Error(
        error.response?.data?.message || 'Search failed'
      );
    }
  }
}

// Hook-friendly exports for React Query
export const voiceAnalysisQueries = {
  all: ['voice-analysis'] as const,
  calls: () => [...voiceAnalysisQueries.all, 'calls'] as const,
  callsList: (params: { page?: number; page_size?: number }) => 
    [...voiceAnalysisQueries.calls(), 'list', params] as const,
  callAnalysis: (fileName: string) => 
    [...voiceAnalysisQueries.calls(), 'analysis', fileName] as const,
};

// Default export
export default VoiceAnalysisAPI;
