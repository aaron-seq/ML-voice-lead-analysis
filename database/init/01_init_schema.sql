-- ML Voice Lead Analysis Database Schema
-- PostgreSQL Initialization Script
-- Version: 1.0.0

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create custom types
DO $$ BEGIN
    CREATE TYPE lead_classification AS ENUM ('Hot', 'Warm', 'Cold', 'Unknown');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE processing_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'reprocessing');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE sentiment_category AS ENUM ('Positive', 'Neutral', 'Negative', 'Mixed');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Audio Files Table
CREATE TABLE IF NOT EXISTS audio_files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(512) NOT NULL UNIQUE,
    s3_key VARCHAR(1024),
    s3_bucket VARCHAR(256),
    file_size_bytes BIGINT,
    file_size_mb DECIMAL(10, 2) GENERATED ALWAYS AS (file_size_bytes / 1048576.0) STORED,
    duration_seconds DECIMAL(10, 2),
    duration_minutes DECIMAL(10, 2) GENERATED ALWAYS AS (duration_seconds / 60.0) STORED,
    audio_format VARCHAR(50),
    sample_rate INTEGER,
    channels INTEGER,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    uploaded_by VARCHAR(256),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Analysis Results Table
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    audio_file_id UUID REFERENCES audio_files(id) ON DELETE CASCADE,
    file_identifier VARCHAR(512) NOT NULL,
    processing_status processing_status DEFAULT 'pending',
    
    -- Sentiment Analysis
    sentiment_score DECIMAL(5, 4) CHECK (sentiment_score >= -1 AND sentiment_score <= 1),
    sentiment_category sentiment_category,
    sentiment_confidence DECIMAL(5, 4),
    sentiment_progression JSONB,
    
    -- Lead Scoring
    lead_classification lead_classification,
    lead_confidence_score DECIMAL(5, 4),
    engagement_level DECIMAL(4, 2),
    
    -- Content Analysis
    transcript_text TEXT,
    key_phrases JSONB DEFAULT '[]'::jsonb,
    topics JSONB DEFAULT '[]'::jsonb,
    technical_terms JSONB DEFAULT '[]'::jsonb,
    interest_moments JSONB DEFAULT '[]'::jsonb,
    
    -- Processing Metadata
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    processing_duration_seconds DECIMAL(10, 2),
    ml_model_version VARCHAR(100),
    analysis_metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT unique_audio_file_analysis UNIQUE(audio_file_id),
    CONSTRAINT unique_file_identifier UNIQUE(file_identifier)
);

-- Lead Scoring Details Table
CREATE TABLE IF NOT EXISTS lead_scoring_details (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_result_id UUID REFERENCES analysis_results(id) ON DELETE CASCADE,
    
    interest_signals JSONB DEFAULT '[]'::jsonb,
    concern_flags JSONB DEFAULT '[]'::jsonb,
    followup_priority VARCHAR(50),
    next_actions JSONB DEFAULT '[]'::jsonb,
    
    buying_intent_score DECIMAL(5, 4),
    decision_maker_confidence DECIMAL(5, 4),
    budget_discussion_detected BOOLEAN DEFAULT false,
    timeline_mentioned BOOLEAN DEFAULT false,
    competitor_mentioned BOOLEAN DEFAULT false,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_analysis_scoring UNIQUE(analysis_result_id)
);

-- Processing Logs Table
CREATE TABLE IF NOT EXISTS processing_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_result_id UUID REFERENCES analysis_results(id) ON DELETE CASCADE,
    log_level VARCHAR(20) NOT NULL,
    log_message TEXT NOT NULL,
    error_details JSONB,
    processing_stage VARCHAR(100),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User Sessions Table (for caching and rate limiting)
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_key VARCHAR(256) UNIQUE NOT NULL,
    user_identifier VARCHAR(256),
    session_data JSONB DEFAULT '{}'::jsonb,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_audio_files_uploaded_at ON audio_files(uploaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_audio_files_filename ON audio_files(filename);
CREATE INDEX IF NOT EXISTS idx_audio_files_s3_key ON audio_files(s3_key);

CREATE INDEX IF NOT EXISTS idx_analysis_results_audio_file_id ON analysis_results(audio_file_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_file_identifier ON analysis_results(file_identifier);
CREATE INDEX IF NOT EXISTS idx_analysis_results_status ON analysis_results(processing_status);
CREATE INDEX IF NOT EXISTS idx_analysis_results_classification ON analysis_results(lead_classification);
CREATE INDEX IF NOT EXISTS idx_analysis_results_created_at ON analysis_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_results_sentiment ON analysis_results(sentiment_category);

CREATE INDEX IF NOT EXISTS idx_lead_scoring_analysis_result_id ON lead_scoring_details(analysis_result_id);
CREATE INDEX IF NOT EXISTS idx_lead_scoring_priority ON lead_scoring_details(followup_priority);

CREATE INDEX IF NOT EXISTS idx_processing_logs_analysis_result_id ON processing_logs(analysis_result_id);
CREATE INDEX IF NOT EXISTS idx_processing_logs_timestamp ON processing_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_processing_logs_level ON processing_logs(log_level);

CREATE INDEX IF NOT EXISTS idx_user_sessions_session_key ON user_sessions(session_key);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_analysis_results_transcript_fts ON analysis_results USING gin(to_tsvector('english', transcript_text));

-- JSON indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_analysis_results_key_phrases_gin ON analysis_results USING gin(key_phrases);
CREATE INDEX IF NOT EXISTS idx_analysis_results_topics_gin ON analysis_results USING gin(topics);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic updated_at
DROP TRIGGER IF EXISTS update_audio_files_updated_at ON audio_files;
CREATE TRIGGER update_audio_files_updated_at
    BEFORE UPDATE ON audio_files
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_analysis_results_updated_at ON analysis_results;
CREATE TRIGGER update_analysis_results_updated_at
    BEFORE UPDATE ON analysis_results
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_lead_scoring_details_updated_at ON lead_scoring_details;
CREATE TRIGGER update_lead_scoring_details_updated_at
    BEFORE UPDATE ON lead_scoring_details
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to clean expired sessions
CREATE OR REPLACE FUNCTION clean_expired_sessions()
RETURNS void AS $$
BEGIN
    DELETE FROM user_sessions WHERE expires_at < CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO voice_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO voice_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO voice_user;

-- Insert sample data for testing (optional)
INSERT INTO audio_files (filename, s3_key, s3_bucket, file_size_bytes, duration_seconds, audio_format, sample_rate, channels)
VALUES 
    ('sample-call-001.wav', 'transcripts/sample-call-001.wav', 'ml-voice-analysis-bucket', 2621440, 912.5, 'wav', 44100, 2)
ON CONFLICT (filename) DO NOTHING;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'ML Voice Lead Analysis database schema initialized successfully';
END $$;
