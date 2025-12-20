-- PostgreSQL initialization script for ML Voice Lead Analysis
-- This script runs automatically when the container is first created

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create voice_calls table for storing call metadata
CREATE TABLE IF NOT EXISTS voice_calls (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_name VARCHAR(255) NOT NULL UNIQUE,
    file_key VARCHAR(500),
    upload_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    file_size_bytes BIGINT,
    duration_seconds FLOAT,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create analysis_results table for storing ML analysis
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id UUID REFERENCES voice_calls(id) ON DELETE CASCADE,
    sentiment_score FLOAT,
    lead_classification VARCHAR(20),
    confidence_score FLOAT,
    keywords JSONB,
    topics JSONB,
    wow_moments JSONB,
    processing_time_seconds FLOAT,
    model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_voice_calls_status ON voice_calls(status);
CREATE INDEX IF NOT EXISTS idx_voice_calls_upload ON voice_calls(upload_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_lead ON analysis_results(lead_classification);
CREATE INDEX IF NOT EXISTS idx_analysis_sentiment ON analysis_results(sentiment_score);

-- Create function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for auto-updating timestamp
DROP TRIGGER IF EXISTS update_voice_calls_updated_at ON voice_calls;
CREATE TRIGGER update_voice_calls_updated_at
    BEFORE UPDATE ON voice_calls
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data for testing (optional, only in development)
-- Commented out for production safety
-- INSERT INTO voice_calls (file_name, status, duration_seconds) 
-- VALUES ('sample-call-001.json', 'completed', 912.5);

-- Grant permissions (adjust as needed for your security requirements)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO voice_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO voice_user;
