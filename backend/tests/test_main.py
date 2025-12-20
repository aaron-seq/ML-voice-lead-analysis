"""
Comprehensive Test Suite for ML Voice Lead Analysis API
Modern testing patterns with pytest, async support, and comprehensive coverage.
Aligned with main.py implementation v4.0.0
"""

import pytest
from datetime import datetime
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient
import os

# Set testing environment before importing app
os.environ["ENVIRONMENT"] = "testing"
os.environ["DISABLE_AWS_CHECKS"] = "true"

from main import app, settings, aws_connector, analysis_service


class TestApplicationConfiguration:
    """Test application configuration and settings."""
    
    def test_settings_initialization(self):
        """Test that settings are properly initialized."""
        assert settings.app_name == "ML Voice Lead Analysis API"
        assert settings.version == "4.0.0"
        assert settings.api_version == "v1"
    
    def test_environment_detection(self):
        """Test environment detection logic."""
        assert hasattr(settings, 'environment')
        assert hasattr(settings, 'debug_mode')
        assert hasattr(settings, 'is_testing_environment')
    
    def test_cors_configuration(self):
        """Test CORS settings for different deployment scenarios."""
        assert isinstance(settings.cors_origins, list)
        assert len(settings.cors_origins) > 0


class TestHealthCheckEndpoint:
    """Test health check functionality."""
    
    def setup_method(self):
        """Setup test client for each test method."""
        self.client = TestClient(app)
    
    def test_basic_health_check(self):
        """Test basic health check endpoint."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "environment" in data
        assert "uptime_seconds" in data
        assert "service_checks" in data
    
    @pytest.mark.asyncio
    async def test_health_check_with_service_status(self):
        """Test health check with mocked service status."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]


class TestAPIRootEndpoint:
    """Test API root endpoint functionality."""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_api_root_response(self):
        """Test API root endpoint response structure."""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        required_fields = [
            "service_name", "version", "environment", "status", 
            "api_documentation", "health_endpoint", "timestamp"
        ]
        
        for field in required_fields:
            assert field in data
        
        # Test status field
        assert data["status"] == "operational"


class TestCallListingEndpoint:
    """Test call listing and pagination functionality."""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_call_listings_basic(self):
        """Test basic call listings endpoint."""
        response = self.client.get("/v1/calls")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "call_summaries" in data
        assert "pagination_info" in data
        assert "success" in data
    
    def test_call_listings_pagination_parameters(self):
        """Test pagination parameter validation."""
        # Test valid pagination parameters
        response = self.client.get("/v1/calls?page=2&page_size=10")
        # Should not return validation errors for valid params
        assert response.status_code != 422
        
        # Test invalid pagination parameters
        response = self.client.get("/v1/calls?page=0")  # Invalid page
        assert response.status_code == 422
        
        response = self.client.get("/v1/calls?page_size=101")  # Exceeds max
        assert response.status_code == 422
    
    def test_call_listings_sorting_parameters(self):
        """Test sorting parameters."""
        response = self.client.get("/v1/calls?sort_by=upload_timestamp&sort_order=asc")
        assert response.status_code == 200
        
        response = self.client.get("/v1/calls?sort_order=invalid")
        assert response.status_code == 422


class TestCallAnalysisDetailEndpoint:
    """Test detailed call analysis endpoint."""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_call_analysis_details_success(self):
        """Test successful call analysis details retrieval."""
        response = self.client.get("/v1/calls/test-call.json")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure matches ComprehensiveCallAnalysis model
        assert "fileName" in data or "file_identifier" in data
        assert "sentiment" in data or "sentiment_score" in data
        assert "leadScore" in data or "lead_scoring_analysis" in data


class TestCallReprocessingEndpoint:
    """Test call reprocessing functionality."""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_trigger_reprocessing_success(self):
        """Test successful reprocessing trigger."""
        response = self.client.post("/v1/calls/test-call.json/reprocess")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "message" in data
        assert "processing_status" in data
        assert data["processing_status"] == "queued"


class TestAnalyticsEndpoints:
    """Test analytics and system information endpoints."""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_analytics_dashboard(self):
        """Test analytics dashboard endpoint."""
        response = self.client.get("/v1/analytics/dashboard")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert "analytics_data" in data
        
        analytics_data = data["analytics_data"]
        assert "total_calls_analyzed" in analytics_data
        assert "lead_score_distribution" in analytics_data


class TestErrorHandling:
    """Test error handling and exception management."""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_404_error_handling(self):
        """Test 404 error handling."""
        response = self.client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test method not allowed error handling."""
        response = self.client.post("/health")  # Health endpoint only supports GET
        
        assert response.status_code == 405


class TestAWSServiceConnector:
    """Test AWS service connector functionality."""
    
    @pytest.mark.asyncio
    async def test_service_connectivity_in_testing_mode(self):
        """Test AWS connectivity returns mock data in testing environment."""
        result = await aws_connector.verify_service_connectivity()
        
        assert isinstance(result, dict)
        assert result.get('testing_mode', False) is True
        assert result.get('s3_service') is True
        assert result.get('transcribe_service') is True


class TestVoiceAnalysisService:
    """Test voice analysis service functionality."""
    
    def setup_method(self):
        self.service = analysis_service
    
    @pytest.mark.asyncio
    async def test_get_call_list_paginated_mock_data(self):
        """Test that mock data is returned in testing environment."""
        result = await self.service.get_call_list_paginated(
            page_number=1,
            items_per_page=20
        )
        
        assert result.success is True
        assert len(result.call_summaries) > 0
        assert result.pagination_info.current_page == 1
    
    @pytest.mark.asyncio
    async def test_get_detailed_call_analysis_mock_data(self):
        """Test detailed analysis returns mock data in testing environment."""
        result = await self.service.get_detailed_call_analysis("test-call.json")
        
        # Check the mock data structure
        assert result.file_identifier == "test-call.json"
        assert result.sentiment_score >= -1.0 and result.sentiment_score <= 1.0


# Pytest configuration and fixtures
@pytest.fixture(scope="session", autouse=True)
def configure_testing_environment():
    """Configure testing environment variables."""
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["DISABLE_AWS_CHECKS"] = "true"
    yield


@pytest.fixture
def test_client():
    """Provide test client fixture."""
    return TestClient(app)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main(["-v", __file__])