"""
Comprehensive Test Suite for ML Voice Lead Analysis API
Modern testing patterns with pytest, async support, and comprehensive coverage.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json

from main import app, settings, aws_connector, analysis_service


class TestApplicationConfiguration:
    """Test application configuration and settings."""
    
    def test_settings_initialization(self):
        """Test that settings are properly initialized."""
        assert settings.APP_NAME == "ML Voice Lead Analysis API"
        assert settings.VERSION == "3.1.0"
        assert settings.API_VERSION == "v1"
    
    def test_environment_detection(self):
        """Test environment detection logic."""
        # Test development environment
        assert hasattr(settings, 'is_production')
        assert hasattr(settings, 'is_cloud_deployment')
    
    def test_cors_configuration(self):
        """Test CORS settings for different deployment scenarios."""
        assert isinstance(settings.CORS_ORIGINS, list)
        assert len(settings.CORS_ORIGINS) > 0


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
        assert "version" in data
        assert "uptime_seconds" in data
        assert "services" in data
    
    @patch.object(aws_connector, 'check_service_health')
    async def test_health_check_with_service_status(self, mock_service_health):
        """Test health check with mocked service status."""
        # Mock AWS service health
        mock_service_health.return_value = {
            "s3_service": True,
            "bucket_access": True
        }
        
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
            "service", "version", "environment", "status", 
            "uptime_seconds", "endpoints", "timestamp"
        ]
        
        for field in required_fields:
            assert field in data
        
        # Test endpoints structure
        endpoints = data["endpoints"]
        assert "documentation" in endpoints
        assert "health_check" in endpoints
        assert "call_listings" in endpoints


class TestCallListingEndpoint:
    """Test call listing and pagination functionality."""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    @patch.object(analysis_service, 'retrieve_call_listings')
    async def test_call_listings_basic(self, mock_retrieve_calls):
        """Test basic call listings endpoint."""
        # Mock service response
        mock_retrieve_calls.return_value = {
            "data": [],
            "pagination": {
                "current_page": 1,
                "page_size": 20,
                "total_items": 0,
                "total_pages": 0,
                "has_next_page": False,
                "has_previous_page": False
            },
            "success": True,
            "message": "Retrieved 0 call analysis records",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "3.1.0"
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/v1/calls")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "data" in data
        assert "pagination" in data
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


class TestCallAnalysisDetailEndpoint:
    """Test detailed call analysis endpoint."""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    @patch.object(analysis_service, 'get_detailed_call_analysis')
    async def test_call_analysis_details_success(self, mock_get_details):
        """Test successful call analysis details retrieval."""
        # Mock detailed analysis response
        mock_get_details.return_value = {
            "file_name": "test-call.json",
            "original_transcript": "This is a test transcript.",
            "processing_metadata": {"duration": 1.5},
            "sentiment_score": 0.5,
            "lead_score_details": {
                "primary_classification": "Warm",
                "numerical_score": 65,
                "confidence": 0.8
            },
            "extracted_topics": ["product demo", "pricing"],
            "key_phrases": ["interested", "pricing options"],
            "insights_and_moments": [],
            "conversation_metrics": {"word_count": 150},
            "recommendation_summary": {"priority_level": "medium"}
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/v1/calls/test-call.json")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "file_name" in data
        assert "sentiment_score" in data
        assert "lead_score_details" in data
    
    @patch.object(analysis_service, 'get_detailed_call_analysis')
    async def test_call_analysis_details_not_found(self, mock_get_details):
        """Test call analysis details for non-existent file."""
        from fastapi import HTTPException
        
        # Mock not found exception
        mock_get_details.side_effect = HTTPException(
            status_code=404, 
            detail="Analysis results for 'nonexistent.json' not found"
        )
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/v1/calls/nonexistent.json")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["success"] is False


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
        assert "status" in data
        assert "processing_id" in data
        assert data["status"] == "queued"


class TestAnalyticsEndpoints:
    """Test analytics and system information endpoints."""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    @patch.object(aws_connector, 'check_service_health')
    async def test_performance_analytics(self, mock_service_health):
        """Test performance analytics endpoint."""
        mock_service_health.return_value = {
            "s3_service": True,
            "bucket_access": True
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/v1/analytics/performance")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert "data" in data
        
        performance_data = data["data"]
        expected_metrics = [
            "system_uptime_seconds", "total_api_requests", 
            "successful_requests", "success_rate_percentage",
            "average_response_time_seconds", "service_health"
        ]
        
        for metric in expected_metrics:
            assert metric in performance_data
    
    def test_system_information(self):
        """Test system information endpoint."""
        response = self.client.get("/v1/system/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "application" in data
        assert "deployment" in data
        assert "features" in data
        
        # Test application info structure
        app_info = data["application"]
        assert "name" in app_info
        assert "version" in app_info
        assert "environment" in app_info
        
        # Test deployment info structure
        deployment_info = data["deployment"]
        assert "cloud_platform_detected" in deployment_info
        assert "aws_region" in deployment_info


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
    
    @patch('main.analysis_service.retrieve_call_listings')
    async def test_internal_server_error_handling(self, mock_retrieve):
        """Test internal server error handling."""
        # Mock an internal server error
        mock_retrieve.side_effect = Exception("Internal processing error")
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/v1/calls")
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["success"] is False


class TestAWSServiceConnector:
    """Test AWS service connector functionality."""
    
    @patch('boto3.Session')
    def test_s3_client_initialization(self, mock_session):
        """Test S3 client initialization."""
        mock_boto_session = Mock()
        mock_session.return_value = mock_boto_session
        
        # Test that S3 client can be accessed
        connector = aws_connector
        assert hasattr(connector, 's3_client')
    
    @patch.object(aws_connector, 's3_client')
    async def test_service_health_check_success(self, mock_s3_client):
        """Test successful AWS service health check."""
        # Mock successful S3 response
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': []
        }
        
        health_status = await aws_connector.check_service_health()
        
        assert isinstance(health_status, dict)
        assert "s3_service" in health_status
        assert "bucket_access" in health_status
    
    @patch.object(aws_connector, 's3_client')
    async def test_service_health_check_failure(self, mock_s3_client):
        """Test AWS service health check with failures."""
        from botocore.exceptions import ClientError
        
        # Mock S3 client error
        mock_s3_client.list_objects_v2.side_effect = ClientError(
            {'Error': {'Code': 'NoSuchBucket'}}, 
            'list_objects_v2'
        )
        
        health_status = await aws_connector.check_service_health()
        
        assert isinstance(health_status, dict)
        # Should handle error gracefully
        assert "s3_service" in health_status


class TestCallAnalysisService:
    """Test call analysis service functionality."""
    
    def setup_method(self):
        self.service = analysis_service
    
    def test_extract_lead_classification(self):
        """Test lead classification extraction from analysis data."""
        # Test with new format
        data_new_format = {
            "lead_scoring": {
                "primary_score": "Hot",
                "confidence_level": 0.9
            }
        }
        
        result = self.service._extract_lead_classification(data_new_format)
        assert result == "Hot"
        
        # Test with legacy format
        data_legacy_format = {
            "leadScore": {
                "score": "Warm",
                "confidence": 0.7
            }
        }
        
        result = self.service._extract_lead_classification(data_legacy_format)
        assert result == "Warm"
        
        # Test with missing data
        result = self.service._extract_lead_classification({})
        assert result is None
    
    def test_extract_sentiment_overview(self):
        """Test sentiment overview extraction."""
        # Test positive sentiment
        data_positive = {
            "sentiment_analysis": {
                "overall_score": 0.8
            }
        }
        
        result = self.service._extract_sentiment_overview(data_positive)
        assert result == "Positive"
        
        # Test negative sentiment
        data_negative = {
            "sentiment": -0.6
        }
        
        result = self.service._extract_sentiment_overview(data_negative)
        assert result == "Negative"
        
        # Test neutral sentiment
        data_neutral = {
            "sentiment": 0.1
        }
        
        result = self.service._extract_sentiment_overview(data_neutral)
        assert result == "Neutral"
    
    def test_count_insights(self):
        """Test insight counting functionality."""
        data_with_insights = {
            "wow_moments": [
                {"trigger_phrase": "amazing"},
                {"trigger_phrase": "perfect"}
            ],
            "key_phrases": ["pricing", "demo", "timeline"]
        }
        
        count = self.service._count_insights(data_with_insights)
        assert count == 5  # 2 wow moments + 3 key phrases
        
        # Test with empty data
        count = self.service._count_insights({})
        assert count == 0


# Pytest configuration and fixtures
@pytest.fixture
def mock_aws_credentials():
    """Mock AWS credentials for testing."""
    import os
    os.environ["AWS_ACCESS_KEY_ID"] = "test_access_key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test_secret_key"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    yield
    # Cleanup
    for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]:
        os.environ.pop(key, None)


@pytest.fixture
def test_settings():
    """Provide test-specific settings."""
    original_environment = settings.ENVIRONMENT
    settings.ENVIRONMENT = "testing"
    yield settings
    settings.ENVIRONMENT = original_environment


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main(["-v", __file__])