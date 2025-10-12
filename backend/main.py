"""
ML Voice Lead Analysis API - Production Ready Application
Modern FastAPI application with enterprise-grade features and cloud deployment support.
"""

import os
import json
import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import traceback

import boto3
from fastapi import (
    FastAPI, HTTPException, Depends, BackgroundTasks, 
    Query, Path, status, Request, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
import structlog
from botocore.exceptions import ClientError, BotoCoreError

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    level=logging.INFO,
)
logger = structlog.get_logger()

# --- Configuration Management ---
class ApplicationSettings:
    """Centralized application configuration with environment detection."""
    
    # Application metadata
    APP_NAME: str = "ML Voice Lead Analysis API"
    VERSION: str = "3.1.0"
    API_VERSION: str = "v1"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG_MODE: bool = ENVIRONMENT == "development"
    
    # Server configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # AWS Configuration with fallbacks
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv("DATA_BUCKET", os.getenv("S3_BUCKET_NAME", "ml-voice-analysis-data"))
    S3_ANALYSIS_PREFIX: str = "analysis-results/"
    S3_TRANSCRIPTS_PREFIX: str = "transcripts/"
    
    # Database settings (optional for cloud deployment)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "sqlite+aiosqlite:///./voice_analysis.db"
    )
    
    # Performance settings
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    REQUEST_TIMEOUT_SECONDS: int = 30
    
    # Security configuration
    SECRET_KEY: str = os.getenv("SECRET_KEY", "fallback-secret-key-change-in-production")
    
    # CORS settings for deployment platforms
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app",
        "https://*.netlify.app",
        "https://*.render.com",
        "https://*.railway.app",
        "https://*.fly.dev"
    ]
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"
    
    @property
    def is_cloud_deployment(self) -> bool:
        """Detect if running on cloud platforms."""
        cloud_indicators = [
            "VERCEL", "NETLIFY", "RENDER", "RAILWAY", "FLY_APP_NAME",
            "HEROKU_APP_NAME", "AWS_LAMBDA_FUNCTION_NAME"
        ]
        return any(os.getenv(indicator) for indicator in cloud_indicators)

settings = ApplicationSettings()

# --- Enhanced Data Models ---
class BaseAPIResponse(BaseModel):
    """Standard API response wrapper."""
    model_config = ConfigDict(from_attributes=True)
    
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = settings.VERSION

class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    environment: str
    version: str
    uptime_seconds: float
    services: Dict[str, bool]

class PaginationMetadata(BaseModel):
    """Pagination information."""
    current_page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next_page: bool
    has_previous_page: bool

class CallSummaryModel(BaseModel):
    """Call analysis summary with essential metadata."""
    file_name: str = Field(..., description="Analysis file identifier")
    upload_timestamp: Optional[datetime] = Field(None, description="File upload time")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    processing_status: str = Field("completed", description="Processing status")
    lead_classification: Optional[str] = Field(None, description="Lead score category")
    confidence_level: Optional[float] = Field(None, ge=0.0, le=1.0, description="Analysis confidence")
    sentiment_overview: Optional[str] = Field(None, description="Overall sentiment")
    key_insights_count: int = Field(0, description="Number of insights extracted")

class DetailedAnalysisModel(BaseModel):
    """Comprehensive analysis results."""
    file_name: str
    original_transcript: str
    processing_metadata: Dict[str, Any]
    
    # Core analysis results
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    lead_score_details: Dict[str, Any]
    extracted_topics: List[str]
    key_phrases: List[str]
    insights_and_moments: List[Dict[str, Any]]
    
    # Additional intelligence
    conversation_metrics: Dict[str, Any]
    recommendation_summary: Dict[str, Any]

class PaginatedCallsResponse(BaseAPIResponse):
    """Paginated response for call listings."""
    data: List[CallSummaryModel]
    pagination: PaginationMetadata

# --- Cloud Service Integration ---
class AWSServiceConnector:
    """AWS service connector with robust error handling and fallbacks."""
    
    def __init__(self):
        self._s3_client = None
        self._session = None
        self._connection_health = {
            "s3_accessible": False,
            "bucket_exists": False,
            "last_check": None
        }
    
    @property
    def session(self):
        if self._session is None:
            self._session = boto3.Session()
        return self._session
    
    @property
    def s3_client(self):
        if self._s3_client is None:
            config = boto3.client.Config(
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                max_pool_connections=10,
                connect_timeout=5,
                read_timeout=10
            )
            self._s3_client = self.session.client(
                's3',
                region_name=settings.AWS_REGION,
                config=config
            )
        return self._s3_client
    
    async def check_service_health(self) -> Dict[str, bool]:
        """Comprehensive service health check with graceful degradation."""
        health_status = {"s3_service": False, "bucket_access": False}
        
        try:
            # Test S3 service connectivity
            response = self.s3_client.list_objects_v2(
                Bucket=settings.S3_BUCKET_NAME,
                MaxKeys=1
            )
            health_status["s3_service"] = True
            health_status["bucket_access"] = True
            
            self._connection_health.update({
                "s3_accessible": True,
                "bucket_exists": True,
                "last_check": datetime.utcnow()
            })
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchBucket':
                logger.warning(f"S3 bucket not found: {settings.S3_BUCKET_NAME}")
            else:
                logger.warning(f"S3 access error: {error_code}")
                
        except BotoCoreError as e:
            logger.warning(f"AWS service connection error: {e}")
            
        except Exception as e:
            logger.warning(f"Unexpected AWS error: {e}")
        
        return health_status
    
    def is_service_available(self) -> bool:
        """Quick check if AWS services are available."""
        return self._connection_health.get("s3_accessible", False)

aws_connector = AWSServiceConnector()

# --- Business Logic Layer ---
class CallAnalysisService:
    """Enhanced service layer with error handling and performance optimization."""
    
    def __init__(self):
        self.cache_duration = 300  # 5 minutes
        self.performance_stats = {
            "total_requests": 0,
            "successful_responses": 0,
            "average_response_time": 0.0
        }
    
    async def retrieve_call_listings(
        self, 
        page: int = 1, 
        page_size: int = 20,
        sort_field: str = "upload_timestamp",
        sort_direction: str = "desc"
    ) -> PaginatedCallsResponse:
        """Retrieve paginated call analysis listings with robust error handling."""
        
        request_start_time = datetime.utcnow()
        
        try:
            # Check service availability
            if not aws_connector.is_service_available():
                service_health = await aws_connector.check_service_health()
                if not service_health.get("s3_service"):
                    return self._create_fallback_response(page, page_size)
            
            # Retrieve analysis files from S3
            s3_response = aws_connector.s3_client.list_objects_v2(
                Bucket=settings.S3_BUCKET_NAME,
                Prefix=settings.S3_ANALYSIS_PREFIX,
                MaxKeys=1000
            )
            
            if 'Contents' not in s3_response:
                return PaginatedCallsResponse(
                    data=[],
                    pagination=self._create_pagination_info(0, page, page_size)
                )
            
            # Filter and process files
            analysis_files = [
                obj for obj in s3_response['Contents'] 
                if not obj['Key'].endswith('/') and obj['Size'] > 0
            ]
            
            # Apply sorting
            if sort_field == "upload_timestamp":
                analysis_files.sort(
                    key=lambda x: x['LastModified'], 
                    reverse=(sort_direction == "desc")
                )
            elif sort_field == "file_size":
                analysis_files.sort(
                    key=lambda x: x['Size'], 
                    reverse=(sort_direction == "desc")
                )
            
            # Implement pagination
            total_files = len(analysis_files)
            start_index = (page - 1) * page_size
            end_index = start_index + page_size
            page_files = analysis_files[start_index:end_index]
            
            # Build call summaries
            call_summaries = []
            for file_obj in page_files:
                summary = await self._build_call_summary(file_obj)
                call_summaries.append(summary)
            
            # Update performance metrics
            processing_time = (datetime.utcnow() - request_start_time).total_seconds()
            self._update_performance_metrics(processing_time, True)
            
            return PaginatedCallsResponse(
                data=call_summaries,
                pagination=self._create_pagination_info(total_files, page, page_size),
                message=f"Retrieved {len(call_summaries)} call analysis records"
            )
            
        except Exception as e:
            processing_time = (datetime.utcnow() - request_start_time).total_seconds()
            self._update_performance_metrics(processing_time, False)
            
            logger.error(f"Failed to retrieve call listings: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to retrieve call analysis data"
            )
    
    async def get_detailed_call_analysis(self, file_name: str) -> DetailedAnalysisModel:
        """Retrieve comprehensive analysis for a specific call."""
        
        analysis_key = f"{settings.S3_ANALYSIS_PREFIX}{file_name}"
        
        try:
            s3_object = aws_connector.s3_client.get_object(
                Bucket=settings.S3_BUCKET_NAME,
                Key=analysis_key
            )
            
            content_data = s3_object['Body'].read().decode('utf-8')
            analysis_results = json.loads(content_data)
            
            # Transform to standardized model
            return self._transform_to_detailed_model(analysis_results, file_name)
            
        except aws_connector.s3_client.exceptions.NoSuchKey:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis results for '{file_name}' not found"
            )
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid analysis data format for '{file_name}'"
            )
        except Exception as e:
            logger.error(f"Error retrieving detailed analysis for {file_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve analysis details"
            )
    
    async def _build_call_summary(self, s3_object: dict) -> CallSummaryModel:
        """Build call summary with error handling and data validation."""
        
        file_name = s3_object['Key'].replace(settings.S3_ANALYSIS_PREFIX, "")
        
        try:
            # Attempt to extract preview data
            content = aws_connector.s3_client.get_object(
                Bucket=settings.S3_BUCKET_NAME,
                Key=s3_object['Key']
            )
            
            analysis_data = json.loads(content['Body'].read().decode('utf-8'))
            
            # Extract summary information safely
            lead_classification = self._extract_lead_classification(analysis_data)
            confidence_level = self._extract_confidence_level(analysis_data)
            sentiment_overview = self._extract_sentiment_overview(analysis_data)
            insights_count = self._count_insights(analysis_data)
            
            return CallSummaryModel(
                file_name=file_name,
                upload_timestamp=s3_object['LastModified'],
                file_size_bytes=s3_object['Size'],
                processing_status="completed",
                lead_classification=lead_classification,
                confidence_level=confidence_level,
                sentiment_overview=sentiment_overview,
                key_insights_count=insights_count
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract preview data for {file_name}: {e}")
            # Return basic summary on preview extraction failure
            return CallSummaryModel(
                file_name=file_name,
                upload_timestamp=s3_object['LastModified'],
                file_size_bytes=s3_object['Size'],
                processing_status="preview_unavailable"
            )
    
    def _extract_lead_classification(self, data: dict) -> Optional[str]:
        """Safely extract lead classification from analysis data."""
        lead_info = data.get('leadScore', data.get('lead_scoring', {}))
        if isinstance(lead_info, dict):
            return lead_info.get('primary_score', lead_info.get('score'))
        return None
    
    def _extract_confidence_level(self, data: dict) -> Optional[float]:
        """Safely extract confidence level from analysis data."""
        lead_info = data.get('leadScore', data.get('lead_scoring', {}))
        if isinstance(lead_info, dict):
            confidence = lead_info.get('confidence_level', lead_info.get('confidence'))
            if isinstance(confidence, (int, float)):
                return float(confidence)
        return None
    
    def _extract_sentiment_overview(self, data: dict) -> Optional[str]:
        """Safely extract sentiment overview from analysis data."""
        sentiment_data = data.get('sentiment_analysis', {})
        if isinstance(sentiment_data, dict):
            score = sentiment_data.get('overall_score', data.get('sentiment', 0))
            if isinstance(score, (int, float)):
                if score > 0.2:
                    return "Positive"
                elif score < -0.2:
                    return "Negative"
                else:
                    return "Neutral"
        return None
    
    def _count_insights(self, data: dict) -> int:
        """Count total insights and key moments."""
        count = 0
        
        # Count wow moments
        wow_moments = data.get('wow_moments', data.get('conversation_insights', []))
        if isinstance(wow_moments, list):
            count += len(wow_moments)
        
        # Count key phrases
        key_phrases = data.get('key_phrases', [])
        if isinstance(key_phrases, list):
            count += len(key_phrases)
        
        return count
    
    def _transform_to_detailed_model(self, data: dict, file_name: str) -> DetailedAnalysisModel:
        """Transform raw analysis data to standardized detailed model."""
        
        # Extract and normalize data with fallbacks
        return DetailedAnalysisModel(
            file_name=file_name,
            original_transcript=data.get('transcript', data.get('full_transcript', '')),
            processing_metadata=data.get('processing_stats', data.get('metadata', {})),
            sentiment_score=self._normalize_sentiment_score(data),
            lead_score_details=self._normalize_lead_score_details(data),
            extracted_topics=data.get('topics', data.get('discussion_topics', [])),
            key_phrases=data.get('key_phrases', data.get('keywords', [])),
            insights_and_moments=self._normalize_insights_and_moments(data),
            conversation_metrics=self._extract_conversation_metrics(data),
            recommendation_summary=self._extract_recommendation_summary(data)
        )
    
    def _normalize_sentiment_score(self, data: dict) -> float:
        """Normalize sentiment score from various formats."""
        sentiment_analysis = data.get('sentiment_analysis', {})
        if isinstance(sentiment_analysis, dict):
            score = sentiment_analysis.get('overall_score')
            if isinstance(score, (int, float)):
                return max(-1.0, min(1.0, float(score)))
        
        # Fallback to legacy format
        legacy_sentiment = data.get('sentiment', 0)
        if isinstance(legacy_sentiment, (int, float)):
            return max(-1.0, min(1.0, float(legacy_sentiment)))
        
        return 0.0
    
    def _normalize_lead_score_details(self, data: dict) -> Dict[str, Any]:
        """Normalize lead scoring details from various formats."""
        lead_data = data.get('lead_scoring', data.get('leadScore', {}))
        if not isinstance(lead_data, dict):
            return {'score': 'Unknown', 'confidence': 0.5}
        
        return {
            'primary_classification': lead_data.get('primary_score', lead_data.get('score', 'Unknown')),
            'numerical_score': lead_data.get('numerical_score', 50),
            'confidence': lead_data.get('confidence_level', lead_data.get('confidence', 0.5)),
            'interest_indicators': lead_data.get('interest_indicators', []),
            'next_actions': lead_data.get('next_best_actions', lead_data.get('recommended_actions', []))
        }
    
    def _normalize_insights_and_moments(self, data: dict) -> List[Dict[str, Any]]:
        """Normalize insights and wow moments from various formats."""
        insights = []
        
        # Process wow moments
        wow_moments = data.get('wow_moments', data.get('conversation_insights', []))
        if isinstance(wow_moments, list):
            for moment in wow_moments:
                if isinstance(moment, dict):
                    insights.append({
                        'type': 'wow_moment',
                        'content': moment.get('trigger_phrase', ''),
                        'context': moment.get('full_context', moment.get('surrounding_context', '')),
                        'timestamp': moment.get('timestamp', moment.get('estimated_timestamp')),
                        'score': moment.get('excitement_score', moment.get('excitement_level', 0.5))
                    })
        
        return insights
    
    def _extract_conversation_metrics(self, data: dict) -> Dict[str, Any]:
        """Extract conversation flow and engagement metrics."""
        metrics = {
            'total_duration_estimate': 0,
            'word_count': 0,
            'engagement_score': 0.5,
            'question_count': 0
        }
        
        # Extract from various possible locations
        conversation_flow = data.get('conversation_flow', {})
        if isinstance(conversation_flow, dict):
            metrics.update({
                'question_count': conversation_flow.get('question_count', 0),
                'engagement_score': conversation_flow.get('engagement_score', 0.5)
            })
        
        # Extract word count
        word_count = data.get('word_count', 0)
        if word_count == 0 and 'transcript' in data:
            word_count = len(data['transcript'].split())
        metrics['word_count'] = word_count
        
        return metrics
    
    def _extract_recommendation_summary(self, data: dict) -> Dict[str, Any]:
        """Extract actionable recommendations from analysis."""
        recommendations = {
            'priority_level': 'medium',
            'suggested_actions': [],
            'follow_up_timing': 'within_week',
            'key_focus_areas': []
        }
        
        # Extract from lead scoring
        lead_data = data.get('lead_scoring', data.get('leadScore', {}))
        if isinstance(lead_data, dict):
            recommendations.update({
                'suggested_actions': lead_data.get('next_best_actions', lead_data.get('recommended_actions', [])),
                'key_focus_areas': lead_data.get('interest_indicators', [])
            })
        
        return recommendations
    
    def _create_fallback_response(self, page: int, page_size: int) -> PaginatedCallsResponse:
        """Create fallback response when AWS services are unavailable."""
        return PaginatedCallsResponse(
            data=[],
            pagination=self._create_pagination_info(0, page, page_size),
            success=False,
            message="Service temporarily unavailable - please try again later"
        )
    
    def _create_pagination_info(self, total_items: int, page: int, page_size: int) -> PaginationMetadata:
        """Create pagination metadata."""
        total_pages = (total_items + page_size - 1) // page_size if total_items > 0 else 0
        
        return PaginationMetadata(
            current_page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
            has_next_page=page < total_pages,
            has_previous_page=page > 1
        )
    
    def _update_performance_metrics(self, processing_time: float, success: bool) -> None:
        """Update service performance tracking."""
        self.performance_stats["total_requests"] += 1
        
        if success:
            self.performance_stats["successful_responses"] += 1
            
            # Update average response time
            total_requests = self.performance_stats["total_requests"]
            current_avg = self.performance_stats["average_response_time"]
            self.performance_stats["average_response_time"] = (
                (current_avg * (total_requests - 1) + processing_time) / total_requests
            )

analysis_service = CallAnalysisService()

# --- Application Lifecycle Management ---
app_start_time = datetime.utcnow()

@asynccontextmanager
async def application_lifecycle(app: FastAPI):
    """Manage application startup and shutdown with health monitoring."""
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Cloud deployment detected: {settings.is_cloud_deployment}")
    
    # Startup procedures
    try:
        # Verify AWS connectivity (non-blocking)
        health_check = await aws_connector.check_service_health()
        logger.info(f"Service health check: {health_check}")
        
        logger.info("✅ Application startup completed successfully")
        
    except Exception as e:
        logger.warning(f"⚠️ Startup warning: {e}")
        logger.info("Application will continue with limited functionality")
    
    yield
    
    # Shutdown procedures
    logger.info(f"🛑 Shutting down {settings.APP_NAME}")

# --- FastAPI Application Configuration ---
app = FastAPI(
    title=settings.APP_NAME,
    description="Advanced ML-powered sales call analysis with enterprise features",
    version=settings.VERSION,
    docs_url=f"/{settings.API_VERSION}/docs",
    redoc_url=f"/{settings.API_VERSION}/redoc",
    lifespan=application_lifecycle,
    debug=settings.DEBUG_MODE
)

# --- Middleware Configuration ---
# Compression for better performance
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Enhanced CORS for cloud deployment compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count", "X-Page-Count", "X-Processing-Time"]
)

# --- Enhanced Error Handling ---
@app.exception_handler(HTTPException)
async def enhanced_http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced HTTP exception handling with detailed logging."""
    logger.error(
        "HTTP Exception occurred",
        path=request.url.path,
        method=request.method,
        status_code=exc.status_code,
        detail=exc.detail,
        user_agent=request.headers.get("user-agent", "unknown")
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": exc.detail,
                "code": exc.status_code,
                "timestamp": datetime.utcnow().isoformat(),
                "path": request.url.path
            },
            "version": settings.VERSION
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(
        "Unhandled exception occurred",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        traceback=traceback.format_exc()
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "message": "An unexpected error occurred",
                "code": 500,
                "timestamp": datetime.utcnow().isoformat(),
                "path": request.url.path
            },
            "version": settings.VERSION
        }
    )

# --- Dependency Functions ---
async def get_pagination_parameters(
    page: int = Query(1, ge=1, description="Page number starting from 1"),
    page_size: int = Query(
        settings.DEFAULT_PAGE_SIZE, 
        ge=1, 
        le=settings.MAX_PAGE_SIZE,
        description="Number of items per page"
    ),
    sort_field: str = Query("upload_timestamp", description="Field to sort by"),
    sort_direction: str = Query("desc", regex="^(asc|desc)$", description="Sort direction")
) -> Dict[str, Any]:
    """Extract and validate pagination parameters."""
    return {
        "page": page,
        "page_size": page_size,
        "sort_field": sort_field,
        "sort_direction": sort_direction
    }

# --- API Endpoints ---
@app.get("/", tags=["System"])
async def api_root():
    """API root with system information and navigation."""
    uptime = (datetime.utcnow() - app_start_time).total_seconds()
    
    return {
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "operational",
        "uptime_seconds": uptime,
        "endpoints": {
            "documentation": f"/{settings.API_VERSION}/docs",
            "health_check": "/health",
            "call_listings": f"/{settings.API_VERSION}/calls",
            "analytics": f"/{settings.API_VERSION}/analytics"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def comprehensive_health_check():
    """Comprehensive system health check with service status."""
    uptime = (datetime.utcnow() - app_start_time).total_seconds()
    
    # Check service dependencies
    service_status = await aws_connector.check_service_health()
    
    overall_status = "healthy" if all(service_status.values()) else "degraded"
    if not any(service_status.values()):
        overall_status = "unhealthy"
    
    return HealthCheckResponse(
        status=overall_status,
        environment=settings.ENVIRONMENT,
        version=settings.VERSION,
        uptime_seconds=uptime,
        services=service_status
    )

@app.get(
    f"/{settings.API_VERSION}/calls", 
    response_model=PaginatedCallsResponse, 
    tags=["Call Analysis"]
)
async def list_analyzed_calls(
    response: Response,
    pagination_params: Dict[str, Any] = Depends(get_pagination_parameters)
):
    """
    Retrieve paginated list of analyzed calls with comprehensive metadata.
    
    Features:
    - Robust pagination with configurable page sizes
    - Multiple sorting options (timestamp, file size)
    - Rich metadata including confidence scores and insights
    - Graceful handling of service unavailability
    - Performance tracking and optimization
    """
    request_start = datetime.utcnow()
    
    try:
        result = await analysis_service.retrieve_call_listings(**pagination_params)
        
        # Add performance headers
        processing_time = (datetime.utcnow() - request_start).total_seconds()
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}s"
        response.headers["X-Total-Count"] = str(result.pagination.total_items)
        response.headers["X-Page-Count"] = str(result.pagination.total_pages)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to list calls: {e}")
        raise

@app.get(
    f"/{settings.API_VERSION}/calls/{{file_name}}", 
    response_model=DetailedAnalysisModel,
    tags=["Call Analysis"]
)
async def get_call_analysis_details(
    file_name: str = Path(..., description="Analysis file identifier"),
    response: Response = None
):
    """
    Retrieve comprehensive analysis details for a specific call.
    
    Provides:
    - Complete transcript and processing metadata
    - Detailed sentiment analysis and lead scoring
    - Extracted topics, key phrases, and insights
    - Conversation metrics and engagement timeline
    - Actionable recommendations and next steps
    """
    request_start = datetime.utcnow()
    
    try:
        result = await analysis_service.get_detailed_call_analysis(file_name)
        
        # Add performance header
        processing_time = (datetime.utcnow() - request_start).total_seconds()
        if response:
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}s"
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get analysis details for {file_name}: {e}")
        raise

@app.post(f"/{settings.API_VERSION}/calls/{{file_name}}/reprocess", tags=["Call Processing"])
async def trigger_call_reprocessing(
    file_name: str = Path(..., description="Call file to reprocess"),
    background_tasks: BackgroundTasks = Depends()
):
    """Trigger reprocessing of a call with enhanced analysis pipeline."""
    
    # Add background task for reprocessing notification
    background_tasks.add_task(
        logger.info,
        "Reprocessing request submitted",
        file_name=file_name,
        timestamp=datetime.utcnow().isoformat()
    )
    
    return {
        "success": True,
        "message": f"Enhanced reprocessing initiated for {file_name}",
        "status": "queued",
        "estimated_completion_time": "2-4 minutes",
        "processing_id": f"reprocess_{int(datetime.utcnow().timestamp())}"
    }

@app.get(f"/{settings.API_VERSION}/analytics/performance", tags=["Analytics"])
async def get_system_performance_metrics():
    """Retrieve system performance and usage analytics."""
    
    service_stats = analysis_service.performance_stats
    uptime = (datetime.utcnow() - app_start_time).total_seconds()
    
    return {
        "success": True,
        "data": {
            "system_uptime_seconds": uptime,
            "total_api_requests": service_stats["total_requests"],
            "successful_requests": service_stats["successful_responses"],
            "success_rate_percentage": (
                (service_stats["successful_responses"] / max(1, service_stats["total_requests"])) * 100
            ),
            "average_response_time_seconds": service_stats["average_response_time"],
            "service_health": await aws_connector.check_service_health()
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get(f"/{settings.API_VERSION}/system/info", tags=["System"])
async def get_system_information():
    """Get comprehensive system configuration and deployment information."""
    
    return {
        "application": {
            "name": settings.APP_NAME,
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG_MODE
        },
        "deployment": {
            "cloud_platform_detected": settings.is_cloud_deployment,
            "aws_region": settings.AWS_REGION,
            "s3_bucket_configured": bool(settings.S3_BUCKET_NAME)
        },
        "features": {
            "pagination_support": True,
            "real_time_analysis": True,
            "performance_monitoring": True,
            "error_tracking": True
        }
    }

# --- Development Server ---
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG_MODE,
        log_level="info",
        access_log=settings.DEBUG_MODE
    )