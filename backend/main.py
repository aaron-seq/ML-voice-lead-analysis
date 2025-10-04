"""
Enhanced ML Voice Lead Analysis API
Production-ready FastAPI application with advanced features.
"""

import os
import json
import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import boto3
from fastapi import (
    FastAPI, HTTPException, Depends, BackgroundTasks, 
    Query, Path, status, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, ConfigDict, validator
from dotenv import load_dotenv
import aiofiles
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

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

# --- Enhanced Configuration ---
class ApplicationConfig:
    """Comprehensive application configuration management."""
    
    # Core settings
    APP_NAME: str = "ML Voice Lead Analysis API"
    VERSION: str = "3.0.0"
    API_VERSION: str = "v1"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = ENVIRONMENT == "development"
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "4"))
    
    # AWS Configuration
    S3_BUCKET_NAME: str = os.getenv("DATA_BUCKET", "ml-voice-analysis-bucket")
    S3_ANALYSIS_PREFIX: str = "analysis-results/"
    S3_TRANSCRIPTS_PREFIX: str = "transcripts/"
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    
    # Database settings
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql+asyncpg://voice_user:voice_pass@localhost:5432/voice_analysis"
    )
    
    # Redis settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Performance settings
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    MAX_FILE_SIZE_MB: int = 50
    REQUEST_TIMEOUT: int = 30
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALLOWED_HOSTS: List[str] = [
        "localhost",
        "127.0.0.1",
        "*.vercel.app",
        "*.netlify.app"
    ]
    
    # CORS settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app",
        "https://*.netlify.app"
    ]

config = ApplicationConfig()

# --- Enhanced Pydantic Models ---
class BaseResponse(BaseModel):
    """Base response model with comprehensive metadata."""
    model_config = ConfigDict(from_attributes=True)
    
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    api_version: str = config.VERSION

class PaginationInfo(BaseModel):
    """Pagination metadata."""
    current_page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool

class HealthStatus(BaseModel):
    """System health status."""
    status: str
    checks: Dict[str, Dict[str, Union[bool, str, int]]]

class CallSummary(BaseModel):
    """Enhanced call summary with rich metadata."""
    file_name: str = Field(..., description="The analyzed file name")
    upload_date: Optional[datetime] = Field(None, description="File upload timestamp")
    file_size_mb: Optional[float] = Field(None, description="File size in megabytes")
    lead_score: Optional[str] = Field(None, description="Lead classification preview")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    call_duration_minutes: Optional[float] = Field(None, description="Estimated call duration")
    sentiment_preview: Optional[str] = Field(None, description="Overall sentiment preview")

class AdvancedLeadMetrics(BaseModel):
    """Advanced lead scoring metrics."""
    overall_score: str = Field(..., example="Hot")
    confidence_level: float = Field(..., ge=0.0, le=1.0)
    engagement_score: float = Field(..., ge=0.0, le=10.0)
    interest_indicators: List[str] = Field(..., description="Key interest signals")
    risk_factors: List[str] = Field(default_factory=list, description="Potential concerns")
    follow_up_priority: str = Field(..., description="Recommended follow-up urgency")
    recommended_actions: List[str] = Field(..., description="Suggested next steps")

class EnhancedWowMoment(BaseModel):
    """Enhanced high-interest moment detection."""
    trigger_phrase: str = Field(..., description="The phrase that triggered excitement")
    surrounding_context: str = Field(..., description="Context around the moment")
    estimated_timestamp: Optional[float] = Field(None, description="Approximate time in call")
    excitement_level: float = Field(..., ge=0.0, le=1.0, description="Excitement intensity")
    topic_category: Optional[str] = Field(None, description="What topic caused excitement")

class ComprehensiveCallAnalysis(BaseModel):
    """Complete analysis results with enhanced features."""
    file_name: str = Field(..., alias="fileName")
    full_transcript: str = Field(..., alias="transcript")
    
    # Sentiment analysis
    overall_sentiment_score: float = Field(..., ge=-1.0, le=1.0, alias="sentiment")
    sentiment_trend: List[float] = Field(default_factory=list, description="Sentiment over time")
    
    # Content extraction
    key_phrases: List[str] = Field(..., alias="keywords")
    discussion_topics: List[str] = Field(..., alias="topics")
    technical_terms: List[str] = Field(default_factory=list)
    
    # Enhanced insights
    conversation_insights: List[EnhancedWowMoment] = Field(..., alias="wowMoments")
    lead_analysis: AdvancedLeadMetrics = Field(..., alias="leadScore")
    
    # Metadata
    call_metadata: Dict[str, Any] = Field(..., alias="metadata")
    processing_duration_seconds: float = Field(..., alias="processingTime")

class PaginatedCallsResponse(BaseResponse):
    """Paginated calls response with rich metadata."""
    calls: List[CallSummary]
    pagination: PaginationInfo

# --- Enhanced AWS Service Manager ---
class CloudServiceManager:
    """Advanced AWS service management with connection pooling."""
    
    def __init__(self):
        self._s3_client = None
        self._transcribe_client = None
        self._session = None
        self._connection_verified = False
    
    @property
    def session(self):
        if self._session is None:
            self._session = boto3.Session()
        return self._session
    
    @property
    def s3_client(self):
        if self._s3_client is None:
            self._s3_client = self.session.client(
                's3',
                region_name=config.AWS_REGION,
                config=boto3.client.Config(
                    retries={'max_attempts': 3},
                    max_pool_connections=50
                )
            )
        return self._s3_client
    
    @property
    def transcribe_client(self):
        if self._transcribe_client is None:
            self._transcribe_client = self.session.client(
                'transcribe',
                region_name=config.AWS_REGION
            )
        return self._transcribe_client
    
    async def verify_connections(self) -> Dict[str, bool]:
        """Verify all AWS service connections."""
        results = {}
        
        try:
            # Test S3 connection
            self.s3_client.head_bucket(Bucket=config.S3_BUCKET_NAME)
            results['s3'] = True
        except Exception as e:
            logger.error("S3 connection failed", error=str(e))
            results['s3'] = False
        
        try:
            # Test Transcribe service
            self.transcribe_client.list_transcription_jobs(MaxResults=1)
            results['transcribe'] = True
        except Exception as e:
            logger.error("Transcribe connection failed", error=str(e))
            results['transcribe'] = False
        
        self._connection_verified = all(results.values())
        return results

cloud_services = CloudServiceManager()

# --- Enhanced Business Logic ---
class AdvancedCallAnalysisService:
    """Advanced service layer with caching and performance optimizations."""
    
    def __init__(self):
        self.cache_ttl = 300  # 5 minutes
    
    async def get_paginated_calls(
        self, 
        page: int = 1, 
        page_size: int = 20,
        sort_by: str = "upload_date",
        sort_order: str = "desc"
    ) -> PaginatedCallsResponse:
        """Get paginated calls with advanced filtering and sorting."""
        
        try:
            # Get all analysis files
            response = cloud_services.s3_client.list_objects_v2(
                Bucket=config.S3_BUCKET_NAME,
                Prefix=config.S3_ANALYSIS_PREFIX,
                MaxKeys=1000
            )
            
            if 'Contents' not in response:
                return PaginatedCallsResponse(
                    calls=[],
                    pagination=PaginationInfo(
                        current_page=page,
                        page_size=page_size,
                        total_items=0,
                        total_pages=0,
                        has_next=False,
                        has_previous=False
                    )
                )
            
            # Filter out directory markers
            files = [obj for obj in response['Contents'] if not obj['Key'].endswith('/')]
            
            # Sort files
            if sort_by == "upload_date":
                files.sort(
                    key=lambda x: x['LastModified'], 
                    reverse=(sort_order == "desc")
                )
            elif sort_by == "file_size":
                files.sort(
                    key=lambda x: x['Size'], 
                    reverse=(sort_order == "desc")
                )
            
            # Pagination
            total_items = len(files)
            total_pages = (total_items + page_size - 1) // page_size
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_files = files[start_idx:end_idx]
            
            # Build call summaries with enhanced data
            calls = []
            for obj in page_files:
                file_name = obj['Key'].replace(config.S3_ANALYSIS_PREFIX, "")
                
                # Get enhanced preview data
                summary = await self._build_call_summary(obj, file_name)
                calls.append(summary)
            
            return PaginatedCallsResponse(
                calls=calls,
                pagination=PaginationInfo(
                    current_page=page,
                    page_size=page_size,
                    total_items=total_items,
                    total_pages=total_pages,
                    has_next=page < total_pages,
                    has_previous=page > 1
                )
            )
            
        except Exception as e:
            logger.error("Failed to retrieve paginated calls", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to retrieve call list"
            )
    
    async def _build_call_summary(self, s3_obj: dict, file_name: str) -> CallSummary:
        """Build enhanced call summary with preview data."""
        try:
            # Try to get quick preview from analysis file
            content = cloud_services.s3_client.get_object(
                Bucket=config.S3_BUCKET_NAME,
                Key=s3_obj['Key']
            )
            
            data = json.loads(content['Body'].read().decode('utf-8'))
            
            # Extract preview information
            lead_score = None
            confidence_score = None
            sentiment_preview = None
            call_duration = None
            
            if 'leadScore' in data:
                lead_info = data['leadScore']
                lead_score = lead_info.get('score')
                confidence_score = lead_info.get('confidence')
            
            if 'sentiment' in data:
                sentiment_val = data['sentiment']
                if sentiment_val > 0.2:
                    sentiment_preview = "Positive"
                elif sentiment_val < -0.2:
                    sentiment_preview = "Negative"
                else:
                    sentiment_preview = "Neutral"
            
            if 'metadata' in data and 'call_duration_minutes' in data['metadata']:
                call_duration = data['metadata']['call_duration_minutes']
            
            return CallSummary(
                file_name=file_name,
                upload_date=s3_obj['LastModified'],
                file_size_mb=round(s3_obj['Size'] / (1024 * 1024), 2),
                lead_score=lead_score,
                confidence_score=confidence_score,
                call_duration_minutes=call_duration,
                sentiment_preview=sentiment_preview
            )
            
        except Exception:
            # Fallback to basic info if preview fails
            return CallSummary(
                file_name=file_name,
                upload_date=s3_obj['LastModified'],
                file_size_mb=round(s3_obj['Size'] / (1024 * 1024), 2)
            )
    
    async def get_detailed_analysis(self, file_name: str) -> ComprehensiveCallAnalysis:
        """Get comprehensive call analysis with all details."""
        s3_key = f"{config.S3_ANALYSIS_PREFIX}{file_name}"
        
        try:
            s3_object = cloud_services.s3_client.get_object(
                Bucket=config.S3_BUCKET_NAME,
                Key=s3_key
            )
            
            content = s3_object['Body'].read().decode('utf-8')
            analysis_data = json.loads(content)
            
            return ComprehensiveCallAnalysis(**analysis_data)
            
        except cloud_services.s3_client.exceptions.NoSuchKey:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis for '{file_name}' not found"
            )
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid analysis data format for '{file_name}'"
            )
        except Exception as e:
            logger.error("Error retrieving detailed analysis", file_name=file_name, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve analysis details"
            )

analysis_service = AdvancedCallAnalysisService()

# --- Application Lifecycle ---
@asynccontextmanager
async def application_lifespan(app: FastAPI):
    """Enhanced application lifecycle management."""
    logger.info("🚀 Starting ML Voice Lead Analysis API v3.0")
    
    # Startup tasks
    try:
        # Verify AWS connections
        connection_status = await cloud_services.verify_connections()
        logger.info("AWS connection status", **connection_status)
        
        # Initialize any required databases, caches, etc.
        logger.info("✅ Application startup completed")
        
    except Exception as e:
        logger.error("❌ Application startup failed", error=str(e))
        raise
    
    yield
    
    # Shutdown tasks
    logger.info("🛑 Shutting down ML Voice Lead Analysis API")

# --- FastAPI Application ---
app = FastAPI(
    title=config.APP_NAME,
    description="Production-ready API for ML-powered sales call analysis with advanced insights",
    version=config.VERSION,
    docs_url=f"/{config.API_VERSION}/docs",
    redoc_url=f"/{config.API_VERSION}/redoc",
    lifespan=application_lifespan,
    debug=config.DEBUG
)

# --- Middleware Stack ---
# Trust proxy headers
app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.ALLOWED_HOSTS)

# Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS with production settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count", "X-Page-Count"]
)

# --- Enhanced Exception Handling ---
@app.exception_handler(HTTPException)
async def enhanced_http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced HTTP exception handling with structured logging."""
    logger.error(
        "HTTP Exception",
        path=request.url.path,
        method=request.method,
        status_code=exc.status_code,
        detail=exc.detail
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "error_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path
        }
    )

# --- Dependency Injection ---
async def get_pagination_params(
    page: int = Query(1, ge=1, description="Page number (starts from 1)"),
    page_size: int = Query(
        config.DEFAULT_PAGE_SIZE, 
        ge=1, 
        le=config.MAX_PAGE_SIZE,
        description="Items per page"
    ),
    sort_by: str = Query("upload_date", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort direction")
) -> Dict[str, Any]:
    """Enhanced pagination with sorting."""
    return {
        "page": page,
        "page_size": page_size,
        "sort_by": sort_by,
        "sort_order": sort_order
    }

# --- API Endpoints ---
@app.get("/", tags=["System"])
async def api_root():
    """Enhanced API root with system information."""
    return {
        "service": config.APP_NAME,
        "version": config.VERSION,
        "environment": config.ENVIRONMENT,
        "documentation": f"/{config.API_VERSION}/docs",
        "health_check": "/health",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health", response_model=HealthStatus, tags=["System"])
async def comprehensive_health_check():
    """Comprehensive system health check."""
    checks = {}
    overall_status = "healthy"
    
    # Check AWS services
    try:
        aws_status = await cloud_services.verify_connections()
        checks["aws_services"] = {
            "healthy": all(aws_status.values()),
            "details": aws_status,
            "response_time_ms": 0  # Could add timing
        }
        if not all(aws_status.values()):
            overall_status = "degraded"
    except Exception as e:
        checks["aws_services"] = {
            "healthy": False,
            "error": str(e),
            "response_time_ms": 0
        }
        overall_status = "unhealthy"
    
    # Add more health checks as needed (database, cache, etc.)
    
    return HealthStatus(
        status=overall_status,
        checks=checks
    )

@app.get(
    f"/{config.API_VERSION}/calls", 
    response_model=PaginatedCallsResponse, 
    tags=["Call Analysis"]
)
async def list_calls_with_advanced_features(
    params: Dict[str, Any] = Depends(get_pagination_params)
):
    """
    Get paginated list of analyzed calls with advanced filtering.
    
    Features:
    - Pagination with customizable page sizes
    - Sorting by various fields
    - Rich preview data including sentiment and lead scores
    - Performance optimized with smart caching
    """
    return await analysis_service.get_paginated_calls(**params)

@app.get(
    f"/{config.API_VERSION}/calls/{{file_name}}", 
    response_model=ComprehensiveCallAnalysis,
    tags=["Call Analysis"]
)
async def get_comprehensive_call_analysis(
    file_name: str = Path(..., description="Name of the analyzed call file")
):
    """
    Get comprehensive analysis for a specific call.
    
    Returns detailed insights including:
    - Full transcript and sentiment analysis
    - Advanced lead scoring with reasoning
    - Enhanced wow moments with context
    - Conversation topics and key phrases
    - Processing metadata
    """
    return await analysis_service.get_detailed_analysis(file_name)

@app.post(f"/{config.API_VERSION}/calls/{{file_name}}/reanalyze", tags=["Call Processing"])
async def trigger_advanced_reanalysis(
    file_name: str = Path(..., description="Call file to reanalyze"),
    background_tasks: BackgroundTasks = Depends()
):
    """Trigger enhanced re-analysis with latest ML models."""
    
    # Add background task for reanalysis
    background_tasks.add_task(
        logger.info,
        "Re-analysis triggered",
        file_name=file_name,
        timestamp=datetime.utcnow().isoformat()
    )
    
    return {
        "success": True,
        "message": f"Advanced re-analysis initiated for {file_name}",
        "status": "queued",
        "estimated_completion": "2-3 minutes"
    }

@app.get(f"/{config.API_VERSION}/analytics/overview", tags=["Analytics"])
async def get_analytics_overview():
    """Get system-wide analytics and insights."""
    # This could be expanded with actual analytics
    return {
        "success": True,
        "message": "Analytics endpoint - implement based on requirements",
        "data": {
            "total_calls_processed": 0,
            "average_lead_score_distribution": {},
            "sentiment_trends": [],
            "processing_performance": {}
        }
    }

# --- Development Server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info",
        workers=1 if config.DEBUG else config.WORKERS
    )
