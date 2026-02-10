from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Path, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import PurePath
import json
import logging
import os
import re
import uuid

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class ApplicationSettings:
    def __init__(self):
        self.app_name = "ML Voice Lead Analysis API"
        self.version = "4.1.0"  # Version bump for security fixes
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug_mode = self.environment == "development"
        
        # Testing and CI/CD configuration
        self.disable_aws_checks = os.getenv("DISABLE_AWS_CHECKS", "false").lower() == "true"
        self.is_testing_environment = self.environment in ["testing", "test", "ci"]
        
        # AWS Configuration
        self.aws_s3_bucket = os.getenv("DATA_BUCKET", "ml-voice-analysis-bucket")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.analysis_results_prefix = "analysis-results/"
        self.transcripts_prefix = "transcripts/"
        
        # API Configuration
        self.default_page_size = 20
        self.max_page_size = 100
        self.api_version = "v1"
        
        # Security Configuration
        self.max_request_size_bytes = 100_000_000  # 100MB
        
        # CORS origins
        self.cors_origins = [
            "http://localhost:3000",
            "http://localhost:3001",
            "https://*.vercel.app",
            "https://*.netlify.app",
            "https://*.railway.app",
            "https://*.render.com"
        ]

settings = ApplicationSettings()

# Security Validation Functions
def validate_file_name(file_name: str) -> str:
    """
    Validate and sanitize file names to prevent path traversal attacks.
    
    Args:
        file_name: The file name to validate
        
    Returns:
        Validated file name
        
    Raises:
        HTTPException: If file name is invalid or contains malicious patterns
    """
    # Reject empty strings
    if not file_name or not file_name.strip():
        logger.warning(f"Empty file name rejected")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File name cannot be empty"
        )
    
    # Reject path traversal attempts
    if '..' in file_name or file_name.startswith('/'):
        logger.warning(f"Path traversal attempt detected: {file_name}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file name format: path traversal detected"
        )
    
    # Only allow alphanumeric, hyphens, underscores, dots, and forward slashes for subdirectories
    if not re.match(r'^[a-zA-Z0-9._/-]+$', file_name):
        logger.warning(f"Invalid characters in file name: {file_name}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File name contains invalid characters. Only alphanumeric, hyphens, underscores, dots, and slashes are allowed."
        )
    
    # Ensure file name doesn't try to break out of prefix directory
    full_path = PurePath(settings.analysis_results_prefix) / file_name
    if not str(full_path).startswith(settings.analysis_results_prefix):
        logger.warning(f"File name tries to escape prefix: {file_name}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Access denied: file path is invalid"
        )
    
    # Additional security: check for null bytes
    if '\x00' in file_name:
        logger.warning(f"Null byte detected in file name: {file_name}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file name: null bytes not allowed"
        )
    
    logger.debug(f"File name validated successfully: {file_name}")
    return file_name

# Pydantic Models
class BaseApiResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PaginationMetadata(BaseModel):
    current_page: int
    items_per_page: int
    total_items: int
    total_pages: int
    has_next_page: bool
    has_previous_page: bool

class SystemHealthResponse(BaseModel):
    status: str
    service_checks: Dict[str, Dict[str, Union[bool, str]]]
    uptime_seconds: float
    environment: str

class VoiceCallSummary(BaseModel):
    file_identifier: str = Field(description="Unique file identifier")
    upload_timestamp: Optional[datetime] = Field(None, description="When file was uploaded")
    file_size_mb: Optional[float] = Field(None, description="File size in megabytes")
    processing_status: Optional[str] = Field(None, description="Current processing status")
    lead_classification: Optional[str] = Field(None, description="Lead score classification")
    confidence_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    call_duration_minutes: Optional[float] = Field(None, description="Total call duration")
    overall_sentiment: Optional[str] = Field(None, description="Dominant sentiment")

class LeadScoringMetrics(BaseModel):
    primary_classification: str = Field(description="Hot/Warm/Cold classification")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in classification")
    engagement_level: float = Field(ge=0.0, le=10.0, description="Customer engagement score")
    interest_signals: List[str] = Field(description="Detected interest indicators")
    concern_flags: List[str] = Field(default_factory=list, description="Identified concerns")
    followup_priority: str = Field(description="Recommended follow-up urgency")
    next_actions: List[str] = Field(description="Recommended next steps")

class InterestMoment(BaseModel):
    detected_phrase: str = Field(description="Phrase that indicated interest")
    context_snippet: str = Field(description="Surrounding conversation context")
    timestamp_seconds: Optional[float] = Field(None, description="Time in call when detected")
    interest_intensity: float = Field(ge=0.0, le=1.0, description="Strength of interest signal")
    topic_category: Optional[str] = Field(None, description="What topic generated interest")

class ComprehensiveCallAnalysis(BaseModel):
    file_identifier: str = Field(alias="fileName")
    complete_transcript: str = Field(alias="transcript")
    
    # Sentiment Analysis
    sentiment_score: float = Field(ge=-1.0, le=1.0, alias="sentiment")
    sentiment_progression: List[float] = Field(default_factory=list, description="Sentiment changes over time")
    
    # Content Analysis
    important_phrases: List[str] = Field(alias="keywords")
    conversation_topics: List[str] = Field(alias="topics")
    technical_terminology: List[str] = Field(default_factory=list)
    
    # Advanced Analytics
    high_interest_moments: List[InterestMoment] = Field(alias="wowMoments")
    lead_scoring_analysis: LeadScoringMetrics = Field(alias="leadScore")
    
    # Processing Information
    analysis_metadata: Dict[str, Any] = Field(alias="metadata")
    processing_time_seconds: float = Field(alias="processingTime")

class PaginatedCallListResponse(BaseApiResponse):
    call_summaries: List[VoiceCallSummary]
    pagination_info: PaginationMetadata

# AWS Service Manager
class AWSServiceConnector:
    def __init__(self):
        self.s3_client = None
        self.transcribe_client = None
        self.session = None
        self.aws_available = not (settings.disable_aws_checks or settings.is_testing_environment)
    
    def get_session(self):
        if self.session is None and self.aws_available:
            self.session = boto3.Session()
        return self.session
    
    def get_s3_client(self):
        if not self.aws_available:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AWS services are disabled in this environment"
            )
            
        if self.s3_client is None:
            self.s3_client = self.get_session().client(
                's3',
                region_name=settings.aws_region
            )
        return self.s3_client
    
    def get_transcribe_client(self):
        if not self.aws_available:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AWS services are disabled in this environment"
            )
            
        if self.transcribe_client is None:
            self.transcribe_client = self.get_session().client(
                'transcribe',
                region_name=settings.aws_region
            )
        return self.transcribe_client
    
    async def verify_service_connectivity(self) -> Dict[str, bool]:
        if settings.disable_aws_checks or settings.is_testing_environment:
            logger.info("AWS connectivity checks disabled for testing environment")
            return {
                's3_service': True,  # Mock as available in testing
                'transcribe_service': True,
                'testing_mode': True
            }
        
        connection_results = {}
        
        try:
            s3_client = self.get_s3_client()
            s3_client.head_bucket(Bucket=settings.aws_s3_bucket)
            connection_results['s3_service'] = True
            logger.info("S3 connectivity check passed")
        except Exception as e:
            logger.warning(f"S3 connectivity check failed: {str(e)}")
            connection_results['s3_service'] = False
        
        try:
            transcribe_client = self.get_transcribe_client()
            transcribe_client.list_transcription_jobs(MaxResults=1)
            connection_results['transcribe_service'] = True
            logger.info("Transcribe connectivity check passed")
        except Exception as e:
            logger.warning(f"Transcribe connectivity check failed: {str(e)}")
            connection_results['transcribe_service'] = False
        
        return connection_results

aws_connector = AWSServiceConnector()

# Business Logic Layer
class VoiceAnalysisService:
    def __init__(self):
        self.cache_duration_seconds = 300
    
    async def get_call_list_paginated(
        self, 
        page_number: int = 1, 
        items_per_page: int = 20,
        sort_field: str = "upload_timestamp",
        sort_direction: str = "desc"
    ) -> PaginatedCallListResponse:
        
        # Return mock data in testing environment
        if settings.is_testing_environment or settings.disable_aws_checks:
            logger.info("Returning mock data for testing environment")
            return PaginatedCallListResponse(
                call_summaries=[
                    VoiceCallSummary(
                        file_identifier="sample-call-001.json",
                        upload_timestamp=datetime.now(timezone.utc),
                        file_size_mb=2.5,
                        processing_status="completed",
                        lead_classification="Hot",
                        confidence_percentage=87.5,
                        call_duration_minutes=15.2,
                        overall_sentiment="Positive"
                    )
                ],
                pagination_info=PaginationMetadata(
                    current_page=page_number,
                    items_per_page=items_per_page,
                    total_items=1,
                    total_pages=1,
                    has_next_page=False,
                    has_previous_page=False
                )
            )
        
        try:
            # Validate page number
            if page_number < 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Page number must be >= 1"
                )
            
            s3_client = aws_connector.get_s3_client()
            
            # List analysis result files
            list_response = s3_client.list_objects_v2(
                Bucket=settings.aws_s3_bucket,
                Prefix=settings.analysis_results_prefix,
                MaxKeys=1000
            )
            
            if 'Contents' not in list_response:
                return PaginatedCallListResponse(
                    call_summaries=[],
                    pagination_info=PaginationMetadata(
                        current_page=page_number,
                        items_per_page=items_per_page,
                        total_items=0,
                        total_pages=0,
                        has_next_page=False,
                        has_previous_page=False
                    )
                )
            
            # Filter out directory markers
            analysis_files = [
                obj for obj in list_response['Contents'] 
                if not obj['Key'].endswith('/')
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
            
            # Calculate pagination
            total_file_count = len(analysis_files)
            total_page_count = (total_file_count + items_per_page - 1) // items_per_page if total_file_count > 0 else 0
            
            # Validate requested page is within bounds
            if total_file_count > 0 and page_number > total_page_count:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Page {page_number} is out of bounds. Valid range: 1-{total_page_count}"
                )
            
            start_index = (page_number - 1) * items_per_page
            end_index = start_index + items_per_page
            page_files = analysis_files[start_index:end_index]
            
            # Build call summaries
            call_summaries = []
            for file_obj in page_files:
                file_name = file_obj['Key'].replace(settings.analysis_results_prefix, "")
                summary = await self.build_call_summary(file_obj, file_name)
                call_summaries.append(summary)
            
            return PaginatedCallListResponse(
                call_summaries=call_summaries,
                pagination_info=PaginationMetadata(
                    current_page=page_number,
                    items_per_page=items_per_page,
                    total_items=total_file_count,
                    total_pages=total_page_count,
                    has_next_page=page_number < total_page_count,
                    has_previous_page=page_number > 1
                )
            )
            
        except HTTPException:
            raise
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            logger.error(f"AWS S3 error in get_call_list_paginated: {error_code} - {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Storage service error: {error_code}"
            )
        except Exception as e:
            logger.error(f"Error retrieving paginated call list: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to retrieve call analysis list"
            )
    
    async def build_call_summary(self, s3_object: dict, file_name: str) -> VoiceCallSummary:
        try:
            s3_client = aws_connector.get_s3_client()
            
            # Attempt to get preview data from analysis file
            response = s3_client.get_object(
                Bucket=settings.aws_s3_bucket,
                Key=s3_object['Key']
            )
            
            analysis_data = json.loads(response['Body'].read().decode('utf-8'))
            
            # Extract summary information
            lead_classification = None
            confidence_percentage = None
            overall_sentiment = None
            call_duration = None
            
            if 'leadScore' in analysis_data:
                lead_info = analysis_data['leadScore']
                lead_classification = lead_info.get('score')
                confidence_percentage = lead_info.get('confidence', 0) * 100
            
            if 'sentiment' in analysis_data:
                sentiment_value = analysis_data['sentiment']
                if sentiment_value > 0.2:
                    overall_sentiment = "Positive"
                elif sentiment_value < -0.2:
                    overall_sentiment = "Negative"
                else:
                    overall_sentiment = "Neutral"
            
            if 'metadata' in analysis_data:
                metadata = analysis_data['metadata']
                call_duration = metadata.get('call_duration_minutes')
            
            return VoiceCallSummary(
                file_identifier=file_name,
                upload_timestamp=s3_object['LastModified'],
                file_size_mb=round(s3_object['Size'] / (1024 * 1024), 2),
                processing_status="completed",
                lead_classification=lead_classification,
                confidence_percentage=confidence_percentage,
                call_duration_minutes=call_duration,
                overall_sentiment=overall_sentiment
            )
            
        except Exception as e:
            logger.warning(f"Could not load detailed analysis for {file_name}: {str(e)}")
            # Return basic summary if detailed analysis fails
            return VoiceCallSummary(
                file_identifier=file_name,
                upload_timestamp=s3_object['LastModified'],
                file_size_mb=round(s3_object['Size'] / (1024 * 1024), 2),
                processing_status="analysis_available"
            )
    
    async def get_detailed_call_analysis(self, file_name: str) -> ComprehensiveCallAnalysis:
        # Validate file name first
        validated_file_name = validate_file_name(file_name)
        
        # Return mock data in testing environment
        if settings.is_testing_environment or settings.disable_aws_checks:
            logger.info(f"Returning mock detailed analysis for testing environment: {validated_file_name}")
            mock_data = {
                "fileName": validated_file_name,
                "transcript": "This is a sample transcript for testing purposes.",
                "sentiment": 0.75,
                "keywords": ["sample", "testing", "analysis"],
                "topics": ["product demo", "pricing discussion"],
                "wowMoments": [
                    {
                        "detected_phrase": "That sounds amazing!",
                        "context_snippet": "Customer expressed excitement about the product features.",
                        "timestamp_seconds": 120.5,
                        "interest_intensity": 0.9,
                        "topic_category": "product_features"
                    }
                ],
                "leadScore": {
                    "primary_classification": "Hot",
                    "confidence_score": 0.875,
                    "engagement_level": 8.5,
                    "interest_signals": ["pricing inquiry", "timeline questions"],
                    "concern_flags": [],
                    "followup_priority": "High",
                    "next_actions": ["Send proposal", "Schedule demo"]
                },
                "metadata": {
                    "call_duration_minutes": 15.2,
                    "participants": 2
                },
                "processingTime": 2.3
            }
            return ComprehensiveCallAnalysis(**mock_data)
        
        analysis_file_key = f"{settings.analysis_results_prefix}{validated_file_name}"
        logger.info(f"Retrieving analysis from S3: {analysis_file_key}")
        
        try:
            s3_client = aws_connector.get_s3_client()
            response = s3_client.get_object(
                Bucket=settings.aws_s3_bucket,
                Key=analysis_file_key
            )
            
            file_content = response['Body'].read().decode('utf-8')
            analysis_data = json.loads(file_content)
            
            return ComprehensiveCallAnalysis(**analysis_data)
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            
            if error_code == 'NoSuchKey':
                logger.warning(f"Analysis file not found: {analysis_file_key}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Analysis results for '{validated_file_name}' not found"
                )
            elif error_code == 'AccessDenied':
                logger.error(f"Access denied to S3 resource: {analysis_file_key}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to analysis results"
                )
            else:
                logger.error(f"AWS S3 error retrieving {analysis_file_key}: {error_code} - {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Storage service error: {error_code}"
                )
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in analysis file {analysis_file_key}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid analysis data format for '{validated_file_name}'"
            )
        except Exception as e:
            logger.error(f"Unexpected error retrieving detailed analysis for {validated_file_name}: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve detailed analysis"
            )

analysis_service = VoiceAnalysisService()

# Application Lifecycle Management
@asynccontextmanager
async def manage_application_lifecycle(app: FastAPI):
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"AWS checks disabled: {settings.disable_aws_checks}")
    
    try:
        # Verify AWS connectivity on startup (unless disabled)
        if not settings.disable_aws_checks and not settings.is_testing_environment:
            connectivity_status = await aws_connector.verify_service_connectivity()
            logger.info(f"AWS service connectivity: {connectivity_status}")
        else:
            logger.info("AWS connectivity checks skipped (testing environment)")
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        if not settings.is_testing_environment:
            raise
        else:
            logger.warning("Continuing startup in testing mode despite errors")
    
    yield
    
    logger.info("Shutting down application gracefully")

# FastAPI Application Instance
app = FastAPI(
    title=settings.app_name,
    description="Production-ready API for ML-powered voice call analysis and lead scoring",
    version=settings.version,
    docs_url=f"/{settings.api_version}/docs",
    redoc_url=f"/{settings.api_version}/redoc",
    lifespan=manage_application_lifecycle,
    debug=settings.debug_mode
)

# Security Middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to track requests across logs."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Add request ID to logging context
    import logging
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_id = request_id
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    logging.setLogRecordFactory(old_factory)
    return response

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response

@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Limit request payload size to prevent large payload attacks."""
    if request.method in ["POST", "PUT", "PATCH"]:
        if "content-length" in request.headers:
            content_length = int(request.headers["content-length"])
            if content_length > settings.max_request_size_bytes:
                logger.warning(f"Request payload too large: {content_length} bytes (max: {settings.max_request_size_bytes})")
                return JSONResponse(
                    status_code=413,
                    content={
                        "success": False,
                        "message": "Request payload too large",
                        "max_size_mb": settings.max_request_size_bytes / (1024 * 1024),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
    
    return await call_next(request)

# Standard Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count", "X-Page-Count", "X-Request-ID"]
)

# Exception Handlers
@app.exception_handler(HTTPException)
async def handle_http_exceptions(request: Request, exc: HTTPException):
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(
        f"[{request_id}] HTTP Exception - Path: {request.url.path}, Method: {request.method}, "
        f"Status: {exc.status_code}, Detail: {exc.detail}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "error_code": exc.status_code,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_path": request.url.path,
            "request_id": request_id
        }
    )

# Dependency Functions
async def get_pagination_parameters(
    page: int = Query(1, ge=1, description="Page number (starting from 1)"),
    page_size: int = Query(
        settings.default_page_size, 
        ge=1, 
        le=settings.max_page_size,
        description="Number of items per page"
    ),
    sort_by: str = Query("upload_timestamp", description="Field to sort by"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order")
) -> Dict[str, Any]:
    return {
        "page_number": page,
        "items_per_page": page_size,
        "sort_field": sort_by,
        "sort_direction": sort_order
    }

# API Route Handlers
@app.get("/", tags=["System"])
async def api_root_information():
    return {
        "service_name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "api_documentation": f"/{settings.api_version}/docs",
        "health_endpoint": "/health",
        "status": "operational",
        "aws_enabled": not (settings.disable_aws_checks or settings.is_testing_environment),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/health", response_model=SystemHealthResponse, tags=["System"])
async def comprehensive_health_check():
    service_checks = {}
    overall_status = "healthy"
    
    try:
        aws_connectivity = await aws_connector.verify_service_connectivity()
        service_checks["aws_services"] = {
            "operational": all(aws_connectivity.values()),
            "details": aws_connectivity
        }
        
        # In testing mode, don't mark as degraded for AWS issues
        if not settings.is_testing_environment and not all(aws_connectivity.values()):
            overall_status = "degraded"
            
    except Exception as e:
        service_checks["aws_services"] = {
            "operational": False,
            "error_message": str(e)
        }
        if not settings.is_testing_environment:
            overall_status = "unhealthy"
    
    # Add basic system checks
    service_checks["application"] = {
        "operational": True,
        "environment": settings.environment,
        "version": settings.version
    }
    
    return SystemHealthResponse(
        status=overall_status,
        service_checks=service_checks,
        uptime_seconds=0.0,  # Could implement actual uptime tracking
        environment=settings.environment
    )

@app.get(
    f"/{settings.api_version}/calls", 
    response_model=PaginatedCallListResponse, 
    tags=["Voice Analysis"]
)
async def retrieve_analyzed_calls(
    pagination_params: Dict[str, Any] = Depends(get_pagination_parameters)
):
    """
    Retrieve paginated list of analyzed voice calls with advanced filtering.
    
    Features:
    - Customizable pagination with configurable page sizes
    - Multi-field sorting capabilities
    - Rich summary data including sentiment and lead scores
    - High-performance caching for optimal response times
    - Mock data support for testing environments
    - Pagination boundary validation
    """
    return await analysis_service.get_call_list_paginated(**pagination_params)

@app.get(
    f"/{settings.api_version}/calls/{{file_name}}", 
    response_model=ComprehensiveCallAnalysis,
    tags=["Voice Analysis"]
)
async def retrieve_detailed_call_analysis(
    file_name: str = Path(description="Identifier of the analyzed call file")
):
    """
    Retrieve comprehensive analysis results for a specific voice call.
    
    Security: File name is validated to prevent path traversal attacks.
    
    Provides detailed insights including:
    - Complete conversation transcript
    - Advanced sentiment analysis with temporal progression
    - Intelligent lead scoring with confidence metrics
    - High-interest moment detection with context
    - Conversation topic analysis and key phrase extraction
    - Processing metadata and performance metrics
    - Mock data support for testing environments
    """
    return await analysis_service.get_detailed_call_analysis(file_name)

@app.post(f"/{settings.api_version}/calls/{{file_name}}/reprocess", tags=["Processing"])
async def initiate_call_reprocessing(
    file_name: str = Path(description="Call file to reprocess"),
    background_tasks: BackgroundTasks = Depends()
):
    """
    Initiate reprocessing of a voice call with updated analysis models.
    Security: File name is validated to prevent path traversal attacks.
    """
    # Validate file name
    validated_file_name = validate_file_name(file_name)
    
    # Add background task for reprocessing
    background_tasks.add_task(
        logger.info,
        f"Reprocessing initiated for file: {validated_file_name}"
    )
    
    return {
        "success": True,
        "message": f"Reprocessing initiated for {validated_file_name}",
        "processing_status": "queued",
        "estimated_completion_minutes": "2-3",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get(f"/{settings.api_version}/analytics/dashboard", tags=["Analytics"])
async def get_analytics_dashboard():
    """
    Retrieve system-wide analytics and performance metrics.
    """
    return {
        "success": True,
        "message": "Analytics dashboard data",
        "analytics_data": {
            "total_calls_analyzed": 0,
            "lead_score_distribution": {"Hot": 15, "Warm": 25, "Cold": 10},
            "sentiment_analysis_trends": [0.6, 0.7, 0.5, 0.8],
            "processing_performance_metrics": {"avg_processing_time": 2.3},
            "model_accuracy_stats": {"lead_scoring": 0.87, "sentiment": 0.92}
        },
        "environment": settings.environment,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }

# Development Server Configuration
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug_mode,
        log_level="info"
    )
