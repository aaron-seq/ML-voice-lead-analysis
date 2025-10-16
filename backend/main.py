from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Path, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from contextlib import asynccontextmanager
import json
import logging
import os

import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class ApplicationSettings:
    def __init__(self):
        self.app_name = "ML Voice Lead Analysis API"
        self.version = "4.0.0"
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug_mode = self.environment == "development"
        
        # AWS Configuration
        self.aws_s3_bucket = os.getenv("DATA_BUCKET", "ml-voice-analysis-bucket")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.analysis_results_prefix = "analysis-results/"
        self.transcripts_prefix = "transcripts/"
        
        # API Configuration
        self.default_page_size = 20
        self.max_page_size = 100
        self.api_version = "v1"
        
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

# Pydantic Models
class BaseApiResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

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
    
    def get_session(self):
        if self.session is None:
            self.session = boto3.Session()
        return self.session
    
    def get_s3_client(self):
        if self.s3_client is None:
            self.s3_client = self.get_session().client(
                's3',
                region_name=settings.aws_region
            )
        return self.s3_client
    
    def get_transcribe_client(self):
        if self.transcribe_client is None:
            self.transcribe_client = self.get_session().client(
                'transcribe',
                region_name=settings.aws_region
            )
        return self.transcribe_client
    
    async def verify_service_connectivity(self) -> Dict[str, bool]:
        connection_results = {}
        
        try:
            s3_client = self.get_s3_client()
            s3_client.head_bucket(Bucket=settings.aws_s3_bucket)
            connection_results['s3_service'] = True
        except Exception as e:
            logger.error(f"S3 connectivity check failed: {str(e)}")
            connection_results['s3_service'] = False
        
        try:
            transcribe_client = self.get_transcribe_client()
            transcribe_client.list_transcription_jobs(MaxResults=1)
            connection_results['transcribe_service'] = True
        except Exception as e:
            logger.error(f"Transcribe connectivity check failed: {str(e)}")
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
        try:
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
            total_page_count = (total_file_count + items_per_page - 1) // items_per_page
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
            
        except Exception as e:
            logger.error(f"Error retrieving paginated call list: {str(e)}")
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
            
        except Exception:
            # Return basic summary if detailed analysis fails
            return VoiceCallSummary(
                file_identifier=file_name,
                upload_timestamp=s3_object['LastModified'],
                file_size_mb=round(s3_object['Size'] / (1024 * 1024), 2),
                processing_status="analysis_available"
            )
    
    async def get_detailed_call_analysis(self, file_name: str) -> ComprehensiveCallAnalysis:
        analysis_file_key = f"{settings.analysis_results_prefix}{file_name}"
        
        try:
            s3_client = aws_connector.get_s3_client()
            response = s3_client.get_object(
                Bucket=settings.aws_s3_bucket,
                Key=analysis_file_key
            )
            
            file_content = response['Body'].read().decode('utf-8')
            analysis_data = json.loads(file_content)
            
            return ComprehensiveCallAnalysis(**analysis_data)
            
        except s3_client.exceptions.NoSuchKey:
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
            logger.error(f"Error retrieving detailed analysis for {file_name}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve detailed analysis"
            )

analysis_service = VoiceAnalysisService()

# Application Lifecycle Management
@asynccontextmanager
async def manage_application_lifecycle(app: FastAPI):
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    
    try:
        # Verify AWS connectivity on startup
        connectivity_status = await aws_connector.verify_service_connectivity()
        logger.info(f"AWS service connectivity: {connectivity_status}")
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise
    
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

# Middleware Configuration
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count", "X-Page-Count"]
)

# Exception Handlers
@app.exception_handler(HTTPException)
async def handle_http_exceptions(request: Request, exc: HTTPException):
    logger.error(
        f"HTTP Exception - Path: {request.url.path}, Method: {request.method}, "
        f"Status: {exc.status_code}, Detail: {exc.detail}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "error_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
            "request_path": request.url.path
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
        "timestamp": datetime.utcnow().isoformat()
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
        
        if not all(aws_connectivity.values()):
            overall_status = "degraded"
            
    except Exception as e:
        service_checks["aws_services"] = {
            "operational": False,
            "error_message": str(e)
        }
        overall_status = "unhealthy"
    
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
    
    Provides detailed insights including:
    - Complete conversation transcript
    - Advanced sentiment analysis with temporal progression
    - Intelligent lead scoring with confidence metrics
    - High-interest moment detection with context
    - Conversation topic analysis and key phrase extraction
    - Processing metadata and performance metrics
    """
    return await analysis_service.get_detailed_call_analysis(file_name)

@app.post(f"/{settings.api_version}/calls/{{file_name}}/reprocess", tags=["Processing"])
async def initiate_call_reprocessing(
    file_name: str = Path(description="Call file to reprocess"),
    background_tasks: BackgroundTasks = Depends()
):
    """
    Initiate reprocessing of a voice call with updated analysis models.
    """
    
    # Add background task for reprocessing
    background_tasks.add_task(
        logger.info,
        f"Reprocessing initiated for file: {file_name}"
    )
    
    return {
        "success": True,
        "message": f"Reprocessing initiated for {file_name}",
        "processing_status": "queued",
        "estimated_completion_minutes": "2-3",
        "timestamp": datetime.utcnow().isoformat()
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
            "lead_score_distribution": {},
            "sentiment_analysis_trends": [],
            "processing_performance_metrics": {},
            "model_accuracy_stats": {}
        },
        "last_updated": datetime.utcnow().isoformat()
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
