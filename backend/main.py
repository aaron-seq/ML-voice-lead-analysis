"""
ML Voice Lead Analysis API
A FastAPI application for retrieving and analyzing sales call data.
"""

import os
import json
import asyncio
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import boto3
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
import aiofiles

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuration ---
class AppConfig:
    """Application configuration management."""
    
    S3_BUCKET_NAME: str = os.getenv("DATA_BUCKET", "ml-voice-analysis-bucket")
    S3_ANALYSIS_PREFIX: str = "analysis-results/"
    S3_TRANSCRIPTS_PREFIX: str = "transcripts/"
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    API_VERSION: str = "v1"
    MAX_FILE_SIZE_MB: int = 50
    
    # Performance settings
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100

config = AppConfig()

# --- AWS Service Clients ---
class AWSServiceManager:
    """Manages AWS service connections."""
    
    def __init__(self):
        self._s3_client = None
        self._transcribe_client = None
    
    @property
    def s3_client(self):
        if self._s3_client is None:
            self._s3_client = boto3.client(
                's3',
                region_name=config.AWS_REGION
            )
        return self._s3_client
    
    @property
    def transcribe_client(self):
        if self._transcribe_client is None:
            self._transcribe_client = boto3.client(
                'transcribe',
                region_name=config.AWS_REGION
            )
        return self._transcribe_client

aws_services = AWSServiceManager()

# --- Pydantic Models ---
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    model_config = ConfigDict(from_attributes=True)
    
    success: bool = True
    message: Optional[str] = None

class CallSummary(BaseModel):
    """Summary information for a call in listings."""
    file_name: str = Field(..., description="The analyzed file name")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    lead_score: Optional[str] = Field(None, description="Quick lead score preview")

class LeadScoreDetails(BaseModel):
    """Detailed lead scoring information."""
    score: str = Field(..., example="Hot", description="Lead classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    reasoning: Optional[str] = Field(None, description="Why this score was assigned")

class HighInterestMoment(BaseModel):
    """Represents a moment of high interest in the conversation."""
    keyword: str = Field(..., description="The trigger keyword")
    context: str = Field(..., description="Surrounding context")
    timestamp: Optional[float] = Field(None, description="Time in call (seconds)")
    sentiment_score: Optional[float] = Field(None, description="Local sentiment")

class CallAnalysisResult(BaseModel):
    """Complete analysis results for a call."""
    file_name: str = Field(..., alias="fileName")
    transcript: str = Field(..., description="Full conversation transcript")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Overall sentiment")
    key_phrases: List[str] = Field(..., alias="keywords", description="Extracted key phrases")
    discussion_topics: List[str] = Field(..., alias="topics", description="Main topics discussed")
    high_interest_moments: List[HighInterestMoment] = Field(..., alias="wowMoments")
    lead_classification: LeadScoreDetails = Field(..., alias="leadScore")
    call_duration: Optional[float] = Field(None, description="Duration in minutes")
    participant_count: Optional[int] = Field(None, description="Number of speakers")

class CallListResponse(BaseResponse):
    """Response model for call listings with pagination."""
    calls: List[CallSummary]
    total_count: int
    page: int
    page_size: int
    has_next: bool

class HealthCheckResponse(BaseResponse):
    """Health check response model."""
    version: str
    environment: str
    aws_connection: bool

# --- Application Lifecycle ---
@asynccontextmanager
async def application_lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    logger.info("🚀 Starting ML Voice Lead Analysis API")
    
    # Startup tasks
    try:
        # Test AWS connectivity
        aws_services.s3_client.head_bucket(Bucket=config.S3_BUCKET_NAME)
        logger.info("✅ AWS S3 connection verified")
    except Exception as e:
        logger.warning(f"⚠️ AWS S3 connection issue: {e}")
    
    yield
    
    # Shutdown tasks
    logger.info("🛑 Shutting down ML Voice Lead Analysis API")

# --- FastAPI Application ---
app = FastAPI(
    title="ML Voice Lead Analysis API",
    description="Advanced API for analyzing sales call transcripts and extracting actionable insights",
    version="2.0.0",
    docs_url=f"/{config.API_VERSION}/docs",
    redoc_url=f"/{config.API_VERSION}/redoc",
    lifespan=application_lifespan
)

# --- Middleware Configuration ---
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app",
        "https://*.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# --- Exception Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "error_code": exc.status_code
        }
    )

# --- Dependency Functions ---
async def get_pagination_params(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(config.DEFAULT_PAGE_SIZE, ge=1, le=config.MAX_PAGE_SIZE)
) -> Dict[str, int]:
    """Extract and validate pagination parameters."""
    return {"page": page, "page_size": page_size}

# --- Service Functions ---
class CallAnalysisService:
    """Service class for call analysis operations."""
    
    @staticmethod
    async def list_analyzed_calls(page: int = 1, page_size: int = 20) -> CallListResponse:
        """Retrieve paginated list of analyzed calls."""
        try:
            response = aws_services.s3_client.list_objects_v2(
                Bucket=config.S3_BUCKET_NAME,
                Prefix=config.S3_ANALYSIS_PREFIX,
                MaxKeys=1000  # Get more to handle pagination
            )
            
            if 'Contents' not in response:
                return CallListResponse(
                    calls=[],
                    total_count=0,
                    page=page,
                    page_size=page_size,
                    has_next=False
                )
            
            # Filter and process files
            all_files = [
                obj for obj in response['Contents'] 
                if obj['Key'] != config.S3_ANALYSIS_PREFIX
            ]
            
            total_count = len(all_files)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_files = all_files[start_idx:end_idx]
            
            calls = []
            for obj in page_files:
                file_name = obj['Key'].replace(config.S3_ANALYSIS_PREFIX, "")
                
                # Try to get lead score preview
                lead_score = None
                try:
                    # Quick peek at the file for lead score
                    content = aws_services.s3_client.get_object(
                        Bucket=config.S3_BUCKET_NAME,
                        Key=obj['Key']
                    )
                    data = json.loads(content['Body'].read().decode('utf-8'))
                    lead_score = data.get('leadScore', {}).get('score')
                except:
                    pass  # If we can't get it, just skip
                
                calls.append(CallSummary(
                    file_name=file_name,
                    created_at=obj['LastModified'].isoformat(),
                    file_size=obj['Size'],
                    lead_score=lead_score
                ))
            
            return CallListResponse(
                calls=calls,
                total_count=total_count,
                page=page,
                page_size=page_size,
                has_next=end_idx < total_count
            )
            
        except Exception as e:
            logger.error(f"Failed to list calls: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Unable to retrieve call list: {str(e)}"
            )
    
    @staticmethod
    async def get_call_analysis(file_name: str) -> CallAnalysisResult:
        """Retrieve detailed analysis for a specific call."""
        s3_key = f"{config.S3_ANALYSIS_PREFIX}{file_name}"
        
        try:
            s3_object = aws_services.s3_client.get_object(
                Bucket=config.S3_BUCKET_NAME,
                Key=s3_key
            )
            
            content = s3_object['Body'].read().decode('utf-8')
            raw_data = json.loads(content)
            
            # Transform the data to match our response model
            return CallAnalysisResult(**raw_data)
            
        except aws_services.s3_client.exceptions.NoSuchKey:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis for call '{file_name}' not found"
            )
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid analysis data format for '{file_name}'"
            )
        except Exception as e:
            logger.error(f"Error retrieving analysis for {file_name}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve analysis: {str(e)}"
            )

# --- API Endpoints ---
@app.get("/", response_model=Dict[str, str], tags=["General"])
async def api_root():
    """API root endpoint with welcome message."""
    return {
        "message": "ML Voice Lead Analysis API",
        "version": "2.0.0",
        "docs": f"/{config.API_VERSION}/docs"
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["General"])
async def health_check():
    """Comprehensive health check endpoint."""
    aws_healthy = True
    try:
        aws_services.s3_client.head_bucket(Bucket=config.S3_BUCKET_NAME)
    except:
        aws_healthy = False
    
    return HealthCheckResponse(
        version="2.0.0",
        environment=config.ENVIRONMENT,
        aws_connection=aws_healthy
    )

@app.get(f"/{config.API_VERSION}/calls", response_model=CallListResponse, tags=["Call Analysis"])
async def list_analyzed_calls(
    pagination: Dict[str, int] = Depends(get_pagination_params)
):
    """
    Retrieve a paginated list of all analyzed call files.
    
    - **page**: Page number (starts from 1)
    - **page_size**: Number of items per page (max 100)
    """
    return await CallAnalysisService.list_analyzed_calls(
        page=pagination["page"],
        page_size=pagination["page_size"]
    )

@app.get(f"/{config.API_VERSION}/calls/{{file_name}}", response_model=CallAnalysisResult, tags=["Call Analysis"])
async def get_call_analysis_details(file_name: str):
    """
    Retrieve comprehensive analysis details for a specific call.
    
    - **file_name**: The name of the analyzed call file
    """
    return await CallAnalysisService.get_call_analysis(file_name)

@app.post(f"/{config.API_VERSION}/calls/{{file_name}}/reanalyze", tags=["Call Analysis"])
async def trigger_call_reanalysis(file_name: str, background_tasks: BackgroundTasks):
    """
    Trigger re-analysis of a specific call.
    
    - **file_name**: The name of the call file to re-analyze
    """
    # This would trigger the ML pipeline - implement based on your setup
    background_tasks.add_task(
        logger.info, 
        f"Re-analysis triggered for {file_name}"
    )
    
    return {
        "success": True,
        "message": f"Re-analysis initiated for {file_name}",
        "status": "processing"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=config.ENVIRONMENT == "development",
        log_level="info"
    )
