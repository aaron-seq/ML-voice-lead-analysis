import os
import json
import boto3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# --- Application Configuration ---
# Use environment variables for flexible configuration
S3_BUCKET_NAME = os.environ.get("DATA_BUCKET", "your-ml-pipeline-bucket")
S3_ANALYSIS_PREFIX = "analysis-results/"

# Initialize FastAPI app
app = FastAPI(
    title="ML Voice Lead Analysis API",
    description="API to retrieve and view analysis results of sales calls.",
    version="1.0.0"
)

# --- CORS Configuration ---
# Allows the React frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Add deployed frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AWS S3 Client ---
s3_client = boto3.client('s3')

# --- Pydantic Models for Data Validation ---
# Ensures response data has a consistent and predictable structure

class CallListItem(BaseModel):
    """Represents a single analyzed call in a list."""
    file_name: str = Field(..., description="The name of the analyzed file.", example="call_123.json")

class LeadScore(BaseModel):
    """Detailed lead score information."""
    score: str = Field(..., example="Hot")
    confidence: float = Field(..., example=0.92)

class WowMoment(BaseModel):
    """Represents a moment of high interest in the call."""
    keyword: str
    context: str

class CallAnalysisDetails(BaseModel):
    """Full analysis details for a specific call."""
    file_name: str = Field(..., alias="fileName")
    transcript: str
    sentiment: float
    keywords: List[str]
    topics: List[str]
    wow_moments: List[WowMoment] = Field(..., alias="wowMoments")
    lead_score: LeadScore = Field(..., alias="leadScore")

# --- API Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    """Root endpoint with a welcome message."""
    return {"message": "Welcome to the ML Voice Lead Analysis API"}

@app.get("/calls", response_model=List[CallListItem], tags=["Calls"])
def get_analyzed_calls_list():
    """
    Retrieves a list of all analyzed call files from the S3 bucket.
    """
    try:
        s3_response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_ANALYSIS_PREFIX)
        if 'Contents' not in s3_response:
            return [] # No files found, return empty list

        analyzed_files = [
            {"file_name": obj['Key'].replace(S3_ANALYSIS_PREFIX, "")}
            for obj in s3_response['Contents'] if obj['Key'] != S3_ANALYSIS_PREFIX
        ]
        return analyzed_files
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to list files from S3: {str(error)}")

@app.get("/calls/{file_name}", response_model=CallAnalysisDetails, tags=["Calls"])
def get_analysis_details_for_call(file_name: str):
    """
    Retrieves the detailed analysis JSON for a specific call file.
    """
    s3_key = f"{S3_ANALYSIS_PREFIX}{file_name}"
    try:
        s3_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        content = s3_object['Body'].read().decode('utf-8')
        # Pydantic will automatically validate the structure of the JSON content
        return json.loads(content)
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"Analysis for file '{file_name}' not found.")
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(error)}")
