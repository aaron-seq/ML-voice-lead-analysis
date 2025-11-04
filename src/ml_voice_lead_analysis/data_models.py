from typing import Any, List
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Request model for voice call analysis endpoint."""
    transcript_text: str = Field(min_length=1, description="Voice call transcript text to analyze")
    include_audio_features: bool = Field(default=False, description="Whether to include audio feature analysis")


class AnalyzedItem(BaseModel):
    """Analysis result model with classification and confidence metrics."""
    label: str = Field(description="Lead classification: Hot, Warm, or Cold")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score between 0 and 1")
    features_used: list[str] = Field(description="List of feature types used in analysis")
    model_version: str = Field(description="Version of the ML model used for classification")


class AnalysisResponse(BaseModel):
    """Response model for analysis endpoints."""
    result: AnalyzedItem


class TrainExample(BaseModel):
    """Training example model for supervised learning."""
    transcript_text: str = Field(min_length=1, description="Training transcript text")
    label: str = Field(description="Training label: Hot, Warm, or Cold")


class TrainRequest(BaseModel):
    """Request model for training the classifier with new examples."""
    examples: List[TrainExample] = Field(min_items=1, description="List of training examples")


class BatchItem(BaseModel):
    """Individual item for batch analysis processing."""
    id: str = Field(description="Unique identifier for this batch item")
    transcript_text: str = Field(min_length=1, description="Transcript text to analyze")


class BatchRequest(BaseModel):
    """Request model for batch analysis of multiple transcripts."""
    items: List[BatchItem] = Field(min_items=1, description="List of items to analyze in batch")


class BatchResult(BaseModel):
    """Result model for individual batch analysis item."""
    id: str = Field(description="Identifier matching the request item")
    label: str = Field(description="Classification result")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence score")


class BatchResponse(BaseModel):
    """Response model for batch analysis endpoint."""
    results: List[BatchResult] = Field(description="Analysis results for each batch item")
    count: int = Field(description="Total number of items processed")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(description="Service status: ok, degraded, or error")
    features: dict[str, Any] = Field(description="Available features and configuration")