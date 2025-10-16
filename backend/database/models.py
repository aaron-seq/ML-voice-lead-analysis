"""
Modern Database Models for ML Voice Lead Analysis
SQLAlchemy 2.0 compatible models with comprehensive relationships and indexing.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    String, Text, Integer, Float, Boolean, DateTime,
    JSON, ForeignKey, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class DatabaseBase(DeclarativeBase):
    """Base class for all database models."""
    pass


class TimestampMixin:
    """Mixin for timestamp fields."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )


class CallTranscript(DatabaseBase, TimestampMixin):
    """Model for storing call transcript metadata and analysis results."""
    
    __tablename__ = "call_transcripts"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # File identification
    file_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True
    )
    s3_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        unique=True
    )
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Content
    original_transcript: Mapped[Optional[str]] = mapped_column(Text)
    processed_transcript: Mapped[Optional[str]] = mapped_column(Text)
    
    # Metadata
    upload_timestamp: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    processing_status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False
    )
    processing_duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    
    # Analysis results (stored as JSON for flexibility)
    analysis_results: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    
    # Key metrics (denormalized for quick access)
    word_count: Mapped[Optional[int]] = mapped_column(Integer)
    estimated_duration_minutes: Mapped[Optional[float]] = mapped_column(Float)
    confidence_score: Mapped[Optional[float]] = mapped_column(
        Float,
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1")
    )
    
    # Relationships
    sentiment_analysis: Mapped[Optional["SentimentAnalysis"]] = relationship(
        "SentimentAnalysis",
        back_populates="call_transcript",
        uselist=False,
        cascade="all, delete-orphan"
    )
    lead_scoring: Mapped[Optional["LeadScore"]] = relationship(
        "LeadScore",
        back_populates="call_transcript",
        uselist=False,
        cascade="all, delete-orphan"
    )
    interest_moments: Mapped[List["InterestMoment"]] = relationship(
        "InterestMoment",
        back_populates="call_transcript",
        cascade="all, delete-orphan"
    )
    topics: Mapped[List["CallTopic"]] = relationship(
        "CallTopic",
        back_populates="call_transcript",
        cascade="all, delete-orphan"
    )
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_file_name_status", "file_name", "processing_status"),
        Index("idx_upload_timestamp", "upload_timestamp"),
        Index("idx_confidence_score", "confidence_score"),
    )
    
    def __repr__(self) -> str:
        return f"<CallTranscript(id={self.id}, file_name='{self.file_name}')>"


class SentimentAnalysis(DatabaseBase, TimestampMixin):
    """Model for storing sentiment analysis results."""
    
    __tablename__ = "sentiment_analyses"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Foreign key to call transcript
    call_transcript_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("call_transcripts.id"),
        nullable=False
    )
    
    # Sentiment scores
    overall_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        CheckConstraint("overall_score >= -1.0 AND overall_score <= 1.0")
    )
    confidence_rating: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        CheckConstraint("confidence_rating >= 0.0 AND confidence_rating <= 1.0")
    )
    
    # Emotion breakdown (JSON field for flexibility)
    emotion_categories: Mapped[Dict[str, float]] = mapped_column(
        JSON,
        nullable=False,
        default=dict
    )
    
    # Analysis method metadata
    analysis_method: Mapped[str] = mapped_column(
        String(100),
        default="ensemble",
        nullable=False
    )
    model_versions: Mapped[Dict[str, str]] = mapped_column(
        JSON,
        default=dict
    )
    
    # Relationships
    call_transcript: Mapped["CallTranscript"] = relationship(
        "CallTranscript",
        back_populates="sentiment_analysis"
    )
    
    def __repr__(self) -> str:
        return f"<SentimentAnalysis(id={self.id}, score={self.overall_score:.2f})>"


class LeadScore(DatabaseBase, TimestampMixin):
    """Model for storing lead scoring results."""
    
    __tablename__ = "lead_scores"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Foreign key to call transcript
    call_transcript_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("call_transcripts.id"),
        nullable=False
    )
    
    # Lead scoring results
    primary_classification: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True
    )
    numerical_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        CheckConstraint("numerical_score >= 0 AND numerical_score <= 100")
    )
    confidence_level: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        CheckConstraint("confidence_level >= 0.0 AND confidence_level <= 1.0")
    )
    
    # Detailed analysis (stored as JSON)
    interest_signals: Mapped[List[str]] = mapped_column(
        JSON,
        default=list
    )
    concern_indicators: Mapped[List[str]] = mapped_column(
        JSON,
        default=list
    )
    buying_signals: Mapped[List[str]] = mapped_column(
        JSON,
        default=list
    )
    
    # AI-generated insights
    personality_assessment: Mapped[Optional[str]] = mapped_column(String(100))
    decision_making_style: Mapped[Optional[str]] = mapped_column(String(100))
    recommended_strategy: Mapped[Optional[str]] = mapped_column(Text)
    next_actions: Mapped[List[str]] = mapped_column(
        JSON,
        default=list
    )
    
    # Risk assessment
    risk_factors: Mapped[List[str]] = mapped_column(
        JSON,
        default=list
    )
    conversion_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        CheckConstraint("conversion_probability >= 0.0 AND conversion_probability <= 1.0")
    )
    
    # Relationships
    call_transcript: Mapped["CallTranscript"] = relationship(
        "CallTranscript",
        back_populates="lead_scoring"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_classification_score", "primary_classification", "numerical_score"),
        Index("idx_conversion_probability", "conversion_probability"),
    )
    
    def __repr__(self) -> str:
        return f"<LeadScore(id={self.id}, classification='{self.primary_classification}')>"


class InterestMoment(DatabaseBase, TimestampMixin):
    """Model for storing detected interest moments."""
    
    __tablename__ = "interest_moments"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Foreign key to call transcript
    call_transcript_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("call_transcripts.id"),
        nullable=False
    )
    
    # Interest moment details
    trigger_phrase: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    contextual_content: Mapped[Optional[str]] = mapped_column(Text)
    estimated_timestamp: Mapped[Optional[float]] = mapped_column(Float)
    
    # Scoring
    intensity_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        CheckConstraint("intensity_score >= 0.0 AND intensity_score <= 1.0")
    )
    
    # Categorization
    topic_category: Mapped[Optional[str]] = mapped_column(String(100))
    speaker_identifier: Mapped[Optional[str]] = mapped_column(String(50))
    follow_up_recommendation: Mapped[Optional[str]] = mapped_column(Text)
    
    # Relationships
    call_transcript: Mapped["CallTranscript"] = relationship(
        "CallTranscript",
        back_populates="interest_moments"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_intensity_score", "intensity_score"),
        Index("idx_timestamp", "estimated_timestamp"),
    )
    
    def __repr__(self) -> str:
        return f"<InterestMoment(id={self.id}, intensity={self.intensity_score:.2f})>"


class CallTopic(DatabaseBase, TimestampMixin):
    """Model for storing identified call topics."""
    
    __tablename__ = "call_topics"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Foreign key to call transcript
    call_transcript_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("call_transcripts.id"),
        nullable=False
    )
    
    # Topic details
    topic_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        index=True
    )
    relevance_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        CheckConstraint("relevance_score >= 0.0 AND relevance_score <= 1.0")
    )
    
    # Associated data
    keywords: Mapped[List[str]] = mapped_column(
        JSON,
        default=list
    )
    discussion_duration_percentage: Mapped[Optional[float]] = mapped_column(Float)
    topic_sentiment: Mapped[Optional[float]] = mapped_column(
        Float,
        CheckConstraint("topic_sentiment >= -1.0 AND topic_sentiment <= 1.0")
    )
    
    # Relationships
    call_transcript: Mapped["CallTranscript"] = relationship(
        "CallTranscript",
        back_populates="topics"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_topic_relevance", "topic_name", "relevance_score"),
        Index("idx_relevance_score", "relevance_score"),
    )
    
    def __repr__(self) -> str:
        return f"<CallTopic(id={self.id}, topic='{self.topic_name}')>"


class ProcessingJob(DatabaseBase, TimestampMixin):
    """Model for tracking background processing jobs."""
    
    __tablename__ = "processing_jobs"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Job details
    job_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        nullable=False,
        index=True
    )
    
    # Associated data
    input_data: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict
    )
    output_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    
    # Progress tracking
    progress_percentage: Mapped[int] = mapped_column(
        Integer,
        default=0,
        CheckConstraint("progress_percentage >= 0 AND progress_percentage <= 100")
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_job_type_status", "job_type", "status"),
        Index("idx_created_at", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<ProcessingJob(id={self.id}, type='{self.job_type}', status='{self.status}')>"


class SystemMetric(DatabaseBase, TimestampMixin):
    """Model for storing system performance metrics."""
    
    __tablename__ = "system_metrics"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Metric details
    metric_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True
    )
    metric_value: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )
    metric_unit: Mapped[Optional[str]] = mapped_column(String(20))
    
    # Context
    component: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True
    )
    environment: Mapped[str] = mapped_column(
        String(20),
        default="production",
        nullable=False
    )
    
    # Additional data
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    
    # Indexes
    __table_args__ = (
        Index("idx_metric_component_time", "metric_name", "component", "created_at"),
        Index("idx_created_at_metric", "created_at", "metric_name"),
    )
    
    def __repr__(self) -> str:
        return f"<SystemMetric(metric='{self.metric_name}', value={self.metric_value})>"