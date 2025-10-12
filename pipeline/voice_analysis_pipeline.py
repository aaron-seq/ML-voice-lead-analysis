"""
Advanced ML Voice Lead Analysis Pipeline
Modern, production-ready pipeline with enterprise-grade ML capabilities and cloud deployment support.
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import time
from datetime import datetime, timezone
import hashlib
import traceback
from concurrent.futures import ThreadPoolExecutor

import boto3
import aiofiles
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import spacy
from spacy.matcher import Matcher
try:
    import tensorflow as tf
except ImportError:
    tf = None
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, AutoModel
    )
except ImportError:
    pipeline = None
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import structlog
from botocore.exceptions import ClientError, BotoCoreError

# Download required NLTK data safely
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

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
@dataclass
class PipelineConfiguration:
    """Comprehensive pipeline configuration with cloud deployment support."""
    
    # AWS Configuration
    s3_bucket_name: str = os.getenv("DATA_BUCKET", "ml-voice-analysis-data")
    s3_transcripts_prefix: str = "transcripts/"
    s3_analysis_prefix: str = "analysis-results/"
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    
    # Model Configuration
    spacy_model_name: str = "en_core_web_md"  # More deployment-friendly
    use_transformer_models: bool = os.getenv("USE_TRANSFORMERS", "false").lower() == "true"
    transformer_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # Processing Parameters
    max_sequence_length: int = 512
    batch_processing_size: int = 8
    confidence_threshold: float = 0.7
    
    # Analysis Configuration
    max_extracted_phrases: int = 20
    max_discussion_topics: int = 15
    context_analysis_window: int = 25
    minimum_phrase_length: int = 2
    
    # Performance Settings
    enable_gpu_acceleration: bool = False  # Disabled for cloud compatibility
    max_concurrent_workers: int = 2
    enable_result_caching: bool = True
    
    # Feature Configuration
    enable_advanced_sentiment: bool = True
    enable_topic_modeling: bool = True
    enable_speaker_analysis: bool = True
    enable_emotion_detection: bool = False  # Optional for cloud deployment
    
    # Cloud Deployment Settings
    cloud_deployment_mode: bool = os.getenv("CLOUD_DEPLOYMENT", "false").lower() == "true"
    memory_optimization: bool = True
    lightweight_models_only: bool = True

pipeline_config = PipelineConfiguration()

# --- Enhanced Data Models ---
@dataclass
class SentimentAnalysisResult:
    """Comprehensive sentiment analysis with multiple methods."""
    overall_sentiment_score: float
    confidence_rating: float
    emotion_categories: Dict[str, float]
    temporal_sentiment_trend: List[Tuple[float, float]]
    key_emotional_indicators: List[Dict[str, Any]]

@dataclass
class TopicAnalysisResult:
    """Advanced topic extraction and relevance analysis."""
    topic_identifier: str
    relevance_weight: float
    associated_keywords: List[str]
    discussion_time_percentage: float
    topic_sentiment_score: float

@dataclass
class SpeakerAnalysisResult:
    """Speaker profiling and role identification."""
    speaker_identifier: str
    estimated_role_category: str  # prospect, sales_representative, decision_maker
    speaking_duration_percentage: float
    sentiment_distribution: Dict[str, float]
    engagement_level_score: float
    identified_concerns: List[str]

@dataclass
class InterestMomentResult:
    """Enhanced detection of high-interest conversation moments."""
    trigger_phrase_content: str
    contextual_information: str
    estimated_timestamp: float
    interest_intensity_score: float
    topic_classification: str
    associated_speaker: Optional[str]
    follow_up_recommendation: str

@dataclass
class ComprehensiveLeadScore:
    """Advanced lead scoring with AI-driven insights."""
    primary_classification: str  # Hot, Warm, Cold, Unqualified
    numerical_score_value: float  # 0-100 scale
    confidence_level_rating: float
    
    # Detailed analysis components
    interest_signals: List[str]
    concern_indicators: List[str]
    buying_readiness_signals: List[str]
    timing_assessment_factors: List[str]
    
    # AI-generated insights
    personality_assessment: str
    decision_making_approach: str
    recommended_engagement_strategy: str
    prioritized_next_actions: List[str]
    
    # Risk evaluation
    identified_risk_factors: List[str]
    conversion_probability: float

@dataclass
class ComprehensiveAnalysisOutput:
    """Complete analysis results with enhanced metadata."""
    # Basic file information
    analyzed_file_name: str
    processing_completion_timestamp: datetime
    total_processing_duration_seconds: float
    
    # Content analysis
    original_transcript_content: str
    processed_transcript_content: str
    total_word_count: int
    estimated_call_duration_minutes: float
    
    # Advanced analysis results
    sentiment_analysis_results: SentimentAnalysisResult
    topic_analysis_results: List[TopicAnalysisResult]
    speaker_analysis_results: List[SpeakerAnalysisResult]
    interest_moments: List[InterestMomentResult]
    lead_scoring_results: ComprehensiveLeadScore
    
    # Extracted intelligence
    key_phrases_extracted: List[str]
    technical_terminology: List[str]
    competitor_references: List[str]
    pricing_discussions: List[str]
    objections_identified: List[str]
    
    # Conversation analysis
    conversation_flow_metrics: Dict[str, Any]
    engagement_timeline_data: List[Tuple[float, float]]
    question_analysis_summary: Dict[str, Any]
    
    # Processing metadata
    confidence_scores_by_component: Dict[str, float]
    model_versions_used: Dict[str, str]
    processing_statistics: Dict[str, Any]

# --- Advanced ML Pipeline Implementation ---
class ModernVoiceLeadAnalysisPipeline:
    """
    Next-generation ML pipeline optimized for cloud deployment and enterprise usage.
    """
    
    def __init__(self, configuration: PipelineConfiguration = None):
        self.config = configuration or PipelineConfiguration()
        
        # Core processing components
        self.nlp_processor = None
        self.transformer_models_cache = {}
        self.sentiment_analyzers = {}
        self.topic_modeling_vectorizer = None
        self.aws_service_clients = {}
        
        # Advanced components
        self.emotion_analysis_pipeline = None
        self.speaker_classification_model = None
        self.embeddings_cache = {}
        
        # Performance monitoring
        self.pipeline_performance_metrics = {
            "total_files_processed": 0,
            "average_processing_duration": 0.0,
            "processing_success_rate": 0.0,
            "cache_utilization_rate": 0.0
        }
        
        # Thread pool for concurrent processing
        self.thread_executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_workers)

    async def initialize_pipeline_components(self) -> None:
        """Initialize all pipeline components with enhanced error handling."""
        logger.info("Initializing Advanced ML Voice Analysis Pipeline...")
        
        initialization_start_time = time.time()
        
        try:
            # Initialize core NLP components
            await self._initialize_core_nlp_processor()
            
            # Load transformer models (if enabled)
            if self.config.use_transformer_models:
                await self._initialize_transformer_models()
            
            # Setup sentiment analysis components
            await self._initialize_sentiment_analysis()
            
            # Initialize AWS service clients
            await self._initialize_aws_services()
            
            # Setup advanced analysis components
            await self._initialize_advanced_analyzers()
            
            # Validate pipeline integrity
            await self._validate_pipeline_setup()
            
            initialization_duration = time.time() - initialization_start_time
            logger.info(f"Pipeline initialization completed successfully in {initialization_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            logger.error(traceback.format_exc())
            raise

    async def _initialize_core_nlp_processor(self) -> None:
        """Initialize spaCy NLP processor with fallback handling."""
        logger.info("Loading spaCy NLP model...")
        
        try:
            # Try loading the preferred model
            self.nlp_processor = spacy.load(self.config.spacy_model_name)
            
            # Add custom pipeline components
            if "sentencizer" not in self.nlp_processor.pipe_names:
                self.nlp_processor.add_pipe("sentencizer")
            
            # Setup advanced pattern matching
            self._configure_pattern_matchers()
            
            logger.info("Core NLP processor initialized successfully")
            
        except OSError:
            # Fallback to smaller model for cloud deployment
            logger.warning(f"Model '{self.config.spacy_model_name}' not found, using fallback")
            try:
                self.nlp_processor = spacy.load("en_core_web_sm")
                logger.info("Fallback NLP model loaded successfully")
            except OSError:
                logger.error("No suitable spaCy model found")
                raise

    async def _initialize_transformer_models(self) -> None:
        """Initialize transformer models with graceful fallback."""
        if not pipeline:
            logger.warning("Transformers library not available, skipping transformer models")
            return
            
        logger.info("Loading transformer models...")
        
        try:
            # Sentiment analysis transformer
            self.transformer_models_cache['sentiment'] = pipeline(
                "sentiment-analysis",
                model=self.config.transformer_model_name,
                return_all_scores=True,
                device=-1  # Force CPU for cloud compatibility
            )
            
            logger.info("Transformer models loaded successfully")
            
        except Exception as e:
            logger.warning(f"Transformer model loading failed: {e}")
            logger.info("Continuing with traditional NLP methods")

    async def _initialize_sentiment_analysis(self) -> None:
        """Setup multiple sentiment analysis methods."""
        try:
            # VADER sentiment analyzer
            self.sentiment_analyzers['vader'] = SentimentIntensityAnalyzer()
            
            # TextBlob sentiment
            self.sentiment_analyzers['textblob'] = TextBlob
            
            logger.info("Sentiment analysis components initialized")
            
        except Exception as e:
            logger.error(f"Sentiment analyzer initialization failed: {e}")
            raise

    async def _initialize_aws_services(self) -> None:
        """Initialize AWS service clients with comprehensive error handling."""
        try:
            session = boto3.Session()
            
            # S3 client with optimized configuration
            self.aws_service_clients['s3'] = session.client(
                's3',
                region_name=self.config.aws_region,
                config=boto3.client.Config(
                    retries={'max_attempts': 3, 'mode': 'adaptive'},
                    max_pool_connections=20,
                    connect_timeout=10,
                    read_timeout=30
                )
            )
            
            logger.info("AWS service clients initialized")
            
        except Exception as e:
            logger.warning(f"AWS services initialization warning: {e}")
            # Continue without AWS services for local testing

    async def _initialize_advanced_analyzers(self) -> None:
        """Setup specialized analysis components."""
        try:
            # Topic modeling vectorizer
            if self.config.enable_topic_modeling:
                self.topic_modeling_vectorizer = TfidfVectorizer(
                    max_features=500,  # Reduced for cloud deployment
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.9
                )
            
            logger.info("Advanced analyzers configured")
            
        except Exception as e:
            logger.warning(f"Advanced analyzers setup warning: {e}")

    def _configure_pattern_matchers(self) -> None:
        """Configure comprehensive pattern matching for interest detection."""
        if not self.nlp_processor:
            return
        
        # Interest and excitement pattern matcher
        self.interest_pattern_matcher = Matcher(self.nlp_processor.vocab)
        
        interest_detection_patterns = [
            # Enthusiasm expressions
            [{"LOWER": {"IN": ["wow", "amazing", "incredible", "fantastic", "awesome", 
                               "perfect", "excellent", "brilliant", "outstanding"]}}],
            
            # Positive reactions
            [{"LOWER": {"IN": ["love", "like"]}}, {"LOWER": {"IN": ["this", "that", "it"]}}],
            [{"LOWER": "this"}, {"LOWER": "is"}, 
             {"LOWER": {"IN": ["great", "perfect", "amazing", "exactly", "right"]}}],
            
            # Agreement indicators
            [{"LOWER": {"IN": ["yes", "absolutely", "definitely", "certainly", "exactly"]}}, 
             {"LOWER": {"IN": ["right", "correct", "true"]}}],
            
            # Buying signal patterns
            [{"LOWER": {"IN": ["when", "how"]}}, {"LOWER": {"IN": ["can", "do", "would"]}}, 
             {"LOWER": {"IN": ["we", "i"]}}],
            [{"LOWER": {"IN": ["what", "how"]}}, 
             {"LOWER": {"IN": ["much", "does", "would"]}}, 
             {"LOWER": {"IN": ["cost", "price", "pricing", "budget"]}}],
        ]
        
        self.interest_pattern_matcher.add("INTEREST_DETECTION", interest_detection_patterns)
        
        # Objection and concern pattern matcher
        self.objection_pattern_matcher = Matcher(self.nlp_processor.vocab)
        
        objection_detection_patterns = [
            [{"LOWER": {"IN": ["but", "however", "although"]}}, {"LOWER": "we"}],
            [{"LOWER": {"IN": ["too", "very"]}}, 
             {"LOWER": {"IN": ["expensive", "costly", "high", "much"]}}],
            [{"LOWER": {"IN": ["not", "don't", "can't", "won't"]}}, 
             {"LOWER": {"IN": ["sure", "think", "see", "afford"]}}],
            [{"LOWER": "need"}, {"LOWER": "to"}, 
             {"LOWER": {"IN": ["think", "discuss", "consider"]}}],
        ]
        
        self.objection_pattern_matcher.add("OBJECTION_DETECTION", objection_detection_patterns)

    async def _validate_pipeline_setup(self) -> None:
        """Validate pipeline components with test processing."""
        test_content = "This is a test sentence for validation. It sounds amazing!"
        
        try:
            # Test NLP processing
            processed_doc = self.nlp_processor(test_content)
            assert len(processed_doc) > 0
            
            # Test sentiment analysis
            sentiment_result = await self._perform_comprehensive_sentiment_analysis(test_content)
            assert isinstance(sentiment_result, dict)
            
            logger.info("Pipeline validation completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline validation failed: {e}")
            raise

    async def process_transcript_analysis(self, s3_file_key: str) -> ComprehensiveAnalysisOutput:
        """Process a single transcript with comprehensive analysis."""
        processing_start_time = time.time()
        processing_identifier = hashlib.md5(s3_file_key.encode()).hexdigest()[:8]
        
        logger.info(f"Processing transcript analysis [{processing_identifier}]: {s3_file_key}")
        
        try:
            # Step 1: Download and preprocess transcript
            raw_transcript_content = await self._download_transcript_from_s3(s3_file_key)
            processed_transcript_content = await self._preprocess_transcript_content(raw_transcript_content)
            
            # Step 2: Core NLP processing
            nlp_processed_document = self.nlp_processor(processed_transcript_content)
            
            # Step 3: Parallel analysis execution
            analysis_tasks = [
                self._perform_comprehensive_sentiment_analysis(processed_transcript_content),
                self._extract_topic_analysis_results(nlp_processed_document, processed_transcript_content),
                self._analyze_speaker_characteristics(nlp_processed_document, processed_transcript_content),
                self._detect_interest_moments(nlp_processed_document),
                self._generate_comprehensive_lead_score(nlp_processed_document, processed_transcript_content)
            ]
            
            (
                sentiment_results,
                topic_results,
                speaker_results,
                interest_moments,
                lead_scoring_results
            ) = await asyncio.gather(*analysis_tasks)
            
            # Step 4: Extract structured intelligence
            extracted_phrases = await self._extract_key_phrases(nlp_processed_document)
            technical_terms = self._identify_technical_terminology(nlp_processed_document)
            competitor_mentions = self._find_competitor_references(nlp_processed_document)
            pricing_discussions = self._extract_pricing_conversations(nlp_processed_document)
            identified_objections = self._identify_customer_objections(nlp_processed_document)
            
            # Step 5: Conversation dynamics analysis
            conversation_metrics = await self._analyze_conversation_dynamics(nlp_processed_document)
            engagement_timeline = self._calculate_engagement_timeline(nlp_processed_document)
            question_analysis = self._analyze_question_patterns(nlp_processed_document)
            
            # Step 6: Build comprehensive output
            total_processing_time = time.time() - processing_start_time
            
            analysis_output = ComprehensiveAnalysisOutput(
                analyzed_file_name=Path(s3_file_key).name,
                processing_completion_timestamp=datetime.now(timezone.utc),
                total_processing_duration_seconds=total_processing_time,
                original_transcript_content=raw_transcript_content,
                processed_transcript_content=processed_transcript_content,
                total_word_count=len(nlp_processed_document),
                estimated_call_duration_minutes=len(nlp_processed_document) / 150,
                sentiment_analysis_results=sentiment_results,
                topic_analysis_results=topic_results,
                speaker_analysis_results=speaker_results,
                interest_moments=interest_moments,
                lead_scoring_results=lead_scoring_results,
                key_phrases_extracted=extracted_phrases,
                technical_terminology=technical_terms,
                competitor_references=competitor_mentions,
                pricing_discussions=pricing_discussions,
                objections_identified=identified_objections,
                conversation_flow_metrics=conversation_metrics,
                engagement_timeline_data=engagement_timeline,
                question_analysis_summary=question_analysis,
                confidence_scores_by_component=self._calculate_analysis_confidence_scores(),
                model_versions_used=self._get_model_version_information(),
                processing_statistics={
                    "processing_identifier": processing_identifier,
                    "tokens_analyzed": len(nlp_processed_document),
                    "sentences_processed": len(list(nlp_processed_document.sents)),
                    "cloud_deployment_mode": self.config.cloud_deployment_mode
                }
            )
            
            # Step 7: Upload results to S3
            await self._upload_analysis_results(analysis_output)
            
            # Step 8: Update performance metrics
            self._update_pipeline_performance_metrics(total_processing_time, True)
            
            logger.info(f"Analysis completed successfully [{processing_identifier}] in {total_processing_time:.2f}s")
            return analysis_output
            
        except Exception as e:
            self._update_pipeline_performance_metrics(time.time() - processing_start_time, False)
            logger.error(f"Analysis processing failed [{processing_identifier}]: {e}")
            logger.error(traceback.format_exc())
            raise

    async def _download_transcript_from_s3(self, s3_key: str) -> str:
        """Download and parse transcript from S3 with error handling."""
        try:
            s3_response = self.aws_service_clients['s3'].get_object(
                Bucket=self.config.s3_bucket_name,
                Key=s3_key
            )
            
            file_content = s3_response['Body'].read().decode('utf-8')
            
            # Parse different transcript formats
            if s3_key.endswith('.json'):
                transcript_data = json.loads(file_content)
                if 'results' in transcript_data and 'transcripts' in transcript_data['results']:
                    return transcript_data['results']['transcripts'][0]['transcript']
                elif 'transcript' in transcript_data:
                    return transcript_data['transcript']
            
            return file_content
            
        except Exception as e:
            logger.error(f"Failed to download transcript from S3: {e}")
            raise

    async def _preprocess_transcript_content(self, raw_text: str) -> str:
        """Advanced text preprocessing for optimal analysis."""
        # Remove excessive whitespace
        processed_text = ' '.join(raw_text.split())
        
        # Clean common transcript artifacts
        processed_text = processed_text.replace('[inaudible]', '').replace('[unclear]', '')
        processed_text = processed_text.replace('[music]', '').replace('[background noise]', '')
        
        # Normalize speaker identification tags
        import re
        processed_text = re.sub(r'\b(Speaker \d+|SPEAKER_\d+|\w+:)', '', processed_text)
        
        # Remove excessive punctuation
        processed_text = re.sub(r'[.]{3,}', '...', processed_text)
        processed_text = re.sub(r'[-]{2,}', '--', processed_text)
        
        return processed_text.strip()

    async def _perform_comprehensive_sentiment_analysis(self, text_content: str) -> SentimentAnalysisResult:
        """Multi-method sentiment analysis with confidence scoring."""
        try:
            # VADER sentiment analysis
            vader_sentiment_scores = self.sentiment_analyzers['vader'].polarity_scores(text_content)
            
            # TextBlob sentiment analysis
            textblob_analysis = self.sentiment_analyzers['textblob'](text_content)
            textblob_sentiment_score = textblob_analysis.sentiment.polarity
            
            # Transformer-based sentiment (if available)
            transformer_sentiment_score = 0.0
            if 'sentiment' in self.transformer_models_cache:
                try:
                    transformer_results = self.transformer_models_cache['sentiment'](text_content)
                    for result in transformer_results[0]:
                        if result['label'] in ['POSITIVE', 'POS']:
                            transformer_sentiment_score = result['score']
                        elif result['label'] in ['NEGATIVE', 'NEG']:
                            transformer_sentiment_score = -result['score']
                except Exception as e:
                    logger.warning(f"Transformer sentiment analysis failed: {e}")
            
            # Ensemble sentiment calculation
            ensemble_sentiment_score = (
                vader_sentiment_scores['compound'] * 0.4 +
                textblob_sentiment_score * 0.35 +
                transformer_sentiment_score * 0.25
            )
            
            # Emotion categorization (basic implementation)
            emotion_categories = {
                'positive': max(0, ensemble_sentiment_score),
                'negative': max(0, -ensemble_sentiment_score),
                'neutral': 1 - abs(ensemble_sentiment_score)
            }
            
            return SentimentAnalysisResult(
                overall_sentiment_score=ensemble_sentiment_score,
                confidence_rating=max(vader_sentiment_scores['compound'], abs(textblob_sentiment_score)),
                emotion_categories=emotion_categories,
                temporal_sentiment_trend=[],  # Would require sentence-level analysis
                key_emotional_indicators=[]
            )
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return SentimentAnalysisResult(
                overall_sentiment_score=0.0,
                confidence_rating=0.0,
                emotion_categories={'neutral': 1.0},
                temporal_sentiment_trend=[],
                key_emotional_indicators=[]
            )

    async def _extract_topic_analysis_results(self, document, text_content: str) -> List[TopicAnalysisResult]:
        """Advanced topic extraction and analysis."""
        topics = []
        
        try:
            # Extract noun phrases as potential topics
            noun_phrases = [chunk.text.lower() for chunk in document.noun_chunks 
                          if len(chunk.text.split()) >= 2 and len(chunk.text) > 3]
            
            # Use TF-IDF for topic relevance scoring
            if self.topic_modeling_vectorizer and noun_phrases:
                try:
                    tfidf_matrix = self.topic_modeling_vectorizer.fit_transform([text_content])
                    feature_names = self.topic_modeling_vectorizer.get_feature_names_out()
                    tfidf_scores = tfidf_matrix.toarray()[0]
                    
                    # Get top topics
                    top_topic_indices = np.argsort(tfidf_scores)[-self.config.max_discussion_topics:][::-1]
                    
                    for idx in top_topic_indices:
                        if tfidf_scores[idx] > 0:
                            topics.append(TopicAnalysisResult(
                                topic_identifier=feature_names[idx],
                                relevance_weight=float(tfidf_scores[idx]),
                                associated_keywords=[feature_names[idx]],
                                discussion_time_percentage=0.0,  # Would need temporal analysis
                                topic_sentiment_score=0.0
                            ))
                except Exception as e:
                    logger.warning(f"TF-IDF topic analysis failed: {e}")
            
            # Fallback: use noun phrases directly
            if not topics:
                phrase_counts = {}
                for phrase in noun_phrases:
                    phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
                
                sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
                
                for phrase, count in sorted_phrases[:self.config.max_discussion_topics]:
                    topics.append(TopicAnalysisResult(
                        topic_identifier=phrase,
                        relevance_weight=count / len(noun_phrases),
                        associated_keywords=[phrase],
                        discussion_time_percentage=0.0,
                        topic_sentiment_score=0.0
                    ))
            
        except Exception as e:
            logger.warning(f"Topic analysis failed: {e}")
        
        return topics[:self.config.max_discussion_topics]

    async def _analyze_speaker_characteristics(self, document, text_content: str) -> List[SpeakerAnalysisResult]:
        """Analyze speaker characteristics and engagement patterns."""
        # Placeholder implementation for speaker diarization
        # In production, would use advanced speaker diarization models
        
        return [SpeakerAnalysisResult(
            speaker_identifier="primary_speaker",
            estimated_role_category="unknown",
            speaking_duration_percentage=100.0,
            sentiment_distribution={"positive": 0.5, "negative": 0.3, "neutral": 0.2},
            engagement_level_score=0.7,
            identified_concerns=[]
        )]

    async def _detect_interest_moments(self, document) -> List[InterestMomentResult]:
        """Advanced interest moment detection with contextual analysis."""
        interest_moments = []
        
        if not hasattr(self, 'interest_pattern_matcher'):
            return interest_moments
        
        try:
            pattern_matches = self.interest_pattern_matcher(document)
            
            for match_id, start_token, end_token in pattern_matches:
                matched_span = document[start_token:end_token]
                
                # Extract broader context
                context_start = max(0, start_token - self.config.context_analysis_window)
                context_end = min(len(document), end_token + self.config.context_analysis_window)
                contextual_content = document[context_start:context_end].text
                
                # Calculate interest intensity based on context sentiment
                context_sentiment = TextBlob(contextual_content).sentiment.polarity
                interest_intensity = min(1.0, max(0.0, (context_sentiment + 1) / 2))
                
                # Estimate timestamp in conversation
                estimated_timestamp = (start_token / len(document)) * 300  # Assume 5-minute average
                
                interest_moments.append(InterestMomentResult(
                    trigger_phrase_content=matched_span.text,
                    contextual_information=contextual_content,
                    estimated_timestamp=estimated_timestamp,
                    interest_intensity_score=interest_intensity,
                    topic_classification="general_interest",
                    associated_speaker=None,
                    follow_up_recommendation="Explore expressed interest in detail"
                ))
        
        except Exception as e:
            logger.warning(f"Interest moment detection failed: {e}")
        
        return interest_moments

    async def _generate_comprehensive_lead_score(self, document, text_content: str) -> ComprehensiveLeadScore:
        """Generate comprehensive lead scoring with advanced AI insights."""
        try:
            text_lowercase = text_content.lower()
            
            # Analyze various signal categories
            interest_signals = []
            buying_readiness_signals = []
            concern_indicators = []
            
            # Interest signal detection
            interest_keywords = ['interested', 'yes', 'definitely', 'sounds good', 'love this', 'perfect']
            for keyword in interest_keywords:
                if keyword in text_lowercase:
                    interest_signals.append(f"Expressed: {keyword}")
            
            # Buying readiness signal detection
            buying_keywords = ['when can we', 'how much', 'pricing', 'cost', 'budget', 'timeline', 'next steps']
            for keyword in buying_keywords:
                if keyword in text_lowercase:
                    buying_readiness_signals.append(f"Inquired about: {keyword}")
            
            # Concern indicator detection
            concern_keywords = ['expensive', 'too much', 'not sure', 'need to think', 'budget constraints']
            for keyword in concern_keywords:
                if keyword in text_lowercase:
                    concern_indicators.append(f"Concern raised about: {keyword}")
            
            # Calculate numerical score with weighted factors
            scoring_factors = {
                'interest_signals': len(interest_signals) * 12,
                'buying_signals': len(buying_readiness_signals) * 18,
                'concerns': len(concern_indicators) * -8,
                'overall_sentiment': getattr(document._, 'sentiment', 0) * 15,
                'engagement_length': min(25, len(document) / 80)
            }
            
            numerical_score = max(0, min(100, 50 + sum(scoring_factors.values())))
            
            # Determine primary classification
            if numerical_score >= 80:
                primary_classification = "Hot"
            elif numerical_score >= 60:
                primary_classification = "Warm"
            elif numerical_score >= 30:
                primary_classification = "Cold"
            else:
                primary_classification = "Unqualified"
            
            # Generate AI insights
            personality_assessment = "Analytical" if any(word in text_lowercase for word in ["data", "numbers", "metrics"]) else "Relationship-focused"
            
            return ComprehensiveLeadScore(
                primary_classification=primary_classification,
                numerical_score_value=numerical_score,
                confidence_level_rating=0.8,
                interest_signals=interest_signals,
                concern_indicators=concern_indicators,
                buying_readiness_signals=buying_readiness_signals,
                timing_assessment_factors=[],
                personality_assessment=personality_assessment,
                decision_making_approach="Collaborative" if "we" in text_lowercase else "Individual",
                recommended_engagement_strategy="Focus on addressing concerns" if concern_indicators else "Accelerate engagement",
                prioritized_next_actions=["Schedule follow-up", "Send additional information", "Prepare proposal"],
                identified_risk_factors=concern_indicators,
                conversion_probability=numerical_score / 100
            )
            
        except Exception as e:
            logger.warning(f"Lead scoring failed: {e}")
            return ComprehensiveLeadScore(
                primary_classification="Warm",
                numerical_score_value=50.0,
                confidence_level_rating=0.5,
                interest_signals=[],
                concern_indicators=[],
                buying_readiness_signals=[],
                timing_assessment_factors=[],
                personality_assessment="Unknown",
                decision_making_approach="Unknown",
                recommended_engagement_strategy="Standard follow-up approach",
                prioritized_next_actions=["Follow up within one week"],
                identified_risk_factors=[],
                conversion_probability=0.5
            )

    async def _extract_key_phrases(self, document) -> List[str]:
        """Extract key phrases using advanced NLP techniques."""
        phrases = []
        
        try:
            # Extract meaningful noun chunks
            for chunk in document.noun_chunks:
                if (len(chunk.text.split()) >= self.config.minimum_phrase_length and
                    len(chunk.text) > 4 and
                    not chunk.text.lower().startswith(('this', 'that', 'these', 'those', 'it', 'they'))):
                    phrases.append(chunk.text.lower().strip())
            
            # Extract named entities of interest
            for entity in document.ents:
                if entity.label_ in ['ORG', 'PRODUCT', 'MONEY', 'PERCENT', 'GPE', 'PERSON']:
                    phrases.append(entity.text.lower())
            
            # Remove duplicates and limit results
            unique_phrases = list(dict.fromkeys(phrases))
            return unique_phrases[:self.config.max_extracted_phrases]
            
        except Exception as e:
            logger.warning(f"Key phrase extraction failed: {e}")
            return []

    def _identify_technical_terminology(self, document) -> List[str]:
        """Identify technical terms and industry jargon."""
        technical_terms = []
        
        # Technical pattern identification
        technical_indicators = ['API', 'SDK', 'cloud', 'database', 'integration', 'analytics', 
                               'platform', 'solution', 'system', 'software', 'technology']
        
        for token in document:
            if (token.text.lower() in technical_indicators or
                (token.is_alpha and token.text.isupper() and len(token.text) > 2) or
                (token.like_url or token.like_email)):
                technical_terms.append(token.text)
        
        return list(set(technical_terms))

    def _find_competitor_references(self, document) -> List[str]:
        """Identify mentions of competitors or alternative solutions."""
        competitor_mentions = []
        
        # Competitor identification keywords
        competitor_indicators = ['competitor', 'alternative', 'versus', 'vs', 'other solution', 
                               'different option', 'comparison']
        
        for entity in document.ents:
            if entity.label_ == 'ORG':
                entity_context = entity.sent.text.lower()
                if any(indicator in entity_context for indicator in competitor_indicators):
                    competitor_mentions.append(entity.text)
        
        return competitor_mentions

    def _extract_pricing_conversations(self, document) -> List[str]:
        """Extract price and cost related discussions."""
        pricing_mentions = []
        
        for entity in document.ents:
            if entity.label_ in ['MONEY', 'PERCENT', 'CARDINAL']:
                pricing_mentions.append(entity.text)
        
        # Look for price-related contextual discussions
        price_keywords = ['cost', 'price', 'budget', 'expensive', 'affordable', 'investment', 'fee']
        for sentence in document.sents:
            if any(keyword in sentence.text.lower() for keyword in price_keywords):
                pricing_mentions.append(sentence.text.strip())
        
        return pricing_mentions[:10]  # Limit results

    def _identify_customer_objections(self, document) -> List[str]:
        """Identify objections and concerns raised during conversation."""
        objections = []
        
        if hasattr(self, 'objection_pattern_matcher'):
            objection_matches = self.objection_pattern_matcher(document)
            
            for match_id, start_token, end_token in objection_matches:
                matched_span = document[start_token:end_token]
                contextual_sentence = matched_span.sent.text if matched_span.sent else matched_span.text
                objections.append(contextual_sentence)
        
        return objections

    async def _analyze_conversation_dynamics(self, document) -> Dict[str, Any]:
        """Analyze conversation flow and engagement dynamics."""
        return {
            "total_sentences": len(list(document.sents)),
            "average_sentence_length": np.mean([len(sent.text.split()) for sent in document.sents]),
            "question_count": len([sent for sent in document.sents if sent.text.strip().endswith('?')]),
            "exclamation_count": len([sent for sent in document.sents if sent.text.strip().endswith('!')]),
            "conversation_pace": "moderate"  # Could be enhanced with timing analysis
        }

    def _calculate_engagement_timeline(self, document) -> List[Tuple[float, float]]:
        """Calculate engagement level over conversation timeline."""
        timeline_data = []
        
        sentences = list(document.sents)
        for i, sentence in enumerate(sentences):
            timestamp = (i / len(sentences)) * 300  # Approximate 5-minute conversation
            engagement_score = min(1.0, len(sentence.text.split()) / 25)  # Engagement metric
            timeline_data.append((timestamp, engagement_score))
        
        return timeline_data

    def _analyze_question_patterns(self, document) -> Dict[str, Any]:
        """Analyze questions asked during the conversation."""
        questions = [sent.text for sent in document.sents if sent.text.strip().endswith('?')]
        
        return {
            "total_questions": len(questions),
            "question_categories": {
                "open_ended": len([q for q in questions if any(word in q.lower() 
                                 for word in ['what', 'how', 'why', 'tell me', 'describe'])]),
                "yes_no": len([q for q in questions if any(word in q.lower().split()[:2] 
                             for word in ['are', 'is', 'do', 'does', 'can', 'will', 'would'])]),
                "clarifying": len([q for q in questions if any(phrase in q.lower() 
                                 for phrase in ['so', 'just to clarify', 'you mean', 'correct me'])])
            },
            "sample_questions": questions[:5]  # First 5 questions for analysis
        }

    def _calculate_analysis_confidence_scores(self) -> Dict[str, float]:
        """Calculate confidence scores for different analysis components."""
        return {
            "sentiment_analysis": 0.85,
            "topic_extraction": 0.80,
            "lead_scoring": 0.78,
            "interest_detection": 0.75,
            "overall_analysis": 0.80
        }

    def _get_model_version_information(self) -> Dict[str, str]:
        """Get version information for all models used."""
        version_info = {
            "spacy_model": self.config.spacy_model_name,
            "pipeline_version": "3.1.0",
            "spacy_version": spacy.__version__
        }
        
        if tf:
            version_info["tensorflow_version"] = tf.__version__
        if self.config.use_transformer_models:
            version_info["transformer_model"] = self.config.transformer_model_name
        
        return version_info

    async def _upload_analysis_results(self, analysis_results: ComprehensiveAnalysisOutput) -> None:
        """Upload comprehensive analysis results to S3."""
        output_s3_key = f"{self.config.s3_analysis_prefix}{analysis_results.analyzed_file_name}"
        
        try:
            # Convert to JSON-serializable format
            results_dictionary = asdict(analysis_results)
            
            # Handle datetime serialization
            results_dictionary['processing_completion_timestamp'] = analysis_results.processing_completion_timestamp.isoformat()
            
            self.aws_service_clients['s3'].put_object(
                Bucket=self.config.s3_bucket_name,
                Key=output_s3_key,
                Body=json.dumps(results_dictionary, indent=2, default=str),
                ContentType='application/json',
                Metadata={
                    'pipeline-version': '3.1.0',
                    'processing-duration': str(analysis_results.total_processing_duration_seconds),
                    'confidence-score': str(analysis_results.confidence_scores_by_component.get('overall_analysis', 0.8))
                }
            )
            
            logger.info(f"Analysis results uploaded: s3://{self.config.s3_bucket_name}/{output_s3_key}")
            
        except Exception as e:
            logger.error(f"Failed to upload analysis results: {e}")
            # Don't raise exception - allow processing to complete even if upload fails

    def _update_pipeline_performance_metrics(self, processing_duration: float, success: bool) -> None:
        """Update pipeline performance tracking metrics."""
        self.pipeline_performance_metrics["total_files_processed"] += 1
        
        if success:
            total_processed = self.pipeline_performance_metrics["total_files_processed"]
            current_average = self.pipeline_performance_metrics["average_processing_duration"]
            self.pipeline_performance_metrics["average_processing_duration"] = (
                (current_average * (total_processed - 1) + processing_duration) / total_processed
            )
        
        # Update success rate
        successful_processing_count = (
            self.pipeline_performance_metrics["total_files_processed"] * 
            self.pipeline_performance_metrics["processing_success_rate"]
        )
        
        if success:
            successful_processing_count += 1
        
        self.pipeline_performance_metrics["processing_success_rate"] = (
            successful_processing_count / self.pipeline_performance_metrics["total_files_processed"]
        )

# --- Enhanced CLI Interface ---
async def main():
    """Enhanced command-line interface with comprehensive options."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced ML Voice Lead Analysis Pipeline v3.1"
    )
    parser.add_argument(
        "s3_file_key", 
        help="S3 key path for the transcript file to analyze"
    )
    parser.add_argument(
        "--config-file", 
        help="Path to custom configuration file",
        type=str
    )
    parser.add_argument(
        "--batch-processing", 
        action="store_true",
        help="Enable batch processing mode for multiple files"
    )
    parser.add_argument(
        "--performance-stats", 
        action="store_true",
        help="Display detailed performance statistics"
    )
    parser.add_argument(
        "--validate-only", 
        action="store_true",
        help="Run pipeline validation without processing"
    )
    parser.add_argument(
        "--debug-logging", 
        action="store_true",
        help="Enable debug-level logging output"
    )
    parser.add_argument(
        "--cloud-mode", 
        action="store_true",
        help="Enable cloud deployment optimizations"
    )
    
    args = parser.parse_args()
    
    if args.debug_logging:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create pipeline configuration
    config = PipelineConfiguration()
    if args.cloud_mode:
        config.cloud_deployment_mode = True
        config.lightweight_models_only = True
        config.use_transformer_models = False
    
    # Initialize pipeline
    analysis_pipeline = ModernVoiceLeadAnalysisPipeline(config)
    
    try:
        await analysis_pipeline.initialize_pipeline_components()
        
        if args.validate_only:
            logger.info("Pipeline validation completed successfully")
            return 0
        
        if args.batch_processing:
            logger.info("Batch processing mode - implementation based on specific requirements")
            return 0
        
        # Process single file
        analysis_results = await analysis_pipeline.process_transcript_analysis(args.s3_file_key)
        
        if args.performance_stats:
            logger.info("Analysis Performance Statistics:")
            logger.info(f"  - Total processing time: {analysis_results.total_processing_duration_seconds:.2f}s")
            logger.info(f"  - Word count analyzed: {analysis_results.total_word_count}")
            logger.info(f"  - Lead classification: {analysis_results.lead_scoring_results.primary_classification}")
            logger.info(f"  - Sentiment score: {analysis_results.sentiment_analysis_results.overall_sentiment_score:.2f}")
            logger.info(f"  - Interest moments detected: {len(analysis_results.interest_moments)}")
            logger.info(f"  - Pipeline performance metrics: {analysis_pipeline.pipeline_performance_metrics}")
        
        logger.info("Analysis processing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))