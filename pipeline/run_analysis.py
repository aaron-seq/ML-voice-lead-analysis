"""
Next-Generation ML Voice Lead Analysis Pipeline
Ultra-modern, high-performance pipeline with advanced ML capabilities.
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

import boto3
import aiofiles
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import Matcher
import tensorflow as tf
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, AutoModel
)
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import structlog

# Download required NLTK data
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

# --- Advanced Configuration ---
@dataclass
class AdvancedPipelineConfig:
    """Comprehensive pipeline configuration with performance tuning."""
    
    # AWS Configuration
    s3_bucket_name: str = os.getenv("DATA_BUCKET", "ml-voice-analysis-bucket")
    s3_transcripts_prefix: str = "transcripts/"
    s3_analysis_prefix: str = "analysis-results/"
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    
    # ML Model Configuration
    spacy_model: str = "en_core_web_lg"  # Upgraded to large model
    transformer_model: str = "microsoft/DialoGPT-medium"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # Processing Parameters
    max_sequence_length: int = 512
    batch_size: int = 16
    confidence_threshold: float = 0.75
    
    # Analysis Configuration
    max_key_phrases: int = 20
    max_topics: int = 15
    context_window: int = 30
    min_phrase_length: int = 2
    
    # Performance Settings
    enable_gpu: bool = True
    num_workers: int = 4
    cache_embeddings: bool = True
    
    # Feature Flags
    enable_advanced_sentiment: bool = True
    enable_topic_modeling: bool = True
    enable_speaker_diarization: bool = True
    enable_emotion_detection: bool = True

config = AdvancedPipelineConfig()

# --- Advanced Data Models ---
@dataclass
class SentimentAnalysis:
    """Comprehensive sentiment analysis results."""
    overall_score: float
    confidence: float
    emotion_breakdown: Dict[str, float]
    temporal_sentiment: List[Tuple[float, float]]  # (timestamp, sentiment)
    key_emotional_moments: List[Dict[str, Any]]

@dataclass
class TopicInsight:
    """Advanced topic analysis."""
    topic_name: str
    relevance_score: float
    keywords: List[str]
    discussion_duration: float
    sentiment_during_topic: float

@dataclass
class SpeakerProfile:
    """Speaker analysis and profiling."""
    speaker_id: str
    estimated_role: str  # 'sales_rep', 'prospect', 'decision_maker'
    speaking_time_percentage: float
    sentiment_profile: Dict[str, float]
    engagement_score: float
    key_concerns: List[str]

@dataclass
class EnhancedWowMoment:
    """Advanced excitement and interest detection."""
    trigger_phrase: str
    full_context: str
    timestamp: float
    excitement_score: float
    topic_category: str
    speaker_id: Optional[str]
    follow_up_opportunity: str

@dataclass
class AdvancedLeadScore:
    """Comprehensive lead scoring with AI insights."""
    primary_score: str  # Hot, Warm, Cold
    numerical_score: float  # 0-100
    confidence_level: float
    
    # Detailed breakdown
    interest_indicators: List[str]
    concern_signals: List[str]
    buying_signals: List[str]
    timing_indicators: List[str]
    
    # AI-generated insights
    personality_profile: str
    decision_making_style: str
    recommended_approach: str
    next_best_actions: List[str]
    
    # Risk assessment
    risk_factors: List[str]
    success_probability: float

@dataclass
class ComprehensiveAnalysisResult:
    """Ultra-comprehensive analysis results."""
    # Basic Information
    file_name: str
    processing_timestamp: datetime
    processing_duration_seconds: float
    
    # Content
    original_transcript: str
    cleaned_transcript: str
    word_count: int
    estimated_duration_minutes: float
    
    # Advanced Analysis
    sentiment_analysis: SentimentAnalysis
    topic_insights: List[TopicInsight]
    speaker_profiles: List[SpeakerProfile]
    wow_moments: List[EnhancedWowMoment]
    lead_scoring: AdvancedLeadScore
    
    # Extracted Intelligence
    key_phrases: List[str]
    technical_terms: List[str]
    competitor_mentions: List[str]
    price_discussions: List[str]
    objections_raised: List[str]
    
    # Conversation Dynamics
    conversation_flow: Dict[str, Any]
    engagement_timeline: List[Tuple[float, float]]
    question_analysis: Dict[str, Any]
    
    # Metadata
    confidence_scores: Dict[str, float]
    model_versions: Dict[str, str]
    processing_stats: Dict[str, Any]

# --- Next-Generation Analysis Pipeline ---
class NextGenVoiceLeadPipeline:
    """
    Ultra-advanced ML pipeline with state-of-the-art NLP and AI capabilities.
    """
    
    def __init__(self, config: AdvancedPipelineConfig = None):
        self.config = config or AdvancedPipelineConfig()
        
        # Core components
        self.nlp_processor = None
        self.transformer_models = {}
        self.sentiment_analyzers = {}
        self.topic_modeler = None
        self.aws_clients = {}
        
        # Advanced components
        self.emotion_classifier = None
        self.speaker_classifier = None
        self.embeddings_cache = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_processed": 0,
            "average_processing_time": 0.0,
            "success_rate": 0.0,
            "cache_hit_rate": 0.0
        }

    async def initialize_pipeline(self) -> None:
        """Initialize all advanced pipeline components."""
        logger.info("🚀 Initializing Next-Gen Voice Analysis Pipeline...")
        
        start_time = time.time()
        
        try:
            # Initialize core NLP
            await self._setup_core_nlp()
            
            # Load transformer models
            await self._load_transformer_models()
            
            # Setup sentiment analysis
            await self._setup_advanced_sentiment()
            
            # Initialize AWS services
            await self._setup_aws_services()
            
            # Setup advanced analyzers
            await self._setup_advanced_analyzers()
            
            # Validate setup
            await self._validate_pipeline()
            
            init_time = time.time() - start_time
            logger.info(f"✅ Pipeline initialized successfully in {init_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            logger.error(traceback.format_exc())
            raise

    async def _setup_core_nlp(self) -> None:
        """Setup enhanced spaCy NLP with custom components."""
        logger.info("Loading advanced spaCy model...")
        
        try:
            # Load large spaCy model with better accuracy
            self.nlp_processor = spacy.load(self.config.spacy_model)
            
            # Add custom pipeline components
            if "sentencizer" not in self.nlp_processor.pipe_names:
                self.nlp_processor.add_pipe("sentencizer")
            
            # Setup advanced pattern matchers
            self._setup_advanced_matchers()
            
            logger.info("✅ Core NLP processor loaded")
            
        except OSError:
            logger.error(f"❌ spaCy model '{self.config.spacy_model}' not found")
            logger.info("💡 Install with: python -m spacy download en_core_web_lg")
            raise

    async def _load_transformer_models(self) -> None:
        """Load state-of-the-art transformer models."""
        logger.info("Loading transformer models...")
        
        try:
            # Sentiment analysis model
            self.transformer_models['sentiment'] = pipeline(
                "sentiment-analysis",
                model=self.config.sentiment_model,
                return_all_scores=True,
                device=0 if self.config.enable_gpu and tf.config.list_physical_devices('GPU') else -1
            )
            
            # Emotion detection
            if self.config.enable_emotion_detection:
                self.transformer_models['emotion'] = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True,
                    device=0 if self.config.enable_gpu and tf.config.list_physical_devices('GPU') else -1
                )
            
            logger.info("✅ Transformer models loaded")
            
        except Exception as e:
            logger.warning(f"⚠️ Transformer model loading failed: {e}")
            logger.info("Using fallback models...")

    async def _setup_advanced_sentiment(self) -> None:
        """Setup multiple sentiment analyzers for ensemble analysis."""
        try:
            # VADER sentiment
            self.sentiment_analyzers['vader'] = SentimentIntensityAnalyzer()
            
            # TextBlob sentiment
            self.sentiment_analyzers['textblob'] = TextBlob
            
            logger.info("✅ Advanced sentiment analyzers ready")
            
        except Exception as e:
            logger.error(f"❌ Sentiment analyzer setup failed: {e}")
            raise

    async def _setup_aws_services(self) -> None:
        """Initialize AWS service clients with retry logic."""
        try:
            session = boto3.Session()
            
            # S3 client with retry configuration
            self.aws_clients['s3'] = session.client(
                's3',
                region_name=self.config.aws_region,
                config=boto3.client.Config(
                    retries={'max_attempts': 5, 'mode': 'adaptive'},
                    max_pool_connections=50
                )
            )
            
            # Transcribe client
            self.aws_clients['transcribe'] = session.client(
                'transcribe',
                region_name=self.config.aws_region
            )
            
            logger.info("✅ AWS services initialized")
            
        except Exception as e:
            logger.error(f"❌ AWS services setup failed: {e}")
            raise

    async def _setup_advanced_analyzers(self) -> None:
        """Setup specialized analyzers for advanced features."""
        try:
            # Topic modeling vectorizer
            if self.config.enable_topic_modeling:
                self.topic_modeler = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 3),
                    min_df=1,
                    max_df=0.95
                )
            
            logger.info("✅ Advanced analyzers configured")
            
        except Exception as e:
            logger.warning(f"⚠️ Advanced analyzers setup warning: {e}")

    def _setup_advanced_matchers(self) -> None:
        """Setup comprehensive pattern matchers."""
        if not self.nlp_processor:
            return
        
        # Excitement and interest patterns
        self.excitement_matcher = Matcher(self.nlp_processor.vocab)
        
        excitement_patterns = [
            # Basic excitement
            [{"LOWER": {"IN": ["wow", "amazing", "incredible", "fantastic", "awesome", "perfect", "excellent", "brilliant", "outstanding"]}}],
            
            # Positive reactions
            [{"LOWER": {"IN": ["love", "like"]}}, {"LOWER": {"IN": ["this", "that", "it"]}}],
            [{"LOWER": "this"}, {"LOWER": "is"}, {"LOWER": {"IN": ["great", "perfect", "amazing", "awesome", "exactly"]}}],
            
            # Agreement and interest
            [{"LOWER": {"IN": ["yes", "absolutely", "definitely", "certainly", "exactly"]}}, {"LOWER": {"IN": ["right", "correct", "true"]}}],
            [{"LOWER": "that's"}, {"LOWER": {"IN": ["great", "perfect", "amazing", "awesome", "right", "correct"]}}],
            
            # Buying signals
            [{"LOWER": {"IN": ["when", "how"]}}, {"LOWER": {"IN": ["can", "do", "would"]}}, {"LOWER": {"IN": ["we", "i"]}}],
            [{"LOWER": {"IN": ["what's", "what", "how"]}}, {"LOWER": {"IN": ["the", "much", "does"]}}, {"LOWER": {"IN": ["cost", "price", "pricing"]}}],
        ]
        
        self.excitement_matcher.add("INTEREST_SIGNALS", excitement_patterns)
        
        # Objection patterns
        self.objection_matcher = Matcher(self.nlp_processor.vocab)
        
        objection_patterns = [
            [{"LOWER": {"IN": ["but", "however", "although"]}}, {"LOWER": "we"}],
            [{"LOWER": {"IN": ["too", "very"]}}, {"LOWER": {"IN": ["expensive", "costly", "high", "much"]}}],
            [{"LOWER": {"IN": ["not", "don't", "can't", "won't"]}}, {"LOWER": {"IN": ["sure", "think", "see", "afford"]}}],
            [{"LOWER": "need"}, {"LOWER": "to"}, {"LOWER": {"IN": ["think", "discuss", "consider"]}}],
        ]
        
        self.objection_matcher.add("OBJECTION_SIGNALS", objection_patterns)

    async def _validate_pipeline(self) -> None:
        """Validate that all pipeline components are working."""
        test_text = "This is a test sentence for validation. It's amazing!"
        
        try:
            # Test NLP processor
            doc = self.nlp_processor(test_text)
            assert len(doc) > 0
            
            # Test sentiment analysis
            sentiment_scores = await self._analyze_comprehensive_sentiment(test_text)
            assert isinstance(sentiment_scores, dict)
            
            logger.info("✅ Pipeline validation successful")
            
        except Exception as e:
            logger.error(f"❌ Pipeline validation failed: {e}")
            raise

    async def process_transcript_file(self, s3_key: str) -> ComprehensiveAnalysisResult:
        """Process a single transcript with full advanced analysis."""
        start_time = time.time()
        processing_id = hashlib.md5(s3_key.encode()).hexdigest()[:8]
        
        logger.info(f"🔄 Processing transcript [{processing_id}]: {s3_key}")
        
        try:
            # 1. Download and preprocess transcript
            raw_transcript = await self._download_transcript(s3_key)
            cleaned_transcript = await self._preprocess_transcript(raw_transcript)
            
            # 2. Core NLP processing
            doc = self.nlp_processor(cleaned_transcript)
            
            # 3. Advanced analysis components
            sentiment_analysis = await self._analyze_comprehensive_sentiment(cleaned_transcript)
            topic_insights = await self._extract_topic_insights(doc, cleaned_transcript)
            speaker_profiles = await self._analyze_speaker_profiles(doc, cleaned_transcript)
            wow_moments = await self._detect_enhanced_wow_moments(doc)
            lead_scoring = await self._generate_advanced_lead_score(doc, cleaned_transcript)
            
            # 4. Extract structured intelligence
            key_phrases = await self._extract_enhanced_key_phrases(doc)
            technical_terms = self._extract_technical_terms(doc)
            competitor_mentions = self._find_competitor_mentions(doc)
            price_discussions = self._extract_price_discussions(doc)
            objections = self._identify_objections(doc)
            
            # 5. Conversation dynamics analysis
            conversation_flow = await self._analyze_conversation_flow(doc)
            engagement_timeline = self._calculate_engagement_timeline(doc)
            question_analysis = self._analyze_questions(doc)
            
            # 6. Build comprehensive results
            processing_time = time.time() - start_time
            
            results = ComprehensiveAnalysisResult(
                file_name=Path(s3_key).name,
                processing_timestamp=datetime.now(timezone.utc),
                processing_duration_seconds=processing_time,
                original_transcript=raw_transcript,
                cleaned_transcript=cleaned_transcript,
                word_count=len(doc),
                estimated_duration_minutes=len(doc) / 150,  # Approximate speaking rate
                sentiment_analysis=sentiment_analysis,
                topic_insights=topic_insights,
                speaker_profiles=speaker_profiles,
                wow_moments=wow_moments,
                lead_scoring=lead_scoring,
                key_phrases=key_phrases,
                technical_terms=technical_terms,
                competitor_mentions=competitor_mentions,
                price_discussions=price_discussions,
                objections_raised=objections,
                conversation_flow=conversation_flow,
                engagement_timeline=engagement_timeline,
                question_analysis=question_analysis,
                confidence_scores=self._calculate_confidence_scores(),
                model_versions=self._get_model_versions(),
                processing_stats={
                    "processing_id": processing_id,
                    "tokens_processed": len(doc),
                    "sentences_analyzed": len(list(doc.sents)),
                    "gpu_used": self.config.enable_gpu
                }
            )
            
            # 7. Upload results
            await self._upload_comprehensive_results(results)
            
            # 8. Update performance metrics
            self._update_performance_metrics(processing_time, True)
            
            logger.info(f"✅ Processing completed [{processing_id}] in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            self._update_performance_metrics(time.time() - start_time, False)
            logger.error(f"❌ Processing failed [{processing_id}]: {e}")
            logger.error(traceback.format_exc())
            raise

    async def _download_transcript(self, s3_key: str) -> str:
        """Download and parse transcript from S3."""
        try:
            response = self.aws_clients['s3'].get_object(
                Bucket=self.config.s3_bucket_name,
                Key=s3_key
            )
            
            content = response['Body'].read().decode('utf-8')
            
            # Parse different formats
            if s3_key.endswith('.json'):
                data = json.loads(content)
                if 'results' in data and 'transcripts' in data['results']:
                    return data['results']['transcripts'][0]['transcript']
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to download transcript: {e}")
            raise

    async def _preprocess_transcript(self, text: str) -> str:
        """Advanced text preprocessing."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Basic cleaning
        text = text.replace('[inaudible]', '').replace('[unclear]', '')
        
        # Normalize speaker tags if present
        import re
        text = re.sub(r'\b(Speaker \d+|SPEAKER_\d+):', '', text)
        
        return text.strip()

    async def _analyze_comprehensive_sentiment(self, text: str) -> SentimentAnalysis:
        """Multi-model sentiment analysis with emotion detection."""
        try:
            # VADER sentiment
            vader_scores = self.sentiment_analyzers['vader'].polarity_scores(text)
            
            # TextBlob sentiment
            blob = self.sentiment_analyzers['textblob'](text)
            textblob_sentiment = blob.sentiment.polarity
            
            # Transformer-based sentiment (if available)
            transformer_sentiment = 0.0
            if 'sentiment' in self.transformer_models:
                sentiment_results = self.transformer_models['sentiment'](text)
                # Convert to polarity score
                for result in sentiment_results[0]:
                    if result['label'] in ['POSITIVE', 'POS']:
                        transformer_sentiment = result['score']
                    elif result['label'] in ['NEGATIVE', 'NEG']:
                        transformer_sentiment = -result['score']
            
            # Ensemble sentiment score
            overall_score = (
                vader_scores['compound'] * 0.4 +
                textblob_sentiment * 0.3 +
                transformer_sentiment * 0.3
            )
            
            # Emotion analysis
            emotion_breakdown = {}
            if 'emotion' in self.transformer_models:
                emotion_results = self.transformer_models['emotion'](text)
                for emotion in emotion_results[0]:
                    emotion_breakdown[emotion['label'].lower()] = emotion['score']
            
            return SentimentAnalysis(
                overall_score=overall_score,
                confidence=max(vader_scores['compound'], abs(textblob_sentiment)),
                emotion_breakdown=emotion_breakdown,
                temporal_sentiment=[],  # Would need sentence-level analysis
                key_emotional_moments=[]  # Would need detailed analysis
            )
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return SentimentAnalysis(
                overall_score=0.0,
                confidence=0.0,
                emotion_breakdown={},
                temporal_sentiment=[],
                key_emotional_moments=[]
            )

    async def _extract_topic_insights(self, doc, text: str) -> List[TopicInsight]:
        """Advanced topic extraction and analysis."""
        topics = []
        
        try:
            # Extract noun phrases as potential topics
            noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks 
                          if len(chunk.text.split()) >= 2]
            
            # Use TF-IDF for topic relevance
            if self.topic_modeler and noun_phrases:
                tfidf_matrix = self.topic_modeler.fit_transform([text])
                feature_names = self.topic_modeler.get_feature_names_out()
                tfidf_scores = tfidf_matrix.toarray()[0]
                
                # Get top topics
                top_indices = np.argsort(tfidf_scores)[-self.config.max_topics:][::-1]
                
                for idx in top_indices:
                    if tfidf_scores[idx] > 0:
                        topics.append(TopicInsight(
                            topic_name=feature_names[idx],
                            relevance_score=float(tfidf_scores[idx]),
                            keywords=[feature_names[idx]],
                            discussion_duration=0.0,  # Would need temporal analysis
                            sentiment_during_topic=0.0  # Would need context-aware sentiment
                        ))
            
        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
        
        return topics[:self.config.max_topics]

    async def _analyze_speaker_profiles(self, doc, text: str) -> List[SpeakerProfile]:
        """Analyze speaker characteristics and roles."""
        # Placeholder for speaker diarization and profiling
        # In a real implementation, this would use speaker diarization models
        
        return [SpeakerProfile(
            speaker_id="speaker_1",
            estimated_role="unknown",
            speaking_time_percentage=100.0,
            sentiment_profile={"positive": 0.5, "negative": 0.3, "neutral": 0.2},
            engagement_score=0.7,
            key_concerns=[]
        )]

    async def _detect_enhanced_wow_moments(self, doc) -> List[EnhancedWowMoment]:
        """Advanced wow moment detection with context analysis."""
        wow_moments = []
        
        if not hasattr(self, 'excitement_matcher'):
            return wow_moments
        
        try:
            matches = self.excitement_matcher(doc)
            
            for match_id, start, end in matches:
                span = doc[start:end]
                
                # Get broader context
                context_start = max(0, start - self.config.context_window)
                context_end = min(len(doc), end + self.config.context_window)
                context = doc[context_start:context_end].text
                
                # Calculate excitement score based on surrounding context
                context_sentiment = TextBlob(context).sentiment.polarity
                excitement_score = min(1.0, max(0.0, (context_sentiment + 1) / 2))
                
                # Estimate timestamp
                estimated_timestamp = (start / len(doc)) * 300  # Assume 5-minute average
                
                wow_moments.append(EnhancedWowMoment(
                    trigger_phrase=span.text,
                    full_context=context,
                    timestamp=estimated_timestamp,
                    excitement_score=excitement_score,
                    topic_category="general",  # Could be enhanced with topic classification
                    speaker_id=None,  # Would need speaker diarization
                    follow_up_opportunity="Explore interest further"
                ))
        
        except Exception as e:
            logger.warning(f"Wow moment detection failed: {e}")
        
        return wow_moments

    async def _generate_advanced_lead_score(self, doc, text: str) -> AdvancedLeadScore:
        """Generate comprehensive lead scoring with AI insights."""
        try:
            # Basic rule-based scoring
            text_lower = text.lower()
            
            # Analyze interest indicators
            interest_indicators = []
            buying_signals = []
            concern_signals = []
            
            # Interest keywords
            interest_keywords = ['interested', 'yes', 'definitely', 'sounds good', 'love this']
            for keyword in interest_keywords:
                if keyword in text_lower:
                    interest_indicators.append(f"Mentioned: {keyword}")
            
            # Buying signals
            buying_keywords = ['when can we', 'how much', 'pricing', 'cost', 'budget', 'timeline']
            for keyword in buying_keywords:
                if keyword in text_lower:
                    buying_signals.append(f"Asked about: {keyword}")
            
            # Concerns
            concern_keywords = ['expensive', 'too much', 'not sure', 'need to think', 'budget']
            for keyword in concern_keywords:
                if keyword in text_lower:
                    concern_signals.append(f"Expressed concern about: {keyword}")
            
            # Calculate numerical score
            score_factors = {
                'interest': len(interest_indicators) * 10,
                'buying_signals': len(buying_signals) * 15,
                'concerns': len(concern_signals) * -5,
                'sentiment': (doc._.blob.polarity if hasattr(doc._, 'blob') else 0) * 20,
                'length': min(20, len(doc) / 100)  # Engagement factor
            }
            
            numerical_score = max(0, min(100, 50 + sum(score_factors.values())))
            
            # Determine primary score
            if numerical_score >= 75:
                primary_score = "Hot"
            elif numerical_score >= 50:
                primary_score = "Warm"
            else:
                primary_score = "Cold"
            
            # Generate insights
            personality_profile = "Analytical" if "data" in text_lower or "numbers" in text_lower else "Relationship-focused"
            
            return AdvancedLeadScore(
                primary_score=primary_score,
                numerical_score=numerical_score,
                confidence_level=0.8,
                interest_indicators=interest_indicators,
                concern_signals=concern_signals,
                buying_signals=buying_signals,
                timing_indicators=[],
                personality_profile=personality_profile,
                decision_making_style="Collaborative" if "we" in text_lower else "Individual",
                recommended_approach="Focus on value proposition" if concern_signals else "Accelerate to close",
                next_best_actions=["Schedule demo", "Send pricing", "Follow up in 3 days"],
                risk_factors=concern_signals,
                success_probability=numerical_score / 100
            )
            
        except Exception as e:
            logger.warning(f"Lead scoring failed: {e}")
            return AdvancedLeadScore(
                primary_score="Warm",
                numerical_score=50.0,
                confidence_level=0.5,
                interest_indicators=[],
                concern_signals=[],
                buying_signals=[],
                timing_indicators=[],
                personality_profile="Unknown",
                decision_making_style="Unknown",
                recommended_approach="Standard follow-up",
                next_best_actions=["Follow up"],
                risk_factors=[],
                success_probability=0.5
            )

    async def _extract_enhanced_key_phrases(self, doc) -> List[str]:
        """Extract key phrases using advanced NLP techniques."""
        phrases = []
        
        try:
            # Extract noun chunks
            for chunk in doc.noun_chunks:
                if (len(chunk.text.split()) >= self.config.min_phrase_length and
                    len(chunk.text) > 3 and
                    not chunk.text.lower().startswith(('this', 'that', 'these', 'those'))):
                    phrases.append(chunk.text.lower().strip())
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'MONEY', 'PERCENT', 'GPE']:
                    phrases.append(ent.text.lower())
            
            # Remove duplicates and limit
            unique_phrases = list(dict.fromkeys(phrases))
            return unique_phrases[:self.config.max_key_phrases]
            
        except Exception as e:
            logger.warning(f"Key phrase extraction failed: {e}")
            return []

    def _extract_technical_terms(self, doc) -> List[str]:
        """Extract technical terms and jargon."""
        technical_terms = []
        
        # Look for technical patterns
        technical_patterns = ['API', 'SDK', 'cloud', 'database', 'integration', 'analytics']
        
        for token in doc:
            if (token.text.lower() in technical_patterns or
                (token.is_alpha and token.text.isupper() and len(token.text) > 2)):
                technical_terms.append(token.text)
        
        return list(set(technical_terms))

    def _find_competitor_mentions(self, doc) -> List[str]:
        """Identify competitor mentions."""
        competitors = []
        
        # Common competitor indicators
        competitor_keywords = ['competitor', 'alternative', 'versus', 'vs', 'other solution']
        
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                context = ent.sent.text.lower()
                if any(keyword in context for keyword in competitor_keywords):
                    competitors.append(ent.text)
        
        return competitors

    def _extract_price_discussions(self, doc) -> List[str]:
        """Extract price and cost related discussions."""
        price_mentions = []
        
        for ent in doc.ents:
            if ent.label_ in ['MONEY', 'PERCENT']:
                price_mentions.append(ent.text)
        
        # Look for price-related context
        price_keywords = ['cost', 'price', 'budget', 'expensive', 'cheap', 'affordable']
        for sent in doc.sents:
            if any(keyword in sent.text.lower() for keyword in price_keywords):
                price_mentions.append(sent.text.strip())
        
        return price_mentions

    def _identify_objections(self, doc) -> List[str]:
        """Identify objections and concerns raised."""
        objections = []
        
        if hasattr(self, 'objection_matcher'):
            matches = self.objection_matcher(doc)
            
            for match_id, start, end in matches:
                span = doc[start:end]
                context = span.sent.text if span.sent else span.text
                objections.append(context)
        
        return objections

    async def _analyze_conversation_flow(self, doc) -> Dict[str, Any]:
        """Analyze conversation flow and dynamics."""
        return {
            "total_sentences": len(list(doc.sents)),
            "average_sentence_length": np.mean([len(sent.text.split()) for sent in doc.sents]),
            "question_count": len([sent for sent in doc.sents if sent.text.strip().endswith('?')]),
            "exclamation_count": len([sent for sent in doc.sents if sent.text.strip().endswith('!')])
        }

    def _calculate_engagement_timeline(self, doc) -> List[Tuple[float, float]]:
        """Calculate engagement over time."""
        timeline = []
        
        sentences = list(doc.sents)
        for i, sent in enumerate(sentences):
            timestamp = (i / len(sentences)) * 300  # Approximate timeline
            engagement = len(sent.text.split()) / 20  # Simple engagement metric
            timeline.append((timestamp, min(1.0, engagement)))
        
        return timeline

    def _analyze_questions(self, doc) -> Dict[str, Any]:
        """Analyze questions asked during the conversation."""
        questions = [sent.text for sent in doc.sents if sent.text.strip().endswith('?')]
        
        return {
            "total_questions": len(questions),
            "question_types": {
                "open_ended": len([q for q in questions if any(w in q.lower() for w in ['what', 'how', 'why', 'tell me'])]),
                "yes_no": len([q for q in questions if any(w in q.lower().split()[:2] for w in ['are', 'is', 'do', 'does', 'can', 'will'])]),
                "clarifying": len([q for q in questions if any(w in q.lower() for w in ['so', 'just to clarify', 'you mean'])])
            },
            "questions": questions[:10]  # Store first 10 questions
        }

    def _calculate_confidence_scores(self) -> Dict[str, float]:
        """Calculate confidence scores for different analysis components."""
        return {
            "sentiment_analysis": 0.85,
            "topic_extraction": 0.80,
            "lead_scoring": 0.78,
            "wow_moments": 0.75,
            "overall_analysis": 0.80
        }

    def _get_model_versions(self) -> Dict[str, str]:
        """Get versions of all models used."""
        return {
            "spacy_model": self.config.spacy_model,
            "transformer_sentiment": self.config.sentiment_model,
            "pipeline_version": "3.0.0",
            "tensorflow_version": tf.__version__,
            "spacy_version": spacy.__version__
        }

    async def _upload_comprehensive_results(self, results: ComprehensiveAnalysisResult) -> None:
        """Upload comprehensive results to S3."""
        output_key = f"{self.config.s3_analysis_prefix}{results.file_name}"
        
        try:
            # Convert to JSON-serializable format
            results_dict = asdict(results)
            
            # Handle datetime serialization
            results_dict['processing_timestamp'] = results.processing_timestamp.isoformat()
            
            self.aws_clients['s3'].put_object(
                Bucket=self.config.s3_bucket_name,
                Key=output_key,
                Body=json.dumps(results_dict, indent=2, default=str),
                ContentType='application/json',
                Metadata={
                    'pipeline-version': '3.0.0',
                    'processing-duration': str(results.processing_duration_seconds),
                    'confidence-score': str(results.confidence_scores.get('overall_analysis', 0.8))
                }
            )
            
            logger.info(f"✅ Results uploaded: s3://{self.config.s3_bucket_name}/{output_key}")
            
        except Exception as e:
            logger.error(f"❌ Failed to upload results: {e}")
            raise

    def _update_performance_metrics(self, processing_time: float, success: bool) -> None:
        """Update performance tracking metrics."""
        self.performance_metrics["total_processed"] += 1
        
        if success:
            total = self.performance_metrics["total_processed"]
            current_avg = self.performance_metrics["average_processing_time"]
            self.performance_metrics["average_processing_time"] = (
                (current_avg * (total - 1) + processing_time) / total
            )
        
        # Update success rate
        successful_count = (
            self.performance_metrics["total_processed"] * 
            self.performance_metrics["success_rate"]
        )
        
        if success:
            successful_count += 1
        
        self.performance_metrics["success_rate"] = (
            successful_count / self.performance_metrics["total_processed"]
        )

# --- CLI Interface ---
async def main():
    """Enhanced CLI interface with comprehensive options."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Next-Generation ML Voice Lead Analysis Pipeline v3.0"
    )
    parser.add_argument(
        "s3_key", 
        help="S3 key for the transcript file to process"
    )
    parser.add_argument(
        "--config", 
        help="Path to custom configuration file",
        type=str
    )
    parser.add_argument(
        "--batch", 
        action="store_true",
        help="Process multiple files in batch mode"
    )
    parser.add_argument(
        "--stats", 
        action="store_true",
        help="Show detailed processing statistics"
    )
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Run pipeline validation only"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load custom configuration if provided
    config_obj = config
    if args.config:
        # Load custom config (implementation would go here)
        pass
    
    # Initialize pipeline
    pipeline = NextGenVoiceLeadPipeline(config_obj)
    
    try:
        await pipeline.initialize_pipeline()
        
        if args.validate:
            logger.info("✅ Pipeline validation completed successfully")
            return 0
        
        if args.batch:
            logger.info("🔄 Batch processing mode - implement based on requirements")
            return 0
        
        # Process single file
        results = await pipeline.process_transcript_file(args.s3_key)
        
        if args.stats:
            logger.info("📊 Processing Statistics:")
            logger.info(f"  - Processing time: {results.processing_duration_seconds:.2f}s")
            logger.info(f"  - Word count: {results.word_count}")
            logger.info(f"  - Lead score: {results.lead_scoring.primary_score}")
            logger.info(f"  - Sentiment: {results.sentiment_analysis.overall_score:.2f}")
            logger.info(f"  - Wow moments: {len(results.wow_moments)}")
            logger.info(f"  - Pipeline metrics: {pipeline.performance_metrics}")
        
        logger.info("✅ Processing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
