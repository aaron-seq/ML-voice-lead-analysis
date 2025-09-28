"""
Enhanced ML Voice Lead Analysis Pipeline
Modern, efficient pipeline for processing sales call transcripts.
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import time

import boto3
import spacy
from spacy.matcher import Matcher
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from spacy_textblob.spacytextblob import SpacyTextBlob
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuration ---
@dataclass
class PipelineConfig:
    """Pipeline configuration settings."""
    s3_bucket_name: str = os.getenv("DATA_BUCKET", "ml-voice-analysis-bucket")
    model_path: str = "models/lead_score_model.h5"
    tokenizer_path: str = "models/tokenizer.json"
    max_sequence_length: int = 200
    spacy_model: str = "en_core_web_sm"
    
    # Performance settings
    batch_size: int = 32
    confidence_threshold: float = 0.7
    
    # Analysis settings
    max_key_phrases: int = 15
    max_topics: int = 10
    context_window: int = 20  # Words around wow moments

config = PipelineConfig()

# --- Data Models ---
@dataclass
class WowMoment:
    """Represents a high-interest moment in the conversation."""
    keyword: str
    context: str
    timestamp: Optional[float] = None
    confidence: float = 1.0

@dataclass
class LeadScore:
    """Lead scoring results."""
    score: str
    confidence: float
    reasoning: str = ""

@dataclass
class AnalysisResult:
    """Complete analysis results."""
    fileName: str
    transcript: str
    sentiment: float
    keywords: List[str]
    topics: List[str]
    wowMoments: List[Dict[str, Any]]
    leadScore: Dict[str, Any]
    processingTime: float
    metadata: Dict[str, Any]

# --- Enhanced ML Pipeline ---
class VoiceLeadAnalysisPipeline:
    """
    Enhanced ML pipeline for voice lead analysis with improved performance 
    and modern practices.
    """
    
    def __init__(self):
        self.config = config
        self.nlp_processor = None
        self.lead_scoring_model = None
        self.text_tokenizer = None
        self.wow_moment_matcher = None
        self.s3_client = None
        
        # Performance tracking
        self.processing_stats = {
            "total_processed": 0,
            "average_processing_time": 0,
            "success_rate": 0
        }
    
    async def initialize_pipeline(self) -> None:
        """Initialize all pipeline components asynchronously."""
        logger.info("🔄 Initializing ML Voice Analysis Pipeline...")
        
        try:
            # Initialize NLP components
            await self._setup_nlp_processor()
            
            # Load ML models
            await self._load_ml_models()
            
            # Setup AWS services
            self._setup_aws_services()
            
            # Setup pattern matchers
            self._setup_pattern_matchers()
            
            logger.info("✅ Pipeline initialization completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            raise
    
    async def _setup_nlp_processor(self) -> None:
        """Setup spaCy NLP processor with required components."""
        logger.info("Loading spaCy NLP model...")
        
        try:
            self.nlp_processor = spacy.load(self.config.spacy_model)
            
            # Add sentiment analysis pipeline component
            if 'spacytextblob' not in self.nlp_processor.pipe_names:
                self.nlp_processor.add_pipe('spacytextblob')
            
            logger.info("✅ NLP processor loaded successfully")
            
        except OSError:
            logger.error(f"❌ spaCy model '{self.config.spacy_model}' not found")
            logger.info("💡 Install with: python -m spacy download en_core_web_sm")
            raise
    
    async def _load_ml_models(self) -> None:
        """Load TensorFlow models and tokenizer."""
        logger.info("Loading ML models...")
        
        try:
            # Load lead scoring model
            if Path(self.config.model_path).exists():
                self.lead_scoring_model = tf.keras.models.load_model(
                    self.config.model_path
                )
                logger.info("✅ Lead scoring model loaded")
            else:
                logger.warning(f"⚠️ Model file not found: {self.config.model_path}")
                self.lead_scoring_model = None
            
            # Load tokenizer
            if Path(self.config.tokenizer_path).exists():
                with open(self.config.tokenizer_path, 'r') as f:
                    tokenizer_data = json.load(f)
                    self.text_tokenizer = tokenizer_from_json(tokenizer_data)
                logger.info("✅ Text tokenizer loaded")
            else:
                logger.warning(f"⚠️ Tokenizer file not found: {self.config.tokenizer_path}")
                self.text_tokenizer = None
                
        except Exception as e:
            logger.error(f"❌ Failed to load ML models: {e}")
            raise
    
    def _setup_aws_services(self) -> None:
        """Initialize AWS service clients."""
        try:
            self.s3_client = boto3.client('s3')
            logger.info("✅ AWS S3 client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize AWS services: {e}")
            raise
    
    def _setup_pattern_matchers(self) -> None:
        """Setup pattern matching for wow moments."""
        if not self.nlp_processor:
            return
        
        self.wow_moment_matcher = Matcher(self.nlp_processor.vocab)
        
        # Enhanced patterns for excitement detection
        excitement_patterns = [
            [{"LOWER": {"IN": ["wow", "amazing", "incredible", "fantastic", "awesome", "perfect", "excellent"]}}],
            [{"LOWER": "that's"}, {"LOWER": {"IN": ["great", "perfect", "amazing", "awesome"]}}],
            [{"LOWER": {"IN": ["love", "like"]}}, {"LOWER": "this"}],
            [{"LOWER": "this"}, {"LOWER": "is"}, {"LOWER": {"IN": ["great", "perfect", "amazing"]}}]
        ]
        
        self.wow_moment_matcher.add("HIGH_INTEREST", excitement_patterns)
        logger.info("✅ Pattern matchers configured")
    
    async def download_transcript_from_s3(self, s3_key: str) -> str:
        """Download and parse transcript from S3."""
        logger.info(f"📥 Downloading transcript: s3://{self.config.s3_bucket_name}/{s3_key}")
        
        try:
            response = self.s3_client.get_object(
                Bucket=self.config.s3_bucket_name,
                Key=s3_key
            )
            
            content = response['Body'].read().decode('utf-8')
            
            # Parse AWS Transcribe JSON format
            if s3_key.endswith('.json'):
                data = json.loads(content)
                if 'results' in data and 'transcripts' in data['results']:
                    return data['results']['transcripts'][0]['transcript']
            
            return content
            
        except Exception as e:
            logger.error(f"❌ Failed to download transcript: {e}")
            raise
    
    async def analyze_sentiment_advanced(self, doc) -> float:
        """Enhanced sentiment analysis with context awareness."""
        # Basic sentiment from TextBlob
        base_sentiment = doc._.blob.polarity
        
        # Context-aware adjustments
        question_count = len([sent for sent in doc.sents if sent.text.strip().endswith('?')])
        exclamation_count = len([sent for sent in doc.sents if sent.text.strip().endswith('!')])
        
        # Adjust sentiment based on conversation dynamics
        if question_count > len(list(doc.sents)) * 0.3:  # Lots of questions
            base_sentiment *= 0.9  # Slightly reduce positive sentiment
        
        if exclamation_count > 0:
            base_sentiment += 0.1  # Boost for enthusiasm
        
        return max(-1.0, min(1.0, base_sentiment))
    
    async def extract_enhanced_key_phrases(self, doc) -> Tuple[List[str], List[str]]:
        """Extract key phrases and topics with improved filtering."""
        # Extract noun chunks as key phrases
        key_phrases = []
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip().lower()
            if (len(phrase.split()) > 1 and 
                len(phrase) > 3 and 
                not phrase.startswith(('this', 'that', 'these', 'those'))):
                key_phrases.append(phrase)
        
        # Extract important single entities and topics
        topics = []
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'MONEY', 'PERCENT']:
                topics.append(ent.text.lower())
        
        # Add important nouns
        important_nouns = [
            token.lemma_.lower() for token in doc 
            if (token.pos_ == 'NOUN' and 
                len(token.text) > 3 and 
                not token.is_stop and 
                token.is_alpha)
        ]
        
        topics.extend(important_nouns)
        
        # Remove duplicates and limit results
        key_phrases = list(dict.fromkeys(key_phrases))[:self.config.max_key_phrases]
        topics = list(dict.fromkeys(topics))[:self.config.max_topics]
        
        return key_phrases, topics
    
    async def find_wow_moments_enhanced(self, doc) -> List[WowMoment]:
        """Enhanced wow moment detection with context and timing."""
        if not self.wow_moment_matcher:
            return []
        
        matches = self.wow_moment_matcher(doc)
        wow_moments = []
        
        for match_id, start, end in matches:
            span = doc[start:end]
            
            # Get broader context
            context_start = max(0, start - self.config.context_window)
            context_end = min(len(doc), end + self.config.context_window)
            context = doc[context_start:context_end].text.replace('\n', ' ')
            
            # Calculate approximate timestamp (if transcript had timing)
            estimated_timestamp = (start / len(doc)) * 300  # Assume 5-min average call
            
            wow_moment = WowMoment(
                keyword=span.text,
                context=f"...{context}...",
                timestamp=estimated_timestamp,
                confidence=0.8  # Could be improved with more sophisticated detection
            )
            
            wow_moments.append(wow_moment)
        
        return wow_moments
    
    async def predict_lead_score_enhanced(self, text: str, doc) -> LeadScore:
        """Enhanced lead scoring with reasoning."""
        if not self.lead_scoring_model or not self.text_tokenizer:
            # Fallback rule-based scoring
            return self._rule_based_scoring(text, doc)
        
        try:
            # Prepare text for model
            sequences = self.text_tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequences, maxlen=self.config.max_sequence_length)
            
            # Get prediction
            prediction = self.lead_scoring_model.predict(padded, verbose=0)[0]
            
            # Map to categories
            score_categories = ['Hot', 'Warm', 'Cold']
            predicted_idx = prediction.argmax()
            confidence = float(prediction[predicted_idx])
            
            # Generate reasoning
            reasoning = self._generate_scoring_reasoning(text, doc, predicted_idx, confidence)
            
            return LeadScore(
                score=score_categories[predicted_idx],
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.warning(f"ML scoring failed, using fallback: {e}")
            return self._rule_based_scoring(text, doc)
    
    def _rule_based_scoring(self, text: str, doc) -> LeadScore:
        """Fallback rule-based lead scoring."""
        text_lower = text.lower()
        
        # Positive indicators
        hot_keywords = ['interested', 'yes', 'definitely', 'when can we', 'how much', 'pricing']
        warm_keywords = ['maybe', 'possibly', 'thinking about', 'considering', 'let me check']
        cold_keywords = ['not interested', 'no thanks', 'too expensive', 'not now', 'call back later']
        
        hot_score = sum(1 for keyword in hot_keywords if keyword in text_lower)
        warm_score = sum(1 for keyword in warm_keywords if keyword in text_lower)
        cold_score = sum(1 for keyword in cold_keywords if keyword in text_lower)
        
        if hot_score >= 2:
            return LeadScore(score="Hot", confidence=0.75, reasoning="Multiple positive indicators found")
        elif cold_score >= 1:
            return LeadScore(score="Cold", confidence=0.7, reasoning="Negative indicators detected")
        else:
            return LeadScore(score="Warm", confidence=0.6, reasoning="Mixed or neutral indicators")
    
    def _generate_scoring_reasoning(self, text: str, doc, predicted_idx: int, confidence: float) -> str:
        """Generate human-readable reasoning for the score."""
        categories = ['Hot', 'Warm', 'Cold']
        category = categories[predicted_idx]
        
        reasoning_parts = [f"Model confidence: {confidence:.2%}"]
        
        # Add sentiment context
        sentiment = doc._.blob.polarity
        if sentiment > 0.2:
            reasoning_parts.append("Positive conversation tone")
        elif sentiment < -0.2:
            reasoning_parts.append("Negative conversation tone")
        
        # Add length context
        word_count = len(doc)
        if word_count > 500:
            reasoning_parts.append("Extended conversation suggests engagement")
        
        return "; ".join(reasoning_parts)
    
    async def upload_results_to_s3(self, results: AnalysisResult) -> None:
        """Upload analysis results to S3."""
        output_key = f"analysis-results/{results.fileName}"
        
        try:
            self.s3_client.put_object(
                Bucket=self.config.s3_bucket_name,
                Key=output_key,
                Body=json.dumps(asdict(results), indent=2, default=str),
                ContentType='application/json'
            )
            
            logger.info(f"✅ Results uploaded: s3://{self.config.s3_bucket_name}/{output_key}")
            
        except Exception as e:
            logger.error(f"❌ Failed to upload results: {e}")
            raise
    
    async def process_single_transcript(self, s3_key: str) -> AnalysisResult:
        """Process a single transcript through the complete pipeline."""
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Processing transcript: {s3_key}")
            
            # 1. Download transcript
            transcript_text = await self.download_transcript_from_s3(s3_key)
            
            # 2. Process with NLP
            doc = self.nlp_processor(transcript_text)
            
            # 3. Run analysis components
            sentiment = await self.analyze_sentiment_advanced(doc)
            key_phrases, topics = await self.extract_enhanced_key_phrases(doc)
            wow_moments = await self.find_wow_moments_enhanced(doc)
            lead_score = await self.predict_lead_score_enhanced(transcript_text, doc)
            
            # 4. Calculate processing time
            processing_time = time.time() - start_time
            
            # 5. Create results object
            results = AnalysisResult(
                fileName=Path(s3_key).name,
                transcript=transcript_text,
                sentiment=sentiment,
                keywords=key_phrases,
                topics=topics,
                wowMoments=[asdict(moment) for moment in wow_moments],
                leadScore=asdict(lead_score),
                processingTime=processing_time,
                metadata={
                    "word_count": len(doc),
                    "sentence_count": len(list(doc.sents)),
                    "processed_at": time.time()
                }
            )
            
            # 6. Upload results
            await self.upload_results_to_s3(results)
            
            # 7. Update stats
            self._update_processing_stats(processing_time, True)
            
            logger.info(f"✅ Processing completed in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            self._update_processing_stats(time.time() - start_time, False)
            logger.error(f"❌ Processing failed for {s3_key}: {e}")
            raise
    
    def _update_processing_stats(self, processing_time: float, success: bool) -> None:
        """Update internal processing statistics."""
        self.processing_stats["total_processed"] += 1
        
        if success:
            current_avg = self.processing_stats["average_processing_time"]
            total = self.processing_stats["total_processed"]
            self.processing_stats["average_processing_time"] = (
                (current_avg * (total - 1) + processing_time) / total
            )
        
        success_count = self.processing_stats["total_processed"] * self.processing_stats["success_rate"]
        if success:
            success_count += 1
        
        self.processing_stats["success_rate"] = success_count / self.processing_stats["total_processed"]

# --- CLI Interface ---
async def main():
    """Main CLI interface for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced ML Voice Lead Analysis Pipeline")
    parser.add_argument("s3_key", help="S3 key for the transcript file")
    parser.add_argument("--batch", action="store_true", help="Process multiple files")
    parser.add_argument("--stats", action="store_true", help="Show processing statistics")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VoiceLeadAnalysisPipeline()
    await pipeline.initialize_pipeline()
    
    try:
        if args.batch:
            # Process multiple files (implement based on your needs)
            logger.info("Batch processing not yet implemented")
        else:
            # Process single file
            await pipeline.process_single_transcript(args.s3_key)
        
        if args.stats:
            logger.info(f"Processing stats: {pipeline.processing_stats}")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
