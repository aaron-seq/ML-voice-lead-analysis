"""Feature Extraction for Audio and Text Analysis.

Provides utilities for extracting ML features from:
- Audio files and transcripts
- Text sentiment and linguistic features
- Conversation patterns and engagement metrics
"""

import logging
from typing import Dict, Any, List, Optional
import re

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """Extract features from audio transcripts and conversation data."""
    
    def __init__(self):
        self.positive_indicators = [
            'interested', 'sounds good', 'perfect', 'great', 'excellent',
            'love it', 'amazing', 'fantastic', 'definitely', 'absolutely'
        ]
        self.question_indicators = ['what', 'when', 'where', 'why', 'how', 'can', 'could', 'would']
    
    def extract_text_features(self, transcript: str) -> Dict[str, Any]:
        """Extract linguistic features from transcript text.
        
        Args:
            transcript: Full conversation transcript
            
        Returns:
            Dictionary of extracted features
        """
        sentences = [s.strip() for s in transcript.split('.') if s.strip()]
        words = transcript.lower().split()
        
        features = {
            'total_word_count': len(words),
            'unique_word_count': len(set(words)),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'question_count': sum(1 for s in sentences if '?' in s),
            'exclamation_count': sum(1 for s in sentences if '!' in s),
        }
        
        # Positive indicators
        positive_count = sum(1 for word in words if any(pos in word for pos in self.positive_indicators))
        features['positive_word_ratio'] = positive_count / max(len(words), 1)
        
        # Question indicators
        question_word_count = sum(1 for word in words if word in self.question_indicators)
        features['question_word_ratio'] = question_word_count / max(len(words), 1)
        
        return features
    
    def extract_conversation_features(self, analysis_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract conversation-level features.
        
        Args:
            analysis_data: Full analysis results with metadata
            
        Returns:
            Dictionary of conversation features
        """
        metadata = analysis_data.get('metadata', {})
        
        features = {
            'call_duration_minutes': metadata.get('call_duration_minutes', 15.0),
            'speaker_changes': metadata.get('speaker_changes', 20),
            'engagement_score': metadata.get('engagement_score', 5.0),
            'response_time_avg': metadata.get('avg_response_time', 5.0),
        }
        
        # Interest signals from wow moments
        features['interest_signals'] = len(analysis_data.get('wowMoments', []))
        
        # Technical terms from topics
        features['technical_terms_count'] = len(analysis_data.get('topics', []))
        
        return features
    
    def extract_all_features(self, analysis_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract complete feature set for ML model.
        
        Args:
            analysis_data: Complete voice analysis results
            
        Returns:
            Dictionary with all extracted features
        """
        features = {}
        
        # Sentiment score
        features['sentiment_score'] = analysis_data.get('sentiment', 0.0)
        
        # Text features from transcript
        transcript = analysis_data.get('transcript', '')
        if transcript:
            text_features = self.extract_text_features(transcript)
            features.update(text_features)
        
        # Conversation features
        conv_features = self.extract_conversation_features(analysis_data)
        features.update(conv_features)
        
        return features
