"""ML Model Inference Engine for Voice Lead Analysis.

Provides production-ready inference capabilities with:
- Model loading and caching
- Feature extraction and preprocessing  
- Prediction with confidence scores
- Error handling and validation
"""

import pickle
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for model prediction results."""
    classification: str
    confidence: float
    class_probabilities: Dict[str, float]
    feature_contributions: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelNotFoundError(Exception):
    """Raised when model files cannot be found."""
    pass


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class InferenceError(Exception):
    """Raised when inference fails."""
    pass


class ModelInference:
    """Handles ML model inference for voice analysis."""
    
    def __init__(self, model_dir: str = "backend/models", version: str = "v1"):
        self.model_dir = Path(model_dir)
        self.version = version
        
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        self.metadata = {}
        
        self.is_loaded = False
    
    def load_models(self) -> bool:
        """Load trained models and artifacts from disk.
        
        Returns:
            True if models loaded successfully
            
        Raises:
            ModelNotFoundError: If model files don't exist
            ModelLoadError: If loading fails
        """
        logger.info(f"Loading models (version: {self.version})")
        
        # Check if model directory exists
        if not self.model_dir.exists():
            raise ModelNotFoundError(
                f"Model directory not found: {self.model_dir}. "
                "Please train models first using 'python -m ml.train'"
            )
        
        # Define model file paths
        model_path = self.model_dir / f"lead_classifier_{self.version}.pkl"
        scaler_path = self.model_dir / f"feature_scaler_{self.version}.pkl"
        encoder_path = self.model_dir / f"label_encoder_{self.version}.pkl"
        metadata_path = self.model_dir / f"model_metadata_{self.version}.json"
        
        # Check if files exist
        missing_files = []
        for path in [model_path, scaler_path, encoder_path]:
            if not path.exists():
                missing_files.append(str(path))
        
        if missing_files:
            raise ModelNotFoundError(
                f"Missing model files: {', '.join(missing_files)}. "
                "Please train models first using 'python -m ml.train'"
            )
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Loaded model from {model_path}")
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Loaded scaler from {scaler_path}")
            
            # Load label encoder
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            logger.info(f"Loaded encoder from {encoder_path}")
            
            # Load metadata if available
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    self.feature_names = self.metadata.get('feature_names', [])
                logger.info(f"Loaded metadata from {metadata_path}")
            
            self.is_loaded = True
            logger.info("All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise ModelLoadError(f"Model loading failed: {str(e)}")
    
    def extract_features(self, analysis_data: Dict[str, Any]) -> pd.DataFrame:
        """Extract features from voice analysis data.
        
        Args:
            analysis_data: Dictionary containing voice analysis results
            
        Returns:
            DataFrame with extracted features
        """
        # Extract features from analysis data
        features = {
            'sentiment_score': analysis_data.get('sentiment', 0.0),
            'avg_sentence_length': analysis_data.get('metadata', {}).get('avg_sentence_length', 15.0),
            'question_count': len([k for k in analysis_data.get('keywords', []) if '?' in k]),
            'positive_word_ratio': analysis_data.get('metadata', {}).get('positive_ratio', 0.0),
            'engagement_score': analysis_data.get('metadata', {}).get('engagement', 5.0),
            'response_time_avg': analysis_data.get('metadata', {}).get('response_time', 5.0),
            'technical_terms_count': len(analysis_data.get('topics', [])),
            'call_duration_minutes': analysis_data.get('metadata', {}).get('call_duration_minutes', 15.0),
            'speaker_changes': analysis_data.get('metadata', {}).get('speaker_changes', 20),
            'interest_signals': len(analysis_data.get('wowMoments', [])),
        }
        
        return pd.DataFrame([features])
    
    def predict(self, features: pd.DataFrame) -> PredictionResult:
        """Make prediction on input features.
        
        Args:
            features: DataFrame with feature values
            
        Returns:
            PredictionResult with classification and confidence
            
        Raises:
            InferenceError: If prediction fails
        """
        if not self.is_loaded:
            raise InferenceError("Models not loaded. Call load_models() first.")
        
        try:
            # Ensure features are in correct order
            if self.feature_names:
                features = features[self.feature_names]
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction_encoded = self.model.predict(features_scaled)[0]
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            
            # Decode prediction
            classification = self.label_encoder.inverse_transform([prediction_encoded])[0]
            confidence = float(max(prediction_proba))
            
            # Get class probabilities
            class_probabilities = dict(zip(
                self.label_encoder.classes_,
                [float(p) for p in prediction_proba]
            ))
            
            # Calculate feature contributions if available
            feature_contributions = None
            if hasattr(self.model, 'feature_importances_'):
                feature_contributions = dict(zip(
                    self.feature_names,
                    [float(imp) for imp in self.model.feature_importances_]
                ))
            
            result = PredictionResult(
                classification=classification,
                confidence=confidence,
                class_probabilities=class_probabilities,
                feature_contributions=feature_contributions,
                metadata={'model_version': self.version}
            )
            
            logger.info(f"Prediction: {classification} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise InferenceError(f"Prediction failed: {str(e)}")
    
    def predict_from_analysis(self, analysis_data: Dict[str, Any]) -> PredictionResult:
        """End-to-end prediction from voice analysis data.
        
        Args:
            analysis_data: Complete voice analysis results
            
        Returns:
            PredictionResult with lead classification
        """
        if not self.is_loaded:
            self.load_models()
        
        # Extract features
        features = self.extract_features(analysis_data)
        
        # Make prediction
        return self.predict(features)


def create_mock_models(model_dir: str = "backend/models", version: str = "v1"):
    """Create mock/placeholder models for development and testing.
    
    Args:
        model_dir: Directory to save mock models
        version: Version identifier
    """
    from ml.train import ModelTrainer
    
    logger.info("Creating mock models for development...")
    
    trainer = ModelTrainer(model_dir=model_dir)
    X, y = trainer.create_synthetic_training_data(n_samples=500)
    trainer.train_model(X, y)
    trainer.save_models(version=version)
    
    logger.info("Mock models created successfully")


if __name__ == "__main__":
    # Create mock models if they don't exist
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    create_mock_models()
