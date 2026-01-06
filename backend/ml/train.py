"""ML Model Training Pipeline for Voice Lead Analysis.

This module provides comprehensive training functionality for:
- Lead scoring classification models
- Sentiment analysis models  
- Feature extraction and preprocessing
- Model evaluation and metrics tracking
"""

import pickle
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles training and evaluation of ML models for voice analysis."""
    
    def __init__(self, model_dir: str = "backend/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        self.training_metrics = {}
        
    def create_synthetic_training_data(self, n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic training data for initial model development.
        
        Args:
            n_samples: Number of training samples to generate
            
        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        logger.info(f"Generating {n_samples} synthetic training samples")
        
        np.random.seed(42)
        
        # Generate synthetic features
        features = {
            'sentiment_score': np.random.uniform(-1, 1, n_samples),
            'avg_sentence_length': np.random.uniform(5, 30, n_samples),
            'question_count': np.random.randint(0, 15, n_samples),
            'positive_word_ratio': np.random.uniform(0, 0.5, n_samples),
            'engagement_score': np.random.uniform(0, 10, n_samples),
            'response_time_avg': np.random.uniform(1, 10, n_samples),
            'technical_terms_count': np.random.randint(0, 20, n_samples),
            'call_duration_minutes': np.random.uniform(5, 60, n_samples),
            'speaker_changes': np.random.randint(10, 100, n_samples),
            'interest_signals': np.random.randint(0, 10, n_samples),
        }
        
        df = pd.DataFrame(features)
        
        # Generate labels based on feature combinations (synthetic logic)
        labels = []
        for idx, row in df.iterrows():
            score = (
                row['sentiment_score'] * 0.3 +
                row['positive_word_ratio'] * 0.2 +
                row['engagement_score'] / 10 * 0.3 +
                row['interest_signals'] / 10 * 0.2
            )
            
            if score > 0.6:
                labels.append('Hot')
            elif score > 0.3:
                labels.append('Warm')
            else:
                labels.append('Cold')
        
        y = pd.Series(labels, name='lead_classification')
        
        logger.info(f"Generated data shape: {df.shape}")
        logger.info(f"Label distribution: {y.value_counts().to_dict()}")
        
        return df, y
    
    def train_model(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        model_type: str = 'random_forest',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train ML model for lead classification.
        
        Args:
            X: Feature DataFrame
            y: Target labels
            model_type: Type of model ('random_forest' or 'gradient_boosting')
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing training metrics and evaluation results
        """
        logger.info(f"Training {model_type} model with {len(X)} samples")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Initialize scaler and encode labels
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        logger.info("Training model...")
        self.model.fit(X_train_scaled, y_train_encoded)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'model_type': model_type,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(self.feature_names),
            'classes': self.label_encoder.classes_.tolist(),
            
            'train_accuracy': accuracy_score(y_train_encoded, y_pred_train),
            'test_accuracy': accuracy_score(y_test_encoded, y_pred_test),
            
            'precision': precision_score(y_test_encoded, y_pred_test, average='weighted'),
            'recall': recall_score(y_test_encoded, y_pred_test, average='weighted'),
            'f1_score': f1_score(y_test_encoded, y_pred_test, average='weighted'),
            
            'confusion_matrix': confusion_matrix(y_test_encoded, y_pred_test).tolist(),
            'classification_report': classification_report(
                y_test_encoded, y_pred_test, 
                target_names=self.label_encoder.classes_,
                output_dict=True
            ),
            
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train_encoded, cv=5)
        metrics['cv_scores'] = cv_scores.tolist()
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_.tolist()
            ))
            metrics['feature_importance'] = feature_importance
        
        self.training_metrics = metrics
        
        logger.info(f"Training completed - Test Accuracy: {metrics['test_accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def save_models(self, version: str = "v1") -> Dict[str, str]:
        """Save trained models and artifacts to disk.
        
        Args:
            version: Version identifier for the models
            
        Returns:
            Dictionary with paths to saved files
        """
        if self.model is None:
            raise ValueError("No trained model to save. Train a model first.")
        
        logger.info(f"Saving models (version: {version})")
        
        paths = {}
        
        # Save main model
        model_path = self.model_dir / f"lead_classifier_{version}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        paths['model'] = str(model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save scaler
        scaler_path = self.model_dir / f"feature_scaler_{version}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        paths['scaler'] = str(scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save label encoder
        encoder_path = self.model_dir / f"label_encoder_{version}.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        paths['encoder'] = str(encoder_path)
        logger.info(f"Saved encoder to {encoder_path}")
        
        # Save model metadata
        metadata = {
            'version': version,
            'feature_names': self.feature_names,
            'classes': self.label_encoder.classes_.tolist() if self.label_encoder else [],
            'training_metrics': self.training_metrics,
            'created_at': datetime.utcnow().isoformat()
        }
        
        metadata_path = self.model_dir / f"model_metadata_{version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        paths['metadata'] = str(metadata_path)
        logger.info(f"Saved metadata to {metadata_path}")
        
        return paths
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model on new test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("No trained model available")
        
        logger.info(f"Evaluating model on {len(X_test)} test samples")
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Metrics
        eval_metrics = {
            'accuracy': accuracy_score(y_test_encoded, y_pred),
            'precision': precision_score(y_test_encoded, y_pred, average='weighted'),
            'recall': recall_score(y_test_encoded, y_pred, average='weighted'),
            'f1_score': f1_score(y_test_encoded, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test_encoded, y_pred).tolist(),
            'sample_predictions': [
                {
                    'true_label': self.label_encoder.inverse_transform([y_test_encoded[i]])[0],
                    'predicted_label': self.label_encoder.inverse_transform([y_pred[i]])[0],
                    'confidence': float(max(y_pred_proba[i]))
                }
                for i in range(min(10, len(y_test)))  # First 10 samples
            ]
        }
        
        logger.info(f"Evaluation completed - Accuracy: {eval_metrics['accuracy']:.4f}")
        
        return eval_metrics


def main():
    """Main training script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("ML Voice Lead Analysis - Model Training Pipeline")
    logger.info("=" * 60)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Generate synthetic training data
    logger.info("\nStep 1: Generating training data")
    X, y = trainer.create_synthetic_training_data(n_samples=2000)
    
    # Train model
    logger.info("\nStep 2: Training model")
    metrics = trainer.train_model(X, y, model_type='random_forest')
    
    # Display results
    logger.info("\nStep 3: Training Results")
    logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"Cross-validation Mean: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
    
    # Save models
    logger.info("\nStep 4: Saving models")
    saved_paths = trainer.save_models(version="v1")
    logger.info("Models saved successfully:")
    for key, path in saved_paths.items():
        logger.info(f"  {key}: {path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
