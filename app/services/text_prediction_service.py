"""
Service for predicting sentiment from raw text.

This service processes raw text through NLP pipeline and uses a trained model to predict sentiment.
"""
from typing import Dict, Any
import pandas as pd
import logging
from pathlib import Path
import joblib
from textblob import TextBlob

from app.services.text_normalization_service import TextNormalizationService


class TextPredictionService:
    """Service for end-to-end text sentiment prediction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.normalization_service = TextNormalizationService()
    
    def predict_from_text(self, model_name: str, text: str) -> Dict[str, Any]:
        """
        Predict sentiment from raw text.
        
        Args:
            model_name: Name of the trained model
            text: Raw text to analyze
            
        Returns:
            Dictionary with prediction, probabilities, and confidence
        """
        self.logger.info(f"📝 Text Prediction - Processing text: '{text[:50]}...'")
        
        # 1. Load model metadata
        # Use absolute path relative to backend directory
        backend_dir = Path(__file__).parent.parent.parent
        model_dir = backend_dir / "data" / "models"
        metadata_path = model_dir / f"{model_name}_metadata.json"
        
        if not metadata_path.exists():
            raise ValueError(f"Model '{model_name}' not found")
        
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # 2. Load the actual model
        model_path = model_dir / f"{model_name}.pkl"
        model = joblib.load(model_path)
        
        self.logger.info(f"✅ Model loaded: {model_name}")
        
        # 3. Process text through NLP pipeline
        features = self._extract_features_from_text(text)
        
        self.logger.info(f"🔬 Features extracted: {features}")
        
        # 4. Validate features match model expectations
        expected_features = metadata.get("feature_columns", [])
        feature_df = pd.DataFrame([features])
        
        # Reorder columns to match training
        feature_df = feature_df[expected_features]
        
        # 5. Make prediction
        prediction = model.predict(feature_df)[0]
        
        # Convert numpy types to Python native types
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
        else:
            prediction = str(prediction)
        
        # 6. Get probabilities if available
        probabilities = None
        confidence = None
        classes = None
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feature_df)[0]
            probabilities = proba.tolist()
            confidence = float(max(proba))
            classes = [str(c) for c in model.classes_]
        
        result = {
            "prediction": prediction,
            "target": metadata.get("target_column", "sentiment"),
            "model_name": model_name,
            "model_type": metadata.get("model_type", "classification"),
            "processed_text": features.get("normalized_text", text),
            "features_used": features,
            "probabilities": probabilities,
            "classes": classes,
            "confidence": confidence
        }
        
        self.logger.info(f"🎯 Prediction: {prediction} (confidence: {confidence:.2%})")
        
        return result
    
    def _extract_features_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract features from raw text using NLP pipeline.
        
        Features extracted:
        - review_text_normalized_polarity: Sentiment polarity (-1 to 1)
        - review_text_normalized_subjectivity: Subjectivity (0 to 1)
        
        Args:
            text: Raw text to process
            
        Returns:
            Dictionary with extracted features
        """
        # Step 1: Text Normalization
        df = pd.DataFrame([{"text": text}])
        
        normalization_config = {
            "text_columns": ["text"],
            "lowercase": True,
            "remove_html": True,
            "remove_urls": True,
            "remove_emails": True,
            "remove_punctuation": False,
            "remove_numbers": False,
            "remove_accents": True,
            "normalize_whitespace": True,
            "output_suffix": "_normalized",
            "inplace": False
        }
        
        df = self.normalization_service.execute(df, normalization_config)
        normalized_text = df["text_normalized"].iloc[0]
        
        # Step 2: Sentiment Analysis using TextBlob
        blob = TextBlob(normalized_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Return features that match model training
        features = {
            "review_text_normalized_polarity": float(polarity),
            "review_text_normalized_subjectivity": float(subjectivity),
            "normalized_text": normalized_text
        }
        
        return features


# Global instance
text_prediction_service = TextPredictionService()
