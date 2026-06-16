"""
Intent Classification Service

Classifies user intent from text input using machine learning.
Supports training custom intent classifiers and predicting intents.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import json
from pathlib import Path
from ..core.logger import get_logger

logger = get_logger(__name__)


class IntentClassificationService:
    """Service for classifying user intents from text"""
    
    def __init__(self):
        self.model: Optional[Pipeline] = None
        self.intents: List[str] = []
        self.confidence_threshold = 0.3
        
    def train_intent_classifier(
        self,
        training_data: List[Dict[str, str]],
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train an intent classifier from labeled examples.
        
        Args:
            training_data: List of {"text": "...", "intent": "..."}
            test_size: Proportion of data for testing
            
        Returns:
            Training metrics and report
        """
        logger.info(f"Training intent classifier with {len(training_data)} examples")
        
        # Extract texts and labels
        texts = [item["text"] for item in training_data]
        labels = [item["intent"] for item in training_data]
        
        # Get unique intents
        self.intents = sorted(list(set(labels)))
        logger.info(f"Found {len(self.intents)} unique intents: {self.intents}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Create pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate accuracy
        accuracy = np.mean(np.array(y_pred) == np.array(y_test))
        
        logger.info(f"Intent classifier trained - Accuracy: {accuracy:.2%}")
        
        return {
            "status": "success",
            "accuracy": float(accuracy),
            "num_intents": len(self.intents),
            "intents": self.intents,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "classification_report": report
        }
    
    def predict_intent(self, text: str) -> Dict[str, Any]:
        """
        Predict the intent of a text input.
        
        Args:
            text: Input text to classify
            
        Returns:
            Intent prediction with confidence scores
        """
        if self.model is None:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "all_intents": [],
                "error": "Model not trained yet"
            }
        
        try:
            # Get probabilities for all intents
            probabilities = self.model.predict_proba([text])[0]
            
            # Get predicted intent
            predicted_intent = self.model.predict([text])[0]
            
            # Get confidence
            max_confidence = float(np.max(probabilities))
            
            # Create sorted list of all intents with probabilities
            intent_scores = [
                {"intent": intent, "confidence": float(prob)}
                for intent, prob in zip(self.model.classes_, probabilities)
            ]
            intent_scores.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Check if confidence is above threshold
            is_confident = max_confidence >= self.confidence_threshold
            
            result = {
                "intent": predicted_intent if is_confident else "unknown",
                "confidence": max_confidence,
                "is_confident": is_confident,
                "all_intents": intent_scores,
                "threshold": self.confidence_threshold
            }
            
            logger.debug(f"Predicted intent '{predicted_intent}' with confidence {max_confidence:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting intent: {e}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "all_intents": [],
                "error": str(e)
            }
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict intents for multiple texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of predictions
        """
        return [self.predict_intent(text) for text in texts]
    
    def save_model(self, file_path: str) -> Dict[str, Any]:
        """
        Save the trained model to disk.
        
        Args:
            file_path: Path to save the model
            
        Returns:
            Save status
        """
        if self.model is None:
            return {"status": "error", "message": "No model to save"}
        
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model and metadata
            model_data = {
                "model": self.model,
                "intents": self.intents,
                "confidence_threshold": self.confidence_threshold
            }
            
            joblib.dump(model_data, file_path)
            logger.info(f"Model saved to {file_path}")
            
            return {
                "status": "success",
                "file_path": file_path,
                "num_intents": len(self.intents)
            }
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return {"status": "error", "message": str(e)}
    
    def load_model(self, file_path: str) -> Dict[str, Any]:
        """
        Load a trained model from disk.
        
        Args:
            file_path: Path to the saved model
            
        Returns:
            Load status
        """
        try:
            model_data = joblib.load(file_path)
            
            self.model = model_data["model"]
            self.intents = model_data["intents"]
            self.confidence_threshold = model_data.get("confidence_threshold", 0.3)
            
            logger.info(f"Model loaded from {file_path}")
            logger.info(f"Available intents: {self.intents}")
            
            return {
                "status": "success",
                "file_path": file_path,
                "num_intents": len(self.intents),
                "intents": self.intents
            }
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return {"status": "error", "message": str(e)}
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the minimum confidence threshold for predictions"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold set to {self.confidence_threshold}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if self.model is None:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "num_intents": len(self.intents),
            "intents": self.intents,
            "confidence_threshold": self.confidence_threshold
        }


# Singleton instance
_intent_service_instance = None


def get_intent_classification_service() -> IntentClassificationService:
    """Get or create the singleton intent classification service"""
    global _intent_service_instance
    if _intent_service_instance is None:
        _intent_service_instance = IntentClassificationService()
    return _intent_service_instance
