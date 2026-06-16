"""
Intent Classification Step Service
Classifies text intent for pipeline processing
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from app.services.base_service import BasePipelineService
from app.services.intent_classification_service import get_intent_classification_service
from app.core.logger import get_logger

logger = get_logger(__name__)


class IntentClassificationStepService(BasePipelineService):
    """
    Pipeline step that classifies text intent
    
    Input: DataFrame with text column
    Output: DataFrame with added intent and confidence columns
    """
    
    def __init__(self):
        self.intent_service = get_intent_classification_service()
        # Use absolute path relative to backend directory
        backend_dir = Path(__file__).parent.parent.parent
        self.models_dir = backend_dir / "data" / "models"
        logger.info("IntentClassificationStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute intent classification on DataFrame
        
        Args:
            data: DataFrame with text column
            config: Configuration with:
                - text_column: column name with text to classify
                - model_name: name of trained model to load (optional)
                - min_confidence: minimum confidence threshold (0-1)
                - output_intent_column: name for intent output column
                - output_confidence_column: name for confidence output column
        
        Returns:
            DataFrame with added intent and confidence columns
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for intent classification")
        
        text_column = config.get('text_column', 'message')
        model_name = config.get('model_name', None)
        min_confidence = float(config.get('min_confidence', 0.4))
        output_intent = config.get('output_intent_column', 'intent')
        output_confidence = config.get('output_confidence_column', 'intent_confidence')
        fallback_intent = config.get('fallback_intent', 'unknown')
        
        if text_column not in data.columns:
            raise ValueError(f"Text column '{text_column}' not found in data. Available: {list(data.columns)}")
        
        # Load model if specified
        if model_name:
            logger.info(f"Loading intent model: {model_name}")
            try:
                model_path = self.models_dir / f"{model_name}.joblib"
                self.intent_service.load_model(str(model_path))
                logger.info(f"Model '{model_name}' loaded successfully from '{model_path}'")
            except Exception as e:
                logger.error(f"Failed to load model '{model_name}': {e}")
                raise ValueError(f"Could not load intent model '{model_name}': {e}")
        
        logger.info(f"Classifying intent for {len(data)} rows from column '{text_column}'")
        
        # Check if model is trained
        if self.intent_service.model is None:
            logger.warning("Intent classifier not trained. All predictions will be 'unknown'")
        
        # Classify each text
        intents = []
        confidences = []
        
        for idx, row in data.iterrows():
            text = str(row[text_column])
            
            if not text or text.strip() == '':
                intents.append(fallback_intent)
                confidences.append(0.0)
                continue
            
            result = self.intent_service.predict_intent(text)
            intent = result.get('intent', fallback_intent)
            confidence = result.get('confidence', 0.0)
            
            # Apply confidence threshold
            if confidence < min_confidence:
                intent = fallback_intent
                confidence = 0.0
            
            intents.append(intent)
            confidences.append(confidence)
        
        # Add new columns to DataFrame
        result = data.copy()
        result[output_intent] = intents
        result[output_confidence] = confidences
        
        # Log statistics
        intent_counts = pd.Series(intents).value_counts()
        logger.info(f"Intent classification complete. Distribution: {intent_counts.to_dict()}")
        logger.info(f"Average confidence: {sum(confidences) / len(confidences):.2f}")
        
        return result


def get_intent_classification_step_service() -> IntentClassificationStepService:
    """Get singleton instance"""
    if not hasattr(get_intent_classification_step_service, '_instance'):
        get_intent_classification_step_service._instance = IntentClassificationStepService()
    return get_intent_classification_step_service._instance
