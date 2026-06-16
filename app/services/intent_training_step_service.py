"""
Intent Training Step Service

Pipeline step for training an intent classification model from labeled data.
"""

from typing import Dict, Any
from pathlib import Path
import pandas as pd
from .base_service import BasePipelineService
from .intent_classification_service import get_intent_classification_service
from ..core.logger import get_logger

logger = get_logger(__name__)


class IntentTrainingStepService(BasePipelineService):
    """Pipeline step for training intent classifier"""
    
    def __init__(self):
        super().__init__()
        self.intent_service = get_intent_classification_service()
        # Use absolute path relative to backend directory
        backend_dir = Path(__file__).parent.parent.parent
        self.models_dir = backend_dir / "data" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def execute(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Train intent classification model from labeled data.
        
        Expected config:
        - text_column: Column containing training text
        - intent_column: Column containing intent labels
        - model_name: Name to save the trained model
        - test_size: Proportion for test split (default: 0.2)
        
        Expected DataFrame columns:
        - text_column: Training texts
        - intent_column: Intent labels
        
        Returns:
            Original DataFrame with training metrics added
        """
        try:
            logger.info("Starting intent classifier training step")
            logger.info(f"Received DataFrame with {len(df) if df is not None else 0} rows")
            logger.info(f"DataFrame columns: {df.columns.tolist() if df is not None else 'None'}")
            
            # Get configuration
            text_column = config.get('text_column', 'text')
            intent_column = config.get('intent_column', 'intent')
            model_name = config.get('model_name', 'intent_model')
            test_size = float(config.get('test_size', 0.2))
            
            logger.info(f"Training config - text_column: {text_column}, intent_column: {intent_column}, model_name: {model_name}, test_size: {test_size}")
            
            # Validate DataFrame
            if df is None:
                raise ValueError("No data received from previous step. DataFrame is None.")
            
            if df.empty:
                raise ValueError("Received empty DataFrame from previous step.")
            
            # Validate columns exist
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in data. Available columns: {df.columns.tolist()}")
            if intent_column not in df.columns:
                raise ValueError(f"Intent column '{intent_column}' not found in data. Available columns: {df.columns.tolist()}")
                
            # Prepare training data
            training_data = [
                {"text": row[text_column], "intent": row[intent_column]}
                for _, row in df.iterrows()
            ]
            
            logger.info(f"Training with {len(training_data)} examples")
            
            # Train the model
            metrics = self.intent_service.train_intent_classifier(
                training_data=training_data,
                test_size=test_size
            )
            
            if metrics.get("status") != "success":
                raise RuntimeError(f"Training failed: {metrics.get('message', 'Unknown error')}")
            
            # Save the model with full path
            model_path = self.models_dir / f"{model_name}.joblib"
            save_result = self.intent_service.save_model(str(model_path))
            
            if save_result.get("status") == "error":
                raise RuntimeError(f"Failed to save model: {save_result.get('message')}")
            
            logger.info(f"Model saved to '{model_path}'")
            
            # Add metrics to DataFrame as metadata
            result_df = df.copy()
            result_df.attrs['training_metrics'] = metrics
            result_df.attrs['model_name'] = model_name
            result_df.attrs['model_path'] = str(model_path)
            
            logger.info(f"Intent training complete. Accuracy: {metrics.get('accuracy', 'N/A')}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Intent training step failed: {str(e)}")
            logger.exception("Full traceback:")
            raise
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate step configuration"""
        required = ['text_column', 'intent_column', 'model_name']
        return all(key in config for key in required)


# Singleton instance
_intent_training_step_service = None


def get_intent_training_step_service() -> IntentTrainingStepService:
    """Get singleton instance of intent training step service"""
    global _intent_training_step_service
    if _intent_training_step_service is None:
        _intent_training_step_service = IntentTrainingStepService()
    return _intent_training_step_service
