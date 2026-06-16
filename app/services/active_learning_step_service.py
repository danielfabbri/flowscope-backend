"""
Active Learning Step Service
Collects feedback and improves model through active learning
"""
import pandas as pd
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from app.services.base_service import BasePipelineService
from app.core.logger import get_logger

logger = get_logger(__name__)


class ActiveLearningStepService(BasePipelineService):
    """
    Pipeline step that implements active learning feedback loop
    
    Input: DataFrame with predictions and feedback
    Output: DataFrame with learning signals and retraining triggers
    """
    
    def __init__(self):
        # Feedback storage
        self.feedback_history = []
        self.uncertain_samples = []
        self.retrain_counter = 0
        
        # Use absolute path relative to backend directory
        backend_dir = Path(__file__).parent.parent.parent
        self.feedback_dir = backend_dir / "data" / "feedback"
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ActiveLearningStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute active learning on DataFrame
        
        Args:
            data: DataFrame with predictions and optional feedback
            config: Configuration with:
                - prediction_column: column with model predictions
                - confidence_column: column with prediction confidence
                - feedback_column: column with user feedback (optional)
                - uncertainty_sampling: whether to use uncertainty sampling
                - uncertainty_threshold: threshold for uncertain samples
                - retrain_threshold: number of feedbacks before retraining
                - collect_feedback: whether to collect feedback
                - output_uncertain_column: name for uncertainty flag column
                - output_should_retrain_column: name for retrain flag
                - output_feedback_count_column: name for feedback count
        
        Returns:
            DataFrame with active learning information
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for active learning")
        
        prediction_column = config.get('prediction_column', 'prediction')
        confidence_column = config.get('confidence_column', 'confidence')
        feedback_column = config.get('feedback_column', 'feedback')
        uncertainty_sampling = config.get('uncertainty_sampling', True)
        uncertainty_threshold = float(config.get('uncertainty_threshold', 0.6))
        retrain_threshold = int(config.get('retrain_threshold', 100))
        collect_feedback = config.get('collect_feedback', True)
        output_uncertain = config.get('output_uncertain_column', 'is_uncertain')
        output_retrain = config.get('output_should_retrain_column', 'should_retrain')
        output_count = config.get('output_feedback_count_column', 'feedback_count')
        
        logger.info(f"Processing active learning for {len(data)} samples")
        
        # Process each sample
        uncertain_flags = []
        should_retrain = False
        
        for idx, row in data.iterrows():
            confidence = float(row.get(confidence_column, 1.0))
            
            # Check uncertainty
            is_uncertain = False
            if uncertainty_sampling and confidence < uncertainty_threshold:
                is_uncertain = True
                self.uncertain_samples.append({
                    'index': idx,
                    'data': row.to_dict(),
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Collect feedback if available
            if collect_feedback and feedback_column in row and pd.notna(row[feedback_column]):
                feedback = row[feedback_column]
                self._collect_feedback(row, feedback)
            
            uncertain_flags.append(is_uncertain)
        
        # Check if should retrain
        if len(self.feedback_history) >= retrain_threshold:
            should_retrain = True
            logger.info(f"Retrain threshold reached: {len(self.feedback_history)} feedbacks collected")
        
        # Add new columns to DataFrame
        result = data.copy()
        result[output_uncertain] = uncertain_flags
        result[output_retrain] = should_retrain
        result[output_count] = len(self.feedback_history)
        
        uncertain_count = sum(uncertain_flags)
        logger.info(f"Active learning completed. {uncertain_count} uncertain samples identified")
        return result
    
    def _collect_feedback(self, sample: pd.Series, feedback: Any):
        """Collect feedback for a sample"""
        feedback_entry = {
            'sample': sample.to_dict(),
            'feedback': feedback,
            'timestamp': datetime.now().isoformat(),
            'feedback_id': len(self.feedback_history)
        }
        
        self.feedback_history.append(feedback_entry)
        
        # Save feedback to file
        self._save_feedback(feedback_entry)
        
        logger.debug(f"Feedback collected: {feedback}")
    
    def _save_feedback(self, feedback_entry: Dict):
        """Save feedback to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d')
            feedback_file = self.feedback_dir / f"feedback_{timestamp}.jsonl"
            
            with open(feedback_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(feedback_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    def get_uncertain_samples(self, top_k: int = 10) -> List[Dict]:
        """Get most uncertain samples for labeling"""
        # Sort by confidence (ascending)
        sorted_samples = sorted(
            self.uncertain_samples,
            key=lambda x: x['confidence']
        )
        return sorted_samples[:top_k]
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        if not self.feedback_history:
            return {
                'total_feedback': 0,
                'uncertain_samples': 0
            }
        
        return {
            'total_feedback': len(self.feedback_history),
            'uncertain_samples': len(self.uncertain_samples),
            'avg_confidence': sum(s['confidence'] for s in self.uncertain_samples) / len(self.uncertain_samples) if self.uncertain_samples else 0,
            'last_feedback': self.feedback_history[-1]['timestamp'] if self.feedback_history else None
        }
    
    def reset_feedback(self):
        """Reset feedback history (after retraining)"""
        logger.info("Resetting feedback history")
        self.feedback_history = []
        self.uncertain_samples = []
        self.retrain_counter += 1


# Singleton instance
_instance = None

def get_active_learning_step_service():
    global _instance
    if _instance is None:
        _instance = ActiveLearningStepService()
    return _instance
