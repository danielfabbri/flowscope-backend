"""
Anomaly Detection Step Service
Detects anomalous queries that are malicious or out of domain
"""
import pandas as pd
import json
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from app.services.base_service import BasePipelineService
from app.core.logger import get_logger

logger = get_logger(__name__)


class AnomalyDetectionStepService(BasePipelineService):
    """
    Pipeline step that detects anomalous queries
    
    Input: DataFrame with queries
    Output: DataFrame with anomaly flags and scores
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.anomaly_detector = None
        self.is_trained = False
        
        # Known malicious patterns
        self.malicious_patterns = [
            'sql', 'injection', 'script', 'hack', 'exploit',
            'drop table', 'delete from', '<script>', 'alert(',
            '../', 'exec(', 'eval(', '__import__'
        ]
        
        logger.info("AnomalyDetectionStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute anomaly detection on DataFrame
        
        Args:
            data: DataFrame with query text
            config: Configuration with:
                - query_column: column with query text
                - model_type: detector type (isolation_forest, one_class_svm)
                - threshold: anomaly score threshold
                - contamination: expected proportion of anomalies
                - train_model: whether to train new model
                - check_malicious: whether to check for malicious patterns
                - output_is_anomaly_column: name for anomaly flag column
                - output_anomaly_score_column: name for score column
                - output_anomaly_type_column: name for anomaly type
        
        Returns:
            DataFrame with anomaly detection results
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for anomaly detection")
        
        query_column = config.get('query_column', 'message')
        model_type = config.get('model_type', 'isolation_forest')
        threshold = float(config.get('threshold', 0.8))
        contamination = float(config.get('contamination', 0.1))
        train_model = config.get('train_model', True)
        check_malicious = config.get('check_malicious', True)
        output_is_anomaly = config.get('output_is_anomaly_column', 'is_anomaly')
        output_score = config.get('output_anomaly_score_column', 'anomaly_score')
        output_type = config.get('output_anomaly_type_column', 'anomaly_type')
        
        if query_column not in data.columns:
            raise ValueError(f"Query column '{query_column}' not found")
        
        logger.info(f"Detecting anomalies for {len(data)} queries")
        
        # Train model if needed
        if train_model or not self.is_trained:
            self._train_detector(data[query_column].tolist(), model_type, contamination)
        
        # Process each query
        is_anomaly_list = []
        anomaly_scores = []
        anomaly_types = []
        
        for idx, row in data.iterrows():
            query = str(row[query_column])
            
            # Check for malicious patterns first
            if check_malicious:
                is_malicious, mal_type = self._check_malicious(query)
                if is_malicious:
                    is_anomaly_list.append(True)
                    anomaly_scores.append(1.0)
                    anomaly_types.append(mal_type)
                    continue
            
            # ML-based detection
            is_anomaly, score = self._detect_anomaly(query, threshold)
            
            is_anomaly_list.append(is_anomaly)
            anomaly_scores.append(round(score, 4))
            anomaly_types.append('statistical' if is_anomaly else 'normal')
        
        # Add new columns to DataFrame
        result = data.copy()
        result[output_is_anomaly] = is_anomaly_list
        result[output_score] = anomaly_scores
        result[output_type] = anomaly_types
        
        anomaly_count = sum(is_anomaly_list)
        logger.info(f"Anomaly detection completed. Found {anomaly_count} anomalies")
        return result
    
    def _train_detector(self, texts: List[str], model_type: str, contamination: float):
        """Train anomaly detector"""
        logger.info(f"Training {model_type} anomaly detector")
        
        try:
            # Vectorize texts
            X = self.vectorizer.fit_transform(texts)
            
            # Train detector
            if model_type == 'isolation_forest':
                self.anomaly_detector = IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_estimators=100
                )
            else:
                # Default to Isolation Forest
                self.anomaly_detector = IsolationForest(
                    contamination=contamination,
                    random_state=42
                )
            
            self.anomaly_detector.fit(X)
            self.is_trained = True
            
            logger.info("Anomaly detector trained successfully")
        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")
            self.is_trained = False
    
    def _detect_anomaly(self, query: str, threshold: float) -> tuple:
        """Detect if query is anomalous"""
        if not self.is_trained or self.anomaly_detector is None:
            return False, 0.5
        
        try:
            # Vectorize query
            X = self.vectorizer.transform([query])
            
            # Predict
            prediction = self.anomaly_detector.predict(X)[0]
            score = self.anomaly_detector.score_samples(X)[0]
            
            # Convert score to probability-like value
            # Isolation Forest returns negative scores, normalize to 0-1
            normalized_score = 1.0 / (1.0 + np.exp(score))  # Sigmoid-like transform
            
            is_anomaly = bool(prediction == -1)
            
            return is_anomaly, float(normalized_score)
        except Exception as e:
            logger.error(f"Error detecting anomaly: {e}")
            return False, 0.5
    
    def _check_malicious(self, query: str) -> tuple:
        """Check for malicious patterns"""
        query_lower = query.lower()
        
        for pattern in self.malicious_patterns:
            if pattern in query_lower:
                logger.warning(f"Malicious pattern detected: {pattern}")
                return True, 'malicious'
        
        # Check for unusual characters/encoding
        if self._has_unusual_encoding(query):
            return True, 'unusual_encoding'
        
        # Check length anomalies
        if len(query) > 1000:
            return True, 'too_long'
        
        if len(query) < 2:
            return True, 'too_short'
        
        return False, 'normal'
    
    def _has_unusual_encoding(self, text: str) -> bool:
        """Check for unusual character encoding"""
        try:
            # Check for high proportion of non-ASCII characters
            non_ascii = sum(1 for c in text if ord(c) > 127)
            if non_ascii > len(text) * 0.5:
                return True
            
            # Check for control characters
            control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
            if control_chars > 0:
                return True
            
            return False
        except Exception:
            return True


# Singleton instance
_instance = None

def get_anomaly_detection_step_service():
    global _instance
    if _instance is None:
        _instance = AnomalyDetectionStepService()
    return _instance
