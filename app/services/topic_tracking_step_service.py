"""
Topic Tracking Step Service
Tracks and identifies topics in conversations using topic modeling
"""
import pandas as pd
import json
import numpy as np
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from app.services.base_service import BasePipelineService
from app.core.logger import get_logger

logger = get_logger(__name__)


class TopicTrackingStepService(BasePipelineService):
    """
    Pipeline step that tracks conversation topics
    
    Input: DataFrame with message text
    Output: DataFrame with topic labels and changes
    """
    
    def __init__(self):
        self.vectorizer = None
        self.topic_model = None
        self.topic_labels = []
        # Track topics by conversation
        self.conversation_topics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        logger.info("TopicTrackingStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute topic tracking on DataFrame
        
        Args:
            data: DataFrame with message text
            config: Configuration with:
                - message_column: column with message text
                - conversation_id_column: column with conversation ID
                - model_type: topic model type (lda, nmf, lsa)
                - num_topics: number of topics to identify
                - topic_change_threshold: threshold for detecting topic change
                - train_model: whether to train new model or use existing
                - output_topic_column: name for topic label column
                - output_topic_id_column: name for topic ID column
                - output_topic_change_column: name for topic change flag
        
        Returns:
            DataFrame with topic information
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for topic tracking")
        
        message_column = config.get('message_column', 'message')
        conv_id_column = config.get('conversation_id_column', 'conversation_id')
        model_type = config.get('model_type', 'lda')
        num_topics = int(config.get('num_topics', 10))
        change_threshold = float(config.get('topic_change_threshold', 0.3))
        train_model = config.get('train_model', True)
        output_topic = config.get('output_topic_column', 'topic_label')
        output_topic_id = config.get('output_topic_id_column', 'topic_id')
        output_change = config.get('output_topic_change_column', 'topic_changed')
        
        if message_column not in data.columns:
            raise ValueError(f"Message column '{message_column}' not found")
        
        # Set default conversation ID if not present
        if conv_id_column not in data.columns:
            data[conv_id_column] = 'default'
        
        logger.info(f"Tracking topics for {len(data)} messages")
        logger.info(f"Model: {model_type}, Topics: {num_topics}")
        
        # Train model if needed
        if train_model or self.topic_model is None:
            self._train_topic_model(data[message_column].tolist(), model_type, num_topics)
        
        # Process each message
        topic_labels = []
        topic_ids = []
        topic_changes = []
        
        for idx, row in data.iterrows():
            message = str(row[message_column])
            conv_id = str(row.get(conv_id_column, 'default'))
            
            # Predict topic
            topic_id, topic_label, topic_dist = self._predict_topic(message)
            
            # Check for topic change
            history = self.conversation_topics[conv_id]
            topic_changed = False
            
            if history:
                prev_topic_id = history[-1]
                if prev_topic_id != topic_id:
                    # Check if change is significant
                    if len(history) > 0:
                        topic_changed = True
            
            # Update history
            history.append(topic_id)
            
            topic_labels.append(topic_label)
            topic_ids.append(topic_id)
            topic_changes.append(topic_changed)
        
        # Add new columns to DataFrame
        result = data.copy()
        result[output_topic] = topic_labels
        result[output_topic_id] = topic_ids
        result[output_change] = topic_changes
        
        changes_count = sum(topic_changes)
        logger.info(f"Topic tracking completed. Detected {changes_count} topic changes")
        return result
    
    def _train_topic_model(self, texts: List[str], model_type: str, num_topics: int):
        """Train topic model"""
        logger.info(f"Training {model_type} topic model with {num_topics} topics")
        
        try:
            if model_type == 'lda':
                self.vectorizer = CountVectorizer(
                    max_features=1000,
                    stop_words='english',
                    min_df=2
                )
                doc_term_matrix = self.vectorizer.fit_transform(texts)
                
                self.topic_model = LatentDirichletAllocation(
                    n_components=num_topics,
                    random_state=42,
                    max_iter=10
                )
                self.topic_model.fit(doc_term_matrix)
                
            elif model_type == 'nmf':
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    min_df=2
                )
                doc_term_matrix = self.vectorizer.fit_transform(texts)
                
                self.topic_model = NMF(
                    n_components=num_topics,
                    random_state=42,
                    max_iter=200
                )
                self.topic_model.fit(doc_term_matrix)
            
            else:
                # Default to LDA
                self.vectorizer = CountVectorizer(max_features=1000)
                doc_term_matrix = self.vectorizer.fit_transform(texts)
                self.topic_model = LatentDirichletAllocation(
                    n_components=num_topics,
                    random_state=42
                )
                self.topic_model.fit(doc_term_matrix)
            
            # Generate topic labels
            self._generate_topic_labels(num_topics)
            
            logger.info("Topic model trained successfully")
        except Exception as e:
            logger.error(f"Error training topic model: {e}")
            # Create dummy model
            self.topic_labels = [f"topic_{i}" for i in range(num_topics)]
    
    def _generate_topic_labels(self, num_topics: int):
        """Generate human-readable topic labels"""
        self.topic_labels = []
        
        if self.topic_model is None or self.vectorizer is None:
            self.topic_labels = [f"topic_{i}" for i in range(num_topics)]
            return
        
        try:
            feature_names = self.vectorizer.get_feature_names_out()
            
            for topic_idx, topic in enumerate(self.topic_model.components_):
                # Get top words for this topic
                top_indices = topic.argsort()[-5:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                label = '_'.join(top_words[:3])
                self.topic_labels.append(label)
        except Exception as e:
            logger.warning(f"Could not generate topic labels: {e}")
            self.topic_labels = [f"topic_{i}" for i in range(num_topics)]
    
    def _predict_topic(self, text: str) -> tuple:
        """Predict topic for text"""
        if self.topic_model is None or self.vectorizer is None:
            return 0, 'unknown', [1.0]
        
        try:
            # Vectorize text
            text_vector = self.vectorizer.transform([text])
            
            # Predict topic distribution
            topic_dist = self.topic_model.transform(text_vector)[0]
            
            # Get dominant topic
            topic_id = int(np.argmax(topic_dist))
            topic_label = self.topic_labels[topic_id] if topic_id < len(self.topic_labels) else f"topic_{topic_id}"
            
            return topic_id, topic_label, topic_dist.tolist()
        except Exception as e:
            logger.error(f"Error predicting topic: {e}")
            return 0, 'unknown', [1.0]


# Singleton instance
_instance = None

def get_topic_tracking_step_service():
    global _instance
    if _instance is None:
        _instance = TopicTrackingStepService()
    return _instance
