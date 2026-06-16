"""
Answer Reranking Step Service
Reranks answer candidates using multiple signals and learning-to-rank techniques
"""
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional, List
from sklearn.preprocessing import MinMaxScaler
from app.services.base_service import BasePipelineService
from app.core.logger import get_logger

logger = get_logger(__name__)


class AnswerRerankingStepService(BasePipelineService):
    """
    Pipeline step that reranks answer candidates
    
    Input: DataFrame with query and answer candidates
    Output: DataFrame with reranked answers
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_weights = {
            'semantic_similarity': 0.35,
            'entity_overlap': 0.20,
            'intent_match': 0.15,
            'recency': 0.10,
            'popularity': 0.10,
            'length_match': 0.10
        }
        logger.info("AnswerRerankingStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute answer reranking on DataFrame
        
        Args:
            data: DataFrame with candidates column (JSON array of answers)
            config: Configuration with:
                - candidates_column: column with answer candidates (JSON)
                - query_column: column with user query
                - ranking_features: list of features to use
                - feature_weights: custom weights for features
                - top_k: number of top answers to keep
                - output_column: name for reranked answers column
                - output_scores_column: name for scores column
        
        Returns:
            DataFrame with reranked answers
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for answer reranking")
        
        candidates_column = config.get('candidates_column', 'answer_candidates')
        query_column = config.get('query_column', 'message')
        ranking_features = config.get('ranking_features', list(self.feature_weights.keys()))
        custom_weights = config.get('feature_weights', {})
        top_k = int(config.get('top_k', 3))
        output_column = config.get('output_column', 'reranked_answers')
        output_scores = config.get('output_scores_column', 'reranking_scores')
        
        # Update weights if custom provided
        weights = self.feature_weights.copy()
        weights.update(custom_weights)
        
        if candidates_column not in data.columns:
            raise ValueError(f"Candidates column '{candidates_column}' not found")
        
        logger.info(f"Reranking answers for {len(data)} queries")
        logger.info(f"Using features: {ranking_features}")
        
        # Process each row
        reranked_answers = []
        reranking_scores = []
        
        for idx, row in data.iterrows():
            query = str(row.get(query_column, ''))
            
            # Parse candidates
            candidates = []
            if pd.notna(row[candidates_column]):
                try:
                    candidates = json.loads(str(row[candidates_column]))
                    if not isinstance(candidates, list):
                        candidates = [candidates]
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse candidates JSON for row {idx}")
            
            if not candidates:
                reranked_answers.append(json.dumps([]))
                reranking_scores.append(json.dumps([]))
                continue
            
            # Calculate features and rank
            ranked = self._rank_candidates(query, candidates, ranking_features, weights, row)
            
            # Keep top K
            top_ranked = ranked[:top_k]
            
            reranked_answers.append(json.dumps([c['answer'] for c in top_ranked]))
            reranking_scores.append(json.dumps([c['score'] for c in top_ranked]))
        
        # Add new columns to DataFrame
        result = data.copy()
        result[output_column] = reranked_answers
        result[output_scores] = reranking_scores
        
        logger.info(f"Answer reranking completed")
        return result
    
    def _rank_candidates(self, query: str, candidates: List, features: List[str], 
                        weights: Dict, row: pd.Series) -> List[Dict]:
        """Rank candidates by multiple features"""
        scored_candidates = []
        
        for candidate in candidates:
            if isinstance(candidate, str):
                candidate = {'answer': candidate, 'score': 0.5}
            elif isinstance(candidate, dict) and 'answer' not in candidate:
                candidate = {'answer': str(candidate), 'score': 0.5}
            
            feature_scores = {}
            
            # Calculate each feature
            if 'semantic_similarity' in features:
                feature_scores['semantic_similarity'] = candidate.get('similarity', 0.5)
            
            if 'entity_overlap' in features:
                feature_scores['entity_overlap'] = self._calculate_entity_overlap(
                    query, candidate['answer'], row
                )
            
            if 'intent_match' in features:
                feature_scores['intent_match'] = candidate.get('intent_match', 0.5)
            
            if 'recency' in features:
                feature_scores['recency'] = candidate.get('recency', 0.5)
            
            if 'popularity' in features:
                feature_scores['popularity'] = candidate.get('popularity', 0.5)
            
            if 'length_match' in features:
                feature_scores['length_match'] = self._calculate_length_match(
                    query, candidate['answer']
                )
            
            # Calculate weighted score
            total_score = sum(
                feature_scores.get(f, 0) * weights.get(f, 0)
                for f in features
            )
            
            scored_candidates.append({
                'answer': candidate['answer'],
                'score': round(total_score, 4),
                'features': feature_scores
            })
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_candidates
    
    def _calculate_entity_overlap(self, query: str, answer: str, row: pd.Series) -> float:
        """Calculate entity overlap between query and answer"""
        # Simple word overlap as proxy
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & answer_words)
        return min(overlap / len(query_words), 1.0)
    
    def _calculate_length_match(self, query: str, answer: str) -> float:
        """Calculate if answer length is appropriate"""
        query_len = len(query.split())
        answer_len = len(answer.split())
        
        # Prefer answers that are 2-5x the query length
        if answer_len < query_len:
            return 0.3
        elif query_len * 2 <= answer_len <= query_len * 5:
            return 1.0
        elif answer_len > query_len * 10:
            return 0.5
        else:
            return 0.7


# Singleton instance
_instance = None

def get_answer_reranking_step_service():
    global _instance
    if _instance is None:
        _instance = AnswerRerankingStepService()
    return _instance
