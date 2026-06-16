"""
Response Selection Step Service
Selects the best response from a pre-defined set of candidates
"""
import pandas as pd
import json
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.services.base_service import BasePipelineService
from app.core.logger import get_logger

logger = get_logger(__name__)


class ResponseSelectionStepService(BasePipelineService):
    """
    Pipeline step that selects best response from candidates
    
    Input: DataFrame with query and candidate responses
    Output: DataFrame with selected response
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2)
        )
        # Response templates
        self.response_templates = {}
        logger.info("ResponseSelectionStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute response selection on DataFrame
        
        Args:
            data: DataFrame with query column
            config: Configuration with:
                - query_column: column with user query
                - candidate_sources: sources for candidates (faq, templates, previous)
                - candidates_column: column with candidate responses (JSON)
                - context_column: column with conversation context
                - selection_model: model type (tfidf, dual_encoder)
                - min_score: minimum selection score
                - output_response_column: name for selected response
                - output_score_column: name for selection score
        
        Returns:
            DataFrame with selected responses
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for response selection")
        
        query_column = config.get('query_column', 'message')
        candidates_column = config.get('candidates_column', 'response_candidates')
        context_column = config.get('context_column', 'context')
        sources = config.get('candidate_sources', ['faq', 'templates'])
        selection_model = config.get('selection_model', 'tfidf')
        min_score = float(config.get('min_score', 0.6))
        output_response = config.get('output_response_column', 'selected_response')
        output_score = config.get('output_score_column', 'selection_score')
        
        if query_column not in data.columns:
            raise ValueError(f"Query column '{query_column}' not found")
        
        logger.info(f"Selecting responses for {len(data)} queries")
        logger.info(f"Using sources: {sources}, model: {selection_model}")
        
        # Process each query
        selected_responses = []
        selection_scores = []
        
        for idx, row in data.iterrows():
            query = str(row[query_column])
            context = str(row.get(context_column, ''))
            
            # Get candidates
            candidates = self._get_candidates(row, candidates_column, sources, config)
            
            if not candidates:
                selected_responses.append("Desculpe, não encontrei uma resposta adequada.")
                selection_scores.append(0.0)
                continue
            
            # Select best response
            selected, score = self._select_best_response(
                query, candidates, context, selection_model
            )
            
            # Apply minimum score threshold
            if score < min_score:
                selected = "Desculpe, não tenho certeza sobre isso. Pode reformular?"
                score = 0.0
            
            selected_responses.append(selected)
            selection_scores.append(round(score, 4))
        
        # Add new columns to DataFrame
        result = data.copy()
        result[output_response] = selected_responses
        result[output_score] = selection_scores
        
        logger.info(f"Response selection completed")
        return result
    
    def _get_candidates(self, row: pd.Series, candidates_column: str, 
                       sources: List[str], config: Dict) -> List[str]:
        """Get candidate responses from various sources"""
        candidates = []
        
        # From candidates column if available
        if candidates_column in row and pd.notna(row[candidates_column]):
            try:
                cands = json.loads(str(row[candidates_column]))
                if isinstance(cands, list):
                    candidates.extend([str(c) for c in cands])
                else:
                    candidates.append(str(cands))
            except json.JSONDecodeError:
                pass
        
        # From FAQ source
        if 'faq' in sources:
            faq_responses = config.get('faq_responses', [])
            candidates.extend(faq_responses)
        
        # From templates source
        if 'templates' in sources:
            template_responses = config.get('template_responses', [])
            candidates.extend(template_responses)
        
        # From previous answers source
        if 'previous_answers' in sources:
            prev_responses = config.get('previous_answers', [])
            candidates.extend(prev_responses)
        
        return list(set(candidates))  # Remove duplicates
    
    def _select_best_response(self, query: str, candidates: List[str], 
                             context: str, model: str) -> tuple:
        """Select best response from candidates"""
        if not candidates:
            return '', 0.0
        
        if len(candidates) == 1:
            return candidates[0], 0.8
        
        if model == 'tfidf':
            return self._select_with_tfidf(query, candidates, context)
        elif model == 'dual_encoder':
            return self._select_with_dual_encoder(query, candidates, context)
        else:
            return self._select_with_tfidf(query, candidates, context)
    
    def _select_with_tfidf(self, query: str, candidates: List[str], context: str) -> tuple:
        """Select using TF-IDF similarity"""
        try:
            # Combine query with context for better matching
            query_text = f"{query} {context}".strip()
            
            # Vectorize
            all_texts = [query_text] + candidates
            vectors = self.vectorizer.fit_transform(all_texts)
            
            # Calculate similarities
            query_vector = vectors[0:1]
            candidate_vectors = vectors[1:]
            
            similarities = cosine_similarity(query_vector, candidate_vectors)[0]
            
            # Get best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            return candidates[best_idx], float(best_score)
        except Exception as e:
            logger.error(f"Error in TF-IDF selection: {e}")
            return candidates[0], 0.5
    
    def _select_with_dual_encoder(self, query: str, candidates: List[str], 
                                  context: str) -> tuple:
        """Select using dual encoder (simplified version)"""
        # For now, fall back to TF-IDF
        # In production, this would use trained dual encoders
        return self._select_with_tfidf(query, candidates, context)


# Singleton instance
_instance = None

def get_response_selection_step_service():
    global _instance
    if _instance is None:
        _instance = ResponseSelectionStepService()
    return _instance
