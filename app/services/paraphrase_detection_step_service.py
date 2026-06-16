"""
Paraphrase Detection Step Service
Detects if two texts are semantic paraphrases of each other
"""
import pandas as pd
import json
import numpy as np
from typing import Dict, Any, Optional, List
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.services.base_service import BasePipelineService
from app.core.logger import get_logger

logger = get_logger(__name__)


class ParaphraseDetectionStepService(BasePipelineService):
    """
    Pipeline step that detects paraphrases
    
    Input: DataFrame with text pairs or query text
    Output: DataFrame with paraphrase detection results
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1
        )
        # Cache for quick lookup
        self.paraphrase_cache: Dict[str, List[str]] = defaultdict(list)
        self.response_cache: Dict[str, Any] = {}
        logger.info("ParaphraseDetectionStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute paraphrase detection on DataFrame
        
        Args:
            data: DataFrame with text pairs or queries
            config: Configuration with:
                - text1_column: first text column (or query column)
                - text2_column: second text column (optional)
                - similarity_threshold: threshold for paraphrase detection
                - use_cache: whether to use response cache
                - reference_texts: list of reference texts to compare against
                - output_is_paraphrase_column: name for boolean output
                - output_similarity_column: name for similarity score
                - output_matched_text_column: name for matched text
        
        Returns:
            DataFrame with paraphrase detection results
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for paraphrase detection")
        
        text1_column = config.get('text1_column', 'message')
        text2_column = config.get('text2_column', None)
        threshold = float(config.get('similarity_threshold', 0.85))
        use_cache = config.get('use_cache', True)
        reference_texts = config.get('reference_texts', [])
        output_is_para = config.get('output_is_paraphrase_column', 'is_paraphrase')
        output_similarity = config.get('output_similarity_column', 'paraphrase_similarity')
        output_matched = config.get('output_matched_text_column', 'matched_text')
        
        if text1_column not in data.columns:
            raise ValueError(f"Text column '{text1_column}' not found")
        
        logger.info(f"Detecting paraphrases for {len(data)} texts")
        
        # Two modes: pairwise comparison or reference comparison
        if text2_column and text2_column in data.columns:
            # Pairwise mode
            results = self._pairwise_comparison(data, text1_column, text2_column, threshold)
        else:
            # Reference mode
            results = self._reference_comparison(
                data, text1_column, reference_texts, threshold, use_cache
            )
        
        # Add new columns to DataFrame
        result = data.copy()
        result[output_is_para] = [r['is_paraphrase'] for r in results]
        result[output_similarity] = [r['similarity'] for r in results]
        result[output_matched] = [r['matched_text'] for r in results]
        
        logger.info(f"Paraphrase detection completed. Found {sum(r['is_paraphrase'] for r in results)} paraphrases")
        return result
    
    def _pairwise_comparison(self, data: pd.DataFrame, col1: str, col2: str, 
                            threshold: float) -> List[Dict]:
        """Compare text pairs"""
        results = []
        
        texts1 = data[col1].astype(str).tolist()
        texts2 = data[col2].astype(str).tolist()
        
        # Vectorize all texts
        all_texts = texts1 + texts2
        try:
            vectors = self.vectorizer.fit_transform(all_texts)
            n = len(texts1)
            vectors1 = vectors[:n]
            vectors2 = vectors[n:]
            
            # Calculate similarities
            similarities = np.array([
                cosine_similarity(vectors1[i:i+1], vectors2[i:i+1])[0][0]
                for i in range(n)
            ])
            
            for i in range(len(data)):
                sim = similarities[i]
                results.append({
                    'is_paraphrase': bool(sim >= threshold),
                    'similarity': round(float(sim), 4),
                    'matched_text': texts2[i] if sim >= threshold else ''
                })
        except Exception as e:
            logger.error(f"Error in pairwise comparison: {e}")
            # Return default values
            results = [{
                'is_paraphrase': False,
                'similarity': 0.0,
                'matched_text': ''
            } for _ in range(len(data))]
        
        return results
    
    def _reference_comparison(self, data: pd.DataFrame, text_column: str, 
                            reference_texts: List[str], threshold: float,
                            use_cache: bool) -> List[Dict]:
        """Compare texts against reference corpus"""
        results = []
        
        if not reference_texts:
            # No references, return defaults
            return [{
                'is_paraphrase': False,
                'similarity': 0.0,
                'matched_text': ''
            } for _ in range(len(data))]
        
        texts = data[text_column].astype(str).tolist()
        
        try:
            # Vectorize reference texts
            ref_vectors = self.vectorizer.fit_transform(reference_texts)
            
            for text in texts:
                # Check cache first
                if use_cache and text in self.paraphrase_cache:
                    cached = self.paraphrase_cache[text]
                    if cached:
                        results.append({
                            'is_paraphrase': True,
                            'similarity': 1.0,
                            'matched_text': cached[0]
                        })
                        continue
                
                # Vectorize query
                query_vector = self.vectorizer.transform([text])
                
                # Calculate similarities
                similarities = cosine_similarity(query_vector, ref_vectors)[0]
                max_sim_idx = np.argmax(similarities)
                max_sim = similarities[max_sim_idx]
                
                is_paraphrase = bool(max_sim >= threshold)
                matched = reference_texts[max_sim_idx] if is_paraphrase else ''
                
                # Update cache
                if use_cache and is_paraphrase:
                    self.paraphrase_cache[text].append(matched)
                
                results.append({
                    'is_paraphrase': is_paraphrase,
                    'similarity': round(float(max_sim), 4),
                    'matched_text': matched
                })
        except Exception as e:
            logger.error(f"Error in reference comparison: {e}")
            results = [{
                'is_paraphrase': False,
                'similarity': 0.0,
                'matched_text': ''
            } for _ in range(len(data))]
        
        return results


# Singleton instance
_instance = None

def get_paraphrase_detection_step_service():
    global _instance
    if _instance is None:
        _instance = ParaphraseDetectionStepService()
    return _instance
