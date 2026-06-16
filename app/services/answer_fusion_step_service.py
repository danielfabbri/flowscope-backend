"""
Answer Fusion Step Service
Fuses multiple candidate answers into a single coherent response
"""
import pandas as pd
import json
import numpy as np
from typing import Dict, Any, Optional, List
from collections import Counter
from app.services.base_service import BasePipelineService
from app.core.logger import get_logger

logger = get_logger(__name__)


class AnswerFusionStepService(BasePipelineService):
    """
    Pipeline step that fuses multiple answer candidates
    
    Input: DataFrame with multiple answer candidates
    Output: DataFrame with fused answer
    """
    
    def __init__(self):
        logger.info("AnswerFusionStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute answer fusion on DataFrame
        
        Args:
            data: DataFrame with answer candidates
            config: Configuration with:
                - candidates_column: column with answer candidates (JSON)
                - fusion_strategy: strategy (weighted_voting, concatenate, extract_best)
                - confidence_based: whether to weight by confidence
                - max_length: maximum length of fused answer
                - remove_duplicates: whether to remove duplicate content
                - output_fused_column: name for fused answer column
                - output_sources_column: name for source tracking column
        
        Returns:
            DataFrame with fused answers
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for answer fusion")
        
        candidates_column = config.get('candidates_column', 'answer_candidates')
        strategy = config.get('fusion_strategy', 'weighted_voting')
        confidence_based = config.get('confidence_based', True)
        max_length = int(config.get('max_length', 500))
        remove_duplicates = config.get('remove_duplicates', True)
        output_fused = config.get('output_fused_column', 'fused_answer')
        output_sources = config.get('output_sources_column', 'answer_sources')
        
        if candidates_column not in data.columns:
            raise ValueError(f"Candidates column '{candidates_column}' not found")
        
        logger.info(f"Fusing answers for {len(data)} queries")
        logger.info(f"Strategy: {strategy}")
        
        # Process each row
        fused_answers = []
        sources_list = []
        
        for idx, row in data.iterrows():
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
                fused_answers.append("")
                sources_list.append(json.dumps([]))
                continue
            
            # Fuse answers
            fused, sources = self._fuse_answers(
                candidates, strategy, confidence_based, max_length, remove_duplicates
            )
            
            fused_answers.append(fused)
            sources_list.append(json.dumps(sources))
        
        # Add new columns to DataFrame
        result = data.copy()
        result[output_fused] = fused_answers
        result[output_sources] = sources_list
        
        logger.info(f"Answer fusion completed")
        return result
    
    def _fuse_answers(self, candidates: List, strategy: str, confidence_based: bool,
                     max_length: int, remove_duplicates: bool) -> tuple:
        """Fuse multiple answer candidates"""
        # Normalize candidates to dict format
        normalized = []
        for cand in candidates:
            if isinstance(cand, str):
                normalized.append({'answer': cand, 'confidence': 0.5, 'source': 'unknown'})
            elif isinstance(cand, dict):
                if 'answer' not in cand:
                    cand['answer'] = str(cand)
                if 'confidence' not in cand:
                    cand['confidence'] = 0.5
                if 'source' not in cand:
                    cand['source'] = 'unknown'
                normalized.append(cand)
        
        candidates = normalized
        
        if strategy == 'weighted_voting':
            return self._weighted_voting_fusion(candidates, confidence_based, max_length)
        elif strategy == 'concatenate':
            return self._concatenate_fusion(candidates, max_length, remove_duplicates)
        elif strategy == 'extract_best':
            return self._extract_best_fusion(candidates)
        elif strategy == 'majority_voting':
            return self._majority_voting_fusion(candidates)
        else:
            return self._weighted_voting_fusion(candidates, confidence_based, max_length)
    
    def _weighted_voting_fusion(self, candidates: List[Dict], confidence_based: bool,
                               max_length: int) -> tuple:
        """Fuse using weighted voting"""
        if not candidates:
            return "", []
        
        # Weight by confidence if enabled
        if confidence_based:
            # Sort by confidence
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.get('confidence', 0),
                reverse=True
            )
        else:
            sorted_candidates = candidates
        
        # Take top answer
        best = sorted_candidates[0]
        fused_answer = best['answer']
        
        # Truncate if needed
        if len(fused_answer) > max_length:
            fused_answer = fused_answer[:max_length] + "..."
        
        sources = [c.get('source', 'unknown') for c in sorted_candidates[:3]]
        
        return fused_answer, sources
    
    def _concatenate_fusion(self, candidates: List[Dict], max_length: int,
                           remove_duplicates: bool) -> tuple:
        """Fuse by concatenating answers"""
        answers = []
        sources = []
        
        seen = set()
        for cand in candidates:
            answer = cand['answer']
            
            # Remove duplicates if enabled
            if remove_duplicates:
                answer_lower = answer.lower().strip()
                if answer_lower in seen:
                    continue
                seen.add(answer_lower)
            
            answers.append(answer)
            sources.append(cand.get('source', 'unknown'))
        
        # Concatenate with separator
        fused = " | ".join(answers)
        
        # Truncate if needed
        if len(fused) > max_length:
            fused = fused[:max_length] + "..."
        
        return fused, sources
    
    def _extract_best_fusion(self, candidates: List[Dict]) -> tuple:
        """Extract single best answer"""
        if not candidates:
            return "", []
        
        # Sort by confidence
        best = max(candidates, key=lambda x: x.get('confidence', 0))
        
        return best['answer'], [best.get('source', 'unknown')]
    
    def _majority_voting_fusion(self, candidates: List[Dict]) -> tuple:
        """Fuse using majority voting"""
        if not candidates:
            return "", []
        
        # Count answer occurrences (normalized)
        answer_counts = Counter()
        answer_map = {}  # Map normalized to original
        sources_map = {}
        
        for cand in candidates:
            answer = cand['answer']
            normalized = answer.lower().strip()
            
            answer_counts[normalized] += 1
            
            if normalized not in answer_map:
                answer_map[normalized] = answer
                sources_map[normalized] = []
            
            sources_map[normalized].append(cand.get('source', 'unknown'))
        
        # Get most common
        most_common = answer_counts.most_common(1)[0][0]
        fused_answer = answer_map[most_common]
        sources = sources_map[most_common]
        
        return fused_answer, sources


# Singleton instance
_instance = None

def get_answer_fusion_step_service():
    global _instance
    if _instance is None:
        _instance = AnswerFusionStepService()
    return _instance
