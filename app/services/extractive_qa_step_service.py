"""
Extractive QA Step Service
Extracts exact answer spans from context documents
"""
import pandas as pd
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from app.services.base_service import BasePipelineService
from app.core.logger import get_logger

logger = get_logger(__name__)


class ExtractiveQAStepService(BasePipelineService):
    """
    Pipeline step that extracts exact answers from context
    
    Input: DataFrame with question and context columns
    Output: DataFrame with extracted answer spans
    """
    
    def __init__(self):
        # Question patterns and their typical answer patterns
        self.question_patterns = {
            'quando': ['em ', 'no dia', 'na data', 'às', r'\d{1,2}[/:-]\d{1,2}', r'\d{4}'],
            'onde': ['em ', 'no ', 'na ', 'em casa', 'no local'],
            'quem': ['pessoa', 'time', 'jogador', 'técnico', r'[A-Z][a-z]+ [A-Z][a-z]+'],
            'quanto': [r'\d+', 'reais', 'gols', 'pontos', r'\d+[%]'],
            'qual': ['é ', 'foi ', 'será '],
            'porque': ['porque', 'pois', 'devido a', 'por causa'],
            'como': ['através', 'por meio', 'usando', 'com']
        }
        
        logger.info("ExtractiveQAStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute extractive QA on DataFrame
        
        Args:
            data: DataFrame with question and context columns
            config: Configuration with:
                - question_column: column with question text
                - context_column: column with context text
                - max_answer_length: maximum answer length in words
                - min_confidence: minimum confidence threshold
                - output_answer_column: name for answer column
                - output_confidence_column: name for confidence column
                - output_span_column: name for span position column
        
        Returns:
            DataFrame with extracted answers
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for extractive QA")
        
        question_column = config.get('question_column', 'message')
        context_column = config.get('context_column', 'context')
        max_length = int(config.get('max_answer_length', 50))
        min_confidence = float(config.get('min_confidence', 0.7))
        output_answer = config.get('output_answer_column', 'extracted_answer')
        output_confidence = config.get('output_confidence_column', 'answer_confidence')
        output_span = config.get('output_span_column', 'answer_span')
        
        if question_column not in data.columns:
            raise ValueError(f"Question column '{question_column}' not found")
        
        logger.info(f"Extracting answers for {len(data)} questions")
        
        # Process each question
        answers = []
        confidences = []
        spans = []
        
        for idx, row in data.iterrows():
            question = str(row[question_column])
            context = str(row.get(context_column, ''))
            
            if not context or context == 'nan':
                answers.append('')
                confidences.append(0.0)
                spans.append('')
                continue
            
            # Extract answer
            answer, confidence, span = self._extract_answer(question, context, max_length)
            
            # Apply confidence threshold
            if confidence < min_confidence:
                answer = ''
                span = ''
            
            answers.append(answer)
            confidences.append(round(confidence, 4))
            spans.append(span)
        
        # Add new columns to DataFrame
        result = data.copy()
        result[output_answer] = answers
        result[output_confidence] = confidences
        result[output_span] = spans
        
        logger.info(f"Extractive QA completed. Found {sum(1 for a in answers if a)} answers")
        return result
    
    def _extract_answer(self, question: str, context: str, max_length: int) -> Tuple[str, float, str]:
        """Extract answer span from context"""
        question_lower = question.lower()
        
        # Identify question type
        question_type = self._identify_question_type(question_lower)
        
        # Find answer candidates based on question type
        candidates = self._find_answer_candidates(context, question_type, question_lower, max_length)
        
        if not candidates:
            return '', 0.0, ''
        
        # Rank candidates
        best_candidate = self._rank_candidates(candidates, question_lower, context)
        
        return best_candidate['text'], best_candidate['confidence'], best_candidate['span']
    
    def _identify_question_type(self, question: str) -> str:
        """Identify the type of question"""
        for q_type in self.question_patterns.keys():
            if q_type in question:
                return q_type
        return 'unknown'
    
    def _find_answer_candidates(self, context: str, q_type: str, question: str, 
                                max_length: int) -> List[Dict]:
        """Find candidate answer spans"""
        candidates = []
        
        # Get patterns for this question type
        patterns = self.question_patterns.get(q_type, [])
        
        # Find matches for each pattern
        for pattern in patterns:
            for match in re.finditer(pattern, context, re.IGNORECASE):
                start = match.start()
                end = match.end()
                
                # Expand to word boundaries
                expanded_start = context.rfind(' ', 0, start) + 1
                expanded_end = context.find(' ', end)
                if expanded_end == -1:
                    expanded_end = len(context)
                
                # Extract candidate text
                candidate_text = context[expanded_start:expanded_end].strip()
                
                # Check length
                if len(candidate_text.split()) <= max_length:
                    candidates.append({
                        'text': candidate_text,
                        'span': f"{expanded_start}:{expanded_end}",
                        'pattern': pattern,
                        'confidence': 0.5
                    })
        
        # Also try sentence-level extraction
        sentences = re.split(r'[.!?]+', context)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) <= max_length:
                # Check if sentence contains question keywords
                question_words = set(question.split()) - {'o', 'a', 'é', 'foi', 'quando', 'onde', 'quem'}
                sentence_words = set(sentence.lower().split())
                
                overlap = len(question_words & sentence_words)
                if overlap >= 2:
                    candidates.append({
                        'text': sentence,
                        'span': f"0:{len(sentence)}",
                        'pattern': 'sentence',
                        'confidence': 0.4
                    })
        
        return candidates
    
    def _rank_candidates(self, candidates: List[Dict], question: str, context: str) -> Dict:
        """Rank candidates and return best one"""
        if not candidates:
            return {'text': '', 'confidence': 0.0, 'span': ''}
        
        # Score each candidate
        for candidate in candidates:
            score = candidate['confidence']
            
            # Bonus for exact word matches with question
            question_words = set(question.split())
            candidate_words = set(candidate['text'].lower().split())
            overlap = len(question_words & candidate_words)
            score += overlap * 0.1
            
            # Bonus for shorter answers (more precise)
            length_penalty = len(candidate['text'].split()) / 50
            score -= length_penalty * 0.1
            
            # Bonus for pattern-based matches
            if candidate['pattern'] != 'sentence':
                score += 0.2
            
            candidate['confidence'] = min(score, 1.0)
        
        # Return best candidate
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        return candidates[0]


# Singleton instance
_instance = None

def get_extractive_qa_step_service():
    global _instance
    if _instance is None:
        _instance = ExtractiveQAStepService()
    return _instance
