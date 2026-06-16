"""
Intent Disambiguation Step Service
Disambiguates when multiple intents are detected with similar confidence
"""
import pandas as pd
import json
from typing import Dict, Any, Optional, List
from app.services.base_service import BasePipelineService
from app.core.logger import get_logger

logger = get_logger(__name__)


class IntentDisambiguationStepService(BasePipelineService):
    """
    Pipeline step that disambiguates multiple detected intents
    
    Input: DataFrame with multiple intent predictions
    Output: DataFrame with disambiguated intent
    """
    
    def __init__(self):
        # Disambiguation rules
        self.disambiguation_rules = {}
        # Context-based preferences
        self.context_preferences = {}
        logger.info("IntentDisambiguationStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute intent disambiguation on DataFrame
        
        Args:
            data: DataFrame with intent predictions
            config: Configuration with:
                - intent_column: column with primary intent
                - confidence_column: column with intent confidence
                - all_intents_column: column with all intent predictions (JSON)
                - confidence_gap_threshold: minimum gap between top intents
                - ask_clarification: whether to request clarification
                - use_context_history: whether to use conversation context
                - context_column: column with conversation context
                - output_disambiguated_column: name for disambiguated intent
                - output_clarification_needed_column: name for clarification flag
                - output_clarification_options_column: name for options
        
        Returns:
            DataFrame with disambiguated intents
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for intent disambiguation")
        
        intent_column = config.get('intent_column', 'intent')
        confidence_column = config.get('confidence_column', 'intent_confidence')
        all_intents_column = config.get('all_intents_column', 'all_intents')
        gap_threshold = float(config.get('confidence_gap_threshold', 0.2))
        ask_clarification = config.get('ask_clarification', True)
        use_context = config.get('use_context_history', True)
        context_column = config.get('context_column', 'context')
        output_disambiguated = config.get('output_disambiguated_column', 'disambiguated_intent')
        output_clarification = config.get('output_clarification_needed_column', 'needs_clarification')
        output_options = config.get('output_clarification_options_column', 'clarification_options')
        
        logger.info(f"Disambiguating intents for {len(data)} samples")
        
        # Process each sample
        disambiguated_intents = []
        needs_clarification_list = []
        clarification_options_list = []
        
        for idx, row in data.iterrows():
            primary_intent = str(row.get(intent_column, 'unknown'))
            confidence = float(row.get(confidence_column, 0.0))
            context = str(row.get(context_column, '')) if use_context else ''
            
            # Get all intent predictions
            all_intents = []
            if all_intents_column in row and pd.notna(row[all_intents_column]):
                try:
                    all_intents = json.loads(str(row[all_intents_column]))
                    if not isinstance(all_intents, list):
                        all_intents = [{'intent': primary_intent, 'confidence': confidence}]
                except json.JSONDecodeError:
                    all_intents = [{'intent': primary_intent, 'confidence': confidence}]
            else:
                all_intents = [{'intent': primary_intent, 'confidence': confidence}]
            
            # Disambiguate
            result = self._disambiguate(
                all_intents, gap_threshold, ask_clarification, context
            )
            
            disambiguated_intents.append(result['intent'])
            needs_clarification_list.append(result['needs_clarification'])
            clarification_options_list.append(json.dumps(result['options']))
        
        # Add new columns to DataFrame
        result_df = data.copy()
        result_df[output_disambiguated] = disambiguated_intents
        result_df[output_clarification] = needs_clarification_list
        result_df[output_options] = clarification_options_list
        
        clarification_count = sum(needs_clarification_list)
        logger.info(f"Intent disambiguation completed. {clarification_count} require clarification")
        return result_df
    
    def _disambiguate(self, all_intents: List[Dict], gap_threshold: float,
                     ask_clarification: bool, context: str) -> Dict:
        """Disambiguate between multiple intents"""
        if not all_intents:
            return {
                'intent': 'unknown',
                'needs_clarification': False,
                'options': []
            }
        
        # Sort by confidence
        sorted_intents = sorted(all_intents, key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Single intent case
        if len(sorted_intents) == 1:
            return {
                'intent': sorted_intents[0]['intent'],
                'needs_clarification': False,
                'options': []
            }
        
        # Check confidence gap between top 2
        top_intent = sorted_intents[0]
        second_intent = sorted_intents[1]
        
        confidence_gap = top_intent['confidence'] - second_intent['confidence']
        
        # Clear winner
        if confidence_gap >= gap_threshold:
            return {
                'intent': top_intent['intent'],
                'needs_clarification': False,
                'options': []
            }
        
        # Ambiguous - try context-based disambiguation
        if context:
            disambiguated = self._context_based_disambiguation(sorted_intents, context)
            if disambiguated:
                return {
                    'intent': disambiguated,
                    'needs_clarification': False,
                    'options': []
                }
        
        # Try rule-based disambiguation
        rule_disambiguated = self._rule_based_disambiguation(sorted_intents)
        if rule_disambiguated:
            return {
                'intent': rule_disambiguated,
                'needs_clarification': False,
                'options': []
            }
        
        # Need clarification
        if ask_clarification:
            # Get top N ambiguous intents
            top_ambiguous = sorted_intents[:3]
            return {
                'intent': top_intent['intent'],  # Default to top
                'needs_clarification': True,
                'options': [i['intent'] for i in top_ambiguous]
            }
        else:
            # Default to top intent
            return {
                'intent': top_intent['intent'],
                'needs_clarification': False,
                'options': []
            }
    
    def _context_based_disambiguation(self, intents: List[Dict], context: str) -> Optional[str]:
        """Use context to disambiguate"""
        context_lower = context.lower()
        
        # Check context preferences
        for intent_info in intents:
            intent = intent_info['intent']
            if intent in self.context_preferences:
                keywords = self.context_preferences[intent]
                if any(kw in context_lower for kw in keywords):
                    logger.debug(f"Context-based disambiguation selected: {intent}")
                    return intent
        
        return None
    
    def _rule_based_disambiguation(self, intents: List[Dict]) -> Optional[str]:
        """Use rules to disambiguate"""
        # Check disambiguation rules
        intent_names = [i['intent'] for i in intents]
        intent_set = tuple(sorted(intent_names[:2]))  # Top 2
        
        if intent_set in self.disambiguation_rules:
            selected = self.disambiguation_rules[intent_set]
            logger.debug(f"Rule-based disambiguation selected: {selected}")
            return selected
        
        return None
    
    def add_disambiguation_rule(self, intent1: str, intent2: str, preferred: str):
        """Add a disambiguation rule"""
        key = tuple(sorted([intent1, intent2]))
        self.disambiguation_rules[key] = preferred
        logger.info(f"Added disambiguation rule: {key} -> {preferred}")
    
    def add_context_preference(self, intent: str, keywords: List[str]):
        """Add context-based preference"""
        self.context_preferences[intent] = keywords
        logger.info(f"Added context preference for {intent}: {keywords}")


# Singleton instance
_instance = None

def get_intent_disambiguation_step_service():
    global _instance
    if _instance is None:
        _instance = IntentDisambiguationStepService()
    return _instance
