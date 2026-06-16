"""
Coreference Resolution Step Service
Resolves pronoun references to entities in conversation history
"""
import pandas as pd
import json
import re
from typing import Dict, Any, Optional, List
from collections import deque, defaultdict
from app.services.base_service import BasePipelineService
from app.core.logger import get_logger

logger = get_logger(__name__)


class CoreferenceResolutionStepService(BasePipelineService):
    """
    Pipeline step that resolves coreferences (pronouns to entities)
    
    Input: DataFrame with message and entities columns
    Output: DataFrame with resolved text and resolved_entities columns
    """
    
    def __init__(self):
        # Store entity history by conversation
        self.entity_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        
        # Pronoun patterns
        self.pronouns = {
            'pt': {
                'singular_masculine': ['ele', 'dele', 'lhe', 'o', 'seu', 'este', 'esse', 'aquele'],
                'singular_feminine': ['ela', 'dela', 'lhe', 'a', 'sua', 'esta', 'essa', 'aquela'],
                'plural_masculine': ['eles', 'deles', 'lhes', 'os', 'seus', 'estes', 'esses', 'aqueles'],
                'plural_feminine': ['elas', 'delas', 'lhes', 'as', 'suas', 'estas', 'essas', 'aquelas'],
                'neutral': ['isso', 'isto', 'aquilo', 'o mesmo', 'a mesma']
            }
        }
        
        logger.info("CoreferenceResolutionStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute coreference resolution on DataFrame
        
        Args:
            data: DataFrame with message and entities columns
            config: Configuration with:
                - message_column: column with text message
                - entities_column: column with extracted entities (JSON)
                - conversation_id_column: column with conversation ID
                - max_history_mentions: maximum entities to track
                - resolution_strategy: strategy (entity_based, rule_based)
                - output_resolved_column: name for resolved text column
                - output_entities_column: name for resolved entities column
        
        Returns:
            DataFrame with resolved coreferences
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for coreference resolution")
        
        message_column = config.get('message_column', 'message')
        entities_column = config.get('entities_column', 'entities')
        conv_id_column = config.get('conversation_id_column', 'conversation_id')
        max_history = int(config.get('max_history_mentions', 5))
        strategy = config.get('resolution_strategy', 'entity_based')
        output_resolved = config.get('output_resolved_column', 'resolved_message')
        output_entities = config.get('output_entities_column', 'resolved_entities')
        
        if message_column not in data.columns:
            raise ValueError(f"Message column '{message_column}' not found")
        
        # Set default conversation ID if not present
        if conv_id_column not in data.columns:
            data[conv_id_column] = 'default'
        
        logger.info(f"Resolving coreferences for {len(data)} messages")
        
        # Process each message
        resolved_messages = []
        resolved_entities_list = []
        
        for idx, row in data.iterrows():
            message = str(row[message_column])
            conv_id = str(row.get(conv_id_column, 'default'))
            
            # Get entities if available
            current_entities = {}
            if entities_column in row and pd.notna(row[entities_column]):
                try:
                    current_entities = json.loads(str(row[entities_column]))
                except json.JSONDecodeError:
                    pass
            
            # Get entity history
            history = self.entity_history[conv_id]
            
            # Resolve coreferences
            resolved_msg, resolved_ents = self._resolve_coreferences(
                message, 
                current_entities, 
                list(history)[-max_history:],
                strategy
            )
            
            # Update history with current entities
            if current_entities:
                history.append(current_entities)
            
            resolved_messages.append(resolved_msg)
            resolved_entities_list.append(json.dumps(resolved_ents))
        
        # Add new columns to DataFrame
        result = data.copy()
        result[output_resolved] = resolved_messages
        result[output_entities] = resolved_entities_list
        
        logger.info(f"Coreference resolution completed")
        return result
    
    def _resolve_coreferences(self, text: str, current_entities: Dict, 
                             history: List[Dict], strategy: str) -> tuple:
        """Resolve pronouns to entities"""
        resolved_text = text
        resolved_entities = current_entities.copy()
        
        if strategy == 'entity_based' and history:
            # Get most recent entities
            recent_entities = {}
            for hist_entities in reversed(history):
                for entity_type, values in hist_entities.items():
                    if entity_type not in recent_entities and values:
                        recent_entities[entity_type] = values
            
            # Simple pronoun resolution
            text_lower = text.lower()
            
            # Resolve common pronouns
            for pronoun_list in self.pronouns['pt'].values():
                for pronoun in pronoun_list:
                    if pronoun in text_lower:
                        # Try to find matching entity
                        for entity_type, values in recent_entities.items():
                            if isinstance(values, list) and values:
                                # Replace pronoun with entity
                                pattern = r'\b' + re.escape(pronoun) + r'\b'
                                resolved_text = re.sub(pattern, values[0], resolved_text, flags=re.IGNORECASE)
                                
                                # Add to resolved entities
                                if entity_type not in resolved_entities:
                                    resolved_entities[entity_type] = []
                                if values[0] not in resolved_entities[entity_type]:
                                    resolved_entities[entity_type].append(values[0])
                                break
        
        return resolved_text, resolved_entities


# Singleton instance
_instance = None

def get_coreference_resolution_step_service():
    global _instance
    if _instance is None:
        _instance = CoreferenceResolutionStepService()
    return _instance
