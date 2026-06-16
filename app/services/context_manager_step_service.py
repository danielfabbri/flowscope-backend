"""
Context Manager Step Service
Manages conversation context for pipeline processing
"""
import pandas as pd
import json
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from app.services.base_service import BasePipelineService
from app.core.logger import get_logger

logger = get_logger(__name__)


class ContextManagerStepService(BasePipelineService):
    """
    Pipeline step that manages conversation context
    
    Input: DataFrame with message and conversation_id columns
    Output: DataFrame with added conversation_context column
    """
    
    def __init__(self):
        # Store conversation histories by conversation_id
        self.conversations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.conversation_entities: Dict[str, Dict] = defaultdict(dict)
        logger.info("ContextManagerStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute context management on DataFrame
        
        Args:
            data: DataFrame with message and conversation_id columns
            config: Configuration with:
                - message_column: column name with current message
                - conversation_id_column: column name with conversation ID
                - max_history: maximum messages to keep in context
                - include_entities: whether to include extracted entities
                - merge_strategy: how to merge history (concatenate, summary, last_n)
                - output_column: name for context output column
        
        Returns:
            DataFrame with added conversation_context column
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for context management")
        
        message_column = config.get('message_column', 'message')
        conv_id_column = config.get('conversation_id_column', 'conversation_id')
        max_history = int(config.get('max_history', 10))
        include_entities = config.get('include_entities', True)
        merge_strategy = config.get('merge_strategy', 'concatenate')
        output_column = config.get('output_column', 'conversation_context')
        
        if message_column not in data.columns:
            raise ValueError(f"Message column '{message_column}' not found in data. Available: {list(data.columns)}")
        
        if conv_id_column not in data.columns:
            # Create a default conversation ID if not present
            logger.warning(f"Conversation ID column '{conv_id_column}' not found. Using default ID 'default'")
            data[conv_id_column] = 'default'
        
        logger.info(f"Managing context for {len(data)} messages")
        
        # Process each message
        contexts = []
        
        for idx, row in data.iterrows():
            message = str(row[message_column])
            conv_id = str(row[conv_id_column])
            
            # Get conversation history
            history = self.conversations[conv_id]
            
            # Build context based on merge strategy
            if merge_strategy == 'concatenate':
                # Concatenate all history
                context_text = ' | '.join(list(history)[-max_history:])
            elif merge_strategy == 'last_n':
                # Just last N messages
                context_text = ' | '.join(list(history)[-max_history:])
            elif merge_strategy == 'summary':
                # Simple summary (count + last message)
                if len(history) > 0:
                    context_text = f"Previous messages: {len(history)}. Last: {list(history)[-1]}"
                else:
                    context_text = "No previous messages"
            else:
                context_text = ' | '.join(list(history)[-max_history:])
            
            # Include entities if available and requested
            entities_info = {}
            if include_entities:
                # Check if entities column exists from previous step
                if 'entities' in row:
                    try:
                        entities = json.loads(str(row['entities']))
                        entities_info = entities
                        # Update conversation entities
                        for entity_type, entity_values in entities.items():
                            if isinstance(entity_values, list):
                                for value in entity_values:
                                    if entity_type not in self.conversation_entities[conv_id]:
                                        self.conversation_entities[conv_id][entity_type] = []
                                    if value not in self.conversation_entities[conv_id][entity_type]:
                                        self.conversation_entities[conv_id][entity_type].append(value)
                    except:
                        pass
            
            # Build context object
            context = {
                'history': context_text,
                'message_count': len(history),
                'entities': entities_info,
                'conversation_entities': self.conversation_entities[conv_id]
            }
            
            contexts.append(context)
            
            # Add message to history
            history.append(message)
        
        # Add new column to DataFrame
        result = data.copy()
        result[output_column] = [json.dumps(c) for c in contexts]
        
        # Log statistics
        unique_conversations = len(set(data[conv_id_column]))
        avg_history = sum(c['message_count'] for c in contexts) / len(contexts) if contexts else 0
        logger.info(f"Context management complete. {unique_conversations} conversations, avg history: {avg_history:.1f}")
        
        return result
    
    def clear_conversation(self, conversation_id: str):
        """Clear history for a specific conversation"""
        if conversation_id in self.conversations:
            self.conversations[conversation_id].clear()
            self.conversation_entities[conversation_id].clear()
            logger.info(f"Cleared conversation: {conversation_id}")
    
    def clear_all(self):
        """Clear all conversation histories"""
        self.conversations.clear()
        self.conversation_entities.clear()
        logger.info("Cleared all conversations")


def get_context_manager_step_service() -> ContextManagerStepService:
    """Get singleton instance"""
    if not hasattr(get_context_manager_step_service, '_instance'):
        get_context_manager_step_service._instance = ContextManagerStepService()
    return get_context_manager_step_service._instance
