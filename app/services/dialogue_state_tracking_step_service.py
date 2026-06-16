"""
Dialogue State Tracking Step Service
Tracks dialogue state and manages slot filling for structured conversations
"""
import pandas as pd
import json
from typing import Dict, Any, Optional, List
from collections import defaultdict
from app.services.base_service import BasePipelineService
from app.core.logger import get_logger

logger = get_logger(__name__)


class DialogueStateTrackingStepService(BasePipelineService):
    """
    Pipeline step that tracks dialogue state and fills slots
    
    Input: DataFrame with message, entities, and intent columns
    Output: DataFrame with added dialogue_state and missing_slots columns
    """
    
    def __init__(self):
        # Store dialogue states by conversation_id
        self.dialogue_states: Dict[str, Dict] = defaultdict(lambda: {
            "slots": {},
            "confirmed_slots": set(),
            "required_slots": [],
            "current_intent": None,
            "state": "initial"
        })
        logger.info("DialogueStateTrackingStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute dialogue state tracking on DataFrame
        
        Args:
            data: DataFrame with message, entities, intent columns
            config: Configuration with:
                - conversation_id_column: column with conversation ID
                - intent_column: column with detected intent
                - entities_column: column with extracted entities (JSON)
                - slots: list of slot names to track
                - slot_entity_mapping: mapping of slots to entity types
                - confirmation_required: whether to require confirmation
                - min_confidence_slot: minimum confidence to fill slot
                - output_state_column: name for state output column
                - output_missing_column: name for missing slots column
        
        Returns:
            DataFrame with added dialogue state information
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for dialogue state tracking")
        
        conv_id_column = config.get('conversation_id_column', 'conversation_id')
        intent_column = config.get('intent_column', 'intent')
        entities_column = config.get('entities_column', 'entities')
        slots = config.get('slots', [])
        slot_entity_mapping = config.get('slot_entity_mapping', {})
        confirmation_required = config.get('confirmation_required', True)
        min_confidence = float(config.get('min_confidence_slot', 0.6))
        output_state = config.get('output_state_column', 'dialogue_state')
        output_missing = config.get('output_missing_column', 'missing_slots')
        
        # Set default conversation ID if not present
        if conv_id_column not in data.columns:
            logger.warning(f"Conversation ID column '{conv_id_column}' not found. Using default ID")
            data[conv_id_column] = 'default'
        
        logger.info(f"Tracking dialogue state for {len(data)} messages")
        logger.info(f"Tracking slots: {slots}")
        
        # Process each message
        states = []
        missing_slots_list = []
        
        for idx, row in data.iterrows():
            conv_id = str(row.get(conv_id_column, 'default'))
            intent = str(row.get(intent_column, 'unknown'))
            
            # Get or initialize dialogue state
            state = self.dialogue_states[conv_id]
            
            # Update intent
            if intent != 'unknown':
                state['current_intent'] = intent
                # Set required slots based on intent
                if not state['required_slots']:
                    state['required_slots'] = slots
            
            # Extract and fill slots from entities
            if entities_column in row and pd.notna(row[entities_column]):
                try:
                    entities = json.loads(str(row[entities_column]))
                    self._fill_slots_from_entities(state, entities, slot_entity_mapping, min_confidence)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse entities JSON for row {idx}")
            
            # Determine missing slots
            missing = [s for s in state['required_slots'] if s not in state['slots']]
            
            # Update state status
            if len(missing) == 0:
                if confirmation_required and len(state['confirmed_slots']) < len(state['required_slots']):
                    state['state'] = 'awaiting_confirmation'
                else:
                    state['state'] = 'complete'
            else:
                state['state'] = 'filling'
            
            # Store results
            states.append(json.dumps({
                'slots': state['slots'],
                'required_slots': state['required_slots'],
                'state': state['state'],
                'intent': state['current_intent']
            }))
            missing_slots_list.append(','.join(missing))
        
        # Add new columns to DataFrame
        result = data.copy()
        result[output_state] = states
        result[output_missing] = missing_slots_list
        
        logger.info(f"Dialogue state tracking completed")
        return result
    
    def _fill_slots_from_entities(self, state: Dict, entities: Dict, mapping: Dict, min_confidence: float):
        """Fill slots from extracted entities"""
        for slot, entity_types in mapping.items():
            if not isinstance(entity_types, list):
                entity_types = [entity_types]
            
            for entity_type in entity_types:
                if entity_type in entities:
                    entity_values = entities[entity_type]
                    if isinstance(entity_values, list) and len(entity_values) > 0:
                        # Take first value (could be improved with confidence scoring)
                        state['slots'][slot] = entity_values[0]
                        break


# Singleton instance
_instance = None

def get_dialogue_state_tracking_step_service():
    global _instance
    if _instance is None:
        _instance = DialogueStateTrackingStepService()
    return _instance
