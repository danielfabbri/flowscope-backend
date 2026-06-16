"""
Slot Filling Step Service
Fills form slots automatically from conversation
"""
import pandas as pd
import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from app.services.base_service import BasePipelineService
from app.core.logger import get_logger

logger = get_logger(__name__)


class SlotFillingStepService(BasePipelineService):
    """
    Pipeline step that fills slots/forms from conversation
    
    Input: DataFrame with message and entities
    Output: DataFrame with filled slots and validation status
    """
    
    def __init__(self):
        # Validation patterns
        self.validators = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?[\d\s\-()]{10,}$',
            'date': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'time': r'\d{1,2}:\d{2}',
            'cpf': r'\d{3}\.?\d{3}\.?\d{3}-?\d{2}',
            'cep': r'\d{5}-?\d{3}'
        }
        logger.info("SlotFillingStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute slot filling on DataFrame
        
        Args:
            data: DataFrame with message and entities
            config: Configuration with:
                - message_column: column with message text
                - entities_column: column with extracted entities (JSON)
                - form_schema: definition of form/slots to fill
                - validation_rules: validation rules for slots
                - retry_on_invalid: whether to request retry on invalid data
                - output_slots_column: name for filled slots column
                - output_valid_column: name for validation status column
                - output_missing_column: name for missing required slots
        
        Returns:
            DataFrame with slot filling results
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for slot filling")
        
        message_column = config.get('message_column', 'message')
        entities_column = config.get('entities_column', 'entities')
        form_schema = config.get('form_schema', {})
        validation_rules = config.get('validation_rules', {})
        retry_on_invalid = config.get('retry_on_invalid', True)
        output_slots = config.get('output_slots_column', 'filled_slots')
        output_valid = config.get('output_valid_column', 'slots_valid')
        output_missing = config.get('output_missing_column', 'missing_slots')
        
        if not form_schema:
            raise ValueError("form_schema must be provided in config")
        
        logger.info(f"Filling slots for {len(data)} messages")
        logger.info(f"Form schema: {list(form_schema.keys())}")
        
        # Process each message
        filled_slots_list = []
        valid_list = []
        missing_list = []
        
        for idx, row in data.iterrows():
            message = str(row.get(message_column, ''))
            
            # Get entities if available
            entities = {}
            if entities_column in row and pd.notna(row[entities_column]):
                try:
                    entities = json.loads(str(row[entities_column]))
                except json.JSONDecodeError:
                    pass
            
            # Fill slots
            filled, valid, missing = self._fill_slots(
                message, entities, form_schema, validation_rules, retry_on_invalid
            )
            
            filled_slots_list.append(json.dumps(filled))
            valid_list.append(valid)
            missing_list.append(','.join(missing))
        
        # Add new columns to DataFrame
        result = data.copy()
        result[output_slots] = filled_slots_list
        result[output_valid] = valid_list
        result[output_missing] = missing_list
        
        completed = sum(1 for v in valid_list if v)
        logger.info(f"Slot filling completed. {completed}/{len(data)} forms fully valid")
        return result
    
    def _fill_slots(self, message: str, entities: Dict, schema: Dict, 
                   validation_rules: Dict, retry_on_invalid: bool) -> tuple:
        """Fill slots from message and entities"""
        filled_slots = {}
        missing_slots = []
        all_valid = True
        
        # Get form definition (support multiple forms)
        form_name = list(schema.keys())[0] if schema else 'default'
        required_slots = schema.get(form_name, [])
        
        for slot_name in required_slots:
            value = None
            
            # Try to extract from entities first
            if slot_name in entities:
                value = entities[slot_name]
                if isinstance(value, list) and value:
                    value = value[0]
            
            # Try to extract from message using patterns
            if not value:
                value = self._extract_from_message(message, slot_name)
            
            # Validate if value found
            if value:
                is_valid = self._validate_slot(slot_name, value, validation_rules)
                if is_valid:
                    filled_slots[slot_name] = value
                else:
                    all_valid = False
                    if retry_on_invalid:
                        missing_slots.append(slot_name)
            else:
                missing_slots.append(slot_name)
                all_valid = False
        
        return filled_slots, all_valid, missing_slots
    
    def _extract_from_message(self, message: str, slot_name: str) -> Optional[str]:
        """Extract slot value from message using patterns"""
        # Check common slot types
        if 'email' in slot_name.lower():
            match = re.search(self.validators['email'], message)
            if match:
                return match.group(0)
        
        if 'phone' in slot_name.lower() or 'telefone' in slot_name.lower():
            match = re.search(self.validators['phone'], message)
            if match:
                return match.group(0)
        
        if 'data' in slot_name.lower() or 'date' in slot_name.lower():
            match = re.search(self.validators['date'], message)
            if match:
                return match.group(0)
        
        if 'hora' in slot_name.lower() or 'time' in slot_name.lower():
            match = re.search(self.validators['time'], message)
            if match:
                return match.group(0)
        
        if 'cpf' in slot_name.lower():
            match = re.search(self.validators['cpf'], message)
            if match:
                return match.group(0)
        
        if 'cep' in slot_name.lower():
            match = re.search(self.validators['cep'], message)
            if match:
                return match.group(0)
        
        # For name, try to extract capitalized words
        if 'nome' in slot_name.lower() or 'name' in slot_name.lower():
            words = message.split()
            capitalized = [w for w in words if w and w[0].isupper()]
            if capitalized:
                return ' '.join(capitalized)
        
        return None
    
    def _validate_slot(self, slot_name: str, value: str, rules: Dict) -> bool:
        """Validate slot value"""
        # Check custom rules first
        if slot_name in rules:
            rule = rules[slot_name]
            if 'pattern' in rule:
                if not re.match(rule['pattern'], str(value)):
                    return False
            if 'min_length' in rule:
                if len(str(value)) < rule['min_length']:
                    return False
            if 'max_length' in rule:
                if len(str(value)) > rule['max_length']:
                    return False
            return True
        
        # Check built-in validators
        for validator_name, pattern in self.validators.items():
            if validator_name in slot_name.lower():
                return bool(re.match(pattern, str(value)))
        
        # Default: accept if not empty
        return bool(value and str(value).strip())


# Singleton instance
_instance = None

def get_slot_filling_step_service():
    global _instance
    if _instance is None:
        _instance = SlotFillingStepService()
    return _instance
