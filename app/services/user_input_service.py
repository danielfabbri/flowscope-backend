from typing import Dict, Any
import pandas as pd
import logging
from .base_service import BasePipelineService

class UserInputService(BasePipelineService):
    """Service for manual user input data entry."""
    
    def execute(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a DataFrame from user input.
        
        Config:
        - temperature_celsius: Temperature value (or any other direct field)
        - input_fields: Dict with field names and values (alternative format)
        - use_defaults: Whether to fill missing features with smart defaults
        - default_values: Dict or JSON string of default values for features
        """
        logger = logging.getLogger(__name__)
        
        # Parse input fields - support both direct fields and input_fields dict
        input_fields = config.get('input_fields', {})
        use_defaults = config.get('use_defaults', True)
        default_values_raw = config.get('default_values', {})
        
        # If input_fields is empty, collect all non-system config keys as input
        if not input_fields:
            system_keys = {'input_fields', 'use_defaults', 'default_values', 'prompt_on_run'}
            input_fields = {k: v for k, v in config.items() 
                          if k not in system_keys and v != '' and v is not None}
        
        if not input_fields:
            raise ValueError("No input fields provided. Please add at least one field value.")
        
        # Parse default_values if it's a JSON string
        if isinstance(default_values_raw, str) and default_values_raw.strip():
            try:
                import json
                default_values = json.loads(default_values_raw)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse default_values JSON: {e}. Using empty dict.")
                default_values = {}
        else:
            default_values = default_values_raw if isinstance(default_values_raw, dict) else {}
        
        logger.info(f"Creating DataFrame from user input: {input_fields}")
        
        # Create base data from user input
        data_dict = {}
        for field, value in input_fields.items():
            # Try to convert to appropriate type
            data_dict[field] = [self._convert_value(value)]
        
        # Add smart defaults if enabled
        if use_defaults:
            data_dict = self._add_smart_defaults(data_dict, default_values)
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        logger.info(f"✅ Created input DataFrame with {len(df.columns)} columns: {list(df.columns)}")
        logger.info(f"Sample data: {df.iloc[0].to_dict()}")
        
        return df
    
    def _convert_value(self, value: Any) -> Any:
        """Convert string value to appropriate type."""
        if isinstance(value, str):
            # Try to convert to number
            try:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            except (ValueError, TypeError):
                return value
        return value
    
    def _add_smart_defaults(self, data_dict: Dict, custom_defaults: Dict) -> Dict:
        """Add smart default values for common ML features."""
        logger = logging.getLogger(__name__)
        
        # Default values for common features (can be overridden by custom_defaults)
        smart_defaults = {
            # Time features
            'day_of_week': 3,  # Wednesday
            'day_name': 'Wednesday',
            'month': 6,  # June (summer)
            'is_weekend': 0,
            'is_holiday': 0,
            'peak_hour': 0,
            
            # Weather features
            'weather': 'sunny',
            
            # Business features
            'has_promotion': 0,
            'store_location': 'downtown',
            
            # Calculated features (will be populated later)
            'revenue_usd': 0,
        }
        
        # Override with custom defaults
        smart_defaults.update(custom_defaults)
        
        # Add defaults only for missing fields
        added_defaults = []
        for field, default_value in smart_defaults.items():
            if field not in data_dict:
                data_dict[field] = [default_value]
                added_defaults.append(field)
        
        if added_defaults:
            logger.info(f"📝 Added {len(added_defaults)} default fields: {added_defaults}")
        
        return data_dict
