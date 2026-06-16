from typing import Dict, Any
import pandas as pd
import logging

from app.services.base_service import BasePipelineService


class FilterService(BasePipelineService):
    """Service for filtering/selecting rows based on conditions."""
    
    def execute(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Filter rows based on column conditions.
        
        Config:
        - filter_column: Column name to filter on
        - filter_operator: Operator (=, !=, >, <, >=, <=, contains, not_contains)
        - filter_value: Value to compare against
        - keep_matching: True to keep matching rows, False to remove them
        """
        logger = logging.getLogger(__name__)
        
        filter_column = config.get("filter_column", "").strip()
        filter_operator = config.get("filter_operator", "=")
        filter_value_str = config.get("filter_value", "")
        keep_matching = config.get("keep_matching", True)
        
        if not filter_column:
            raise ValueError("filter_column is required")
        
        if filter_column not in data.columns:
            raise ValueError(f"Column '{filter_column}' not found in data. Available columns: {list(data.columns)}")
        
        logger.info(f"Filtering: {filter_column} {filter_operator} {filter_value_str}, keep_matching={keep_matching}")
        logger.info(f"Initial row count: {len(data)}")
        
        # Convert filter value to appropriate type
        filter_value = self._convert_value(data[filter_column], filter_value_str)
        
        # Apply filter
        if filter_operator == "=":
            mask = data[filter_column] == filter_value
        elif filter_operator == "!=":
            mask = data[filter_column] != filter_value
        elif filter_operator == ">":
            mask = data[filter_column] > filter_value
        elif filter_operator == "<":
            mask = data[filter_column] < filter_value
        elif filter_operator == ">=":
            mask = data[filter_column] >= filter_value
        elif filter_operator == "<=":
            mask = data[filter_column] <= filter_value
        elif filter_operator == "contains":
            mask = data[filter_column].astype(str).str.contains(str(filter_value), case=False, na=False)
        elif filter_operator == "not_contains":
            mask = ~data[filter_column].astype(str).str.contains(str(filter_value), case=False, na=False)
        else:
            raise ValueError(f"Unknown operator: {filter_operator}")
        
        # Apply keep/remove logic
        if not keep_matching:
            mask = ~mask
        
        filtered_data = data[mask].copy()
        
        logger.info(f"Filtered row count: {len(filtered_data)} ({len(filtered_data)/len(data)*100:.1f}% kept)")
        logger.info(f"Removed {len(data) - len(filtered_data)} rows")
        
        if len(filtered_data) == 0:
            logger.warning("⚠️ Filter removed ALL rows! Returning empty DataFrame")
        
        return filtered_data
    
    def _convert_value(self, series: pd.Series, value_str: str) -> Any:
        """Convert string value to the appropriate type based on the series dtype."""
        logger = logging.getLogger(__name__)
        
        # Try to infer type from the series
        dtype = series.dtype
        
        try:
            if pd.api.types.is_numeric_dtype(dtype):
                # Try to convert to float first, then int if possible
                val = float(value_str)
                if val.is_integer():
                    val = int(val)
                logger.info(f"Converted '{value_str}' to numeric: {val} (type: {type(val).__name__})")
                return val
            elif pd.api.types.is_bool_dtype(dtype):
                # Convert to boolean
                val = value_str.lower() in ('true', '1', 'yes', 't')
                logger.info(f"Converted '{value_str}' to boolean: {val}")
                return val
            else:
                # Keep as string
                logger.info(f"Using string value: '{value_str}'")
                return value_str
        except Exception as e:
            logger.warning(f"Failed to convert '{value_str}' to {dtype}: {e}, using as string")
            return value_str
