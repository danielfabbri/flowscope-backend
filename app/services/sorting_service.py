from typing import Dict, Any
import pandas as pd
import logging

from app.services.base_service import BasePipelineService


class SortingService(BasePipelineService):
    """Service for sorting/ordering rows."""
    
    def execute(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Sort rows based on column values.
        
        Config:
        - sort_column: Column name to sort by
        - sort_order: 'ascending' or 'descending' (default: ascending)
        - use_absolute: True to sort by absolute value (useful for errors)
        """
        logger = logging.getLogger(__name__)
        
        sort_column = config.get("sort_column", "").strip()
        sort_order = config.get("sort_order", "ascending")
        use_absolute = config.get("use_absolute", False)
        
        if not sort_column:
            raise ValueError("sort_column is required")
        
        if sort_column not in data.columns:
            raise ValueError(f"Column '{sort_column}' not found in data. Available columns: {list(data.columns)}")
        
        logger.info(f"Sorting by: {sort_column}, order: {sort_order}, absolute: {use_absolute}")
        logger.info(f"Row count: {len(data)}")
        
        # Create copy to avoid modifying original
        sorted_data = data.copy()
        
        # Apply absolute value if requested
        if use_absolute:
            sort_key = sorted_data[sort_column].abs()
            logger.info(f"Using absolute values for sorting")
        else:
            sort_key = sorted_data[sort_column]
        
        # Sort
        ascending = (sort_order == "ascending")
        sorted_data = sorted_data.iloc[sort_key.argsort()]
        
        if not ascending:
            sorted_data = sorted_data.iloc[::-1]
        
        logger.info(f"✅ Sorted {len(sorted_data)} rows by '{sort_column}' ({sort_order})")
        
        # Log sample values
        if len(sorted_data) > 0:
            first_val = sorted_data[sort_column].iloc[0]
            last_val = sorted_data[sort_column].iloc[-1]
            logger.info(f"Value range: {first_val:.2f} to {last_val:.2f}")
        
        return sorted_data
