from typing import Dict, Any, Optional, List
import pandas as pd

from app.services.base_service import BasePipelineService


class ColumnSelectionService(BasePipelineService):
    """Column selection service - select which columns to keep or drop."""
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Select columns to keep or drop.
        
        Config options:
            - mode: 'keep' or 'drop' (default: 'keep')
            - columns: list of column names to keep/drop (default: all)
        """
        if data is None:
            raise ValueError("ColumnSelectionService requires input data")
        
        df = data.copy()
        
        mode = config.get('mode', 'keep')
        columns = config.get('columns', [])
        
        if not columns:
            # No columns specified, return all data
            return df
        
        # Validate columns exist
        invalid_columns = [col for col in columns if col not in df.columns]
        if invalid_columns:
            raise ValueError(f"Columns not found in data: {', '.join(invalid_columns)}")
        
        if mode == 'keep':
            # Keep only specified columns
            df = df[columns]
            from app.core.logger import logger
            logger.info(f"Column Selection: Keeping {len(columns)} columns: {', '.join(columns)}")
        
        elif mode == 'drop':
            # Drop specified columns
            df = df.drop(columns=columns)
            from app.core.logger import logger
            logger.info(f"Column Selection: Dropped {len(columns)} columns, {len(df.columns)} remaining")
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'keep' or 'drop'")
        
        return df
