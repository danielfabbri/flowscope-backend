from typing import Dict, Any, Optional
import pandas as pd

from app.services.base_service import BasePipelineService


class OutputService(BasePipelineService):
    """Output service - finalizes data for consumption."""
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Prepare final output (Gold Layer).
        
        Config options:
            - select_columns: list of columns to keep (default: all)
            - add_metadata: add processing metadata (default: True)
        """
        if data is None:
            raise ValueError("OutputService requires input data")
        
        df = data.copy()
        
        # Select specific columns if configured
        if "select_columns" in config:
            available_cols = [col for col in config["select_columns"] if col in df.columns]
            df = df[available_cols]
        
        # Add metadata
        if config.get("add_metadata", True):
            import datetime
            df["processed_at"] = datetime.datetime.now().isoformat()
        
        return df
