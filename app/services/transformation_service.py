from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from app.services.base_service import BasePipelineService


class TransformationService(BasePipelineService):
    """Data transformation service with column selection and segmentation."""
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Transform data with aggregations and enrichments.
        
        Config options:
            - normalize: boolean - enable normalization (default: False)
            - normalize_columns: list - specific columns to normalize
            - group_by_column: str - column to group by for segmented normalization
            - add_rolling_mean: boolean - enable rolling mean (default: False)
            - rolling_mean_column: str - column to apply rolling mean
            - rolling_window: int - window size for rolling mean (default: 10)
            - add_time_features: boolean - extract time-based features (default: True)
        """
        if data is None:
            raise ValueError("TransformationService requires input data")
        
        from app.core.logger import logger
        df = data.copy()
        
        # Add time-based features
        if config.get("add_time_features", True) and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        # Normalize columns
        if config.get("normalize", False):
            normalize_cols = config.get("normalize_columns", [])
            group_by_col = config.get("group_by_column")
            
            # Validate columns exist
            normalize_cols = [col for col in normalize_cols if col in df.columns]
            
            # Only proceed if columns were explicitly selected
            if not normalize_cols:
                logger.warning("Normalize enabled but no columns selected - skipping normalization")
            elif group_by_col and group_by_col in df.columns:
                # Segmented normalization (normalize within each group)
                logger.info(f"Normalizing {len(normalize_cols)} columns grouped by '{group_by_col}'")
                for col in normalize_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[f"{col}_normalized"] = df.groupby(group_by_col)[col].transform(
                            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                        )
            elif normalize_cols:
                # Global normalization (only if columns selected and no grouping)
                logger.info(f"Normalizing {len(normalize_cols)} columns globally")
                for col in normalize_cols:
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        if std_val > 0:
                            df[f"{col}_normalized"] = (df[col] - mean_val) / std_val
                        else:
                            df[f"{col}_normalized"] = 0
        
        # Add rolling mean
        if config.get("add_rolling_mean", False):
            rolling_col = config.get("rolling_mean_column")
            window = config.get("rolling_window", 10)
            
            if rolling_col and rolling_col in df.columns:
                if pd.api.types.is_numeric_dtype(df[rolling_col]):
                    df[f"{rolling_col}_rolling_mean"] = df[rolling_col].rolling(
                        window=window, min_periods=1
                    ).mean()
                    logger.info(f"Added rolling mean for '{rolling_col}' with window={window}")
                else:
                    logger.warning(f"Column '{rolling_col}' is not numeric, skipping rolling mean")
            else:
                logger.warning(f"Rolling mean column '{rolling_col}' not found in data")
        
        return df
