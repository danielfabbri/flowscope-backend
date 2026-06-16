from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from app.services.base_service import BasePipelineService


class CleaningService(BasePipelineService):
    """Enhanced data cleaning service with per-column configuration."""
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Clean data by handling missing values, duplicates, and outliers.
        
        Config options:
            - remove_nulls: bool or dict - remove null values (default: False)
                * If True: removes rows with ANY null
                * If dict: {'column_name': True/False} for per-column control
            - fill_nulls: bool or dict - fill null values (default: False)
                * If True: fills nulls (numeric=median, categorical=mode)
                * If dict: {'column_name': 'median'/'mean'/'mode'/'zero'/'ffill'/'bfill'/'value'}
            - fill_value: default value when fill_nulls strategy is 'value' (default: 0)
            - drop_null_threshold: float - drop rows with null ratio > threshold (default: 1.0 = never drop)
            - remove_duplicates: whether to remove duplicate rows (default: True)
            - outlier_method: 'none', 'iqr', or 'zscore' (default: 'none')
            - outlier_columns: list of columns or dict with per-column config (default: all numeric)
                * If list: apply outlier detection to these columns
                * If dict: {'column_name': 'iqr'/'zscore'/'none'} for per-column method
        """
        if data is None:
            raise ValueError("CleaningService requires input data")
        
        from app.core.logger import logger
        
        df = data.copy()
        original_rows = len(df)
        
        logger.info(f"🧹 Data Cleaning started: {original_rows} rows")
        logger.info(f"📋 Config: {config}")
        
        # 1. Fill null values (do this BEFORE removing nulls)
        fill_nulls = config.get("fill_nulls", False)
        
        logger.info(f"🔍 fill_nulls parameter: {fill_nulls} (type: {type(fill_nulls)})")
        
        if isinstance(fill_nulls, bool) and fill_nulls:
            # Auto-fill: numeric columns with median, categorical with mode
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"Filled {df[col].isnull().sum()} nulls in '{col}' with median: {median_val:.2f}")
            
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
                    df[col] = df[col].fillna(mode_val)
                    logger.info(f"Filled nulls in '{col}' with mode: {mode_val}")
        
        elif isinstance(fill_nulls, dict):
            # Per-column fill strategy
            for col, strategy in fill_nulls.items():
                if col not in df.columns or df[col].isnull().sum() == 0:
                    continue
                
                null_count = df[col].isnull().sum()
                
                if strategy == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == 'mode':
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
                    df[col] = df[col].fillna(mode_val)
                elif strategy == 'zero':
                    df[col] = df[col].fillna(0)
                elif strategy == 'ffill':
                    df[col] = df[col].fillna(method='ffill')
                elif strategy == 'bfill':
                    df[col] = df[col].fillna(method='bfill')
                elif strategy == 'value':
                    fill_value = config.get('fill_value', 0)
                    df[col] = df[col].fillna(fill_value)
                
                logger.info(f"Filled {null_count} nulls in '{col}' using strategy: {strategy}")
        
        # 2. Drop rows with too many nulls (optional)
        drop_null_threshold = config.get("drop_null_threshold", 1.0)
        if drop_null_threshold < 1.0:
            before = len(df)
            null_ratio = df.isnull().sum(axis=1) / len(df.columns)
            df = df[null_ratio <= drop_null_threshold]
            removed = before - len(df)
            if removed > 0:
                logger.info(f"Removed {removed} rows with null ratio > {drop_null_threshold}")
        
        # 3. Remove null values (if still needed after filling)
        remove_nulls = config.get("remove_nulls", False)
        
        if isinstance(remove_nulls, bool):
            if remove_nulls:
                before = len(df)
                df = df.dropna()
                removed = before - len(df)
                if removed > 0:
                    logger.info(f"Removed {removed} rows with remaining null values")
        
        elif isinstance(remove_nulls, dict):
            # Per-column null removal
            for col, should_remove in remove_nulls.items():
                if should_remove and col in df.columns:
                    before = len(df)
                    df = df[df[col].notna()]
                    removed = before - len(df)
                    if removed > 0:
                        logger.info(f"Removed {removed} rows with null in column '{col}'")
                    if removed > 0:
                        logger.info(f"Removed {removed} rows with null in column '{col}'")
        
        # 4. Remove duplicates
        if config.get("remove_duplicates", True):
            before = len(df)
            df = df.drop_duplicates()
            removed = before - len(df)
            if removed > 0:
                logger.info(f"Removed {removed} duplicate rows")
        
        # 5. Handle outliers
        outlier_method = config.get("outlier_method", "none")
        outlier_columns = config.get("outlier_columns", None)
        
        if outlier_method != "none":
            df = self._handle_outliers(df, outlier_method, outlier_columns, logger)
        
        # Reset index
        df = df.reset_index(drop=True)
        
        total_removed = original_rows - len(df)
        logger.info(f"Data Cleaning complete: {len(df)} rows remaining ({total_removed} removed)")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, method: str, columns_config: Any, logger) -> pd.DataFrame:
        """Detect and remove outliers."""
        
        # Determine which columns to check
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if columns_config is None:
            # Apply to all numeric columns
            columns_to_check = {col: method for col in numeric_cols}
        
        elif isinstance(columns_config, list):
            # Apply same method to specified columns
            columns_to_check = {col: method for col in columns_config if col in numeric_cols}
        
        elif isinstance(columns_config, dict):
            # Per-column method configuration
            columns_to_check = {
                col: col_method 
                for col, col_method in columns_config.items() 
                if col in numeric_cols and col_method != 'none'
            }
        
        else:
            columns_to_check = {}
        
        # Remove outliers for each column
        total_outliers_removed = 0
        
        for col, col_method in columns_to_check.items():
            if col not in df.columns:
                continue
            
            before = len(df)
            outlier_mask = self._detect_outliers(df[col], col_method)
            df = df[~outlier_mask]
            removed = before - len(df)
            
            if removed > 0:
                total_outliers_removed += removed
                logger.info(f"Removed {removed} outliers from column '{col}' using {col_method} method")
        
        if total_outliers_removed > 0:
            logger.info(f"Total outliers removed: {total_outliers_removed}")
        
        return df
    
    def _detect_outliers(self, series: pd.Series, method: str) -> pd.Series:
        """Detect outliers in a series, returning boolean mask."""
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            from app.core.logger import logger
            logger.info(f"IQR Detection - Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}")
            logger.info(f"Bounds - Lower: {lower_bound:.4f}, Upper: {upper_bound:.4f}")
            logger.info(f"Values range: {series.min():.4f} to {series.max():.4f}")
            
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > 3
        
        else:
            return pd.Series([False] * len(series), index=series.index)
