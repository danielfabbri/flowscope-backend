from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from app.services.base_service import BasePipelineService


class ProfilingService(BasePipelineService):
    """Data profiling service - generates statistics for each column."""
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Profile data and return the original data (profiling is informational).
        
        Config options:
            - detect_outliers: whether to detect outliers (default: True)
            - outlier_method: method to use - 'iqr' or 'zscore' (default: 'iqr')
        """
        if data is None:
            raise ValueError("ProfilingService requires input data")
        
        df = data.copy()
        
        # Generate profiling information
        profile = self._generate_profile(df, config)
        
        # Store profiling info in DataFrame metadata (for logging/display)
        # The actual data passes through unchanged
        df.attrs['profile'] = profile
        
        # Log profile summary
        self._log_profile(profile)
        
        return df
    
    def _generate_profile(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed profile for each column."""
        profile = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': {}
        }
        
        detect_outliers = config.get('detect_outliers', True)
        outlier_method = config.get('outlier_method', 'iqr')
        
        for col in df.columns:
            col_profile = {
                'dtype': str(df[col].dtype),
                'non_null_count': int(df[col].count()),
                'null_count': int(df[col].isna().sum()),
                'null_percentage': float(df[col].isna().sum() / len(df) * 100),
                'unique_count': int(df[col].nunique())
            }
            
            # Numeric columns - add statistical info
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                col_profile['is_numeric'] = True
                col_profile['mean'] = float(df[col].mean()) if df[col].count() > 0 else None
                col_profile['median'] = float(df[col].median()) if df[col].count() > 0 else None
                col_profile['std'] = float(df[col].std()) if df[col].count() > 0 else None
                col_profile['min'] = float(df[col].min()) if df[col].count() > 0 else None
                col_profile['max'] = float(df[col].max()) if df[col].count() > 0 else None
                
                # Detect outliers
                if detect_outliers and df[col].count() > 0:
                    outliers = self._detect_outliers(df[col].dropna(), outlier_method)
                    col_profile['outlier_count'] = len(outliers)
                    col_profile['outlier_percentage'] = float(len(outliers) / df[col].count() * 100)
                    if len(outliers) > 0:
                        col_profile['outlier_values'] = [float(x) for x in outliers[:5]]  # First 5
            
            # Categorical columns - add frequency info
            else:
                col_profile['is_numeric'] = False
                if df[col].count() > 0:
                    value_counts = df[col].value_counts().head(5)
                    col_profile['top_values'] = {
                        str(k): int(v) for k, v in value_counts.items()
                    }
            
            profile['columns'][col] = col_profile
        
        return profile
    
    def _detect_outliers(self, series: pd.Series, method: str = 'iqr') -> np.ndarray:
        """Detect outliers in a numeric series."""
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = series[z_scores > 3]
        
        else:
            outliers = pd.Series([], dtype=series.dtype)
        
        return outliers.values
    
    def _log_profile(self, profile: Dict[str, Any]):
        """Log profile summary."""
        from app.core.logger import logger
        
        logger.info(f"Data Profile: {profile['total_rows']} rows, {profile['total_columns']} columns")
        
        for col, stats in profile['columns'].items():
            null_pct = stats['null_percentage']
            logger.info(f"  - {col}: {stats['non_null_count']} values, {stats['null_count']} nulls ({null_pct:.1f}%)")
            
            if stats.get('is_numeric'):
                if stats.get('outlier_count', 0) > 0:
                    logger.info(f"    → {stats['outlier_count']} outliers detected ({stats['outlier_percentage']:.1f}%)")
            else:
                if stats.get('top_values'):
                    top = ', '.join([f"{k}={v}" for k, v in list(stats['top_values'].items())[:3]])
                    logger.info(f"    → Top values: {top}")
