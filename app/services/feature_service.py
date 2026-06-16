from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder

from app.services.base_service import BasePipelineService


class FeatureService(BasePipelineService):
    """Feature engineering service."""
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Engineer features from raw data.
        
        New approach: Each step performs ONE transformation based on transformation_type.
        Supports:
        - scaling: StandardScaler, MinMaxScaler, RobustScaler
        - onehot_encoding: One-Hot Encoding
        - label_encoding: Label Encoding
        - binning: Discretization into bins
        - feature_creation: Custom formulas
        - feature_selection: Remove low variance or high correlation
        - time_features: Extract date components
        """
        if data is None:
            raise ValueError("FeatureService requires input data")
        
        df = data.copy()
        
        # Get transformation type (default to old multi-transform for backward compatibility)
        transformation_type = config.get("transformation_type", None)
        
        # NEW SINGLE-TRANSFORM APPROACH
        if transformation_type:
            if transformation_type == "scaling":
                df = self._apply_scaling(df, config)
            elif transformation_type == "onehot_encoding":
                df = self._apply_onehot_encoding(df, config)
            elif transformation_type == "label_encoding":
                df = self._apply_label_encoding(df, config)
            elif transformation_type == "binning":
                df = self._apply_binning(df, config)
            elif transformation_type == "feature_creation":
                df = self._apply_feature_creation(df, config)
            elif transformation_type == "feature_selection":
                df = self._apply_feature_selection(df, config)
            elif transformation_type == "time_features":
                df = self._apply_time_features(df, config)
        
        # OLD MULTI-TRANSFORM APPROACH (for backward compatibility)
        else:
            df = self._apply_legacy_transforms(df, config)
        
        return df
    
    def _apply_scaling(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply scaling transformation."""
        import logging
        logger = logging.getLogger(__name__)
        
        scaling_cols = self._parse_column_list(config.get("scaling_columns", ""))
        scaling_method = config.get("scaling_method", "standard")
        
        logger.info(f"Scaling config: columns={scaling_cols}, method={scaling_method}")
        
        if scaling_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            logger.info(f"Available numeric columns: {numeric_cols}")
            
            scaling_cols = [col for col in scaling_cols if col in numeric_cols]
            logger.info(f"Columns to scale (filtered): {scaling_cols}")
            
            if scaling_cols:
                if scaling_method == "standard":
                    scaler = StandardScaler()
                elif scaling_method == "minmax":
                    scaler = MinMaxScaler()
                elif scaling_method == "robust":
                    scaler = RobustScaler()
                else:
                    scaler = StandardScaler()
                
                # Apply scaling
                df[scaling_cols] = scaler.fit_transform(df[scaling_cols])
                logger.info(f"Scaling applied successfully to {len(scaling_cols)} columns")
                logger.info(f"Sample values after scaling: {df[scaling_cols].head(1).to_dict()}")
            else:
                logger.warning("No valid columns found to scale")
        else:
            logger.warning("No scaling columns specified in config")
        
        return df
    
    def _apply_onehot_encoding(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply one-hot encoding transformation."""
        onehot_cols = self._parse_column_list(config.get("onehot_columns", ""))
        if onehot_cols:
            onehot_cols = [col for col in onehot_cols if col in df.columns]
            if onehot_cols:
                df = pd.get_dummies(df, columns=onehot_cols, prefix=onehot_cols, drop_first=False)
        return df
    
    def _apply_label_encoding(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply label encoding transformation."""
        label_cols = self._parse_column_list(config.get("label_columns", ""))
        if label_cols:
            label_cols = [col for col in label_cols if col in df.columns]
            for col in label_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        return df
    
    def _apply_binning(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply binning transformation."""
        col = config.get("binning_column", "").strip()
        bins_str = config.get("binning_bins", "").strip()
        labels_str = config.get("binning_labels", "").strip()
        
        if col and col in df.columns and bins_str:
            try:
                # Parse bins
                bins = eval(bins_str) if bins_str.startswith('[') else [float(x.strip()) for x in bins_str.split(',')]
                
                # Parse labels
                labels = None
                if labels_str:
                    labels = [x.strip() for x in labels_str.split(',')]
                
                df[f"{col}_binned"] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
            except Exception as e:
                print(f"Binning error for {col}: {e}")
        
        return df
    
    def _apply_feature_creation(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply feature creation transformation."""
        feature_name = config.get("feature_name", "").strip()
        formula = config.get("feature_formula", "").strip()
        
        if feature_name and formula:
            try:
                # Replace column names with df['column_name'] syntax
                safe_formula = formula
                for col in df.columns:
                    if col in safe_formula:
                        safe_formula = safe_formula.replace(col, f"df['{col}']")
                
                df[feature_name] = eval(safe_formula)
            except Exception as e:
                print(f"Feature creation error for {feature_name}: {e}")
        
        return df
    
    def _apply_feature_selection(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply feature selection transformation."""
        selection_method = config.get("selection_method", "variance")
        
        if selection_method == "variance":
            threshold = float(config.get("variance_threshold", 0.01))
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if df[col].var() < threshold:
                    df = df.drop(columns=[col])
        
        elif selection_method == "correlation":
            corr_threshold = float(config.get("correlation_threshold", 0.95))
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > corr_threshold)]
                df = df.drop(columns=to_drop)
        
        return df
    
    def _apply_time_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply time features transformation."""
        time_col = config.get("time_column", "").strip()
        
        if time_col and time_col in df.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])
            
            # Extract components based on config
            if config.get("extract_year", False):
                df[f"{time_col}_year"] = df[time_col].dt.year
            if config.get("extract_month", False):
                df[f"{time_col}_month"] = df[time_col].dt.month
            if config.get("extract_day", False):
                df[f"{time_col}_day"] = df[time_col].dt.day
            if config.get("extract_dayofweek", False):
                df[f"{time_col}_dayofweek"] = df[time_col].dt.dayofweek
            if config.get("extract_hour", False):
                df[f"{time_col}_hour"] = df[time_col].dt.hour
            if config.get("extract_quarter", False):
                df[f"{time_col}_quarter"] = df[time_col].dt.quarter
        
        return df
    
    def _apply_legacy_transforms(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply legacy multi-transform approach (for backward compatibility with old pipelines)."""
        
        # Old approach: apply all enabled transforms
        
        # 1. Binning
        if config.get("binning_enabled", False):
            for i in range(1, 5):
                col = config.get(f"binning_column_{i}", "").strip()
                bins_str = config.get(f"binning_bins_{i}", "").strip()
                labels_str = config.get(f"binning_labels_{i}", "").strip()
                
                if col and col in df.columns and bins_str:
                    try:
                        bins = eval(bins_str) if bins_str.startswith('[') else [float(x.strip()) for x in bins_str.split(',')]
                        labels = None
                        if labels_str:
                            labels = [x.strip() for x in labels_str.split(',')]
                        df[f"{col}_binned"] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
                    except Exception as e:
                        print(f"Binning error for {col}: {e}")
        
        # 2. Feature Creation
        if config.get("create_features_enabled", False):
            for i in range(1, 6):
                feature_name = config.get(f"feature_name_{i}", "").strip()
                formula = config.get(f"feature_formula_{i}", "").strip()
                
                if feature_name and formula:
                    try:
                        safe_formula = formula
                        for col in df.columns:
                            if col in safe_formula:
                                safe_formula = safe_formula.replace(col, f"df['{col}']")
                        df[feature_name] = eval(safe_formula)
                    except Exception as e:
                        print(f"Feature creation error for {feature_name}: {e}")
        
        # 3. Encoding
        if config.get("encoding_enabled", False):
            onehot_cols = self._parse_column_list(config.get("onehot_columns", ""))
            if onehot_cols:
                onehot_cols = [col for col in onehot_cols if col in df.columns]
                if onehot_cols:
                    df = pd.get_dummies(df, columns=onehot_cols, prefix=onehot_cols, drop_first=False)
            
            label_cols = self._parse_column_list(config.get("label_columns", ""))
            if label_cols:
                label_cols = [col for col in label_cols if col in df.columns]
                for col in label_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
        
        # 4. Scaling
        if config.get("scaling_enabled", False):
            scaling_cols = self._parse_column_list(config.get("scaling_columns", ""))
            scaling_method = config.get("scaling_method", "standard")
            
            if scaling_cols:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                scaling_cols = [col for col in scaling_cols if col in numeric_cols]
                
                if scaling_cols:
                    if scaling_method == "standard":
                        scaler = StandardScaler()
                    elif scaling_method == "minmax":
                        scaler = MinMaxScaler()
                    elif scaling_method == "robust":
                        scaler = RobustScaler()
                    else:
                        scaler = StandardScaler()
                    
                    df[scaling_cols] = scaler.fit_transform(df[scaling_cols])
        
        return df
    
    def _parse_column_list(self, columns_str: str) -> list:
        """Parse comma-separated column list from config."""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Parsing column list from: '{columns_str}'")
        
        if not columns_str:
            logger.warning("Empty column string received")
            return []
        
        # Handle both comma-separated and newline-separated
        if ',' in columns_str:
            result = [col.strip() for col in columns_str.split(',') if col.strip()]
        else:
            result = [col.strip() for col in columns_str.split('\n') if col.strip()]
        
        logger.info(f"Parsed columns: {result}")
        return result
