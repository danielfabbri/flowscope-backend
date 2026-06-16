from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from app.services.base_service import BasePipelineService


class ModelInferenceService(BasePipelineService):
    """Service for making predictions using a loaded model.
    
    This service uses a model loaded by ModelLoadingService to make predictions
    on new data. It expects the model to be in DataFrame.attrs['loaded_model'].
    """
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Make predictions using a loaded model.
        
        Config:
        - prediction_column: Name for prediction column (default: "prediction")
        - include_probabilities: Add probability columns for classification (default: True)
        - include_error: Add prediction error column for regression (default: True)
        - compare_with_actual: If target column exists, compare predictions (default: True)
        - actual_column: Name of actual target column for comparison (optional)
        
        Returns:
            Original data with predictions added
        """
        logger = logging.getLogger(__name__)
        
        if data is None:
            raise ValueError("ModelInferenceService requires input data")
        
        # Check if model was loaded
        if "loaded_model" not in data.attrs:
            raise ValueError(
                "No model found in data. Make sure ModelLoadingService ran before this step."
            )
        
        model = data.attrs["loaded_model"]
        model_name = data.attrs.get("loaded_model_name", "unknown")
        metadata = data.attrs.get("loaded_model_metadata", {})
        
        model_type = metadata.get("model_type", "unknown")
        expected_features = metadata.get("feature_columns", [])
        target_column = metadata.get("target_column", None)
        
        logger.info(f"🔮 Model Inference Service - Using model: {model_name}")
        logger.info(f"  Type: {model_type}")
        logger.info(f"  Features: {len(expected_features)}")
        
        # Get config
        prediction_column = config.get("prediction_column", "prediction")
        include_probabilities = config.get("include_probabilities", True)
        include_error = config.get("include_error", True)
        compare_with_actual = config.get("compare_with_actual", True)
        actual_column = config.get("actual_column", target_column).strip() if config.get("actual_column") else target_column
        
        # Validate features exist
        missing_features = [feat for feat in expected_features if feat not in data.columns]
        if missing_features:
            raise ValueError(
                f"Missing required features for inference: {missing_features[:5]}...\n"
                f"Expected: {expected_features[:10]}...\n"
                f"Available: {list(data.columns)[:10]}..."
            )
        
        # Prepare feature matrix
        df_clean = data.copy()
        
        # Convert boolean and categorical to numeric (same as training)
        df_clean = self._prepare_data_for_ml(df_clean)
        
        # Extract features in correct order
        X = df_clean[expected_features].fillna(0)
        
        logger.info(f"Making predictions on {len(X)} rows...")
        
        # Make predictions
        predictions = model.predict(X)
        
        # Prepare output DataFrame
        df_output = data.copy()
        
        # Add predictions
        if model_type == "classification":
            df_output[prediction_column] = predictions
            
            # Add probabilities if available and requested
            if include_probabilities and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                
                for i, class_label in enumerate(model.classes_):
                    prob_column = f"{prediction_column}_prob_{class_label}"
                    df_output[prob_column] = probabilities[:, i]
                
                # Add confidence (max probability)
                df_output[f"{prediction_column}_confidence"] = np.max(probabilities, axis=1)
                
                logger.info(f"Added probabilities for {len(model.classes_)} classes")
        
        else:  # regression
            df_output[prediction_column] = predictions
            
            # Add prediction error if actual values exist
            if include_error and actual_column and actual_column in df_output.columns:
                df_output[f"{prediction_column}_error"] = df_output[actual_column] - predictions
                df_output[f"{prediction_column}_abs_error"] = np.abs(df_output[f"{prediction_column}_error"])
                df_output[f"{prediction_column}_pct_error"] = (
                    df_output[f"{prediction_column}_error"] / df_output[actual_column] * 100
                )
        
        # Compare with actual if requested and possible
        if compare_with_actual and actual_column and actual_column in df_output.columns:
            logger.info(f"Comparing predictions with actual column: {actual_column}")
            
            if model_type == "classification":
                correct = (df_output[prediction_column] == df_output[actual_column]).astype(int)
                df_output[f"{prediction_column}_correct"] = correct
                
                accuracy = correct.mean()
                logger.info(f"Inference Accuracy: {accuracy:.4f} ({correct.sum()}/{len(correct)} correct)")
            
            else:  # regression
                errors = df_output[f"{prediction_column}_error"]
                mae = np.abs(errors).mean()
                rmse = np.sqrt((errors ** 2).mean())
                
                logger.info(f"Inference MAE: {mae:.4f}")
                logger.info(f"Inference RMSE: {rmse:.4f}")
                
                # Add summary metrics as columns
                df_output[f"{prediction_column}_mae"] = mae
                df_output[f"{prediction_column}_rmse"] = rmse
        
        # Add inference metadata
        df_output["inference_model"] = model_name
        df_output["inference_count"] = len(predictions)
        
        # Keep model in attrs for potential next steps
        df_output.attrs["loaded_model"] = model
        df_output.attrs["loaded_model_name"] = model_name
        df_output.attrs["loaded_model_metadata"] = metadata
        
        logger.info(f"✅ Inference complete: {len(predictions)} predictions generated")
        logger.info(f"Output columns: {list(df_output.columns)[-10:]}")
        
        return df_output
    
    def _prepare_data_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert boolean and categorical columns to numeric (same as training)."""
        logger = logging.getLogger(__name__)
        df_clean = df.copy()
        
        # Convert boolean columns to int
        bool_cols = df_clean.select_dtypes(include=['bool']).columns.tolist()
        if bool_cols:
            for col in bool_cols:
                df_clean[col] = df_clean[col].astype(int)
        
        # Convert categorical columns using label encoding
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if cat_cols:
            from sklearn.preprocessing import LabelEncoder
            for col in cat_cols:
                try:
                    # Skip if it's a datetime-like column
                    if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                        continue
                    
                    le = LabelEncoder()
                    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                except Exception as e:
                    logger.warning(f"Failed to encode {col}: {e}")
        
        return df_clean
