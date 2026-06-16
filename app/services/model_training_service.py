from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from app.services.base_service import BasePipelineService


class ModelTrainingService(BasePipelineService):
    """Service for training and persisting machine learning models."""
    
    def __init__(self):
        super().__init__()
        # Use absolute path relative to backend directory
        backend_dir = Path(__file__).parent.parent.parent
        self.models_dir = backend_dir / "data" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Train a model and save it to disk.
        
        Config:
        - model_name: Unique identifier for the model (required)
        - model_type: classification or regression (required)
        - algorithm: Algorithm to use (random_forest, logistic_regression, linear_regression)
        - target_column: Target variable column name (required)
        - test_size: Train/test split ratio (default: 0.3)
        - random_state: Random seed (default: 42)
        - exclude_features: Comma-separated columns to exclude from training
        - hyperparameters: Dict with algorithm-specific hyperparameters
        - auto_version: Automatically version models if name exists (default: True)
        - save_predictions: Add predictions to output DataFrame (default: True)
        """
        logger = logging.getLogger(__name__)
        
        if data is None:
            raise ValueError("ModelTrainingService requires input data")
        
        # Validate required config
        model_name = config.get("model_name", "").strip()
        model_type = config.get("model_type", "").strip()
        target_column = config.get("target_column", "").strip()
        
        if not model_name:
            raise ValueError("model_name is required")
        if not model_type:
            raise ValueError("model_type is required (classification or regression)")
        if not target_column:
            raise ValueError("target_column is required")
        if model_type not in ["classification", "regression"]:
            raise ValueError(f"model_type must be 'classification' or 'regression', got: {model_type}")
        
        logger.info(f"🏋️ Model Training Service - Training '{model_name}' ({model_type})")
        logger.info(f"Target: {target_column}, Dataset: {len(data)} rows x {len(data.columns)} columns")
        
        # Get config parameters
        algorithm = config.get("algorithm", "random_forest")
        test_size = float(config.get("test_size", 0.3))
        random_state = int(config.get("random_state", 42))
        exclude_features = config.get("exclude_features", "").strip()
        hyperparameters = config.get("hyperparameters", {})
        auto_version = config.get("auto_version", True)
        save_predictions = config.get("save_predictions", True)
        
        # Handle auto-versioning
        if auto_version:
            model_name = self._get_versioned_name(model_name)
            logger.info(f"Auto-versioning enabled: using model name '{model_name}'")
        
        # Validate target column exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data. Available: {list(data.columns)}")
        
        # Prepare data
        df_clean = data.copy()
        exclude_cols = [col.strip() for col in exclude_features.split(',') if col.strip()]
        
        # Convert boolean and categorical to numeric
        df_clean = self._prepare_data_for_ml(df_clean, exclude_cols)
        
        # Extract features (all columns except target and excluded)
        all_potential_features = [col for col in df_clean.columns 
                                 if col != target_column and col not in exclude_cols]
        
        # Filter out datetime columns (scikit-learn can't handle them)
        feature_cols = []
        excluded_datetime = []
        for col in all_potential_features:
            if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                excluded_datetime.append(col)
                logger.warning(f"Excluding datetime column '{col}' from features (not supported by sklearn)")
            else:
                feature_cols.append(col)
        
        if len(feature_cols) == 0:
            raise ValueError(
                f"No features available for training after exclusions.\n"
                f"Target: {target_column}\n"
                f"Excluded by user: {exclude_cols}\n"
                f"Excluded datetime: {excluded_datetime}"
            )
        
        if excluded_datetime:
            logger.info(f"⚠️ Auto-excluded {len(excluded_datetime)} datetime columns: {excluded_datetime}")
        
        logger.info(f"Using {len(feature_cols)} features: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
        
        # Prepare X and y
        X = df_clean[feature_cols].fillna(0)
        y = df_clean[target_column]
        
        logger.info(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")
        
        # Train/test split
        if model_type == "classification":
            # Stratified split for classification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        logger.info(f"Train set: {len(X_train)} rows, Test set: {len(X_test)} rows")
        
        # Train model
        model = self._create_model(model_type, algorithm, hyperparameters, random_state)
        
        logger.info(f"Training {algorithm} model...")
        train_start = datetime.now()
        model.fit(X_train, y_train)
        train_duration = (datetime.now() - train_start).total_seconds()
        logger.info(f"Training completed in {train_duration:.2f} seconds")
        
        # Make predictions
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            model_type, y_train, train_predictions, y_test, test_predictions, model
        )
        
        # Log metrics
        logger.info(f"📊 Model Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
        
        # Save model and metadata
        model_path = self._save_model(
            model=model,
            model_name=model_name,
            model_type=model_type,
            algorithm=algorithm,
            feature_cols=feature_cols,
            target_column=target_column,
            metrics=metrics,
            hyperparameters=hyperparameters,
            train_duration=train_duration,
            dataset_info={
                "total_rows": len(data),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "n_features": len(feature_cols)
            }
        )
        
        logger.info(f"✅ Model saved to: {model_path}")
        
        # Prepare output DataFrame
        df_output = data.copy()
        
        # Add predictions to full dataset if requested
        if save_predictions:
            full_predictions = model.predict(X)
            
            if model_type == "classification":
                df_output["predicted_class"] = full_predictions
                
                # Add probabilities
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X)
                    for i, class_label in enumerate(model.classes_):
                        df_output[f"probability_class_{class_label}"] = probabilities[:, i]
            else:
                df_output["predicted_value"] = full_predictions
                df_output["prediction_error"] = df_output[target_column] - full_predictions
        
        # Add metadata columns
        df_output["model_name"] = model_name
        df_output["model_type"] = model_type
        df_output["model_algorithm"] = algorithm
        
        # Add key metrics as columns
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                df_output[f"model_{key}"] = value
        
        logger.info(f"🎉 Model training complete! Model: {model_name}")
        
        return df_output
    
    def _prepare_data_for_ml(self, df: pd.DataFrame, exclude_cols: list) -> pd.DataFrame:
        """Convert boolean and categorical columns to numeric."""
        logger = logging.getLogger(__name__)
        df_clean = df.copy()
        
        # Convert boolean columns to int
        bool_cols = df_clean.select_dtypes(include=['bool']).columns.tolist()
        if bool_cols:
            for col in bool_cols:
                df_clean[col] = df_clean[col].astype(int)
        
        # Convert categorical columns using label encoding (but skip datetime)
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = [col for col in cat_cols if col not in exclude_cols]
        
        if cat_cols:
            from sklearn.preprocessing import LabelEncoder
            for col in cat_cols:
                try:
                    # Skip if it's a datetime-like string
                    if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                        continue
                    
                    le = LabelEncoder()
                    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                except Exception as e:
                    logger.warning(f"Failed to encode {col}: {e}")
        
        return df_clean
    
    def _create_model(self, model_type: str, algorithm: str, hyperparameters: Dict, random_state: int):
        """Create and configure model based on type and algorithm."""
        logger = logging.getLogger(__name__)
        
        if model_type == "classification":
            if algorithm == "random_forest":
                params = {
                    "n_estimators": hyperparameters.get("n_estimators", 100),
                    "max_depth": hyperparameters.get("max_depth", None),
                    "min_samples_split": hyperparameters.get("min_samples_split", 2),
                    "class_weight": hyperparameters.get("class_weight", None),
                    "random_state": random_state,
                    "n_jobs": -1
                }
                if params["class_weight"] == "none":
                    params["class_weight"] = None
                return RandomForestClassifier(**params)
            
            elif algorithm == "logistic_regression":
                params = {
                    "max_iter": hyperparameters.get("max_iter", 1000),
                    "random_state": random_state
                }
                return LogisticRegression(**params)
            
            else:
                logger.warning(f"Unknown algorithm '{algorithm}', using random_forest")
                return RandomForestClassifier(random_state=random_state, n_jobs=-1)
        
        else:  # regression
            if algorithm == "random_forest":
                params = {
                    "n_estimators": hyperparameters.get("n_estimators", 100),
                    "max_depth": hyperparameters.get("max_depth", None),
                    "min_samples_split": hyperparameters.get("min_samples_split", 2),
                    "random_state": random_state,
                    "n_jobs": -1
                }
                return RandomForestRegressor(**params)
            
            elif algorithm == "linear_regression":
                return LinearRegression()
            
            else:
                logger.warning(f"Unknown algorithm '{algorithm}', using random_forest")
                return RandomForestRegressor(random_state=random_state, n_jobs=-1)
    
    def _calculate_metrics(self, model_type: str, y_train, train_pred, y_test, test_pred, model) -> Dict:
        """Calculate appropriate metrics based on model type."""
        metrics = {}
        
        if model_type == "classification":
            # Train metrics
            metrics["train_accuracy"] = accuracy_score(y_train, train_pred)
            metrics["train_precision"] = precision_score(y_train, train_pred, average='weighted', zero_division=0)
            metrics["train_recall"] = recall_score(y_train, train_pred, average='weighted', zero_division=0)
            metrics["train_f1"] = f1_score(y_train, train_pred, average='weighted', zero_division=0)
            
            # Test metrics
            metrics["test_accuracy"] = accuracy_score(y_test, test_pred)
            metrics["test_precision"] = precision_score(y_test, test_pred, average='weighted', zero_division=0)
            metrics["test_recall"] = recall_score(y_test, test_pred, average='weighted', zero_division=0)
            metrics["test_f1"] = f1_score(y_test, test_pred, average='weighted', zero_division=0)
            
            # Overall metrics (use test)
            metrics["accuracy"] = metrics["test_accuracy"]
            metrics["precision"] = metrics["test_precision"]
            metrics["recall"] = metrics["test_recall"]
            metrics["f1_score"] = metrics["test_f1"]
        
        else:  # regression
            # Train metrics
            metrics["train_mse"] = mean_squared_error(y_train, train_pred)
            metrics["train_mae"] = mean_absolute_error(y_train, train_pred)
            metrics["train_r2"] = r2_score(y_train, train_pred)
            
            # Test metrics
            metrics["test_mse"] = mean_squared_error(y_test, test_pred)
            metrics["test_mae"] = mean_absolute_error(y_test, test_pred)
            metrics["test_r2"] = r2_score(y_test, test_pred)
            
            # Overall metrics (use test)
            metrics["mse"] = metrics["test_mse"]
            metrics["rmse"] = np.sqrt(metrics["test_mse"])
            metrics["mae"] = metrics["test_mae"]
            metrics["r2_score"] = metrics["test_r2"]
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            metrics["has_feature_importance"] = True
        
        return metrics
    
    def _get_versioned_name(self, base_name: str) -> str:
        """Get versioned model name if base_name already exists."""
        # Check if base model exists
        base_path = self.models_dir / f"{base_name}.pkl"
        
        if not base_path.exists():
            return base_name
        
        # Find next available version
        version = 1
        while True:
            versioned_name = f"{base_name}_v{version}"
            versioned_path = self.models_dir / f"{versioned_name}.pkl"
            if not versioned_path.exists():
                return versioned_name
            version += 1
    
    def _save_model(self, model, model_name: str, model_type: str, algorithm: str,
                    feature_cols: list, target_column: str, metrics: Dict,
                    hyperparameters: Dict, train_duration: float, dataset_info: Dict) -> Path:
        """Save model and metadata to disk."""
        # Save model binary
        model_path = self.models_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "model_type": model_type,
            "algorithm": algorithm,
            "target_column": target_column,
            "feature_columns": feature_cols,
            "n_features": len(feature_cols),
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "training_info": {
                "trained_at": datetime.now().isoformat(),
                "train_duration_seconds": train_duration,
                **dataset_info
            },
            "sklearn_version": self._get_sklearn_version(),
            "model_version": "1.0"
        }
        
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return model_path
    
    def _get_sklearn_version(self) -> str:
        """Get scikit-learn version."""
        try:
            import sklearn
            return sklearn.__version__
        except:
            return "unknown"
