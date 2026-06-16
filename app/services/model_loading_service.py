from typing import Dict, Any, Optional
import pandas as pd
import logging
import joblib
import json
from pathlib import Path

from app.services.base_service import BasePipelineService


class ModelLoadingService(BasePipelineService):
    """Service for loading pre-trained models from disk.
    
    This service loads a saved model and its metadata into memory,
    making it available for the next inference step.
    """
    
    def __init__(self):
        super().__init__()
        # Use absolute path relative to backend directory
        backend_dir = Path(__file__).parent.parent.parent
        self.models_dir = backend_dir / "data" / "models"
        self.loaded_models = {}  # Cache for loaded models
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Load a trained model from disk.
        
        Config:
        - model_name: Name of the model to load (required)
        - model_path: Custom path to model file (optional, overrides model_name)
        - validate_features: Check if data has required features (default: True)
        - cache_model: Keep model in memory for reuse (default: True)
        
        Returns:
            Original data with model metadata added
        """
        logger = logging.getLogger(__name__)
        
        if data is None:
            raise ValueError("ModelLoadingService requires input data")
        
        # Get config
        model_name = config.get("model_name", "").strip()
        custom_path = config.get("model_path", "").strip()
        validate_features = config.get("validate_features", True)
        cache_model = config.get("cache_model", True)
        
        if not model_name and not custom_path:
            raise ValueError("Either model_name or model_path is required")
        
        logger.info(f"📂 Model Loading Service - Loading model: {model_name or custom_path}")
        
        # Determine paths
        if custom_path:
            model_path = Path(custom_path)
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            cache_key = custom_path
        else:
            # Try .pkl first, then .joblib
            model_path = self.models_dir / f"{model_name}.pkl"
            if not model_path.exists():
                model_path = self.models_dir / f"{model_name}.joblib"
            
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            cache_key = model_name
        
        # Check if model exists
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Tried extensions: .pkl, .joblib\n"
                f"Available models: {self._list_available_models()}"
            )
        
        # Check if metadata exists
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            metadata = {}
        else:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Check cache first
        if cache_model and cache_key in self.loaded_models:
            logger.info(f"Using cached model: {cache_key}")
            model = self.loaded_models[cache_key]
        else:
            # Load model from disk
            logger.info(f"Loading model from disk: {model_path}")
            model = joblib.load(model_path)
            
            if cache_model:
                self.loaded_models[cache_key] = model
                logger.info(f"Model cached in memory: {cache_key}")
        
        # Extract metadata
        model_type = metadata.get("model_type", "unknown")
        algorithm = metadata.get("algorithm", "unknown")
        expected_features = metadata.get("feature_columns", [])
        target_column = metadata.get("target_column", "unknown")
        training_metrics = metadata.get("metrics", {})
        
        logger.info(f"Model loaded successfully:")
        logger.info(f"  Type: {model_type}")
        logger.info(f"  Algorithm: {algorithm}")
        logger.info(f"  Features: {len(expected_features)}")
        logger.info(f"  Target: {target_column}")
        
        # Log training metrics
        if training_metrics:
            logger.info(f"  Training Metrics:")
            for key, value in training_metrics.items():
                if isinstance(value, (int, float)) and not key.startswith("train_") and not key.startswith("test_"):
                    logger.info(f"    {key}: {value:.4f}")
        
        # Validate features if requested
        if validate_features and expected_features:
            self._validate_features(data, expected_features)
        
        # Prepare output DataFrame
        df_output = data.copy()
        
        # Add model metadata as columns
        df_output.attrs["loaded_model"] = model
        df_output.attrs["loaded_model_name"] = model_name or cache_key
        df_output.attrs["loaded_model_metadata"] = metadata
        
        # Add metadata columns for visibility
        df_output["loaded_model_name"] = model_name or cache_key
        df_output["loaded_model_type"] = model_type
        df_output["loaded_model_algorithm"] = algorithm
        df_output["loaded_model_features"] = len(expected_features)
        
        logger.info(f"✅ Model loaded and ready for inference")
        
        return df_output
    
    def _validate_features(self, data: pd.DataFrame, expected_features: list):
        """Validate that data contains all required features."""
        logger = logging.getLogger(__name__)
        
        missing_features = [feat for feat in expected_features if feat not in data.columns]
        extra_features = [col for col in data.columns if col not in expected_features]
        
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            raise ValueError(
                f"Data is missing {len(missing_features)} required features: {missing_features[:5]}...\n"
                f"Expected features: {expected_features[:10]}...\n"
                f"Available columns: {list(data.columns)[:10]}..."
            )
        
        if extra_features:
            logger.info(f"Data has {len(extra_features)} extra columns (will be ignored during inference)")
        
        logger.info(f"✅ Feature validation passed: all {len(expected_features)} required features found")
    
    def _list_available_models(self) -> list:
        """List all available models in the models directory."""
        if not self.models_dir.exists():
            return []
        
        models = set()  # Use set to avoid duplicates
        
        # Find all .pkl files
        for model_file in self.models_dir.glob("*.pkl"):
            # Skip metadata files
            if not model_file.name.endswith("_metadata.pkl"):
                models.add(model_file.stem)
        
        # Find all .joblib files
        for model_file in self.models_dir.glob("*.joblib"):
            # Skip metadata files
            if not model_file.name.endswith("_metadata.joblib"):
                models.add(model_file.stem)
        
        return sorted(list(models))
    
    def get_loaded_model(self, model_name: str):
        """Get a cached model by name (for external use)."""
        return self.loaded_models.get(model_name)
    
    def clear_cache(self):
        """Clear all cached models from memory."""
        logger = logging.getLogger(__name__)
        count = len(self.loaded_models)
        self.loaded_models.clear()
        logger.info(f"Cleared {count} cached models from memory")
