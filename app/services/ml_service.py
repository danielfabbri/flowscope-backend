from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import logging

from app.services.base_service import BasePipelineService


class MLService(BasePipelineService):
    """Machine learning service for classification, regression, and anomaly detection."""
    
    def _prepare_data_for_ml(self, df: pd.DataFrame, exclude_cols: list) -> pd.DataFrame:
        """Prepare data for ML by converting boolean and categorical columns.
        
        - Converts boolean columns to int (True=1, False=0)
        - Converts categorical text columns to numeric via label encoding
        - Returns full DataFrame with all columns converted to numeric
        """
        logger = logging.getLogger(__name__)
        df_clean = df.copy()
        
        # Convert boolean columns to int
        bool_cols = df_clean.select_dtypes(include=['bool']).columns.tolist()
        if bool_cols:
            logger.info(f"Converting {len(bool_cols)} boolean columns to int: {bool_cols[:10]}")
            for col in bool_cols:
                df_clean[col] = df_clean[col].astype(int)
        
        # Convert categorical columns to numeric (label encoding)
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if cat_cols:
            logger.info(f"Converting {len(cat_cols)} categorical columns to numeric: {cat_cols}")
            from sklearn.preprocessing import LabelEncoder
            for col in cat_cols:
                try:
                    le = LabelEncoder()
                    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                    logger.info(f"  {col}: {len(le.classes_)} categories -> 0-{len(le.classes_)-1}")
                except Exception as e:
                    logger.warning(f"Failed to encode {col}: {e}")
        
        return df_clean
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Apply ML models.
        
        Supports:
        - Classification (Random Forest, Logistic Regression)
        - Regression (Random Forest, Linear Regression)
        - Anomaly Detection (Isolation Forest, Z-Score)
        """
        if data is None:
            raise ValueError("MLService requires input data")
        
        logger = logging.getLogger(__name__)
        df = data.copy()
        
        model_type = config.get("model_type", "anomaly_detection")
        logger.info(f"ML Service - Model Type: {model_type}")
        logger.info(f"Input data shape: {df.shape}")
        logger.info(f"Input columns: {df.columns.tolist()}")
        logger.info(f"Config: {config}")
        
        try:
            if model_type == "classification":
                df = self._apply_classification(df, config)
            elif model_type == "regression":
                df = self._apply_regression(df, config)
            elif model_type == "anomaly_detection":
                df = self._apply_anomaly_detection(df, config)
            elif model_type == "clustering":
                df = self._apply_clustering(df, config)
            else:
                raise ValueError(
                    f"Tipo de modelo desconhecido: '{model_type}'. "
                    f"Valores válidos: classification, regression, anomaly_detection, clustering."
                )
        except Exception as e:
            logger.error(f"ML Service Error: {type(e).__name__}: {str(e)}", exc_info=True)
            raise
        
        return df
    
    def _apply_classification(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply classification model."""
        logger = logging.getLogger(__name__)
        
        # Get config
        target_column = config.get("target_column", "").strip()
        algorithm = config.get("algorithm", "random_forest")
        test_size = float(config.get("test_size", 0.3))
        random_state = int(config.get("random_state", 42))
        exclude_features = config.get("exclude_features", "").strip()
        
        logger.info(f"Classification - Target: {target_column}, Algorithm: {algorithm}")
        
        if not target_column or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Exclude features
        exclude_cols = [col.strip() for col in exclude_features.split(',') if col.strip()]
        
        # Prepare data: convert bool/categorical to numeric
        df_clean = self._prepare_data_for_ml(df, exclude_cols)
        
        # Extract features (all numeric columns except target and excluded)
        feature_cols = [col for col in df_clean.columns 
                       if col != target_column and col not in exclude_cols]
        
        logger.info(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")
        
        if len(feature_cols) == 0:
            raise ValueError("No features available for training")
        
        # Prepare X and y
        X = df_clean[feature_cols].fillna(0)
        y = df_clean[target_column]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        if algorithm == "random_forest":
            n_estimators = int(config.get("n_estimators", 100))
            max_depth = config.get("max_depth")
            if max_depth:
                max_depth = int(max_depth)
            min_samples_split = int(config.get("min_samples_split", 2))
            class_weight = config.get("class_weight", None)
            if class_weight == "none":
                class_weight = None
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=-1
            )
        elif algorithm == "logistic_regression":
            model = LogisticRegression(random_state=random_state, max_iter=1000)
        else:
            model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        
        logger.info(f"Training {algorithm} model...")
        model.fit(X_train, y_train)
        
        # Predict on full dataset
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Add predictions to dataframe
        df["predicted_class"] = predictions
        
        # Add probabilities for each class
        for i, class_label in enumerate(model.classes_):
            df[f"probability_class_{class_label}"] = probabilities[:, i]
        
        # Calculate metrics on test set
        y_pred_test = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        
        logger.info(f"Model Metrics - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # Add metrics as columns (same value for all rows)
        df["model_accuracy"] = accuracy
        df["model_precision"] = precision
        df["model_recall"] = recall
        df["model_f1_score"] = f1
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info(f"Top 5 Important Features:\n{importance.head()}")
            
            # Add top features as metadata columns
            for idx, row in importance.head(10).iterrows():
                df[f"importance_{row['feature']}"] = row['importance']
        
        return df
    
    def _apply_regression(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply regression model."""
        logger = logging.getLogger(__name__)
        
        target_column = config.get("target_column", "").strip()
        algorithm = config.get("algorithm", "random_forest")
        test_size = float(config.get("test_size", 0.3))
        random_state = int(config.get("random_state", 42))
        exclude_features = config.get("exclude_features", "").strip()
        
        logger.info(f"Regression - Target: {target_column}, Algorithm: {algorithm}")
        
        if not target_column or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Exclude features
        exclude_cols = [col.strip() for col in exclude_features.split(',') if col.strip()]
        
        # Prepare data: convert bool/categorical to numeric
        df_clean = self._prepare_data_for_ml(df, exclude_cols)
        
        # Extract features (all numeric columns except target and excluded)
        feature_cols = [col for col in df_clean.columns 
                       if col != target_column and col not in exclude_cols]
        
        logger.info(f"Using {len(feature_cols)} features")
        
        if len(feature_cols) == 0:
            raise ValueError("No features available for training")
        
        # Prepare X and y
        X = df_clean[feature_cols].fillna(0)
        y = df_clean[target_column]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        if algorithm == "random_forest":
            n_estimators = int(config.get("n_estimators", 100))
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        elif algorithm == "linear_regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        
        logger.info(f"Training {algorithm} model...")
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X)
        df["predicted_value"] = predictions
        df["prediction_error"] = df[target_column] - predictions
        
        # Metrics
        y_pred_test = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        logger.info(f"Model Metrics - MSE: {mse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")
        
        df["model_mse"] = mse
        df["model_mae"] = mae
        df["model_r2"] = r2
        
        return df
    
    def _apply_clustering(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply clustering (K-Means or DBSCAN)."""
        algorithm = config.get("algorithm", "kmeans")

        if algorithm == "kmeans":
            return self._kmeans_clustering(df, config)
        elif algorithm == "dbscan":
            return self._dbscan_clustering(df, config)
        else:
            raise ValueError(
                f"Algoritmo de clustering desconhecido: '{algorithm}'. "
                f"Valores válidos: kmeans, dbscan."
            )

    def _kmeans_clustering(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply K-Means clustering."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        logger = logging.getLogger(__name__)

        n_clusters = int(config.get("n_clusters", 3))
        random_state = int(config.get("random_state", 42))
        exclude_features = config.get("exclude_features", "").strip()
        exclude_cols = [col.strip() for col in exclude_features.split(',') if col.strip()]

        df_clean = self._prepare_data_for_ml(df, exclude_cols)
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

        if len(feature_cols) == 0:
            raise ValueError("Nenhuma feature disponível para clustering.")

        if n_clusters < 2:
            raise ValueError("n_clusters deve ser >= 2.")

        if n_clusters > len(df):
            raise ValueError(
                f"n_clusters ({n_clusters}) não pode ser maior que o número de linhas ({len(df)})."
            )

        logger.info(f"K-Means clustering: {n_clusters} clusters, {len(feature_cols)} features")

        X = df_clean[feature_cols].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        labels = model.fit_predict(X_scaled)

        df["cluster"] = labels
        df["cluster_distance"] = np.min(
            np.linalg.norm(X_scaled[:, None] - model.cluster_centers_[None, :], axis=2), axis=1
        ).round(4)

        cluster_counts = pd.Series(labels).value_counts().sort_index()
        logger.info(f"K-Means resultado - inertia: {model.inertia_:.2f}, distribuição: {cluster_counts.to_dict()}")

        return df

    def _dbscan_clustering(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply DBSCAN clustering."""
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        logger = logging.getLogger(__name__)

        eps = float(config.get("eps", 0.5))
        min_samples = int(config.get("min_samples", 5))
        exclude_features = config.get("exclude_features", "").strip()
        exclude_cols = [col.strip() for col in exclude_features.split(',') if col.strip()]

        df_clean = self._prepare_data_for_ml(df, exclude_cols)
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

        if len(feature_cols) == 0:
            raise ValueError("Nenhuma feature disponível para clustering.")

        logger.info(f"DBSCAN clustering: eps={eps}, min_samples={min_samples}, {len(feature_cols)} features")

        X = df_clean[feature_cols].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)

        df["cluster"] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        logger.info(f"DBSCAN resultado - {n_clusters} clusters, {n_noise} pontos de ruído (-1)")

        return df

    def _apply_anomaly_detection(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply anomaly detection (legacy support)."""
        anomaly_method = config.get("anomaly_detection", "isolation_forest")
        
        if anomaly_method == "isolation_forest":
            df = self._isolation_forest_detection(df, config)
        elif anomaly_method == "zscore":
            df = self._zscore_detection(df, config)
        
        return df
    
    def _isolation_forest_detection(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Detect anomalies using Isolation Forest."""
        logger = logging.getLogger(__name__)
        contamination = config.get("contamination", 0.02)
        exclude_features = config.get("exclude_features", "").strip()
        
        # Exclude features
        exclude_cols = [col.strip() for col in exclude_features.split(',') if col.strip()]
        
        # Prepare data: convert bool/categorical to numeric
        df_clean = self._prepare_data_for_ml(df, exclude_cols)
        
        # Select all numeric features
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        if len(feature_cols) == 0:
            raise ValueError("No features available for anomaly detection")
        
        logger.info(f"Anomaly Detection using {len(feature_cols)} features")
        
        X = df_clean[feature_cols].fillna(0).values
        
        # Train Isolation Forest
        model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        predictions = model.fit_predict(X)
        
        # -1 for anomalies, 1 for normal
        df["anomaly_detected"] = (predictions == -1).astype(int)
        df["anomaly_score"] = model.score_samples(X)
        
        return df
    
    def _zscore_detection(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Detect anomalies using Z-score method."""
        threshold = config.get("zscore_threshold", 3)
        
        if "value" in df.columns:
            mean = df["value"].mean()
            std = df["value"].std()
            df["zscore"] = (df["value"] - mean) / std
            df["anomaly_detected"] = (np.abs(df["zscore"]) > threshold).astype(int)
            df["anomaly_score"] = -np.abs(df["zscore"])
        
        return df
    
    def _add_forecast(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add simple rolling average forecast."""
        window = config.get("forecast_window", 24)
        
        # Simple moving average forecast
        df["forecast"] = df["value"].rolling(window=window, min_periods=1).mean().shift(1)
        df["forecast_error"] = df["value"] - df["forecast"]
        
        return df
