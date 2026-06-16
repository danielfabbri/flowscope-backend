from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from app.services.base_service import BasePipelineService


class TextVectorizationService(BasePipelineService):
    """Text vectorization service for converting text to numerical features.
    
    Supports TF-IDF, Bag of Words (Count), and Binary vectorization.
    Creates feature columns that can be used for machine learning.
    """
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Vectorize text data into numerical features.
        
        Config options:
            - text_columns: list of column names to vectorize (required)
            - method: vectorization method (default: 'tfidf')
                * 'tfidf': TF-IDF (Term Frequency-Inverse Document Frequency)
                * 'count': Bag of Words (term counts)
                * 'binary': Binary occurrence (0 or 1)
            - max_features: maximum number of features to keep (default: 100)
            - min_df: minimum document frequency (default: 1)
                * If int: minimum number of documents
                * If float: minimum proportion of documents (0.0 to 1.0)
            - max_df: maximum document frequency (default: 1.0)
                * If int: maximum number of documents
                * If float: maximum proportion of documents (0.0 to 1.0)
            - ngram_range: n-gram range tuple (default: (1, 1) for unigrams)
                * Example: (1, 2) for unigrams and bigrams
            - use_idf: use inverse document frequency (TF-IDF only) (default: True)
            - sublinear_tf: apply sublinear tf scaling (TF-IDF only) (default: False)
            - norm: normalization method for TF-IDF (default: 'l2')
                * 'l1': L1 normalization
                * 'l2': L2 normalization
                * None: no normalization
            - input_format: 'text' or 'tokens' (default: 'text')
                * 'text': expects string columns
                * 'tokens': expects list columns (will be joined)
            - token_separator: separator for joining tokens (default: ' ')
            - output_format: 'sparse' or 'dense' (default: 'dense')
                * 'sparse': separate columns for each feature
                * 'dense': single column with array/list
            - feature_prefix: prefix for feature columns (default: 'tfidf_' or 'count_')
            - top_n_features: return only top N features per document (default: None)
        """
        if data is None:
            raise ValueError("TextVectorizationService requires input data")
        
        from app.core.logger import logger
        
        text_columns = config.get("text_columns")
        if not text_columns:
            raise ValueError("text_columns is required for text vectorization")
        
        if isinstance(text_columns, str):
            text_columns = [text_columns]
        
        # Validate columns exist
        missing_cols = [col for col in text_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        
        df = data.copy()
        
        # Extract config
        method = config.get("method", "tfidf")
        max_features = config.get("max_features", 100)
        min_df = config.get("min_df", 1)
        max_df = config.get("max_df", 1.0)
        ngram_range = tuple(config.get("ngram_range", [1, 1]))
        use_idf = config.get("use_idf", True)
        sublinear_tf = config.get("sublinear_tf", False)
        norm = config.get("norm", "l2")
        input_format = config.get("input_format", "text")
        token_separator = config.get("token_separator", " ")
        output_format = config.get("output_format", "dense")
        top_n_features = config.get("top_n_features", None)
        
        # Determine feature prefix
        if "feature_prefix" in config:
            feature_prefix = config["feature_prefix"]
        else:
            feature_prefix = f"{method}_"
        
        logger.info(f"🔢 Text Vectorization started")
        logger.info(f"Columns: {text_columns}")
        logger.info(f"Method: {method}, max_features: {max_features}, ngram_range: {ngram_range}")
        
        for col in text_columns:
            # Prepare text data
            if input_format == "tokens":
                # Join tokens into strings
                text_data = df[col].apply(
                    lambda x: token_separator.join(x) if isinstance(x, list) else str(x)
                )
            else:
                text_data = df[col].fillna("").astype(str)
            
            # Create vectorizer
            vectorizer = self._create_vectorizer(
                method, max_features, min_df, max_df, ngram_range,
                use_idf, sublinear_tf, norm
            )
            
            # Fit and transform
            try:
                vectors = vectorizer.fit_transform(text_data)
                feature_names = vectorizer.get_feature_names_out()
                
                logger.info(f"✓ Vectorized {col}: {vectors.shape[1]} features created")
                
                # Add features to dataframe
                if output_format == "sparse":
                    # Create separate column for each feature
                    self._add_sparse_features(df, vectors, feature_names, feature_prefix, col)
                else:
                    # Create single column with dense array
                    self._add_dense_features(df, vectors, col, feature_prefix, 
                                            feature_names, top_n_features)
                
            except ValueError as e:
                logger.warning(f"⚠️ Could not vectorize {col}: {str(e)}")
                # Add empty features
                if output_format == "sparse":
                    df[f"{feature_prefix}{col}"] = 0
                else:
                    df[f"{col}_vector"] = [[] for _ in range(len(df))]
        
        logger.info(f"✅ Text vectorization complete")
        return df
    
    def _create_vectorizer(self, method: str, max_features: int, min_df, max_df,
                          ngram_range: tuple, use_idf: bool, sublinear_tf: bool,
                          norm: Optional[str]):
        """Create the appropriate vectorizer."""
        common_params = {
            'max_features': max_features,
            'min_df': min_df,
            'max_df': max_df,
            'ngram_range': ngram_range,
        }
        
        if method == "tfidf":
            return TfidfVectorizer(
                **common_params,
                use_idf=use_idf,
                sublinear_tf=sublinear_tf,
                norm=norm
            )
        elif method == "count":
            return CountVectorizer(**common_params)
        elif method == "binary":
            return CountVectorizer(**common_params, binary=True)
        else:
            raise ValueError(f"Unknown vectorization method: {method}")
    
    def _add_sparse_features(self, df: pd.DataFrame, vectors, feature_names: List[str],
                            prefix: str, col: str):
        """Add features as separate columns (sparse format)."""
        # Convert to dense array
        vectors_dense = vectors.toarray()
        
        # Add each feature as a column
        for i, feature_name in enumerate(feature_names):
            # Clean feature name for column name
            clean_name = feature_name.replace(" ", "_").replace("-", "_")
            col_name = f"{prefix}{col}_{clean_name}"
            df[col_name] = vectors_dense[:, i]
    
    def _add_dense_features(self, df: pd.DataFrame, vectors, col: str, 
                           prefix: str, feature_names: List[str],
                           top_n: Optional[int]):
        """Add features as a single column with arrays (dense format)."""
        vectors_dense = vectors.toarray()
        
        if top_n:
            # Keep only top N features per document
            result = []
            for row in vectors_dense:
                # Get top N indices
                top_indices = np.argsort(row)[-top_n:][::-1]
                # Create sparse representation: [(feature_name, value), ...]
                top_features = [
                    (feature_names[idx], float(row[idx]))
                    for idx in top_indices
                    if row[idx] > 0
                ]
                result.append(top_features)
            
            df[f"{col}_top_features"] = result
        else:
            # Store full vector as list
            df[f"{col}_vector"] = [row.tolist() for row in vectors_dense]
            
            # Also store feature names for reference
            df[f"{col}_feature_names"] = [feature_names.tolist()] * len(df)
