from typing import Dict, Any, Optional, List, Set
import pandas as pd
import nltk
from nltk.corpus import stopwords

from app.services.base_service import BasePipelineService


class StopWordsRemovalService(BasePipelineService):
    """Stop words removal service for filtering common words from tokens.
    
    Supports multiple languages and custom stop word lists.
    """
    
    def __init__(self):
        """Initialize and download required NLTK data."""
        super().__init__()
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """Download required NLTK data packages."""
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Remove stop words from tokenized text.
        
        Config options:
            - text_columns: list of column names containing tokens (required)
            - language: language for stop words (default: 'english')
                * Supported: 'arabic', 'azerbaijani', 'bengali', 'catalan', 'chinese',
                  'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'greek',
                  'hebrew', 'hungarian', 'indonesian', 'italian', 'kazakh', 'nepali',
                  'norwegian', 'portuguese', 'romanian', 'russian', 'spanish', 'swedish',
                  'turkish'
            - custom_stopwords: list of additional stop words to remove
            - exclude_stopwords: list of stop words to keep (whitelist)
            - case_sensitive: whether matching is case-sensitive (default: False)
            - input_format: 'list' or 'string' (default: 'list')
            - output_format: 'list' or 'string' (default: same as input)
            - separator: separator for string format (default: ' ')
            - output_suffix: suffix for output columns (default: '_no_stopwords')
        """
        if data is None:
            raise ValueError("StopWordsRemovalService requires input data")
        
        from app.core.logger import logger
        
        text_columns = config.get("text_columns")
        if not text_columns:
            raise ValueError("text_columns is required for stop words removal")
        
        if isinstance(text_columns, str):
            text_columns = [text_columns]
        
        # Validate columns exist
        missing_cols = [col for col in text_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        
        df = data.copy()
        
        # Extract config
        language = config.get("language", "english")
        custom_stopwords = config.get("custom_stopwords", [])
        exclude_stopwords = config.get("exclude_stopwords", [])
        case_sensitive = config.get("case_sensitive", False)
        input_format = config.get("input_format", "list")
        output_format = config.get("output_format", input_format)
        separator = config.get("separator", " ")
        output_suffix = config.get("output_suffix", "_no_stopwords")
        
        logger.info(f"🚫 Stop Words Removal started")
        logger.info(f"Columns: {text_columns}")
        logger.info(f"Language: {language}, case_sensitive: {case_sensitive}")
        
        # Build stop words set
        stop_words_set = self._build_stopwords_set(
            language, custom_stopwords, exclude_stopwords, case_sensitive
        )
        
        logger.info(f"Stop words count: {len(stop_words_set)}")
        
        for col in text_columns:
            output_col = f"{col}{output_suffix}"
            
            # Process based on input format
            if input_format == "string":
                # Split string into tokens
                df[output_col] = df[col].fillna("").astype(str).apply(
                    lambda text: self._remove_stopwords_from_string(
                        text, stop_words_set, case_sensitive, separator
                    )
                )
            else:
                # Assume list of tokens
                df[output_col] = df[col].apply(
                    lambda tokens: self._remove_stopwords_from_list(
                        tokens, stop_words_set, case_sensitive
                    )
                )
            
            # Convert output format if needed
            if output_format == "string" and input_format == "list":
                df[output_col] = df[output_col].apply(
                    lambda tokens: separator.join(tokens) if isinstance(tokens, list) else ""
                )
            elif output_format == "list" and input_format == "string":
                df[output_col] = df[output_col].apply(
                    lambda text: text.split(separator) if isinstance(text, str) else []
                )
            
            # Calculate removal statistics
            if input_format == "list":
                original_counts = df[col].apply(lambda x: len(x) if isinstance(x, list) else 0)
                filtered_counts = df[output_col].apply(lambda x: len(x) if isinstance(x, list) else 0)
                removed = (original_counts - filtered_counts).sum()
                total = original_counts.sum()
                removal_pct = (removed / total * 100) if total > 0 else 0
                logger.info(f"✓ Removed {removed:,} stop words from {col} ({removal_pct:.1f}%)")
            else:
                logger.info(f"✓ Processed column: {col} -> {output_col}")
        
        logger.info(f"✅ Stop words removal complete")
        return df
    
    def _build_stopwords_set(self, language: str, custom_stopwords: List[str],
                            exclude_stopwords: List[str], case_sensitive: bool) -> Set[str]:
        """Build the set of stop words to remove."""
        try:
            # Get NLTK stop words for language
            stop_words = set(stopwords.words(language))
        except OSError:
            raise ValueError(f"Language '{language}' not supported. Please check NLTK stopwords corpus.")
        
        # Add custom stop words
        if custom_stopwords:
            stop_words.update(custom_stopwords)
        
        # Remove excluded stop words
        if exclude_stopwords:
            stop_words -= set(exclude_stopwords)
        
        # Convert to lowercase if not case-sensitive
        if not case_sensitive:
            stop_words = {word.lower() for word in stop_words}
        
        return stop_words
    
    def _remove_stopwords_from_list(self, tokens: Any, stop_words: Set[str],
                                    case_sensitive: bool) -> List[str]:
        """Remove stop words from a list of tokens."""
        if not isinstance(tokens, list):
            return []
        
        filtered = []
        for token in tokens:
            check_token = token if case_sensitive else token.lower()
            if check_token not in stop_words:
                filtered.append(token)
        
        return filtered
    
    def _remove_stopwords_from_string(self, text: str, stop_words: Set[str],
                                      case_sensitive: bool, separator: str) -> str:
        """Remove stop words from a string."""
        if not text:
            return ""
        
        tokens = text.split(separator)
        filtered = self._remove_stopwords_from_list(tokens, stop_words, case_sensitive)
        return separator.join(filtered)
