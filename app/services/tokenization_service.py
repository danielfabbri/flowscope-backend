from typing import Dict, Any, Optional, List
import pandas as pd
import re
import nltk
from nltk.tokenize import (
    word_tokenize, sent_tokenize, wordpunct_tokenize,
    TweetTokenizer, RegexpTokenizer
)

from app.services.base_service import BasePipelineService


class TokenizationService(BasePipelineService):
    """Tokenization service for splitting text into tokens.
    
    Supports multiple tokenization methods: word, sentence, tweet, regex, and custom.
    """
    
    def __init__(self):
        """Initialize and download required NLTK data."""
        super().__init__()
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """Download required NLTK data packages."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Tokenize text data.
        
        Config options:
            - text_columns: list of column names to tokenize (required)
            - method: tokenization method (default: 'word')
                * 'word': standard word tokenization
                * 'sentence': sentence tokenization
                * 'wordpunct': word tokenization keeping punctuation separate
                * 'tweet': Twitter-aware tokenization (handles @mentions, #hashtags)
                * 'regex': custom regex-based tokenization
                * 'whitespace': simple whitespace splitting
            - regex_pattern: regex pattern for 'regex' method (default: r'\w+')
            - keep_case: preserve original case (default: False, converts to lower)
            - min_token_length: minimum token length to keep (default: 1)
            - max_token_length: maximum token length to keep (default: None)
            - preserve_handles: for tweet method, keep @mentions (default: True)
            - preserve_hashtags: for tweet method, keep #hashtags (default: True)
            - output_suffix: suffix for tokenized columns (default: '_tokens')
            - output_as_string: join tokens back to string (default: False)
            - separator: separator when output_as_string=True (default: ' ')
        """
        if data is None:
            raise ValueError("TokenizationService requires input data")
        
        from app.core.logger import logger
        
        text_columns = config.get("text_columns")
        if not text_columns:
            raise ValueError("text_columns is required for tokenization")
        
        if isinstance(text_columns, str):
            text_columns = [text_columns]
        
        # Validate columns exist
        missing_cols = [col for col in text_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        
        df = data.copy()
        
        # Extract config
        method = config.get("method", "word")
        regex_pattern = config.get("regex_pattern", r'\w+')
        keep_case = config.get("keep_case", False)
        min_token_length = config.get("min_token_length", 1)
        max_token_length = config.get("max_token_length", None)
        preserve_handles = config.get("preserve_handles", True)
        preserve_hashtags = config.get("preserve_hashtags", True)
        output_suffix = config.get("output_suffix", "_tokens")
        output_as_string = config.get("output_as_string", False)
        separator = config.get("separator", " ")
        
        logger.info(f"🔤 Tokenization started")
        logger.info(f"Columns: {text_columns}")
        logger.info(f"Method: {method}, min_length: {min_token_length}")
        
        # Get tokenizer function
        tokenizer = self._get_tokenizer(method, regex_pattern, preserve_handles, preserve_hashtags)
        
        for col in text_columns:
            output_col = f"{col}{output_suffix}"
            
            # Tokenize
            df[output_col] = df[col].fillna("").astype(str).apply(tokenizer)
            
            # Post-process tokens
            df[output_col] = df[output_col].apply(
                lambda tokens: self._post_process_tokens(
                    tokens, keep_case, min_token_length, max_token_length
                )
            )
            
            # Convert back to string if requested
            if output_as_string:
                df[output_col] = df[output_col].apply(lambda tokens: separator.join(tokens))
            
            token_counts = df[output_col].apply(lambda x: len(x) if isinstance(x, list) else 0)
            avg_tokens = token_counts.mean()
            logger.info(f"✓ Tokenized column: {col} -> {output_col} (avg {avg_tokens:.1f} tokens)")
        
        logger.info(f"✅ Tokenization complete")
        return df
    
    def _get_tokenizer(self, method: str, regex_pattern: str, 
                       preserve_handles: bool, preserve_hashtags: bool):
        """Get tokenizer function based on method."""
        if method == "word":
            return word_tokenize
        elif method == "sentence":
            return sent_tokenize
        elif method == "wordpunct":
            return wordpunct_tokenize
        elif method == "tweet":
            tokenizer = TweetTokenizer(
                preserve_case=True,
                reduce_len=True,
                strip_handles=not preserve_handles
            )
            return tokenizer.tokenize
        elif method == "regex":
            tokenizer = RegexpTokenizer(regex_pattern)
            return tokenizer.tokenize
        elif method == "whitespace":
            return lambda text: text.split()
        else:
            raise ValueError(f"Unknown tokenization method: {method}")
    
    def _post_process_tokens(self, tokens: List[str], keep_case: bool,
                            min_length: int, max_length: Optional[int]) -> List[str]:
        """Post-process tokens: case conversion, length filtering."""
        if not isinstance(tokens, list):
            return []
        
        # Case conversion
        if not keep_case:
            tokens = [t.lower() for t in tokens]
        
        # Length filtering
        filtered = []
        for token in tokens:
            token_len = len(token)
            if token_len < min_length:
                continue
            if max_length is not None and token_len > max_length:
                continue
            filtered.append(token)
        
        return filtered
