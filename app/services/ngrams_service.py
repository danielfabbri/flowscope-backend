from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from itertools import combinations

from app.services.base_service import BasePipelineService


class NgramsService(BasePipelineService):
    """N-grams generation service for creating token sequences.
    
    Generates unigrams, bigrams, trigrams, or custom n-grams from tokens.
    Useful for capturing multi-word expressions and context.
    """
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Generate n-grams from tokenized text.
        
        Config options:
            - text_columns: list of column names containing tokens (required)
            - n: n-gram size (default: 2 for bigrams)
                * Can be int (single n) or list of ints (multiple n-gram sizes)
                * Examples: 2 for bigrams, [1,2,3] for unigrams+bigrams+trigrams
            - separator: separator for joining tokens in n-gram (default: '_')
            - skip_ngrams: generate skip-grams (allow gaps) (default: False)
            - skip_distance: maximum distance for skip-grams (default: 1)
            - min_ngram_freq: minimum frequency to keep n-gram (default: 1)
            - keep_original: keep original unigrams along with n-grams (default: False)
            - input_format: 'list' or 'string' (default: 'list')
            - output_format: 'list' or 'string' (default: 'list')
            - input_separator: separator for string input (default: ' ')
            - output_separator: separator for string output (default: ' ')
            - output_suffix: suffix for output columns (default: '_ngrams')
        """
        if data is None:
            raise ValueError("NgramsService requires input data")
        
        from app.core.logger import logger
        
        text_columns = config.get("text_columns")
        if not text_columns:
            raise ValueError("text_columns is required for n-gram generation")
        
        if isinstance(text_columns, str):
            text_columns = [text_columns]
        
        # Validate columns exist
        missing_cols = [col for col in text_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        
        df = data.copy()
        
        # Extract config
        n = config.get("n", 2)
        separator = config.get("separator", "_")
        skip_ngrams = config.get("skip_ngrams", False)
        skip_distance = config.get("skip_distance", 1)
        min_ngram_freq = config.get("min_ngram_freq", 1)
        keep_original = config.get("keep_original", False)
        input_format = config.get("input_format", "list")
        output_format = config.get("output_format", "list")
        input_separator = config.get("input_separator", " ")
        output_separator = config.get("output_separator", " ")
        output_suffix = config.get("output_suffix", "_ngrams")
        
        # Normalize n to list
        if isinstance(n, int):
            n_values = [n]
        else:
            n_values = sorted(n)
        
        logger.info(f"📊 N-grams generation started")
        logger.info(f"Columns: {text_columns}")
        logger.info(f"N-gram sizes: {n_values}, skip_ngrams: {skip_ngrams}")
        
        for col in text_columns:
            output_col = f"{col}{output_suffix}"
            
            # Process based on input format
            if input_format == "string":
                # Split string into tokens
                df[output_col] = df[col].fillna("").astype(str).apply(
                    lambda text: self._generate_ngrams_from_string(
                        text, n_values, separator, skip_ngrams, skip_distance,
                        keep_original, input_separator
                    )
                )
            else:
                # Assume list of tokens
                df[output_col] = df[col].apply(
                    lambda tokens: self._generate_ngrams_from_list(
                        tokens, n_values, separator, skip_ngrams, skip_distance,
                        keep_original
                    )
                )
            
            # Apply frequency filtering if needed
            if min_ngram_freq > 1:
                df[output_col] = self._filter_by_frequency(df[output_col], min_ngram_freq)
            
            # Convert output format if needed
            if output_format == "string":
                df[output_col] = df[output_col].apply(
                    lambda ngrams: output_separator.join(ngrams) if isinstance(ngrams, list) else ""
                )
            
            # Calculate statistics
            if isinstance(df[output_col].iloc[0], list):
                avg_ngrams = df[output_col].apply(len).mean()
                logger.info(f"✓ Generated n-grams: {col} -> {output_col} (avg {avg_ngrams:.1f} n-grams)")
            else:
                logger.info(f"✓ Generated n-grams: {col} -> {output_col}")
        
        logger.info(f"✅ N-grams generation complete")
        return df
    
    def _generate_ngrams_from_list(self, tokens: Any, n_values: List[int],
                                   separator: str, skip_ngrams: bool,
                                   skip_distance: int, keep_original: bool) -> List[str]:
        """Generate n-grams from a list of tokens."""
        if not isinstance(tokens, list) or len(tokens) == 0:
            return []
        
        ngrams = []
        
        # Add original tokens if requested
        if keep_original and 1 not in n_values:
            ngrams.extend(tokens)
        
        # Generate n-grams for each n value
        for n in n_values:
            if n < 1:
                continue
            
            if skip_ngrams:
                ngrams.extend(self._generate_skipgrams(tokens, n, skip_distance, separator))
            else:
                ngrams.extend(self._generate_consecutive_ngrams(tokens, n, separator))
        
        return ngrams
    
    def _generate_ngrams_from_string(self, text: str, n_values: List[int],
                                    separator: str, skip_ngrams: bool,
                                    skip_distance: int, keep_original: bool,
                                    input_separator: str) -> List[str]:
        """Generate n-grams from a string."""
        if not text:
            return []
        
        tokens = text.split(input_separator)
        return self._generate_ngrams_from_list(
            tokens, n_values, separator, skip_ngrams, skip_distance, keep_original
        )
    
    def _generate_consecutive_ngrams(self, tokens: List[str], n: int, 
                                    separator: str) -> List[str]:
        """Generate consecutive n-grams."""
        if n == 1:
            return tokens
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = separator.join(tokens[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def _generate_skipgrams(self, tokens: List[str], n: int, 
                           max_skip: int, separator: str) -> List[str]:
        """Generate skip-grams (n-grams with gaps)."""
        if n == 1:
            return tokens
        
        skipgrams = []
        
        for i in range(len(tokens)):
            # Get all possible combinations within skip distance
            end_idx = min(i + n + max_skip * (n - 1), len(tokens))
            candidates = tokens[i:end_idx]
            
            if len(candidates) >= n:
                # Generate all combinations of size n
                for combo in combinations(range(len(candidates)), n):
                    # Check if first element is at position 0 (start with current token)
                    if combo[0] == 0:
                        # Check if skip distance is within limit
                        max_gap = max(combo[j] - combo[j-1] for j in range(1, len(combo)))
                        if max_gap <= max_skip + 1:
                            skipgram = separator.join(candidates[idx] for idx in combo)
                            skipgrams.append(skipgram)
        
        return skipgrams
    
    def _filter_by_frequency(self, ngrams_series: pd.Series, 
                            min_freq: int) -> pd.Series:
        """Filter n-grams by minimum frequency across all documents."""
        # Count frequency of each n-gram across all documents
        ngram_counts = {}
        for ngram_list in ngrams_series:
            if isinstance(ngram_list, list):
                for ngram in ngram_list:
                    ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        
        # Filter based on frequency
        frequent_ngrams = {ng for ng, count in ngram_counts.items() if count >= min_freq}
        
        # Apply filter to each document
        def filter_doc(ngram_list):
            if not isinstance(ngram_list, list):
                return []
            return [ng for ng in ngram_list if ng in frequent_ngrams]
        
        return ngrams_series.apply(filter_doc)
