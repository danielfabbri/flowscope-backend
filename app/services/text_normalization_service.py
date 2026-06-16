from typing import Dict, Any, Optional
import pandas as pd
import re
import unicodedata
from bs4 import BeautifulSoup

from app.services.base_service import BasePipelineService


class TextNormalizationService(BasePipelineService):
    """Text normalization service for NLP preprocessing.
    
    Handles lowercase conversion, punctuation removal, HTML cleaning,
    whitespace normalization, and accent removal.
    """
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Normalize text data.
        
        Config options:
            - text_columns: list of column names to normalize (required)
            - lowercase: convert to lowercase (default: True)
            - remove_html: remove HTML tags (default: True)
            - remove_urls: remove URLs (default: True)
            - remove_emails: remove email addresses (default: False)
            - remove_punctuation: remove punctuation (default: False)
            - remove_numbers: remove numeric characters (default: False)
            - remove_accents: remove accents/diacritics (default: False)
            - normalize_whitespace: normalize whitespace (default: True)
            - custom_replacements: dict of {pattern: replacement} for custom regex replacements
            - output_suffix: suffix for normalized columns (default: '_normalized')
            - inplace: replace original columns (default: False)
        """
        if data is None:
            raise ValueError("TextNormalizationService requires input data")
        
        from app.core.logger import logger
        
        text_columns = config.get("text_columns")
        if not text_columns:
            raise ValueError("text_columns is required for text normalization")
        
        if isinstance(text_columns, str):
            text_columns = [text_columns]
        
        # Validate columns exist
        missing_cols = [col for col in text_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        
        df = data.copy()
        
        # Extract config
        lowercase = config.get("lowercase", True)
        remove_html = config.get("remove_html", True)
        remove_urls = config.get("remove_urls", True)
        remove_emails = config.get("remove_emails", False)
        remove_punctuation = config.get("remove_punctuation", False)
        remove_numbers = config.get("remove_numbers", False)
        remove_accents = config.get("remove_accents", False)
        normalize_whitespace = config.get("normalize_whitespace", True)
        custom_replacements = config.get("custom_replacements", {})
        output_suffix = config.get("output_suffix", "_normalized")
        inplace = config.get("inplace", False)
        
        logger.info(f"📝 Text Normalization started")
        logger.info(f"Columns: {text_columns}")
        logger.info(f"Options: lowercase={lowercase}, remove_html={remove_html}, "
                   f"remove_urls={remove_urls}, remove_punctuation={remove_punctuation}")
        
        for col in text_columns:
            # Determine output column
            output_col = col if inplace else f"{col}{output_suffix}"
            
            # Convert to string and handle nulls
            df[output_col] = df[col].fillna("").astype(str)
            
            # Apply transformations in order
            if remove_html:
                df[output_col] = df[output_col].apply(self._remove_html)
            
            if remove_urls:
                df[output_col] = df[output_col].apply(self._remove_urls)
            
            if remove_emails:
                df[output_col] = df[output_col].apply(self._remove_emails)
            
            if lowercase:
                df[output_col] = df[output_col].str.lower()
            
            if remove_accents:
                df[output_col] = df[output_col].apply(self._remove_accents)
            
            if remove_numbers:
                df[output_col] = df[output_col].apply(lambda x: re.sub(r'\d+', '', x))
            
            if remove_punctuation:
                df[output_col] = df[output_col].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
            
            # Apply custom replacements
            for pattern, replacement in custom_replacements.items():
                df[output_col] = df[output_col].apply(
                    lambda x: re.sub(pattern, replacement, x)
                )
            
            if normalize_whitespace:
                df[output_col] = df[output_col].apply(self._normalize_whitespace)
            
            logger.info(f"✓ Normalized column: {col} -> {output_col}")
        
        logger.info(f"✅ Text normalization complete")
        return df
    
    @staticmethod
    def _remove_html(text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return text
        try:
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text()
        except:
            # Fallback to regex if BeautifulSoup fails
            return re.sub(r'<[^>]+>', '', text)
    
    @staticmethod
    def _remove_urls(text: str) -> str:
        """Remove URLs from text."""
        # Remove http/https URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        return text
    
    @staticmethod
    def _remove_emails(text: str) -> str:
        """Remove email addresses from text."""
        return re.sub(r'\S+@\S+', '', text)
    
    @staticmethod
    def _remove_accents(text: str) -> str:
        """Remove accents and diacritics from text."""
        if not text:
            return text
        # Normalize to NFD (decomposed form), then filter out combining characters
        nfd = unicodedata.normalize('NFD', text)
        return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    
    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace: remove extra spaces, tabs, newlines."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        return text.strip()
