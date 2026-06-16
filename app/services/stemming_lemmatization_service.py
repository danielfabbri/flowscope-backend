from typing import Dict, Any, Optional, List
import pandas as pd
import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from app.services.base_service import BasePipelineService


class StemmingLemmatizationService(BasePipelineService):
    """Stemming and lemmatization service for word normalization.
    
    Reduces words to their root/base form using various algorithms.
    Stemming: faster, rule-based (may produce non-words)
    Lemmatization: slower, dictionary-based (produces valid words)
    """
    
    def __init__(self):
        """Initialize and download required NLTK data."""
        super().__init__()
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """Download required NLTK data packages."""
        required_data = ['wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
        for data_name in required_data:
            try:
                nltk.data.find(f'corpora/{data_name}')
            except LookupError:
                try:
                    nltk.download(data_name, quiet=True)
                except:
                    pass  # Some data might not be available in all NLTK versions
        
        # Also try averaged_perceptron_tagger_eng for newer NLTK versions
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            try:
                nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            except:
                pass
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Apply stemming or lemmatization to tokens.
        
        Config options:
            - text_columns: list of column names containing tokens (required)
            - method: normalization method (default: 'lemmatize')
                * 'lemmatize': lemmatization (produces valid words)
                * 'stem_porter': Porter stemmer (most common)
                * 'stem_snowball': Snowball stemmer (supports multiple languages)
                * 'stem_lancaster': Lancaster stemmer (most aggressive)
            - language: language for Snowball stemmer (default: 'english')
                * Supported: arabic, danish, dutch, english, finnish, french, german,
                  hungarian, italian, norwegian, portuguese, romanian, russian, spanish, swedish
            - pos_tagging: use POS tagging for better lemmatization (default: False)
                * Only for lemmatize method, improves accuracy but slower
            - input_format: 'list' or 'string' (default: 'list')
            - output_format: 'list' or 'string' (default: same as input)
            - separator: separator for string format (default: ' ')
            - output_suffix: suffix for output columns (default: '_stemmed' or '_lemmatized')
        """
        if data is None:
            raise ValueError("StemmingLemmatizationService requires input data")
        
        from app.core.logger import logger
        
        text_columns = config.get("text_columns")
        if not text_columns:
            raise ValueError("text_columns is required for stemming/lemmatization")
        
        if isinstance(text_columns, str):
            text_columns = [text_columns]
        
        # Validate columns exist
        missing_cols = [col for col in text_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        
        df = data.copy()
        
        # Extract config
        method = config.get("method", "lemmatize")
        language = config.get("language", "english")
        pos_tagging = config.get("pos_tagging", False)
        input_format = config.get("input_format", "list")
        output_format = config.get("output_format", input_format)
        separator = config.get("separator", " ")
        
        # Determine output suffix
        if "output_suffix" in config:
            output_suffix = config["output_suffix"]
        else:
            output_suffix = "_lemmatized" if method == "lemmatize" else "_stemmed"
        
        logger.info(f"🌿 Stemming/Lemmatization started")
        logger.info(f"Columns: {text_columns}")
        logger.info(f"Method: {method}, language: {language}")
        
        # Initialize normalizer
        normalizer = self._get_normalizer(method, language)
        
        for col in text_columns:
            output_col = f"{col}{output_suffix}"
            
            # Process based on input format
            if input_format == "string":
                # Split string into tokens
                df[output_col] = df[col].fillna("").astype(str).apply(
                    lambda text: self._normalize_string(
                        text, normalizer, method, pos_tagging, separator
                    )
                )
            else:
                # Assume list of tokens
                df[output_col] = df[col].apply(
                    lambda tokens: self._normalize_list(
                        tokens, normalizer, method, pos_tagging
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
            
            logger.info(f"✓ Normalized column: {col} -> {output_col}")
        
        logger.info(f"✅ Stemming/Lemmatization complete")
        return df
    
    def _get_normalizer(self, method: str, language: str):
        """Get the appropriate stemmer or lemmatizer."""
        if method == "lemmatize":
            return WordNetLemmatizer()
        elif method == "stem_porter":
            return PorterStemmer()
        elif method == "stem_snowball":
            return SnowballStemmer(language)
        elif method == "stem_lancaster":
            return LancasterStemmer()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _normalize_list(self, tokens: Any, normalizer, method: str, 
                       pos_tagging: bool) -> List[str]:
        """Normalize a list of tokens."""
        if not isinstance(tokens, list):
            return []
        
        if method == "lemmatize" and pos_tagging:
            # Use POS tagging for better lemmatization
            return self._lemmatize_with_pos(tokens, normalizer)
        elif method == "lemmatize":
            # Simple lemmatization without POS
            return [normalizer.lemmatize(token) for token in tokens]
        else:
            # Stemming
            return [normalizer.stem(token) for token in tokens]
    
    def _normalize_string(self, text: str, normalizer, method: str,
                         pos_tagging: bool, separator: str) -> str:
        """Normalize a string."""
        if not text:
            return ""
        
        tokens = text.split(separator)
        normalized = self._normalize_list(tokens, normalizer, method, pos_tagging)
        return separator.join(normalized)
    
    def _lemmatize_with_pos(self, tokens: List[str], lemmatizer) -> List[str]:
        """Lemmatize tokens using POS tagging for better accuracy."""
        try:
            # Get POS tags
            pos_tags = nltk.pos_tag(tokens)
            
            # Lemmatize with appropriate POS tag
            lemmatized = []
            for token, pos in pos_tags:
                wordnet_pos = self._get_wordnet_pos(pos)
                if wordnet_pos:
                    lemmatized.append(lemmatizer.lemmatize(token, pos=wordnet_pos))
                else:
                    lemmatized.append(lemmatizer.lemmatize(token))
            
            return lemmatized
        except:
            # Fallback to simple lemmatization if POS tagging fails
            return [lemmatizer.lemmatize(token) for token in tokens]
    
    @staticmethod
    def _get_wordnet_pos(treebank_tag: str) -> Optional[str]:
        """Convert Treebank POS tag to WordNet POS tag."""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
