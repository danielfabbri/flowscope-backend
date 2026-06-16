from typing import Dict, Any, Optional
import pandas as pd
from textblob import TextBlob

from app.services.base_service import BasePipelineService


class SentimentAnalysisService(BasePipelineService):
    """Sentiment analysis service for determining text polarity and subjectivity.
    
    Uses TextBlob for sentiment analysis:
    - Polarity: -1 (negative) to +1 (positive)
    - Subjectivity: 0 (objective) to 1 (subjective)
    """
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Perform sentiment analysis on text data.
        
        Config options:
            - text_columns: list of column names to analyze (required)
            - metrics: list of metrics to compute (default: ['polarity', 'subjectivity'])
                * 'polarity': sentiment polarity (-1 to +1)
                * 'subjectivity': text subjectivity (0 to 1)
                * 'sentiment_label': categorical label (positive/negative/neutral)
            - polarity_threshold: threshold for neutral sentiment (default: 0.1)
                * Values between -threshold and +threshold are neutral
            - input_format: 'text' or 'tokens' (default: 'text')
            - token_separator: separator for joining tokens (default: ' ')
            - suffix_polarity: suffix for polarity column (default: '_polarity')
            - suffix_subjectivity: suffix for subjectivity column (default: '_subjectivity')
            - suffix_label: suffix for sentiment label column (default: '_sentiment')
        """
        if data is None:
            raise ValueError("SentimentAnalysisService requires input data")
        
        from app.core.logger import logger
        
        text_columns = config.get("text_columns")
        if not text_columns:
            raise ValueError("text_columns is required for sentiment analysis")
        
        if isinstance(text_columns, str):
            text_columns = [text_columns]
        
        # Validate columns exist
        missing_cols = [col for col in text_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        
        df = data.copy()
        
        # Extract config
        metrics = config.get("metrics", ["polarity", "subjectivity"])
        polarity_threshold = config.get("polarity_threshold", 0.1)
        input_format = config.get("input_format", "text")
        token_separator = config.get("token_separator", " ")
        suffix_polarity = config.get("suffix_polarity", "_polarity")
        suffix_subjectivity = config.get("suffix_subjectivity", "_subjectivity")
        suffix_label = config.get("suffix_label", "_sentiment")
        
        logger.info(f"😊 Sentiment Analysis started")
        logger.info(f"Columns: {text_columns}")
        logger.info(f"Metrics: {metrics}")
        
        for col in text_columns:
            # Prepare text data
            if input_format == "tokens":
                text_data = df[col].apply(
                    lambda x: token_separator.join(x) if isinstance(x, list) else str(x)
                )
            else:
                text_data = df[col].fillna("").astype(str)
            
            # Compute sentiment
            sentiments = text_data.apply(self._analyze_sentiment)
            
            # Extract requested metrics
            if "polarity" in metrics:
                df[f"{col}{suffix_polarity}"] = sentiments.apply(lambda x: x['polarity'])
            
            if "subjectivity" in metrics:
                df[f"{col}{suffix_subjectivity}"] = sentiments.apply(lambda x: x['subjectivity'])
            
            if "sentiment_label" in metrics:
                df[f"{col}{suffix_label}"] = sentiments.apply(
                    lambda x: self._get_sentiment_label(x['polarity'], polarity_threshold)
                )
            
            # Calculate statistics
            avg_polarity = sentiments.apply(lambda x: x['polarity']).mean()
            avg_subjectivity = sentiments.apply(lambda x: x['subjectivity']).mean()
            
            logger.info(f"✓ Analyzed {col}: avg polarity={avg_polarity:.3f}, "
                       f"avg subjectivity={avg_subjectivity:.3f}")
        
        logger.info(f"✅ Sentiment analysis complete")
        return df
    
    @staticmethod
    def _analyze_sentiment(text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text."""
        if not text or text.strip() == "":
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except:
            # Return neutral if analysis fails
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    @staticmethod
    def _get_sentiment_label(polarity: float, threshold: float) -> str:
        """Convert polarity score to categorical label."""
        if polarity > threshold:
            return "positive"
        elif polarity < -threshold:
            return "negative"
        else:
            return "neutral"
