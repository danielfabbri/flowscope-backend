"""
N-gram Language Model Training Service for Pipeline.

Integrates N-gram text generation model training into the pipeline system.
"""
from typing import Dict, Any, Optional
import pandas as pd
import logging
from pathlib import Path

from app.services.base_service import BasePipelineService
from app.services.ngram_language_model import NgramLanguageModel


class NgramTrainingService(BasePipelineService):
    """Service for training N-gram language models in a pipeline."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Use absolute path relative to backend directory
        backend_dir = Path(__file__).parent.parent.parent
        self.models_dir = backend_dir / "data" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Train an N-gram language model from text data.
        
        Args:
            data: Input DataFrame containing text data
            config: Configuration with:
                - text_column: Column containing text to train on (required)
                - model_name: Name to save the model (default: "ngram_model")
                - n: N-gram size (default: 5, range: 2-5)
                - min_text_length: Minimum text length to include (default: 10)
                - enable_kneser_ney: Enable Kneser-Ney smoothing (default: True)
                - smoothing_discount: Discount parameter for Kneser-Ney (default: 0.75)
        
        Returns:
            DataFrame with training statistics added as metadata
        """
        if data is None or data.empty:
            raise ValueError("NgramTrainingService requires input data")
        
        # Get configuration
        text_column = config.get("text_column", "").strip()
        model_name = config.get("model_name", "ngram_model").strip()
        n = int(config.get("n", 5))  # Default to 5-grams
        min_text_length = int(config.get("min_text_length", 10))
        enable_kneser_ney = config.get("enable_kneser_ney", True)
        smoothing_discount = float(config.get("smoothing_discount", 0.75))
        
        # Validate
        if not text_column:
            raise ValueError("text_column is required")
        
        if text_column not in data.columns:
            raise ValueError(
                f"Column '{text_column}' not found. Available columns: {list(data.columns)}"
            )
        
        if n < 2 or n > 5:
            raise ValueError(f"n must be between 2 and 5, got {n}")
        
        self.logger.info(f"🎓 N-gram Training Service - Training model '{model_name}'")
        self.logger.info(f"Text column: {text_column}, N-gram size: {n}")
        self.logger.info(f"Kneser-Ney smoothing: {enable_kneser_ney}, Discount: {smoothing_discount}")
        
        # Extract corpus
        corpus = data[text_column].dropna().astype(str).tolist()
        
        # Filter by minimum length
        original_count = len(corpus)
        corpus = [text for text in corpus if len(text) >= min_text_length]
        filtered_count = original_count - len(corpus)
        
        if filtered_count > 0:
            self.logger.info(f"Filtered out {filtered_count} texts shorter than {min_text_length} chars")
        
        if not corpus:
            raise ValueError(
                f"No valid text data found in column '{text_column}' after filtering"
            )
        
        self.logger.info(f"Training on {len(corpus)} documents")
        
        # Create and train model
        model = NgramLanguageModel(n=n)
        stats = model.train(corpus)
        
        # Store training config in model for generation
        model.training_config = {
            "enable_kneser_ney": enable_kneser_ney,
            "smoothing_discount": smoothing_discount
        }
        
        # Save model
        model_path = self.models_dir / f"{model_name}_ngram.pkl"
        model.save(model_path)
        
        self.logger.info(f"✅ Model trained and saved: {model_path}")
        self.logger.info(f"Statistics: {stats['unique_ngrams']} unique n-grams, "
                        f"vocabulary size: {stats['vocabulary_size']}")
        
        # Create output DataFrame with metadata
        output_df = pd.DataFrame({
            "model_name": [model_name],
            "model_path": [str(model_path)],
            "n": [n],
            "enable_kneser_ney": [enable_kneser_ney],
            "smoothing_discount": [smoothing_discount],
            "documents_trained": [stats['documents']],
            "total_tokens": [stats['total_tokens']],
            "unique_ngrams": [stats['unique_ngrams']],
            "vocabulary_size": [stats['vocabulary_size']],
            "text_column": [text_column],
            "training_status": ["success"]
        })
        
        # Add original data as additional context (useful for next steps)
        output_df["original_data_rows"] = len(data)
        output_df["original_data_columns"] = len(data.columns)
        
        return output_df


class NgramGenerationService(BasePipelineService):
    """Service for generating text using trained N-gram models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Use absolute path relative to backend directory
        backend_dir = Path(__file__).parent.parent.parent
        self.models_dir = backend_dir / "data" / "models"
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate text using a trained N-gram model.
        
        Args:
            data: Optional input DataFrame (can be None)
            config: Configuration with:
                - model_name: Name of the model to load (required)
                - prompt: Starting text (default: "")
                - max_length: Maximum words to generate (default: 50)
                - temperature: Randomness (default: 1.0)
                - top_k: Top-k sampling (default: 40)
                - num_samples: Number of texts to generate (default: 3)
                - enable_grammar_postprocessing: Apply grammar fixes (default: True)
        
        Returns:
            DataFrame with generated texts
        """
        # Get configuration
        model_name = config.get("model_name", "").strip()
        prompt = config.get("prompt", "")
        max_length = int(config.get("max_length", 50))
        temperature = float(config.get("temperature", 1.0))
        top_k = config.get("top_k", 40)
        num_samples = int(config.get("num_samples", 3))
        enable_grammar = config.get("enable_grammar_postprocessing", True)
        
        # Validate
        if not model_name:
            raise ValueError("model_name is required")
        
        model_path = self.models_dir / f"{model_name}_ngram.pkl"
        if not model_path.exists():
            raise ValueError(
                f"Model '{model_name}' not found at {model_path}. "
                f"Train the model first using ngram_training step."
            )
        
        if top_k is not None:
            top_k = int(top_k)
        
        self.logger.info(f"🎨 N-gram Generation Service - Loading model '{model_name}'")
        self.logger.info(f"Grammar post-processing: {enable_grammar}")
        
        # Load model
        model = NgramLanguageModel.load(model_path)
        
        self.logger.info(
            f"Generating {num_samples} samples with prompt: '{prompt}', "
            f"temperature: {temperature}, max_length: {max_length}"
        )
        
        # Generate texts
        samples = []
        for i in range(num_samples):
            generated_text, steps = model.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k
            )
            
            # Apply grammar post-processing if enabled
            if enable_grammar:
                generated_text = self._grammar_postprocessing(generated_text)
            
            samples.append({
                "sample_id": i + 1,
                "prompt": prompt,
                "generated_text": generated_text,
                "length": len(generated_text.split()),
                "temperature": temperature,
                "model_name": model_name,
                "grammar_postprocessed": enable_grammar
            })
        
        # Create output DataFrame
        output_df = pd.DataFrame(samples)
        
        self.logger.info(f"✅ Generated {num_samples} text samples")
        
        return output_df
    
    def _grammar_postprocessing(self, text: str) -> str:
        """Apply grammar rules and corrections to generated text
        
        Fixes common issues:
        - Double spaces
        - Repeated words
        - Article agreement (a/an)
        - Common contractions
        - Punctuation spacing
        """
        # Remove double spaces
        while "  " in text:
            text = text.replace("  ", " ")
        
        # Fix repeated words (e.g., "o o buraco" -> "o buraco")
        words = text.split()
        cleaned_words = [words[0]] if words else []
        for i in range(1, len(words)):
            if words[i] != words[i-1]:
                cleaned_words.append(words[i])
        text = " ".join(cleaned_words)
        
        # Portuguese grammar fixes
        text = text.replace(" de o ", " do ")
        text = text.replace(" de a ", " da ")
        text = text.replace(" em o ", " no ")
        text = text.replace(" em a ", " na ")
        text = text.replace(" por o ", " pelo ")
        text = text.replace(" por a ", " pela ")
        text = text.replace(" a o ", " ao ")
        text = text.replace(" a a ", " à ")
        
        # Fix spacing before punctuation
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace(" :", ":")
        text = text.replace(" ;", ";")
        
        # Fix spacing after punctuation
        for punct in [",", ":", ";"]:
            text = text.replace(f"{punct} ", f"{punct} ").replace(f"{punct}", f"{punct} ")
        
        # Remove space before apostrophe
        text = text.replace(" '", "'")
        
        # Fix multiple punctuation
        text = text.replace("?.", "?")
        text = text.replace("!.", "!")
        text = text.replace("..", ".")
        text = text.replace("??", "?")
        text = text.replace("!!", "!")
        
        return text.strip()
