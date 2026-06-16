"""
Text Generation Service using N-gram Language Model.

Trains on review data and generates new text.
"""
from typing import Dict, Any, Optional
import pandas as pd
import logging
from pathlib import Path

from app.services.ngram_language_model import NgramLanguageModel


class TextGenerationService:
    """Service for training and using text generation models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Use absolute path relative to backend directory
        backend_dir = Path(__file__).parent.parent.parent
        self.models_dir = backend_dir / "data" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.current_model: Optional[NgramLanguageModel] = None
        self.current_model_name: Optional[str] = None
    
    def train_model(
        self, 
        corpus: list[str],
        model_name: str = "review_generator",
        n: int = 3
    ) -> Dict[str, Any]:
        """
        Train a new n-gram language model.
        
        Args:
            corpus: List of text documents
            model_name: Name to save the model
            n: N-gram size (3 = trigram)
            
        Returns:
            Training statistics
        """
        self.logger.info(f"🎓 Training text generation model: {model_name}")
        
        # Create and train model
        model = NgramLanguageModel(n=n)
        stats = model.train(corpus)
        
        # Save model
        model_path = self.models_dir / f"{model_name}_ngram.pkl"
        model.save(model_path)
        
        # Set as current
        self.current_model = model
        self.current_model_name = model_name
        
        self.logger.info(f"✅ Model trained and saved: {model_name}")
        
        return {
            "model_name": model_name,
            "model_path": str(model_path),
            **stats
        }
    
    def load_model(self, model_name: str):
        """Load a saved model."""
        # Handle model names that already end with _ngram
        if model_name.endswith('_ngram'):
            model_path = self.models_dir / f"{model_name}.pkl"
        else:
            model_path = self.models_dir / f"{model_name}_ngram.pkl"
        
        if not model_path.exists():
            raise ValueError(f"Model '{model_name}' not found at {model_path}")
        
        self.current_model = NgramLanguageModel.load(model_path)
        self.current_model_name = model_name
        
        self.logger.info(f"📂 Loaded model: {model_name}")
    
    def generate_text(
        self,
        prompt: str = "",
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 40,
        num_samples: int = 1
    ) -> Dict[str, Any]:
        """
        Generate text using the current model.
        
        Args:
            prompt: Starting text
            max_length: Maximum words to generate
            temperature: Randomness (0.5=conservative, 2.0=creative)
            top_k: Only sample from top-k words
            num_samples: Number of different texts to generate
            
        Returns:
            Generation results with metadata
        """
        if not self.current_model:
            raise ValueError("No model loaded. Train or load a model first.")
        
        self.logger.info(f"🎨 Generating {num_samples} sample(s) from prompt: '{prompt}'")
        
        samples = []
        
        for i in range(num_samples):
            generated_text, steps = self.current_model.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k
            )
            
            samples.append({
                "sample_id": i + 1,
                "text": generated_text,
                "length": len(generated_text.split()),
                "generation_steps": steps if i == 0 else None  # Only include steps for first sample
            })
        
        return {
            "model_name": self.current_model_name,
            "prompt": prompt,
            "parameters": {
                "max_length": max_length,
                "temperature": temperature,
                "top_k": top_k,
                "n": self.current_model.n
            },
            "samples": samples
        }
    
    def train_from_csv(
        self,
        csv_path: str,
        text_column: str,
        model_name: str = "review_generator",
        n: int = 3
    ) -> Dict[str, Any]:
        """
        Train model from a CSV file.
        
        Args:
            csv_path: Path to CSV file
            text_column: Column containing text data
            model_name: Name for the model
            n: N-gram size
            
        Returns:
            Training statistics
        """
        self.logger.info(f"📖 Loading corpus from {csv_path}, column: {text_column}")
        
        # Load data
        df = pd.read_csv(csv_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV. Available: {list(df.columns)}")
        
        # Extract text corpus
        corpus = df[text_column].dropna().astype(str).tolist()
        
        self.logger.info(f"📚 Loaded {len(corpus)} documents")
        
        # Train model
        return self.train_model(corpus, model_name=model_name, n=n)
    
    def has_model_for_intent(self, intent: str) -> bool:
        """
        Check if current model is suitable for generating responses for this intent.
        For now, returns True if model is loaded (intent-agnostic generation).
        """
        return self.current_model is not None
    
    def generate_response(
        self,
        intent: str,
        seed: Optional[str] = None,
        max_length: int = 50,
        temperature: float = 0.8,
        context_docs: Optional[list] = None
    ) -> str:
        """
        Generate a conversational response for the given intent.
        
        Args:
            intent: The classified intent
            seed: Starting text (if None, uses context_docs or empty)
            max_length: Maximum words to generate
            temperature: Randomness (0.5=conservative, 2.0=creative)
            context_docs: List of context documents to seed generation
            
        Returns:
            Generated response text
        """
        if not self.current_model:
            return ""
        
        # Determine seed text
        if seed is None:
            if context_docs and len(context_docs) > 0:
                # Use first few words of top context document
                seed = " ".join(context_docs[0].split()[:5])
            else:
                # Use intent-based seed
                seed = ""
        
        try:
            generated_text, _ = self.current_model.generate(
                prompt=seed,
                max_length=max_length,
                temperature=temperature,
                top_k=40
            )
            return generated_text
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return ""


# Global instance
text_generation_service = TextGenerationService()
