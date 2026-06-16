"""
Intent-Conditioned Text Generation Service

Generates responses using n-gram models trained separately for each intent.
Combines intent classification with probabilistic text generation.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import random
import json
import joblib
from pathlib import Path
from collections import defaultdict, Counter
from app.core.logger import get_logger

logger = get_logger(__name__)


class IntentConditionedGenerationService:
    """Service for generating text conditioned on detected intent"""
    
    def __init__(self):
        self.intent_models: Dict[str, Dict] = {}  # {intent: {ngram_model, ...}}
        self.n = 5  # 5-gram by default (improved from trigram)
        self.smoothing_discount = 0.75  # Kneser-Ney discount parameter
        # Use absolute path
        backend_dir = Path(__file__).parent.parent.parent
        self.models_dir = backend_dir / "data" / "models"
        
    def train_intent_models(
        self,
        training_data: Dict[str, List[str]],
        n: int = 5
    ) -> Dict[str, Any]:
        """
        Train separate n-gram models for each intent.
        
        Args:
            training_data: {"intent_name": ["response1", "response2", ...]}
            n: N-gram size (3 = trigram, 4 = 4-gram)
            
        Returns:
            Training metrics
        """
        logger.info(f"Training {n}-gram models for {len(training_data)} intents with Kneser-Ney smoothing")
        
        self.n = n
        self.intent_models = {}
        metrics = {}
        
        for intent, responses in training_data.items():
            if not responses or len(responses) < 2:
                logger.warning(f"Insufficient data for intent '{intent}' ({len(responses)} responses), skipping")
                continue
                
            logger.info(f"Training model for intent '{intent}' with {len(responses)} responses")
            
            # Build n-gram model for this intent
            ngram_counts = defaultdict(Counter)
            vocab = set()
            
            for response in responses:
                tokens = self._tokenize(response)
                vocab.update(tokens)
                
                # Create n-grams
                for i in range(len(tokens) - n + 1):
                    context = tuple(tokens[i:i + n - 1])
                    next_word = tokens[i + n - 1]
                    ngram_counts[context][next_word] += 1
            
            # Apply Kneser-Ney smoothing to probabilities
            ngram_probs = self._apply_kneser_ney_smoothing(ngram_counts, vocab)
            
            # Store model
            self.intent_models[intent] = {
                "ngram_probs": ngram_probs,
                "vocab": list(vocab),
                "n": n,
                "num_responses": len(responses),
                "num_ngrams": len(ngram_probs),
                "seed_contexts": self._extract_seed_contexts(ngram_probs)
            }
            
            metrics[intent] = {
                "vocab_size": len(vocab),
                "num_ngrams": len(ngram_probs),
                "num_responses": len(responses)
            }
        
        logger.info(f"Training complete. Models for {len(self.intent_models)} intents")
        
        return {
            "status": "success",
            "num_intents": len(self.intent_models),
            "n": n,
            "metrics": metrics
        }
    
    def generate_response(
        self,
        intent: str,
        seed: Optional[str] = None,
        max_length: int = 50,
        temperature: float = 0.8,
        context_docs: Optional[List[str]] = None
    ) -> str:
        """
        Generate a response for a given intent using n-grams.
        
        Args:
            intent: The detected intent
            seed: Optional starting text
            max_length: Maximum words to generate
            temperature: Sampling temperature (0.5=conservative, 1.5=creative)
            context_docs: Relevant docs from knowledge base (for future use)
            
        Returns:
            Generated response
        """
        if intent not in self.intent_models:
            logger.warning(f"No generative model for intent '{intent}'")
            return f"Entendi sua pergunta sobre {intent}, mas ainda estou aprendendo a responder."
        
        model = self.intent_models[intent]
        ngram_probs = model["ngram_probs"]
        n = model["n"]
        
        # Initialize generation
        if seed:
            tokens = self._tokenize(seed)
            # Remove end token if present
            if tokens and tokens[-1] == "<END>":
                tokens = tokens[:-1]
        else:
            # Use random seed context from training data
            tokens = list(random.choice(model["seed_contexts"]))
        
        # Ensure enough context
        while len(tokens) < n - 1:
            tokens.insert(0, "<START>")
        
        # Generate tokens
        generated = tokens[:]
        attempts = 0
        max_attempts = max_length * 2
        
        while len(generated) < max_length + len(tokens) and attempts < max_attempts:
            attempts += 1
            context = tuple(generated[-(n-1):])
            
            # Get next word probabilities
            if context in ngram_probs:
                next_word = self._sample_word(ngram_probs[context], temperature)
            else:
                # Backoff to shorter context
                if len(context) > 1:
                    context = (context[-1],)
                    if context in ngram_probs:
                        next_word = self._sample_word(ngram_probs[context], temperature)
                    else:
                        break
                else:
                    break
            
            # Check for end
            if next_word == "<END>":
                break
            
            generated.append(next_word)
            
            # Natural sentence end
            if next_word in [".", "!", "?"] and len(generated) >= 10:
                break
        
        # Clean and format response
        response = self._format_response(generated)
        
        return response
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        text = text.lower().strip()
        
        # Preserve sentence-ending punctuation
        for punct in [".", "!", "?"]:
            text = text.replace(punct, f" {punct}")
        
        # Remove other punctuation
        for punct in [",", ":", ";", "(", ")", "[", "]", '"', "'"]:
            text = text.replace(punct, "")
        
        tokens = ["<START>"] + text.split() + ["<END>"]
        return [t for t in tokens if t]  # Remove empty strings
    
    def _sample_word(self, word_probs: Dict[str, float], temperature: float) -> str:
        """Sample next word with temperature"""
        words = list(word_probs.keys())
        probs = np.array(list(word_probs.values()))
        
        # Apply temperature
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
            probs = probs / probs.sum()
        
        return np.random.choice(words, p=probs)
    
    def _extract_seed_contexts(self, ngram_probs: Dict) -> List[tuple]:
        """Extract good starting contexts (those that begin with <START>)"""
        seed_contexts = [
            ctx for ctx in ngram_probs.keys()
            if ctx and ctx[0] == "<START>"
        ]
        
        if not seed_contexts:
            # Fallback: any context
            seed_contexts = list(ngram_probs.keys())[:5]
        
        return seed_contexts
    
    def _format_response(self, tokens: List[str]) -> str:
        """Format generated tokens into readable text with grammar post-processing"""
        # Remove special tokens
        tokens = [t for t in tokens if t not in ["<START>", "<END>"]]
        
        if not tokens:
            return "Desculpe, não consegui gerar uma resposta adequada."
        
        # Join tokens
        response = " ".join(tokens)
        
        # Fix spacing around punctuation
        for punct in [".", "!", "?", ",", ":", ";"]:
            response = response.replace(f" {punct}", punct)
        
        # Apply grammar post-processing
        response = self._grammar_postprocessing(response)
        
        # Capitalize first letter
        if response:
            response = response[0].upper() + response[1:]
        
        # Ensure ends with punctuation
        if response and response[-1] not in [".","!","?"]:
            response += "."
        
        return response
    
    def save_model(self, model_name: str, metadata: Optional[Dict] = None):
        """Save intent-conditioned models"""
        filepath = self.models_dir / f"{model_name}_intgen.joblib"
        
        save_data = {
            "intent_models": self.intent_models,
            "n": self.n,
            "metadata": metadata or {}
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Intent-conditioned generation model saved: {filepath}")
        logger.info(f"Intents covered: {list(self.intent_models.keys())}")
    
    def load_model(self, model_name: str):
        """Load intent-conditioned models"""
        # Try different possible filenames
        possible_paths = [
            self.models_dir / f"{model_name}_intgen.joblib",
            self.models_dir / f"{model_name}.joblib"
        ]
        
        filepath = None
        for path in possible_paths:
            if path.exists():
                filepath = path
                break
        
        if not filepath:
            raise FileNotFoundError(f"Model '{model_name}' not found in {self.models_dir}")
        
        save_data = joblib.load(filepath)
        
        self.intent_models = save_data["intent_models"]
        self.n = save_data["n"]
        
        logger.info(f"Loaded intent-conditioned generation model: {filepath}")
        logger.info(f"Available intents: {list(self.intent_models.keys())}")
    
    def has_model_for_intent(self, intent: str) -> bool:
        """Check if model exists for given intent"""
        return intent in self.intent_models
    
    def _apply_kneser_ney_smoothing(
        self, 
        ngram_counts: Dict[tuple, Counter], 
        vocab: set
    ) -> Dict[tuple, Dict[str, float]]:
        """Apply Kneser-Ney smoothing to n-gram counts
        
        Kneser-Ney smoothing improves probability estimates by:
        1. Discounting observed counts
        2. Redistributing probability mass to unseen events
        3. Using continuation probability for backoff
        """
        discount = self.smoothing_discount
        ngram_probs = {}
        
        # Calculate continuation counts for lower-order model
        continuation_counts = Counter()
        for context, word_counts in ngram_counts.items():
            for word in word_counts:
                continuation_counts[word] += 1  # How many different contexts word appears in
        
        total_continuations = sum(continuation_counts.values())
        
        for context, word_counts in ngram_counts.items():
            total_count = sum(word_counts.values())
            num_different_words = len(word_counts)  # N1+(context, •)
            
            probs = {}
            for word, count in word_counts.items():
                # Discounted probability
                discounted_prob = max(count - discount, 0) / total_count
                
                # Continuation probability (backoff)
                continuation_prob = continuation_counts[word] / total_continuations
                
                # Interpolation weight (lambda)
                lambda_weight = (discount * num_different_words) / total_count
                
                # Final probability: discounted + backoff
                probs[word] = discounted_prob + (lambda_weight * continuation_prob)
            
            # Normalize to ensure probabilities sum to 1
            total_prob = sum(probs.values())
            if total_prob > 0:
                probs = {word: prob / total_prob for word, prob in probs.items()}
            
            ngram_probs[context] = probs
        
        return ngram_probs
    
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


def get_intent_conditioned_generation_service() -> IntentConditionedGenerationService:
    """Get singleton instance"""
    return IntentConditionedGenerationService()
