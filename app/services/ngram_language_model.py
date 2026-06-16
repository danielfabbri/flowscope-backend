"""
N-gram Language Model for Text Generation.

This implements a simple statistical language model that learns from text
and can generate new text by predicting the next word based on context.

Educational implementation to understand the fundamentals of language modeling.
"""
import random
import logging
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path


class NgramLanguageModel:
    """
    Statistical language model using n-grams.
    
    How it works:
    1. Training: Count all n-gram sequences in corpus
    2. Generation: Given context, sample next word from learned probabilities
    
    Example (trigram, n=3):
        Corpus: "I love this product. I love the quality."
        
        N-grams learned:
        ("I", "love") -> {"this": 1, "the": 1}  # P(this|I,love) = 0.5
        ("love", "this") -> {"product": 1}      # P(product|love,this) = 1.0
        
        Generation from "I love":
        - Sample from {"this": 0.5, "the": 0.5}
        - If sampled "this", next context is ("love", "this")
        - Continue until max_length or end token
    """
    
    def __init__(self, n: int = 3):
        """
        Initialize n-gram model.
        
        Args:
            n: Size of n-gram (3 = trigram, considers 2 previous words)
        """
        self.n = n
        self.ngrams: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.vocabulary = set()
        self.start_tokens = []
        self.logger = logging.getLogger(__name__)
    
    def train(self, corpus: List[str]) -> Dict[str, any]:
        """
        Train the model on a corpus of text.
        
        Args:
            corpus: List of text documents
            
        Returns:
            Training statistics
        """
        self.logger.info(f"🎓 Training {self.n}-gram language model...")
        
        total_ngrams = 0
        total_tokens = 0
        
        for text in corpus:
            # Tokenize (simple word-level)
            tokens = self._tokenize(text)
            total_tokens += len(tokens)
            
            # Add start token
            tokens = ['<START>'] * (self.n - 1) + tokens + ['<END>']
            
            # Extract n-grams
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                next_word = tokens[i + self.n - 1]
                
                self.ngrams[context][next_word] += 1
                self.vocabulary.add(next_word)
                total_ngrams += 1
                
                # Store possible start contexts
                if context[0] == '<START>':
                    self.start_tokens.append(context)
        
        stats = {
            "n": self.n,
            "documents": len(corpus),
            "total_tokens": total_tokens,
            "unique_ngrams": len(self.ngrams),
            "vocabulary_size": len(self.vocabulary),
            "total_ngram_instances": total_ngrams
        }
        
        self.logger.info(f"✅ Model trained: {stats['unique_ngrams']} unique n-grams, "
                        f"vocab size: {stats['vocabulary_size']}")
        
        return stats
    
    def generate(
        self, 
        prompt: str = "", 
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Generate text given a prompt.
        
        Args:
            prompt: Starting text (empty = random start)
            max_length: Maximum number of words to generate
            temperature: Controls randomness (0.0 = greedy, 2.0 = very random)
            top_k: Only sample from top-k most likely words (None = all words)
            
        Returns:
            Tuple of (generated_text, generation_steps)
            where generation_steps shows the decision process
        """
        self.logger.info(f"🎨 Generating text from prompt: '{prompt}'")
        
        # Initialize context
        if prompt:
            tokens = self._tokenize(prompt)
            # Pad if needed
            if len(tokens) < self.n - 1:
                tokens = ['<START>'] * ((self.n - 1) - len(tokens)) + tokens
        else:
            # Random start
            tokens = list(random.choice(self.start_tokens)) if self.start_tokens else ['<START>'] * (self.n - 1)
        
        generated_tokens = list(tokens)
        generation_steps = []
        
        for step in range(max_length):
            # Get context (last n-1 tokens)
            context = tuple(generated_tokens[-(self.n - 1):])
            
            # Get possible next words
            if context not in self.ngrams:
                self.logger.warning(f"Context {context} not found in model, stopping generation")
                break
            
            next_word_counts = self.ngrams[context]
            
            # Sample next word
            next_word, probs = self._sample_next_word(
                next_word_counts, 
                temperature=temperature,
                top_k=top_k
            )
            
            # Record decision
            generation_steps.append({
                "step": step + 1,
                "context": " ".join(context),
                "candidates": probs[:5],  # Top 5 for display
                "chosen": next_word,
                "temperature": temperature
            })
            
            # Stop if end token
            if next_word == '<END>':
                break
            
            generated_tokens.append(next_word)
        
        # Remove special tokens for output
        output_tokens = [t for t in generated_tokens if t not in ['<START>', '<END>']]
        generated_text = " ".join(output_tokens)
        
        self.logger.info(f"✅ Generated {len(output_tokens)} tokens")
        
        return generated_text, generation_steps
    
    def _sample_next_word(
        self, 
        word_counts: Counter, 
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Sample next word from probability distribution.
        
        Args:
            word_counts: Counter of word frequencies
            temperature: Higher = more random, lower = more deterministic
            top_k: Only consider top-k words
            
        Returns:
            Tuple of (sampled_word, probability_distribution)
        """
        # Convert counts to probabilities
        total = sum(word_counts.values())
        words = list(word_counts.keys())
        probs = [word_counts[w] / total for w in words]
        
        # Apply temperature
        if temperature != 1.0:
            probs = [p ** (1 / temperature) for p in probs]
            # Re-normalize
            prob_sum = sum(probs)
            probs = [p / prob_sum for p in probs]
        
        # Apply top-k filtering
        if top_k is not None and len(words) > top_k:
            # Sort by probability
            sorted_pairs = sorted(zip(probs, words), reverse=True)
            probs, words = zip(*sorted_pairs[:top_k])
            # Re-normalize
            prob_sum = sum(probs)
            probs = [p / prob_sum for p in probs]
        
        # Sample
        sampled_word = random.choices(words, weights=probs, k=1)[0]
        
        # Prepare probability distribution for logging
        prob_dist = [
            {"word": w, "probability": round(p, 4)}
            for w, p in sorted(zip(words, probs), key=lambda x: x[1], reverse=True)
        ]
        
        return sampled_word, prob_dist
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple word-level tokenization."""
        # Basic tokenization: lowercase, split on whitespace
        return text.lower().split()
    
    def save(self, path: Path):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'n': self.n,
                'ngrams': dict(self.ngrams),
                'vocabulary': self.vocabulary,
                'start_tokens': self.start_tokens
            }, f)
        self.logger.info(f"💾 Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'NgramLanguageModel':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(n=data['n'])
        model.ngrams = defaultdict(Counter, data['ngrams'])
        model.vocabulary = data['vocabulary']
        model.start_tokens = data['start_tokens']
        
        logging.getLogger(__name__).info(f"📂 Model loaded from {path}")
        return model
