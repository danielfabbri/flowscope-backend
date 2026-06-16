"""
Semantic Search Service

Provides semantic similarity search using sentence embeddings.
Uses lightweight models that run locally without external API calls.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import json
from ..core.logger import get_logger

logger = get_logger(__name__)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")


class SemanticSearchService:
    """Service for semantic similarity search using embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic search service.
        
        Args:
            model_name: Name of the sentence-transformers model to use
                       Default: all-MiniLM-L6-v2 (lightweight, 80MB, fast)
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading semantic model: {model_name}")
                self.model = SentenceTransformer(model_name)
                logger.info("Semantic model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading semantic model: {e}")
        else:
            logger.warning("Semantic search unavailable - sentence-transformers not installed")
    
    def is_available(self) -> bool:
        """Check if semantic search is available"""
        return self.model is not None
    
    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text"
    ) -> Dict[str, Any]:
        """
        Index documents for semantic search.
        
        Args:
            documents: List of documents to index
            text_field: Field containing the text to embed
            
        Returns:
            Indexing status
        """
        if not self.is_available():
            return {
                "status": "error",
                "message": "Semantic search not available"
            }
        
        try:
            logger.info(f"Indexing {len(documents)} documents")
            
            # Store documents
            self.documents = documents
            
            # Extract texts
            texts = [doc.get(text_field, "") for doc in documents]
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            self.embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            
            logger.info(f"Indexed {len(documents)} documents")
            logger.info(f"Embedding shape: {self.embeddings.shape}")
            
            return {
                "status": "success",
                "num_documents": len(documents),
                "embedding_dim": self.embeddings.shape[1]
            }
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return {"status": "error", "message": str(e)}
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for documents semantically similar to the query.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of matching documents with scores
        """
        if not self.is_available():
            return []
        
        if self.embeddings is None or len(self.documents) == 0:
            logger.warning("No documents indexed yet")
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True
            )[0]
            
            # Calculate cosine similarities
            similarities = self._cosine_similarity(query_embedding, self.embeddings)
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Build results
            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score >= min_score:
                    result = self.documents[idx].copy()
                    result["score"] = score
                    result["rank"] = len(results) + 1
                    results.append(result)
            
            logger.debug(f"Found {len(results)} results for query: '{query}'")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not self.is_available():
            return 0.0
        
        try:
            embeddings = self.model.encode([text1, text2], convert_to_numpy=True)
            similarity = self._cosine_similarity(embeddings[0], embeddings[1].reshape(1, -1))[0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for multiple queries at once.
        
        Args:
            queries: List of query texts
            top_k: Number of results per query
            
        Returns:
            List of result lists
        """
        return [self.search(query, top_k) for query in queries]
    
    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between vectors"""
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-9)
        vec2_norm = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-9)
        
        # Compute dot product
        similarity = np.dot(vec2_norm, vec1_norm)
        
        return similarity
    
    def save_index(self, file_path: str) -> Dict[str, Any]:
        """
        Save the document index to disk.
        
        Args:
            file_path: Path to save the index
            
        Returns:
            Save status
        """
        if self.embeddings is None:
            return {"status": "error", "message": "No index to save"}
        
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save as numpy format
            np.savez_compressed(
                file_path,
                embeddings=self.embeddings,
                documents=np.array(self.documents, dtype=object)
            )
            
            # Also save metadata
            metadata = {
                "model_name": self.model_name,
                "num_documents": len(self.documents),
                "embedding_dim": self.embeddings.shape[1]
            }
            
            metadata_path = file_path.replace(".npz", "_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Index saved to {file_path}")
            
            return {
                "status": "success",
                "file_path": file_path,
                "metadata_path": metadata_path
            }
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return {"status": "error", "message": str(e)}
    
    def load_index(self, file_path: str) -> Dict[str, Any]:
        """
        Load a document index from disk.
        
        Args:
            file_path: Path to the saved index
            
        Returns:
            Load status
        """
        try:
            # Load data
            data = np.load(file_path, allow_pickle=True)
            self.embeddings = data["embeddings"]
            self.documents = data["documents"].tolist()
            
            logger.info(f"Index loaded from {file_path}")
            logger.info(f"Loaded {len(self.documents)} documents")
            
            return {
                "status": "success",
                "num_documents": len(self.documents),
                "embedding_dim": self.embeddings.shape[1]
            }
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the current index"""
        if not self.is_available():
            return {
                "status": "unavailable",
                "message": "sentence-transformers not installed"
            }
        
        return {
            "status": "available",
            "model_name": self.model_name,
            "num_documents": len(self.documents),
            "indexed": self.embeddings is not None,
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else None
        }


# Singleton instance
_semantic_search_instance = None


def get_semantic_search_service(model_name: str = "all-MiniLM-L6-v2") -> SemanticSearchService:
    """Get or create the singleton semantic search service"""
    global _semantic_search_instance
    if _semantic_search_instance is None:
        _semantic_search_instance = SemanticSearchService(model_name)
    return _semantic_search_instance
