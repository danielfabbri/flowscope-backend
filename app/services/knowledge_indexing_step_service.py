"""
Knowledge Indexing Step Service

Pipeline step for indexing a knowledge base for semantic search.
"""

from typing import Dict, Any
from pathlib import Path
import pandas as pd
from .base_service import BasePipelineService
from .semantic_search_service import get_semantic_search_service
from ..core.logger import get_logger

logger = get_logger(__name__)


class KnowledgeIndexingStepService(BasePipelineService):
    """Pipeline step for indexing knowledge base documents"""
    
    def __init__(self):
        super().__init__()
        self.semantic_service = get_semantic_search_service()
        # Use absolute path relative to backend directory
        backend_dir = Path(__file__).parent.parent.parent
        self.models_dir = backend_dir / "data" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def execute(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Index documents for semantic search.
        
        Expected config:
        - text_column: Column containing document text
        - id_column: Column for document ID (optional)
        - metadata_columns: List of columns to include as metadata (optional)
        - index_name: Name for this knowledge base index
        
        Expected DataFrame columns:
        - text_column: Document texts to index
        - id_column (optional): Document IDs
        - metadata_columns (optional): Additional metadata
        
        Returns:
            DataFrame with indexing results
        """
        logger.info("Starting knowledge base indexing step")
        
        # Get configuration
        text_column = config.get('text_column', 'text')
        id_column = config.get('id_column', 'id')
        metadata_columns = config.get('metadata_columns', [])
        index_name = config.get('index_name', 'knowledge_base')
        
        # Validate text column exists
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
            
        # Prepare documents for indexing
        documents = []
        for idx, row in df.iterrows():
            doc = {
                "text": row[text_column],
                "id": row[id_column] if id_column in df.columns else idx
            }
            
            # Add metadata columns
            for col in metadata_columns:
                if col in df.columns:
                    doc[col] = row[col]
                    
            documents.append(doc)
        
        logger.info(f"Indexing {len(documents)} documents for '{index_name}'")
        
        # Index the documents
        result = self.semantic_service.index_documents(
            documents=documents,
            text_field="text"
        )
        
        if result.get("status") == "error":
            logger.error(f"Indexing failed: {result.get('message')}")
            raise RuntimeError(f"Knowledge indexing failed: {result.get('message')}")
        
        # Save the index with full path
        index_path = self.models_dir / f"{index_name}.npz"
        self.semantic_service.save_index(str(index_path))
        logger.info(f"Knowledge base '{index_name}' indexed and saved to '{index_path}'")
        
        # Add result metadata to DataFrame
        result_df = df.copy()
        result_df.attrs['index_name'] = index_name
        result_df.attrs['index_path'] = str(index_path)
        result_df.attrs['num_documents'] = len(documents)
        result_df.attrs['indexing_status'] = result.get("status")
        
        logger.info(f"Knowledge indexing complete. {len(documents)} documents indexed")
        
        return result_df
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate step configuration"""
        required = ['text_column', 'index_name']
        return all(key in config for key in required)


# Singleton instance
_knowledge_indexing_step_service = None


def get_knowledge_indexing_step_service() -> KnowledgeIndexingStepService:
    """Get singleton instance of knowledge indexing step service"""
    global _knowledge_indexing_step_service
    if _knowledge_indexing_step_service is None:
        _knowledge_indexing_step_service = KnowledgeIndexingStepService()
    return _knowledge_indexing_step_service
