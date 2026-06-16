"""
Semantic Search Step Service
Performs semantic similarity search for pipeline processing
"""
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from app.services.base_service import BasePipelineService
from app.services.semantic_search_service import get_semantic_search_service
from app.core.logger import get_logger

logger = get_logger(__name__)


class SemanticSearchStepService(BasePipelineService):
    """
    Pipeline step that performs semantic search on a knowledge base
    
    Input: DataFrame with query column
    Output: DataFrame with added search results column
    """
    
    def __init__(self):
        self.search_service = get_semantic_search_service()
        self.knowledge_base_indexed = False
        # Use absolute path relative to backend directory
        backend_dir = Path(__file__).parent.parent.parent
        self.models_dir = backend_dir / "data" / "models"
        logger.info("SemanticSearchStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute semantic search on DataFrame
        
        Args:
            data: DataFrame with query column
            config: Configuration with:
                - query_column: column name with search queries
                - index_name: name of saved index to load (optional)
                - knowledge_base_path: path to knowledge base file (CSV/JSON, optional)
                - text_field: field name in knowledge base with text content
                - top_k: number of results to return
                - min_score: minimum similarity score (0-1)
                - output_column: name for search results output column
        
        Returns:
            DataFrame with added search results column
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for semantic search")
        
        query_column = config.get('query_column', 'message')
        index_name = config.get('index_name', None)
        kb_path = config.get('knowledge_base_path', '')
        text_field = config.get('text_field', 'text')
        top_k = int(config.get('top_k', 3))
        min_score = float(config.get('min_score', 0.3))
        output_column = config.get('output_column', 'search_results')
        
        if query_column not in data.columns:
            raise ValueError(f"Query column '{query_column}' not found in data. Available: {list(data.columns)}")
        
        # Check if semantic search is available
        if not self.search_service.is_available():
            logger.warning("Semantic search service not available (sentence-transformers not installed)")
            result = data.copy()
            result[output_column] = [json.dumps([]) for _ in range(len(data))]
            return result
        
        # Load index by name if specified
        if index_name and not self.knowledge_base_indexed:
            logger.info(f"Loading semantic search index: {index_name}")
            try:
                index_path = self.models_dir / f"{index_name}.npz"
                self.search_service.load_index(str(index_path))
                self.knowledge_base_indexed = True
                logger.info(f"Index '{index_name}' loaded successfully from '{index_path}'")
            except Exception as e:
                logger.error(f"Failed to load index '{index_name}': {e}")
                raise ValueError(f"Could not load semantic index '{index_name}': {e}")
        
        # Load and index knowledge base from file if specified
        elif kb_path and not self.knowledge_base_indexed:
            if not os.path.exists(kb_path):
                raise ValueError(f"Knowledge base file not found: {kb_path}")
            
            logger.info(f"Loading knowledge base from: {kb_path}")
            
            # Load knowledge base based on file extension
            if kb_path.endswith('.csv'):
                kb_data = pd.read_csv(kb_path)
            elif kb_path.endswith('.json'):
                kb_data = pd.read_json(kb_path)
            else:
                raise ValueError(f"Unsupported knowledge base format: {kb_path}")
            
            # Convert to list of dicts
            kb_documents = kb_data.to_dict('records')
            
            # Index documents
            logger.info(f"Indexing {len(kb_documents)} documents")
            self.search_service.index_documents(kb_documents, text_field=text_field)
            self.knowledge_base_indexed = True
        
        if not self.knowledge_base_indexed:
            logger.warning("No knowledge base indexed. Returning empty results.")
            result = data.copy()
            result[output_column] = [json.dumps([]) for _ in range(len(data))]
            return result
        
        logger.info(f"Searching {len(data)} queries from column '{query_column}'")
        
        # Perform search for each query
        all_results = []
        
        for idx, row in data.iterrows():
            query = str(row[query_column])
            
            if not query or query.strip() == '':
                all_results.append([])
                continue
            
            # Search
            search_results = self.search_service.search(
                query=query,
                top_k=top_k,
                min_score=min_score
            )
            
            all_results.append(search_results)
        
        # Add new column to DataFrame
        result = data.copy()
        result[output_column] = [json.dumps(r) for r in all_results]
        
        # Log statistics
        total_results = sum(len(r) for r in all_results)
        avg_results = total_results / len(all_results) if all_results else 0
        logger.info(f"Semantic search complete. Average results per query: {avg_results:.2f}")
        
        return result


def get_semantic_search_step_service() -> SemanticSearchStepService:
    """Get singleton instance"""
    if not hasattr(get_semantic_search_step_service, '_instance'):
        get_semantic_search_step_service._instance = SemanticSearchStepService()
    return get_semantic_search_step_service._instance
