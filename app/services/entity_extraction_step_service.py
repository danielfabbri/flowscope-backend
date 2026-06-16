"""
Entity Extraction Step Service
Extracts named entities from text for pipeline processing
"""
import pandas as pd
import json
from typing import Dict, Any, Optional, List
from app.services.base_service import BasePipelineService
from app.services.entity_extraction_service import get_entity_extraction_service
from app.core.logger import get_logger

logger = get_logger(__name__)


class EntityExtractionStepService(BasePipelineService):
    """
    Pipeline step that extracts named entities from text
    
    Input: DataFrame with text column
    Output: DataFrame with added entities column (JSON)
    """
    
    def __init__(self):
        self.entity_service = get_entity_extraction_service()
        logger.info("EntityExtractionStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute entity extraction on DataFrame
        
        Args:
            data: DataFrame with text column
            config: Configuration with:
                - text_column: column name with text to process
                - entity_types: list of entity types to extract (PERSON, ORG, GPE, DATE, PRODUCT)
                - output_column: name for entities output column
                - extract_keywords: whether to also extract keywords
                - num_keywords: number of keywords to extract
        
        Returns:
            DataFrame with added entities column
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for entity extraction")
        
        text_column = config.get('text_column', 'message')
        entity_types_str = config.get('entity_types', 'PERSON,ORG,GPE,DATE,PRODUCT')
        output_column = config.get('output_column', 'entities')
        extract_keywords = config.get('extract_keywords', False)
        num_keywords = int(config.get('num_keywords', 5))
        
        # Parse entity types
        if isinstance(entity_types_str, str):
            entity_types = [e.strip() for e in entity_types_str.split(',')]
        else:
            entity_types = entity_types_str
        
        if text_column not in data.columns:
            raise ValueError(f"Text column '{text_column}' not found in data. Available: {list(data.columns)}")
        
        logger.info(f"Extracting entities from {len(data)} rows, column '{text_column}'")
        logger.info(f"Entity types: {entity_types}")
        
        # Extract entities for each text
        all_entities = []
        
        for idx, row in data.iterrows():
            text = str(row[text_column])
            
            if not text or text.strip() == '':
                all_entities.append({})
                continue
            
            # Extract entities
            entities = self.entity_service.extract_entities(text)
            
            # Filter by requested entity types
            filtered_entities = {k: v for k, v in entities.items() if k in entity_types}
            
            # Optionally add keywords
            if extract_keywords:
                keywords = self.entity_service.extract_keywords(text, top_n=num_keywords)
                filtered_entities['keywords'] = keywords
            
            all_entities.append(filtered_entities)
        
        # Add new column to DataFrame
        result = data.copy()
        result[output_column] = [json.dumps(e) for e in all_entities]
        
        # Also add individual columns for each entity type
        for entity_type in entity_types:
            col_name = f'{output_column}_{entity_type.lower()}'
            result[col_name] = [
                ','.join(e.get(entity_type, [])) if isinstance(e.get(entity_type, []), list) else ''
                for e in all_entities
            ]
        
        # Log statistics
        total_entities = sum(len([v for vals in e.values() if isinstance(vals, list) for v in vals]) for e in all_entities)
        logger.info(f"Entity extraction complete. Total entities found: {total_entities}")
        
        return result


def get_entity_extraction_step_service() -> EntityExtractionStepService:
    """Get singleton instance"""
    if not hasattr(get_entity_extraction_step_service, '_instance'):
        get_entity_extraction_step_service._instance = EntityExtractionStepService()
    return get_entity_extraction_step_service._instance
