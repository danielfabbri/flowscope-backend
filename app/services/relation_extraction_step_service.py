"""
Relation Extraction Step Service
Extracts relations between entities in text
"""
import pandas as pd
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from app.services.base_service import BasePipelineService
from app.core.logger import get_logger

logger = get_logger(__name__)


class RelationExtractionStepService(BasePipelineService):
    """
    Pipeline step that extracts relations between entities
    
    Input: DataFrame with text and entities
    Output: DataFrame with extracted relations
    """
    
    def __init__(self):
        # Relation patterns
        self.relation_patterns = self._build_relation_patterns()
        logger.info("RelationExtractionStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute relation extraction on DataFrame
        
        Args:
            data: DataFrame with text and entities
            config: Configuration with:
                - text_column: column with text
                - entities_column: column with entities (JSON)
                - relation_types: list of relation types to extract
                - model_type: extraction model (pattern_based, dependency)
                - output_relations_column: name for relations column
                - output_triples_column: name for subject-predicate-object triples
        
        Returns:
            DataFrame with extracted relations
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for relation extraction")
        
        text_column = config.get('text_column', 'message')
        entities_column = config.get('entities_column', 'entities')
        relation_types = config.get('relation_types', [
            'joga_em', 'é_técnico_de', 'venceu', 'perdeu_para', 
            'marcou_gol', 'pertence_a'
        ])
        model_type = config.get('model_type', 'pattern_based')
        output_relations = config.get('output_relations_column', 'relations')
        output_triples = config.get('output_triples_column', 'relation_triples')
        
        if text_column not in data.columns:
            raise ValueError(f"Text column '{text_column}' not found")
        
        logger.info(f"Extracting relations for {len(data)} texts")
        logger.info(f"Relation types: {relation_types}")
        
        # Process each text
        all_relations = []
        all_triples = []
        
        for idx, row in data.iterrows():
            text = str(row[text_column])
            
            # Get entities if available
            entities = {}
            if entities_column in row and pd.notna(row[entities_column]):
                try:
                    entities = json.loads(str(row[entities_column]))
                except json.JSONDecodeError:
                    pass
            
            # Extract relations
            relations, triples = self._extract_relations(
                text, entities, relation_types, model_type
            )
            
            all_relations.append(json.dumps(relations))
            all_triples.append(json.dumps(triples))
        
        # Add new columns to DataFrame
        result = data.copy()
        result[output_relations] = all_relations
        result[output_triples] = all_triples
        
        total_relations = sum(len(json.loads(r)) for r in all_relations)
        logger.info(f"Relation extraction completed. Extracted {total_relations} relations")
        return result
    
    def _extract_relations(self, text: str, entities: Dict, 
                          relation_types: List[str], model_type: str) -> Tuple[List, List]:
        """Extract relations from text"""
        relations = []
        triples = []
        
        if model_type == 'pattern_based':
            relations, triples = self._pattern_based_extraction(text, entities, relation_types)
        elif model_type == 'dependency':
            relations, triples = self._dependency_based_extraction(text, entities)
        else:
            relations, triples = self._pattern_based_extraction(text, entities, relation_types)
        
        return relations, triples
    
    def _pattern_based_extraction(self, text: str, entities: Dict,
                                  relation_types: List[str]) -> Tuple[List, List]:
        """Extract relations using patterns"""
        relations = []
        triples = []
        
        text_lower = text.lower()
        
        # Try each relation pattern
        for relation_type in relation_types:
            if relation_type in self.relation_patterns:
                patterns = self.relation_patterns[relation_type]
                
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        # Extract subject and object
                        groups = match.groups()
                        if len(groups) >= 2:
                            subject = groups[0].strip()
                            obj = groups[-1].strip()
                            
                            # Create relation
                            relation = {
                                'type': relation_type,
                                'subject': subject,
                                'object': obj,
                                'confidence': 0.8
                            }
                            relations.append(relation)
                            
                            # Create triple
                            triple = [subject, relation_type, obj]
                            triples.append(triple)
        
        return relations, triples
    
    def _dependency_based_extraction(self, text: str, entities: Dict) -> Tuple[List, List]:
        """Extract relations using dependency parsing (simplified)"""
        # This would use spaCy or similar for dependency parsing
        # For now, return empty
        return [], []
    
    def _build_relation_patterns(self) -> Dict[str, List[str]]:
        """Build relation extraction patterns"""
        return {
            'joga_em': [
                r'(\w+)\s+joga\s+(?:no|na|pelo|pela)\s+(\w+)',
                r'(\w+)\s+é\s+jogador\s+(?:do|da)\s+(\w+)',
                r'(\w+)\s+atua\s+(?:no|na)\s+(\w+)'
            ],
            'é_técnico_de': [
                r'(\w+)\s+é\s+(?:técnico|treinador)\s+(?:do|da)\s+(\w+)',
                r'(\w+)\s+treina\s+(?:o|a)\s+(\w+)',
                r'(\w+)\s+comanda\s+(?:o|a)\s+(\w+)'
            ],
            'venceu': [
                r'(\w+)\s+venceu\s+(?:o|a)\s+(\w+)',
                r'(\w+)\s+derrotou\s+(?:o|a)\s+(\w+)',
                r'(\w+)\s+ganhou\s+(?:do|da)\s+(\w+)',
                r'(\w+)\s+bateu\s+(?:o|a)\s+(\w+)'
            ],
            'perdeu_para': [
                r'(\w+)\s+perdeu\s+(?:para|pro)\s+(?:o|a)?\s*(\w+)',
                r'(\w+)\s+foi\s+derrotado\s+(?:pelo|pela)\s+(\w+)',
                r'(\w+)\s+caiu\s+(?:para|pro)\s+(?:o|a)?\s*(\w+)'
            ],
            'marcou_gol': [
                r'(\w+)\s+marcou\s+(?:gol|tento)',
                r'(\w+)\s+fez\s+o\s+gol',
                r'gol\s+(?:de|do)\s+(\w+)'
            ],
            'pertence_a': [
                r'(\w+)\s+pertence\s+(?:ao|à)\s+(\w+)',
                r'(\w+)\s+é\s+(?:do|da)\s+(\w+)',
                r'(\w+)\s+faz\s+parte\s+(?:do|da)\s+(\w+)'
            ],
            'é_capitão_de': [
                r'(\w+)\s+é\s+capitão\s+(?:do|da)\s+(\w+)',
                r'(\w+)\s+comanda\s+(?:o|a)\s+(\w+)\s+como\s+capitão'
            ],
            'compete_em': [
                r'(\w+)\s+compete\s+(?:no|na)\s+(\w+)',
                r'(\w+)\s+disputa\s+(?:o|a)\s+(\w+)',
                r'(\w+)\s+participa\s+(?:do|da)\s+(\w+)'
            ]
        }


# Singleton instance
_instance = None

def get_relation_extraction_step_service():
    global _instance
    if _instance is None:
        _instance = RelationExtractionStepService()
    return _instance
