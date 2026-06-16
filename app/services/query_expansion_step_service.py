"""
Query Expansion Step Service
Expands queries with synonyms and related terms to improve search recall
"""
import pandas as pd
import json
from typing import Dict, Any, Optional, List, Set
from collections import defaultdict
from app.services.base_service import BasePipelineService
from app.core.logger import get_logger

logger = get_logger(__name__)


class QueryExpansionStepService(BasePipelineService):
    """
    Pipeline step that expands queries with related terms
    
    Input: DataFrame with query text
    Output: DataFrame with expanded query
    """
    
    def __init__(self):
        # Synonym dictionary (can be loaded from file)
        self.synonyms = self._build_synonym_dict()
        
        # Word associations (co-occurrence based)
        self.associations = defaultdict(list)
        
        logger.info("QueryExpansionStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute query expansion on DataFrame
        
        Args:
            data: DataFrame with query text
            config: Configuration with:
                - query_column: column with query text
                - expansion_methods: list of methods (synonyms, hypernyms, related_terms)
                - max_expansions: maximum number of expansion terms
                - use_word2vec: whether to use word embeddings (requires model)
                - custom_synonyms: custom synonym dictionary
                - output_expanded_column: name for expanded query column
                - output_terms_column: name for expansion terms column
        
        Returns:
            DataFrame with expanded queries
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for query expansion")
        
        query_column = config.get('query_column', 'message')
        methods = config.get('expansion_methods', ['synonyms', 'related_terms'])
        max_expansions = int(config.get('max_expansions', 5))
        use_word2vec = config.get('use_word2vec', False)
        custom_synonyms = config.get('custom_synonyms', {})
        output_expanded = config.get('output_expanded_column', 'expanded_query')
        output_terms = config.get('output_terms_column', 'expansion_terms')
        
        if query_column not in data.columns:
            raise ValueError(f"Query column '{query_column}' not found")
        
        # Update synonyms with custom ones
        if custom_synonyms:
            self.synonyms.update(custom_synonyms)
        
        logger.info(f"Expanding queries for {len(data)} rows")
        logger.info(f"Using methods: {methods}")
        
        # Process each query
        expanded_queries = []
        expansion_terms_list = []
        
        for idx, row in data.iterrows():
            query = str(row[query_column])
            
            # Expand query
            expanded, terms = self._expand_query(query, methods, max_expansions, use_word2vec)
            
            expanded_queries.append(expanded)
            expansion_terms_list.append(json.dumps(terms))
        
        # Add new columns to DataFrame
        result = data.copy()
        result[output_expanded] = expanded_queries
        result[output_terms] = expansion_terms_list
        
        logger.info(f"Query expansion completed")
        return result
    
    def _expand_query(self, query: str, methods: List[str], max_expansions: int,
                     use_word2vec: bool) -> tuple:
        """Expand query with related terms"""
        original_terms = query.lower().split()
        expansion_terms = set()
        
        for term in original_terms:
            # Skip very short terms
            if len(term) < 3:
                continue
            
            # Method 1: Synonyms
            if 'synonyms' in methods:
                synonyms = self._get_synonyms(term)
                expansion_terms.update(synonyms[:max_expansions])
            
            # Method 2: Hypernyms (more general terms)
            if 'hypernyms' in methods:
                hypernyms = self._get_hypernyms(term)
                expansion_terms.update(hypernyms[:max_expansions])
            
            # Method 3: Related terms (co-occurrence)
            if 'related_terms' in methods:
                related = self._get_related_terms(term)
                expansion_terms.update(related[:max_expansions])
            
            # Method 4: Word2Vec similar words
            if use_word2vec and 'word2vec' in methods:
                similar = self._get_similar_words_w2v(term)
                expansion_terms.update(similar[:max_expansions])
        
        # Limit total expansions
        expansion_terms = list(expansion_terms)[:max_expansions]
        
        # Build expanded query
        expanded_query = query + ' ' + ' '.join(expansion_terms)
        
        return expanded_query.strip(), expansion_terms
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for word"""
        return self.synonyms.get(word.lower(), [])
    
    def _get_hypernyms(self, word: str) -> List[str]:
        """Get more general terms (hypernyms)"""
        # Simplified hypernym mapping
        hypernyms = {
            'flamengo': ['time', 'clube', 'equipe'],
            'palmeiras': ['time', 'clube', 'equipe'],
            'pelé': ['jogador', 'atleta'],
            'gol': ['ponto', 'marcação'],
            'campeonato': ['competição', 'torneio'],
        }
        return hypernyms.get(word.lower(), [])
    
    def _get_related_terms(self, word: str) -> List[str]:
        """Get related terms based on co-occurrence"""
        # Use associations dictionary
        return self.associations.get(word.lower(), [])
    
    def _get_similar_words_w2v(self, word: str) -> List[str]:
        """Get similar words using Word2Vec (placeholder)"""
        # This would use a trained Word2Vec model
        # For now, return empty list
        return []
    
    def _build_synonym_dict(self) -> Dict[str, List[str]]:
        """Build initial synonym dictionary"""
        return {
            # Football/Soccer terms
            'jogo': ['partida', 'confronto', 'peleja'],
            'time': ['equipe', 'clube', 'seleção'],
            'gol': ['tento', 'marcação'],
            'jogador': ['atleta', 'craque', 'futebolista'],
            'técnico': ['treinador', 'coach'],
            'campeonato': ['torneio', 'competição', 'copa'],
            'vitória': ['triunfo', 'êxito'],
            'derrota': ['perda', 'revés'],
            'empate': ['igualdade', 'draw'],
            
            # General terms
            'quando': ['data', 'horário', 'dia'],
            'onde': ['local', 'lugar', 'localização'],
            'quem': ['qual', 'pessoa'],
            'como': ['modo', 'maneira', 'forma'],
            'quanto': ['valor', 'preço', 'quantia'],
            
            # Actions
            'ganhar': ['vencer', 'triunfar'],
            'perder': ['ser derrotado'],
            'jogar': ['disputar', 'competir'],
        }


# Singleton instance
_instance = None

def get_query_expansion_step_service():
    global _instance
    if _instance is None:
        _instance = QueryExpansionStepService()
    return _instance
