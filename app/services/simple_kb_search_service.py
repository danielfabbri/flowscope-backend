"""
Simple Knowledge Base Search Service using TF-IDF (no external dependencies)
"""
from typing import Dict, Any
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

class SimpleKnowledgeBaseSearchService:
    """Simple KB search using TF-IDF similarity"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.kb_data: Dict[str, Any] = {}  # {kb_name: {vectorizer, matrix, df}}
        # FIXED: Use correct absolute path
        self.data_dir = Path("c:/dev/flowscope/data/pipeline_data")
    
    def load_kb(self, kb_name: str):
        """Load and index a knowledge base"""
        if kb_name in self.kb_data:
            return  # Already loaded
        
        # Try to find KB file
        kb_file = self.data_dir / f"{kb_name}.csv"
        if not kb_file.exists():
            kb_file = self.data_dir / f"{kb_name}_massive.csv"
        
        if not kb_file.exists():
            self.logger.error(f"KB file not found: {kb_name}")
            return
        
        # Load CSV
        df = pd.read_csv(kb_file, encoding='utf-8-sig')
        
        # Build TF-IDF index on responses
        response_col = 'resposta' if 'resposta' in df.columns else 'answer'
        question_col = 'pergunta' if 'pergunta' in df.columns else 'question'
        
        if response_col not in df.columns:
            self.logger.error(f"No response column found in {kb_name}")
            return
        
        # Create combined text for indexing (question + response)
        texts = []
        for _, row in df.iterrows():
            text = str(row[response_col])
            if question_col in df.columns and pd.notna(row[question_col]):
                text = str(row[question_col]) + " " + text
            texts.append(text.lower())
        
        # Build TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        self.kb_data[kb_name] = {
            'vectorizer': vectorizer,
            'matrix': tfidf_matrix,
            'df': df,
            'response_col': response_col,
            'question_col': question_col
        }
        
        self.logger.info(f"✅ Loaded KB: {kb_name} ({len(df)} entries)")
    
    def search(self, kb_name: str, query: str, top_k: int = 5) -> list:
        """Search KB and return top matches"""
        # Load KB if not loaded
        if kb_name not in self.kb_data:
            self.load_kb(kb_name)
        
        if kb_name not in self.kb_data:
            return []
        
        kb = self.kb_data[kb_name]
        
        # Vectorize query
        query_vector = kb['vectorizer'].transform([query.lower()])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, kb['matrix'])[0]
        
        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > 0.1:  # Minimum threshold
                row = kb['df'].iloc[idx]
                results.append({
                    'resposta': str(row[kb['response_col']]),
                    'answer': str(row[kb['response_col']]),  # Alias
                    'pergunta': str(row.get(kb['question_col'], '')) if kb['question_col'] in kb['df'].columns else '',
                    'score': float(score)
                })
        
        return results
    
    def execute(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Execute search (compatible with StepService interface)"""
        kb_name = config.get('kb_name', 'knowledge_base')
        query_column = config.get('query_column', 'message')
        top_k = config.get('top_k', 5)
        
        results_list = []
        
        for _, row in data.iterrows():
            query = str(row[query_column])
            search_results = self.search(kb_name, query, top_k)
            results_list.append(search_results)
        
        result_df = data.copy()
        result_df['search_results'] = results_list
        
        return result_df


# Global instance
simple_kb_search_service = SimpleKnowledgeBaseSearchService()
