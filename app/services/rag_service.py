"""
RAG (Retrieval-Augmented Generation) Service

Carrega conhecimento de texto ou CSV e responde perguntas buscando
contexto relevante.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import re


class RAGService:
    """
    Sistema RAG que responde perguntas baseado em conhecimento carregado.
    
    Suporta 2 formatos:
    1. TXT: Texto corrido (divide em chunks)
    2. CSV: Perguntas e respostas ou documentos estruturados
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.knowledge_base: List[Dict[str, str]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.current_knowledge_name: Optional[str] = None
        
    def load_knowledge_from_text(
        self, 
        file_path: str, 
        chunk_size: int = 200,
        name: str = None
    ) -> Dict[str, Any]:
        """
        Carrega conhecimento de arquivo TXT.
        Divide em chunks para busca eficiente.
        
        Args:
            file_path: Caminho para arquivo .txt
            chunk_size: Tamanho dos chunks em palavras
            name: Nome do conhecimento
            
        Returns:
            Estatísticas do carregamento
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
        self.logger.info(f"📄 Carregando conhecimento de texto: {path.name}")
        
        # Ler texto
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Dividir em chunks
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append({
                'id': f"chunk_{i // chunk_size}",
                'content': chunk.strip(),
                'source': path.name,
                'type': 'text_chunk'
            })
        
        self.knowledge_base = chunks
        self.current_knowledge_name = name or path.stem
        
        # Indexar com TF-IDF
        self._build_index()
        
        stats = {
            "name": self.current_knowledge_name,
            "source_file": path.name,
            "chunks": len(chunks),
            "total_words": len(words),
            "avg_chunk_size": len(words) // len(chunks) if chunks else 0
        }
        
        self.logger.info(f"✅ Conhecimento carregado: {len(chunks)} chunks")
        return stats
    
    def load_knowledge_from_csv(
        self, 
        file_path: str,
        text_column: str = None,
        question_column: str = None,
        answer_column: str = None,
        name: str = None
    ) -> Dict[str, Any]:
        """
        Carrega conhecimento de CSV.
        
        Suporta 2 modos:
        1. Documentos: CSV com coluna de texto (text_column)
        2. Q&A: CSV com colunas pergunta/resposta (question_column/answer_column)
        
        Args:
            file_path: Caminho para arquivo .csv
            text_column: Coluna com documentos (modo documento)
            question_column: Coluna com perguntas (modo Q&A)
            answer_column: Coluna com respostas (modo Q&A)
            name: Nome do conhecimento
            
        Returns:
            Estatísticas do carregamento
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
        self.logger.info(f"📊 Carregando conhecimento de CSV: {path.name}")
        
        df = pd.read_csv(path, encoding='utf-8')
        
        # Detectar modo automaticamente se não especificado
        if not text_column and not (question_column and answer_column):
            # Tentar detectar
            if 'question' in df.columns and 'answer' in df.columns:
                question_column = 'question'
                answer_column = 'answer'
                mode = 'qa'
            elif 'pergunta' in df.columns and 'resposta' in df.columns:
                question_column = 'pergunta'
                answer_column = 'resposta'
                mode = 'qa'
            elif len(df.columns) == 1:
                text_column = df.columns[0]
                mode = 'documents'
            else:
                # Usar primeira coluna como texto
                text_column = df.columns[0]
                mode = 'documents'
        elif question_column and answer_column:
            mode = 'qa'
        else:
            mode = 'documents'
        
        # Carregar baseado no modo
        if mode == 'qa':
            self.knowledge_base = []
            for idx, row in df.iterrows():
                question = str(row[question_column])
                answer = str(row[answer_column])
                
                self.knowledge_base.append({
                    'id': f"qa_{idx}",
                    'content': question,  # Indexar pela pergunta
                    'answer': answer,     # Resposta associada
                    'source': path.name,
                    'type': 'qa_pair'
                })
        else:
            self.knowledge_base = []
            for idx, row in df.iterrows():
                text = str(row[text_column])
                
                self.knowledge_base.append({
                    'id': f"doc_{idx}",
                    'content': text,
                    'source': path.name,
                    'type': 'document'
                })
        
        self.current_knowledge_name = name or path.stem
        
        # Indexar com TF-IDF
        self._build_index()
        
        stats = {
            "name": self.current_knowledge_name,
            "source_file": path.name,
            "mode": mode,
            "entries": len(self.knowledge_base),
            "columns": list(df.columns)
        }
        
        self.logger.info(f"✅ Conhecimento carregado: {len(self.knowledge_base)} entradas (modo: {mode})")
        return stats
    
    def _build_index(self):
        """Constrói índice TF-IDF para busca eficiente."""
        if not self.knowledge_base:
            return
        
        # Extrair textos
        texts = [item['content'] for item in self.knowledge_base]
        
        # Criar vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,  # Manter stop words em português
            ngram_range=(1, 2),  # Unigrams e bigrams
            max_features=5000
        )
        
        # Criar matriz TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        self.logger.info(f"🔍 Índice construído: {self.tfidf_matrix.shape[0]} documentos, "
                        f"{self.tfidf_matrix.shape[1]} features")
    
    def search(
        self, 
        query: str, 
        top_k: int = 3,
        min_similarity: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Busca contexto relevante para a query.
        
        Args:
            query: Pergunta/query do usuário
            top_k: Número de resultados
            min_similarity: Similaridade mínima
            
        Returns:
            Lista de documentos relevantes com scores
        """
        if not self.vectorizer or not self.knowledge_base:
            raise ValueError("Nenhum conhecimento carregado. Use load_knowledge_* primeiro.")
        
        # Vetorizar query
        query_vector = self.vectorizer.transform([query.lower()])
        
        # Calcular similaridade
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Pegar top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_similarity:
                result = self.knowledge_base[idx].copy()
                result['similarity_score'] = round(score, 4)
                results.append(result)
        
        self.logger.info(f"🔍 Busca: '{query}' → {len(results)} resultados")
        return results
    
    def answer(
        self, 
        question: str,
        top_k: int = 3,
        min_similarity: float = 0.1
    ) -> Dict[str, Any]:
        """
        Responde pergunta usando RAG.
        
        Args:
            question: Pergunta do usuário
            top_k: Número de contextos a considerar
            min_similarity: Similaridade mínima
            
        Returns:
            Resposta com contexto usado
        """
        if not self.knowledge_base:
            return {
                "answer": "❌ Nenhum conhecimento carregado ainda. Carregue um arquivo TXT ou CSV primeiro.",
                "contexts": [],
                "confidence": 0.0
            }
        
        # Buscar contextos relevantes
        contexts = self.search(question, top_k=top_k, min_similarity=min_similarity)
        
        if not contexts:
            return {
                "answer": "❓ Desculpe, não encontrei informação relevante sobre isso na base de conhecimento.",
                "contexts": [],
                "confidence": 0.0
            }
        
        # Gerar resposta baseada no tipo de conhecimento
        best_match = contexts[0]
        
        if best_match['type'] == 'qa_pair':
            # Modo Q&A: retornar resposta direta
            answer = best_match['answer']
            confidence = best_match['similarity_score']
        else:
            # Modo documento/chunk: extrair trecho relevante
            # Para simplificar, retornar o chunk mais relevante
            answer = best_match['content']
            confidence = best_match['similarity_score']
            
            # Tentar extrair sentença mais relevante
            sentences = re.split(r'[.!?]+', answer)
            if len(sentences) > 1:
                # Pegar primeira sentença completa
                answer = sentences[0].strip() + '.'
        
        return {
            "answer": answer,
            "contexts": contexts,
            "confidence": confidence,
            "knowledge_name": self.current_knowledge_name,
            "retrieved_docs": len(contexts)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do conhecimento carregado."""
        if not self.knowledge_base:
            return {
                "loaded": False,
                "message": "Nenhum conhecimento carregado"
            }
        
        return {
            "loaded": True,
            "name": self.current_knowledge_name,
            "total_entries": len(self.knowledge_base),
            "types": {
                item['type']: sum(1 for i in self.knowledge_base if i['type'] == item['type'])
                for item in self.knowledge_base
            },
            "sources": list(set(item['source'] for item in self.knowledge_base))
        }


# Instância global
rag_service = RAGService()
