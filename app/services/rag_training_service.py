"""
RAG Training Service - Pipeline Integration

Cria e treina modelos RAG a partir de DataFrames no pipeline.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any
import pickle
import logging

from app.services.base_service import BasePipelineService
from app.services.rag_service import RAGService


class RAGTrainingService(BasePipelineService):
    """
    Pipeline step para treinar modelos RAG.
    
    Parâmetros esperados:
    - model_name: Nome do modelo a salvar
    - text_column: Nome da coluna com texto (modo documentos)
    - question_column: Nome da coluna com perguntas (modo Q&A)
    - answer_column: Nome da coluna com respostas (modo Q&A)
    - chunk_size: Tamanho dos chunks para texto (padrão: 200)
    """
    
    def __init__(self):
        super().__init__()
        # Use absolute path relative to backend directory
        backend_dir = Path(__file__).parent.parent.parent
        self.models_dir = backend_dir / "data" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Treina modelo RAG a partir do DataFrame.
        
        Args:
            df: DataFrame com dados de treinamento
            params: Parâmetros do step
                - model_name (required): Nome do modelo
                - text_column: Coluna com texto (modo documentos)
                - question_column: Coluna com perguntas (modo Q&A)
                - answer_column: Coluna com respostas (modo Q&A)
                - chunk_size: Tamanho dos chunks (padrão: 200)
        
        Returns:
            DataFrame com estatísticas do treinamento
        """
        model_name = params.get('model_name')
        if not model_name:
            raise ValueError("Parameter 'model_name' is required")
        
        text_column = params.get('text_column')
        question_column = params.get('question_column')
        answer_column = params.get('answer_column')
        chunk_size = params.get('chunk_size', 200)
        
        self.logger.info(f"🤖 Treinando modelo RAG: {model_name}")
        
        # Criar instância RAG
        rag = RAGService()
        
        # Detectar modo
        if question_column and answer_column:
            # Modo Q&A
            if question_column not in df.columns or answer_column not in df.columns:
                raise ValueError(f"Columns '{question_column}' and '{answer_column}' not found in DataFrame")
            
            self.logger.info(f"📊 Modo Q&A: {question_column} → {answer_column}")
            
            # Construir knowledge base
            for idx, row in df.iterrows():
                question = str(row[question_column])
                answer = str(row[answer_column])
                
                rag.knowledge_base.append({
                    'id': f"qa_{idx}",
                    'content': question,  # Indexar pela pergunta
                    'answer': answer,
                    'source': 'pipeline',
                    'type': 'qa_pair'
                })
            
            mode = 'qa'
            
        elif text_column:
            # Modo documentos
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in DataFrame")
            
            self.logger.info(f"📄 Modo Documentos: {text_column}")
            
            # Dividir em chunks se necessário
            for idx, row in df.iterrows():
                text = str(row[text_column])
                words = text.split()
                
                if len(words) > chunk_size:
                    # Dividir em chunks
                    for chunk_idx in range(0, len(words), chunk_size):
                        chunk = ' '.join(words[chunk_idx:chunk_idx + chunk_size])
                        rag.knowledge_base.append({
                            'id': f"doc_{idx}_chunk_{chunk_idx // chunk_size}",
                            'content': chunk.strip(),
                            'source': 'pipeline',
                            'type': 'text_chunk'
                        })
                else:
                    rag.knowledge_base.append({
                        'id': f"doc_{idx}",
                        'content': text,
                        'source': 'pipeline',
                        'type': 'document'
                    })
            
            mode = 'documents'
            
        else:
            # Auto-detect: usar primeira coluna
            text_column = df.columns[0]
            self.logger.info(f"🔍 Auto-detectado: usando coluna '{text_column}'")
            
            for idx, row in df.iterrows():
                text = str(row[text_column])
                rag.knowledge_base.append({
                    'id': f"doc_{idx}",
                    'content': text,
                    'source': 'pipeline',
                    'type': 'document'
                })
            
            mode = 'documents'
        
        # Construir índice TF-IDF
        rag._build_index()
        rag.current_knowledge_name = model_name
        
        # Salvar modelo (evitar duplicação de sufixo)
        if not model_name.endswith('_rag'):
            model_filename = f"{model_name}_rag.pkl"
        else:
            model_filename = f"{model_name}.pkl"
        
        model_path = self.models_dir / model_filename
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'knowledge_base': rag.knowledge_base,
                'vectorizer': rag.vectorizer,
                'tfidf_matrix': rag.tfidf_matrix,
                'model_name': model_name,
                'mode': mode
            }, f)
        
        stats = {
            "model_name": model_name,
            "model_path": str(model_path),
            "mode": mode,
            "entries": len(rag.knowledge_base),
            "vocabulary_size": len(rag.vectorizer.vocabulary_) if rag.vectorizer else 0
        }
        
        self.logger.info(f"✅ Modelo RAG salvo: {model_path} ({stats['entries']} entradas)")
        
        # Retornar estatísticas como DataFrame
        return pd.DataFrame([stats])


class RAGAnswerService(BasePipelineService):
    """
    Pipeline step para gerar respostas usando modelo RAG.
    
    Parâmetros esperados:
    - model_name: Nome do modelo RAG
    - questions: Lista de perguntas OU coluna no DataFrame
    - question_column: Nome da coluna com perguntas (opcional)
    - top_k: Número de contextos (padrão: 3)
    - min_similarity: Similaridade mínima (padrão: 0.1)
    """
    
    def __init__(self):
        super().__init__()
        # Use absolute path relative to backend directory
        backend_dir = Path(__file__).parent.parent.parent
        self.models_dir = backend_dir / "data" / "models"
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Gera respostas usando modelo RAG.
        
        Args:
            df: DataFrame (pode conter perguntas em coluna)
            params: Parâmetros do step
                - model_name (required): Nome do modelo
                - questions: Lista de perguntas fixas
                - question_column: Coluna com perguntas no DataFrame
                - top_k: Número de contextos (padrão: 3)
                - min_similarity: Similaridade mínima (padrão: 0.1)
        
        Returns:
            DataFrame com perguntas e respostas
        """
        model_name = params.get('model_name')
        if not model_name:
            raise ValueError("Parameter 'model_name' is required")
        
        # Carregar modelo (evitar duplicação de sufixo)
        if not model_name.endswith('_rag'):
            model_filename = f"{model_name}_rag.pkl"
        else:
            model_filename = f"{model_name}.pkl"
        
        model_path = self.models_dir / model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.logger.info(f"📂 Carregando modelo RAG: {model_name}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Reconstruir RAG
        rag = RAGService()
        rag.knowledge_base = model_data['knowledge_base']
        rag.vectorizer = model_data['vectorizer']
        rag.tfidf_matrix = model_data['tfidf_matrix']
        rag.current_knowledge_name = model_data['model_name']
        
        # Obter perguntas
        questions = []
        question_column = params.get('question_column')
        fixed_questions = params.get('questions', [])
        
        if question_column and question_column in df.columns:
            questions = df[question_column].tolist()
        elif fixed_questions:
            questions = fixed_questions
        else:
            raise ValueError("Provide 'question_column' or 'questions' parameter")
        
        top_k = params.get('top_k', 3)
        min_similarity = params.get('min_similarity', 0.1)
        
        self.logger.info(f"💬 Gerando {len(questions)} respostas")
        
        # Gerar respostas
        results = []
        for question in questions:
            result = rag.answer(
                question=str(question),
                top_k=top_k,
                min_similarity=min_similarity
            )
            
            results.append({
                'question': question,
                'answer': result['answer'],
                'confidence': result['confidence'],
                'contexts_found': result['retrieved_docs']
            })
        
        return pd.DataFrame(results)
