from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.schemas.pipeline import ChatQuery, ChatResponse
from app.services.chat_service import chat_service
from app.services.rag_service import rag_service
from app.services.conversational_model_service import ConversationalModelService
from app.core.logger import logger

router = APIRouter(prefix="/chat", tags=["chat"])

# Instanciar serviço conversacional
conversational_service = ConversationalModelService()


# ===== RAG Schemas =====

class RAGLoadTextRequest(BaseModel):
    file_path: str
    chunk_size: int = 200
    name: Optional[str] = None

class RAGLoadCSVRequest(BaseModel):
    file_path: str
    text_column: Optional[str] = None
    question_column: Optional[str] = None
    answer_column: Optional[str] = None
    name: Optional[str] = None

class RAGAskRequest(BaseModel):
    question: str
    top_k: int = 3
    min_similarity: float = 0.1


@router.post("/query", response_model=ChatResponse)
async def query_chat(request: ChatQuery):
    """Process a chat query about pipeline data."""
    try:
        response = chat_service.query(request.pipeline_id, request.query)
        
        return ChatResponse(
            answer=response["answer"],
            data=response.get("data")
        )
    except Exception as e:
        logger.error(f"Chat query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== RAG Endpoints =====

@router.post("/rag/load-text")
async def load_rag_text(request: RAGLoadTextRequest):
    """
    Carregar conhecimento de arquivo TXT.
    
    O texto será dividido em chunks para busca eficiente.
    
    Example:
        POST /chat/rag/load-text
        {
            "file_path": "data/use_cases/rag/produto_info.txt",
            "chunk_size": 200,
            "name": "info_produtos"
        }
    """
    try:
        stats = rag_service.load_knowledge_from_text(
            file_path=request.file_path,
            chunk_size=request.chunk_size,
            name=request.name
        )
        
        return {
            "status": "success",
            "message": f"Conhecimento carregado: {stats['chunks']} chunks",
            "stats": stats
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load text knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/load-csv")
async def load_rag_csv(request: RAGLoadCSVRequest):
    """
    Carregar conhecimento de arquivo CSV.
    
    Suporta 2 modos:
    1. Documentos: CSV com coluna de texto
    2. Q&A: CSV com colunas pergunta/resposta
    
    Example (modo Q&A):
        POST /chat/rag/load-csv
        {
            "file_path": "data/use_cases/rag/faq.csv",
            "question_column": "pergunta",
            "answer_column": "resposta",
            "name": "faq_produtos"
        }
    
    Example (modo documentos):
        POST /chat/rag/load-csv
        {
            "file_path": "data/use_cases/rag/docs.csv",
            "text_column": "conteudo",
            "name": "documentos"
        }
    """
    try:
        stats = rag_service.load_knowledge_from_csv(
            file_path=request.file_path,
            text_column=request.text_column,
            question_column=request.question_column,
            answer_column=request.answer_column,
            name=request.name
        )
        
        return {
            "status": "success",
            "message": f"Conhecimento carregado: {stats['entries']} entradas ({stats['mode']})",
            "stats": stats
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load CSV knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/ask")
async def ask_rag(request: RAGAskRequest):
    """
    Fazer pergunta ao sistema RAG.
    
    O sistema busca contexto relevante na base de conhecimento
    e retorna a melhor resposta encontrada.
    
    Example:
        POST /chat/rag/ask
        {
            "question": "Qual a cor da camisa?",
            "top_k": 3,
            "min_similarity": 0.1
        }
    """
    try:
        result = rag_service.answer(
            question=request.question,
            top_k=request.top_k,
            min_similarity=request.min_similarity
        )
        
        return {
            "status": "success",
            "question": request.question,
            **result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/stats")
async def get_rag_stats():
    """
    Obter estatísticas do conhecimento carregado.
    
    Returns:
        Informações sobre a base de conhecimento atual
    """
    try:
        stats = rag_service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get RAG stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Conversational Model Endpoints =====

class SaveConversationalModelRequest(BaseModel):
    model_name: str
    intent_model_name: str
    kb_name: str
    min_confidence: float = 0.4
    top_k: int = 3
    description: str = ""
    default_responses: Optional[dict] = None


class ChatWithModelRequest(BaseModel):
    model_name: str
    message: str
    context: Optional[dict] = None


@router.post("/conversational/save")
async def save_conversational_model(request: SaveConversationalModelRequest):
    """
    Salva um modelo conversacional completo que encapsula:
    - Classificador de intenção
    - Knowledge base
    - Lógica de geração de resposta
    
    Example:
        POST /chat/conversational/save
        {
            "model_name": "space_explorer_chat",
            "intent_model_name": "space_explorer",
            "kb_name": "space_explorer_kb",
            "min_confidence": 0.4,
            "top_k": 3,
            "description": "Chatbot sobre exploração espacial"
        }
    """
    try:
        result = conversational_service.save_conversational_model(
            model_name=request.model_name,
            intent_model_name=request.intent_model_name,
            kb_name=request.kb_name,
            min_confidence=request.min_confidence,
            top_k=request.top_k,
            description=request.description,
            default_responses=request.default_responses
        )
        
        return {
            "status": "success",
            "message": f"Modelo conversacional '{request.model_name}' salvo com sucesso!",
            **result
        }
    except Exception as e:
        logger.error(f"Failed to save conversational model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversational/chat")
async def chat_with_conversational_model(request: ChatWithModelRequest):
    """
    Conversa com um modelo conversacional.
    
    O modelo processa a mensagem através de:
    1. Classificação de intenção
    2. Busca semântica na knowledge base
    3. Geração de resposta contextualizada
    
    Example:
        POST /chat/conversational/chat
        {
            "model_name": "space_explorer_chat",
            "message": "O que são buracos negros?"
        }
    """
    try:
        result = conversational_service.chat(
            model_name=request.model_name,
            message=request.message,
            context=request.context
        )
        
        return {
            "status": "success",
            **result
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Conversational chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversational/list")
async def list_conversational_models():
    """
    Lista todos os modelos conversacionais salvos.
    
    Returns:
        Lista de modelos conversacionais com suas configurações
    """
    try:
        models = conversational_service.list_conversational_models()
        return {
            "status": "success",
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        logger.error(f"Failed to list conversational models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

