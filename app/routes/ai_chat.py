"""
Intelligent Chat Routes

API endpoints for conversational AI chat system.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from ..services.conversational_ai_pipeline import get_conversational_ai_pipeline
from ..core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ai-chat", tags=["ai-chat"])


# Request/Response Models
class ChatMessage(BaseModel):
    """User chat message"""
    message: str = Field(..., description="User message text")
    conversation_id: str = Field(..., description="Unique conversation ID")
    user_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional user info")


class ChatResponse(BaseModel):
    """AI chat response"""
    status: str
    conversation_id: str
    user_message: str
    response: str
    analysis: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    timestamp: str


class TrainingExample(BaseModel):
    """Intent training example"""
    text: str
    intent: str


class TrainIntentRequest(BaseModel):
    """Request to train intent classifier"""
    training_data: List[TrainingExample]
    test_size: float = 0.2


class IndexDocumentRequest(BaseModel):
    """Request to index documents"""
    documents: List[Dict[str, Any]]
    text_field: str = "text"


class CustomEntityRequest(BaseModel):
    """Request to add custom entities"""
    entity_type: str
    entities: List[str]


class ResponseTemplateRequest(BaseModel):
    """Request to add response template"""
    intent: str
    templates: List[str]
    required_slots: Optional[List[str]] = None


class ConfigureRequest(BaseModel):
    """Pipeline configuration"""
    config: Dict[str, Any]


# Get pipeline instance
def get_pipeline():
    """Get conversational AI pipeline"""
    return get_conversational_ai_pipeline()


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatMessage):
    """
    Send a message to the AI chatbot.
    
    The AI will:
    1. Classify the intent
    2. Extract entities
    3. Search knowledge base
    4. Generate contextual response
    5. Update conversation history
    """
    try:
        pipeline = get_pipeline()
        
        result = pipeline.process_message(
            message=request.message,
            conversation_id=request.conversation_id,
            user_metadata=request.user_metadata
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}/history")
async def get_conversation_history(
    conversation_id: str,
    last_n: Optional[int] = None
):
    """
    Get conversation history.
    
    Args:
        conversation_id: Conversation identifier
        last_n: Number of recent messages (optional)
    """
    try:
        pipeline = get_pipeline()
        history = pipeline.get_conversation_history(conversation_id, last_n)
        
        return {
            "status": "success",
            "conversation_id": conversation_id,
            "history": history,
            "num_messages": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history"""
    try:
        pipeline = get_pipeline()
        result = pipeline.clear_conversation(conversation_id)
        return result
        
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/intent")
async def train_intent_classifier(request: TrainIntentRequest):
    """
    Train the intent classification model.
    
    Provide labeled examples to train the AI to recognize user intents.
    """
    try:
        pipeline = get_pipeline()
        
        # Convert to dict format
        training_data = [
            {"text": ex.text, "intent": ex.intent}
            for ex in request.training_data
        ]
        
        result = pipeline.train_intent_classifier(training_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Error training intent classifier: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/documents")
async def index_knowledge_base(request: IndexDocumentRequest):
    """
    Index documents for semantic search.
    
    The AI will use these documents to answer questions.
    """
    try:
        pipeline = get_pipeline()
        
        result = pipeline.index_knowledge_base(
            documents=request.documents,
            text_field=request.text_field
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/entities/custom")
async def add_custom_entities(request: CustomEntityRequest):
    """
    Add custom domain-specific entities.
    
    Examples: product names, store locations, etc.
    """
    try:
        pipeline = get_pipeline()
        
        pipeline.add_custom_entities(
            entity_type=request.entity_type,
            entities=request.entities
        )
        
        return {
            "status": "success",
            "entity_type": request.entity_type,
            "num_entities": len(request.entities)
        }
        
    except Exception as e:
        logger.error(f"Error adding entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates/response")
async def add_response_template(request: ResponseTemplateRequest):
    """
    Add custom response templates for intents.
    
    Templates can use variables like $variable_name.
    """
    try:
        pipeline = get_pipeline()
        
        pipeline.add_response_template(
            intent=request.intent,
            templates=request.templates,
            required_slots=request.required_slots
        )
        
        return {
            "status": "success",
            "intent": request.intent,
            "num_templates": len(request.templates)
        }
        
    except Exception as e:
        logger.error(f"Error adding template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/configure")
async def configure_pipeline(request: ConfigureRequest):
    """
    Configure pipeline settings.
    
    Available settings:
    - use_intent_classification: bool
    - use_entity_extraction: bool
    - use_context: bool
    - use_semantic_search: bool
    - min_intent_confidence: float
    - min_search_score: float
    - max_search_results: int
    - max_context_messages: int
    """
    try:
        pipeline = get_pipeline()
        pipeline.configure(request.config)
        
        return {
            "status": "success",
            "updated_config": request.config
        }
        
    except Exception as e:
        logger.error(f"Error configuring pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_pipeline_info():
    """Get information about the AI pipeline and all services"""
    try:
        pipeline = get_pipeline()
        info = pipeline.get_pipeline_info()
        return info
        
    except Exception as e:
        logger.error(f"Error getting pipeline info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        pipeline = get_pipeline()
        info = pipeline.get_pipeline_info()
        
        return {
            "status": "healthy",
            "pipeline_initialized": True,
            "services_available": {
                "intent_classification": info["services"]["intent_classification"]["status"] != "not_trained",
                "entity_extraction": info["services"]["entity_extraction"]["status"] == "available",
                "semantic_search": info["services"]["semantic_search"]["status"] == "available",
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
