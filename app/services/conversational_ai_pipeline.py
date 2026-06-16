"""
Conversational AI Pipeline

Orchestrates all AI services to create an intelligent conversational agent.
Integrates: Intent Classification, Entity Extraction, Context Management,
Semantic Search, and Response Generation.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from ..core.logger import get_logger
from .intent_classification_service import get_intent_classification_service
from .entity_extraction_service import get_entity_extraction_service
from .context_manager_service import get_context_manager_service
from .semantic_search_service import get_semantic_search_service
from .response_generation_service import get_response_generation_service

logger = get_logger(__name__)


class ConversationalAIPipeline:
    """Main pipeline for conversational AI"""
    
    def __init__(self):
        # Initialize all services
        self.intent_service = get_intent_classification_service()
        self.entity_service = get_entity_extraction_service()
        self.context_service = get_context_manager_service()
        self.semantic_search = get_semantic_search_service()
        self.response_service = get_response_generation_service()
        
        # Pipeline configuration
        self.config = {
            "use_intent_classification": True,
            "use_entity_extraction": True,
            "use_context": True,
            "use_semantic_search": True,
            "min_intent_confidence": 0.4,
            "min_search_score": 0.3,
            "max_search_results": 5,
            "max_context_messages": 5,
        }
        
        logger.info("Conversational AI Pipeline initialized")
    
    def process_message(
        self,
        message: str,
        conversation_id: str,
        user_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message through the full AI pipeline.
        
        Args:
            message: User's input message
            conversation_id: Unique conversation identifier
            user_metadata: Additional user information
            
        Returns:
            Response with AI analysis and generated reply
        """
        start_time = datetime.now()
        
        logger.info(f"Processing message in conversation {conversation_id}")
        logger.debug(f"Message: {message}")
        
        try:
            # Step 1: Get conversation context
            context = self._get_context(conversation_id)
            
            # Step 2: Classify intent
            intent_result = self._classify_intent(message)
            
            # Step 3: Extract entities
            entities_result = self._extract_entities(message, context)
            
            # Step 4: Search for relevant information
            search_results = self._search_knowledge(
                message,
                intent_result,
                entities_result
            )
            
            # Step 5: Generate response
            response = self._generate_response(
                message,
                intent_result,
                entities_result,
                search_results,
                context
            )
            
            # Step 6: Update context
            self._update_context(
                conversation_id,
                message,
                response,
                intent_result,
                entities_result,
                user_metadata
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Build result
            result = {
                "status": "success",
                "conversation_id": conversation_id,
                "user_message": message,
                "response": response,
                "analysis": {
                    "intent": intent_result,
                    "entities": entities_result,
                    "search_results": search_results,
                    "context_used": context.get("has_context", False)
                },
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Message processed successfully in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return {
                "status": "error",
                "conversation_id": conversation_id,
                "user_message": message,
                "response": "Desculpe, ocorreu um erro ao processar sua mensagem. Pode tentar novamente?",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_context(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation context"""
        if not self.config["use_context"]:
            return {"has_context": False}
        
        context = self.context_service.get_context_for_query(
            conversation_id,
            max_messages=self.config["max_context_messages"]
        )
        
        logger.debug(f"Retrieved context: {context.get('conversation_summary', {})}")
        
        return context
    
    def _classify_intent(self, message: str) -> Dict[str, Any]:
        """Classify user intent"""
        if not self.config["use_intent_classification"]:
            return {"intent": "unknown", "confidence": 0.0}
        
        intent_result = self.intent_service.predict_intent(message)
        
        # Check confidence threshold
        if intent_result["confidence"] < self.config["min_intent_confidence"]:
            logger.debug(f"Intent confidence too low: {intent_result['confidence']}")
            intent_result["intent"] = "unknown"
        
        logger.debug(f"Intent: {intent_result['intent']} ({intent_result['confidence']:.2%})")
        
        return intent_result
    
    def _extract_entities(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract entities from message"""
        if not self.config["use_entity_extraction"]:
            return {"entities": [], "entities_by_type": {}}
        
        entities_result = self.entity_service.extract_entities(message)
        
        # Enhance with context entities
        if context.get("has_context"):
            context_entities = context.get("entities", {})
            entities_result["context_entities"] = context_entities
        
        logger.debug(f"Extracted entities: {entities_result.get('entities_by_type', {})}")
        
        return entities_result
    
    def _search_knowledge(
        self,
        message: str,
        intent_result: Dict[str, Any],
        entities_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant information"""
        if not self.config["use_semantic_search"]:
            return []
        
        # Check if semantic search is available and indexed
        if not self.semantic_search.is_available() or self.semantic_search.embeddings is None:
            logger.debug("Semantic search not available or not indexed")
            return []
        
        # Perform semantic search
        search_results = self.semantic_search.search(
            query=message,
            top_k=self.config["max_search_results"],
            min_score=self.config["min_search_score"]
        )
        
        logger.debug(f"Found {len(search_results)} search results")
        
        return search_results
    
    def _generate_response(
        self,
        message: str,
        intent_result: Dict[str, Any],
        entities_result: Dict[str, Any],
        search_results: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> str:
        """Generate appropriate response"""
        
        intent = intent_result.get("intent", "unknown")
        confidence = intent_result.get("confidence", 0.0)
        
        # Handle different scenarios
        
        # 1. Low confidence intent - ask for clarification
        if confidence < self.config["min_intent_confidence"] and intent != "greeting":
            if intent_result.get("all_intents"):
                top_intents = intent_result["all_intents"][:3]
                return self.response_service.generate_clarification(top_intents)
            return "Pode me dar mais detalhes sobre o que você precisa?"
        
        # 2. Greeting - simple greeting response
        if intent == "greeting":
            return self.response_service.generate_response("greeting")
        
        # 3. Thanks - simple thanks response
        if intent == "thanks":
            return self.response_service.generate_response("thanks")
        
        # 4. FAQ/Info Request with search results
        if search_results and intent in ["faq", "info_request", "unknown"]:
            return self.response_service.generate_from_documents(
                search_results,
                message,
                intent
            )
        
        # 5. Intent with entities but no search results
        if entities_result.get("entities"):
            slots = self._prepare_slots(entities_result, search_results)
            return self.response_service.generate_response(
                intent,
                slots,
                context
            )
        
        # 6. No relevant information found
        if not search_results:
            return self.response_service.generate_error_response("not_found")
        
        # 7. Fallback - use template response
        return self.response_service.generate_response(intent, {}, context)
    
    def _prepare_slots(
        self,
        entities_result: Dict[str, Any],
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare slots for template filling"""
        slots = {}
        
        # Add entities as slots
        entities_by_type = entities_result.get("entities_by_type", {})
        for entity_type, values in entities_by_type.items():
            if values:
                slots[entity_type.lower()] = values[0]
        
        # Add search result data if available
        if search_results:
            best_result = search_results[0]
            for key, value in best_result.items():
                if key not in ["score", "rank"]:
                    slots[key] = value
        
        return slots
    
    def _update_context(
        self,
        conversation_id: str,
        user_message: str,
        ai_response: str,
        intent_result: Dict[str, Any],
        entities_result: Dict[str, Any],
        user_metadata: Optional[Dict[str, Any]]
    ):
        """Update conversation context"""
        if not self.config["use_context"]:
            return
        
        # Add user message
        self.context_service.add_message(
            conversation_id,
            "user",
            user_message,
            entities=entities_result.get("entities_by_type", {}),
            metadata={
                "intent": intent_result.get("intent"),
                "intent_confidence": intent_result.get("confidence"),
                **(user_metadata or {})
            }
        )
        
        # Add AI response
        self.context_service.add_message(
            conversation_id,
            "assistant",
            ai_response,
            metadata={
                "generated_from": intent_result.get("intent")
            }
        )
    
    def train_intent_classifier(
        self,
        training_data: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Train the intent classifier with examples.
        
        Args:
            training_data: List of {"text": "...", "intent": "..."}
            
        Returns:
            Training results
        """
        logger.info("Training intent classifier...")
        result = self.intent_service.train_intent_classifier(training_data)
        
        if result["status"] == "success":
            self.config["use_intent_classification"] = True
        
        return result
    
    def index_knowledge_base(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text"
    ) -> Dict[str, Any]:
        """
        Index documents for semantic search.
        
        Args:
            documents: List of documents to index
            text_field: Field containing text to embed
            
        Returns:
            Indexing results
        """
        logger.info("Indexing knowledge base...")
        result = self.semantic_search.index_documents(documents, text_field)
        
        if result["status"] == "success":
            self.config["use_semantic_search"] = True
        
        return result
    
    def add_custom_entities(
        self,
        entity_type: str,
        entities: List[str]
    ):
        """Add custom domain-specific entities"""
        self.entity_service.add_custom_entities(entity_type, entities)
        self.config["use_entity_extraction"] = True
    
    def add_response_template(
        self,
        intent: str,
        templates: List[str],
        required_slots: Optional[List[str]] = None
    ):
        """Add custom response template"""
        self.response_service.add_template(intent, templates, required_slots)
    
    def configure(self, config: Dict[str, Any]):
        """Update pipeline configuration"""
        self.config.update(config)
        logger.info(f"Pipeline configured: {config}")
    
    def get_conversation_history(
        self,
        conversation_id: str,
        last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.context_service.get_conversation_history(conversation_id, last_n)
    
    def clear_conversation(self, conversation_id: str):
        """Clear a conversation's history"""
        return self.context_service.clear_conversation(conversation_id)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline and all services"""
        return {
            "status": "initialized",
            "config": self.config,
            "services": {
                "intent_classification": self.intent_service.get_model_info(),
                "entity_extraction": self.entity_service.get_info(),
                "context_manager": self.context_service.get_info(),
                "semantic_search": self.semantic_search.get_info(),
                "response_generation": self.response_service.get_info()
            }
        }


# Singleton instance
_conversational_ai_pipeline = None


def get_conversational_ai_pipeline() -> ConversationalAIPipeline:
    """Get or create the singleton conversational AI pipeline"""
    global _conversational_ai_pipeline
    if _conversational_ai_pipeline is None:
        _conversational_ai_pipeline = ConversationalAIPipeline()
    return _conversational_ai_pipeline
