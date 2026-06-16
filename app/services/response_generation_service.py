"""
Response Generation Service

Generates natural responses using templates, extractive summarization,
and context-aware composition.
"""

from typing import Dict, List, Any, Optional
import re
from string import Template
from ..core.logger import get_logger

logger = get_logger(__name__)


class ResponseTemplate:
    """Represents a response template for a specific intent"""
    
    def __init__(
        self,
        intent: str,
        templates: List[str],
        required_slots: Optional[List[str]] = None
    ):
        self.intent = intent
        self.templates = templates
        self.required_slots = required_slots or []
        self.current_template_index = 0
    
    def generate(self, slots: Dict[str, Any]) -> str:
        """Generate a response using the template and slots"""
        # Check if all required slots are present
        missing_slots = [s for s in self.required_slots if s not in slots]
        if missing_slots:
            return f"Desculpe, preciso de mais informações: {', '.join(missing_slots)}"
        
        # Get template (rotate through templates for variety)
        template_str = self.templates[self.current_template_index % len(self.templates)]
        self.current_template_index += 1
        
        # Replace slots
        try:
            template = Template(template_str)
            return template.safe_substitute(slots)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return template_str


class ResponseGenerationService:
    """Service for generating natural language responses"""
    
    def __init__(self):
        self.templates: Dict[str, ResponseTemplate] = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default response templates"""
        
        # Greeting intent
        self.add_template(
            "greeting",
            [
                "Olá! Como posso ajudá-lo hoje?",
                "Oi! Em que posso ser útil?",
                "Olá! Estou aqui para ajudar. O que você precisa?",
            ]
        )
        
        # FAQ intent
        self.add_template(
            "faq",
            [
                "Baseado nas informações que tenho: $answer",
                "De acordo com nossa base de conhecimento: $answer",
                "Aqui está o que encontrei: $answer",
            ],
            required_slots=["answer"]
        )
        
        # Information request
        self.add_template(
            "info_request",
            [
                "Sobre $topic: $information",
                "Aqui está o que sei sobre $topic: $information",
                "Posso te informar sobre $topic: $information",
            ],
            required_slots=["topic", "information"]
        )
        
        # Complaint/problem
        self.add_template(
            "complaint",
            [
                "Entendo sua preocupação sobre $issue. Vou verificar isso para você.",
                "Lamento ouvir sobre $issue. Deixe-me ver como posso ajudar.",
                "Agradeço por reportar sobre $issue. Vou investigar isso.",
            ],
            required_slots=["issue"]
        )
        
        # Confirmation
        self.add_template(
            "confirmation",
            [
                "Entendido! $action foi $status.",
                "Confirmado: $action - $status.",
                "Perfeito! $action está $status.",
            ],
            required_slots=["action", "status"]
        )
        
        # Unknown/fallback
        self.add_template(
            "unknown",
            [
                "Desculpe, não entendi completamente. Pode reformular?",
                "Não tenho certeza sobre isso. Pode me dar mais detalhes?",
                "Hmm, não tenho informações suficientes. Pode explicar melhor?",
            ]
        )
        
        # Thanks
        self.add_template(
            "thanks",
            [
                "De nada! Posso ajudar com mais alguma coisa?",
                "Por nada! Estou aqui se precisar de mais ajuda.",
                "Fico feliz em ajudar! Algo mais?",
            ]
        )
    
    def add_template(
        self,
        intent: str,
        templates: List[str],
        required_slots: Optional[List[str]] = None
    ):
        """
        Add or update a response template for an intent.
        
        Args:
            intent: Intent name
            templates: List of template strings
            required_slots: Required slot names
        """
        self.templates[intent] = ResponseTemplate(intent, templates, required_slots)
        logger.debug(f"Added template for intent: {intent}")
    
    def generate_response(
        self,
        intent: str,
        slots: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response for an intent.
        
        Args:
            intent: Detected intent
            slots: Extracted entities/slots
            context: Conversation context
            
        Returns:
            Generated response string
        """
        slots = slots or {}
        context = context or {}
        
        # Get template for intent
        template = self.templates.get(intent)
        
        if not template:
            logger.warning(f"No template found for intent: {intent}")
            template = self.templates.get("unknown")
        
        # Generate response
        response = template.generate(slots)
        
        # Post-process with context if available
        if context:
            response = self._apply_context(response, context)
        
        return response
    
    def generate_from_documents(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        intent: str = "faq"
    ) -> str:
        """
        Generate a response from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            query: Original query
            intent: Intent for response template
            
        Returns:
            Generated response
        """
        if not documents:
            return "Desculpe, não encontrei informações sobre isso."
        
        # Get best document
        best_doc = documents[0]
        
        # Extract answer from document
        answer = self._extract_answer(best_doc, query)
        
        # Add source information if available
        source_info = ""
        if "source" in best_doc:
            source_info = f" (Fonte: {best_doc['source']})"
        
        # Generate response using template
        slots = {
            "answer": answer + source_info,
            "topic": query
        }
        
        return self.generate_response(intent, slots)
    
    def _extract_answer(
        self,
        document: Dict[str, Any],
        query: str
    ) -> str:
        """
        Extract the most relevant answer from a document.
        
        Args:
            document: Retrieved document
            query: Original query
            
        Returns:
            Extracted answer text
        """
        # Try different fields
        for field in ["resposta", "answer", "text", "content"]:
            if field in document:
                return str(document[field])
        
        # Fallback to full document text
        return str(document)
    
    def _apply_context(self, response: str, context: Dict[str, Any]) -> str:
        """Apply conversation context to personalize response"""
        
        # Add user name if available
        if "user_name" in context:
            if not any(greeting in response.lower() for greeting in ["olá", "oi", "bem-vindo"]):
                response = f"{context['user_name']}, {response}"
        
        # Reference previous topics if relevant
        if "last_topic" in context and "$previous" in response:
            response = response.replace("$previous", context["last_topic"])
        
        return response
    
    def generate_multi_document_response(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        max_docs: int = 3
    ) -> str:
        """
        Generate a response synthesizing multiple documents.
        
        Args:
            documents: List of retrieved documents
            query: Original query
            max_docs: Maximum documents to include
            
        Returns:
            Synthesized response
        """
        if not documents:
            return "Não encontrei informações relevantes sobre isso."
        
        # Take top documents
        top_docs = documents[:max_docs]
        
        if len(top_docs) == 1:
            return self.generate_from_documents(top_docs, query)
        
        # Synthesize multiple answers
        intro = f"Encontrei {len(top_docs)} informações relevantes:\n\n"
        
        answers = []
        for i, doc in enumerate(top_docs, 1):
            answer = self._extract_answer(doc, query)
            source = doc.get("source", f"Documento {i}")
            answers.append(f"{i}. {answer} (Fonte: {source})")
        
        return intro + "\n\n".join(answers)
    
    def generate_clarification(
        self,
        ambiguous_intents: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a clarification question when intent is ambiguous.
        
        Args:
            ambiguous_intents: List of possible intents with scores
            
        Returns:
            Clarification question
        """
        if len(ambiguous_intents) < 2:
            return "Pode me dar mais detalhes sobre o que você precisa?"
        
        top_intents = ambiguous_intents[:3]
        
        options = []
        for intent_data in top_intents:
            intent = intent_data["intent"]
            # Generate human-readable option
            option = self._intent_to_readable(intent)
            options.append(option)
        
        options_text = " ou ".join(options)
        return f"Você está perguntando sobre {options_text}?"
    
    def _intent_to_readable(self, intent: str) -> str:
        """Convert intent code to human-readable text"""
        intent_map = {
            "faq": "perguntas frequentes",
            "info_request": "informações gerais",
            "complaint": "um problema",
            "greeting": "saudação",
            "thanks": "agradecimento",
            "product_info": "informações de produto",
            "price_info": "preços",
            "shipping_info": "entrega",
        }
        return intent_map.get(intent, intent.replace("_", " "))
    
    def generate_error_response(self, error_type: str) -> str:
        """Generate user-friendly error responses"""
        error_responses = {
            "no_data": "Desculpe, não tenho dados suficientes para responder isso no momento.",
            "processing_error": "Ocorreu um erro ao processar sua solicitação. Pode tentar novamente?",
            "timeout": "A operação está demorando mais que o esperado. Pode tentar mais tarde?",
            "not_found": "Não encontrei informações sobre isso. Pode reformular a pergunta?",
        }
        return error_responses.get(error_type, "Desculpe, ocorreu um erro inesperado.")
    
    def get_available_intents(self) -> List[str]:
        """Get list of intents with templates"""
        return list(self.templates.keys())
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the response generation service"""
        return {
            "num_templates": len(self.templates),
            "available_intents": self.get_available_intents()
        }


# Singleton instance
_response_generation_instance = None


def get_response_generation_service() -> ResponseGenerationService:
    """Get or create the singleton response generation service"""
    global _response_generation_instance
    if _response_generation_instance is None:
        _response_generation_instance = ResponseGenerationService()
    return _response_generation_instance
