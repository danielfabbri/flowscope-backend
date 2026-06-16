"""
Context Manager Service

Manages conversation context, history, and dialogue state.
Tracks entities, maintains context window, and handles conversation memory.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import deque, defaultdict
import json
from pathlib import Path
from ..core.logger import get_logger

logger = get_logger(__name__)


class ConversationContext:
    """Represents a single conversation context"""
    
    def __init__(self, conversation_id: str, max_history: int = 10):
        self.conversation_id = conversation_id
        self.max_history = max_history
        self.messages: deque = deque(maxlen=max_history)
        self.entities: Dict[str, List[str]] = defaultdict(list)
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def add_entities(self, entity_type: str, entities: List[str]):
        """Track entities mentioned in the conversation"""
        for entity in entities:
            if entity not in self.entities[entity_type]:
                self.entities[entity_type].append(entity)
    
    def get_last_messages(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the last N messages"""
        return list(self.messages)[-n:]
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation context"""
        return {
            "conversation_id": self.conversation_id,
            "num_messages": len(self.messages),
            "entities": dict(self.entities),
            "last_updated": self.updated_at.isoformat(),
            "duration": (self.updated_at - self.created_at).total_seconds()
        }
    
    def clear(self):
        """Clear conversation history"""
        self.messages.clear()
        self.entities.clear()
        self.updated_at = datetime.now()


class ContextManagerService:
    """Service for managing conversation contexts"""
    
    def __init__(self, max_conversations: int = 100):
        self.conversations: Dict[str, ConversationContext] = {}
        self.max_conversations = max_conversations
        self.default_context_window = 10
    
    def create_conversation(
        self,
        conversation_id: str,
        max_history: Optional[int] = None
    ) -> ConversationContext:
        """
        Create a new conversation context.
        
        Args:
            conversation_id: Unique identifier for the conversation
            max_history: Maximum number of messages to keep
            
        Returns:
            New conversation context
        """
        if conversation_id in self.conversations:
            logger.warning(f"Conversation {conversation_id} already exists")
            return self.conversations[conversation_id]
        
        # Cleanup old conversations if at limit
        if len(self.conversations) >= self.max_conversations:
            self._cleanup_old_conversations()
        
        context = ConversationContext(
            conversation_id,
            max_history or self.default_context_window
        )
        self.conversations[conversation_id] = context
        
        logger.info(f"Created conversation: {conversation_id}")
        return context
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get an existing conversation context"""
        return self.conversations.get(conversation_id)
    
    def get_or_create_conversation(
        self,
        conversation_id: str
    ) -> ConversationContext:
        """Get existing conversation or create new one"""
        if conversation_id not in self.conversations:
            return self.create_conversation(conversation_id)
        return self.conversations[conversation_id]
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        entities: Optional[Dict[str, List[str]]] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation identifier
            role: Message role (user/assistant/system)
            content: Message content
            entities: Extracted entities to track
            metadata: Additional metadata
            
        Returns:
            Status of the operation
        """
        try:
            context = self.get_or_create_conversation(conversation_id)
            
            # Add message
            context.add_message(role, content, metadata)
            
            # Track entities
            if entities:
                for entity_type, entity_list in entities.items():
                    context.add_entities(entity_type, entity_list)
            
            logger.debug(f"Added {role} message to conversation {conversation_id}")
            
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "num_messages": len(context.messages)
            }
            
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_conversation_history(
        self,
        conversation_id: str,
        last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            conversation_id: Conversation identifier
            last_n: Number of recent messages to return (None = all)
            
        Returns:
            List of messages
        """
        context = self.get_conversation(conversation_id)
        if not context:
            return []
        
        if last_n:
            return context.get_last_messages(last_n)
        return list(context.messages)
    
    def get_conversation_entities(
        self,
        conversation_id: str
    ) -> Dict[str, List[str]]:
        """Get all entities mentioned in a conversation"""
        context = self.get_conversation(conversation_id)
        if not context:
            return {}
        return dict(context.entities)
    
    def get_context_for_query(
        self,
        conversation_id: str,
        max_messages: int = 5
    ) -> Dict[str, Any]:
        """
        Get relevant context for a new query.
        
        Args:
            conversation_id: Conversation identifier
            max_messages: Maximum recent messages to include
            
        Returns:
            Context information
        """
        context = self.get_conversation(conversation_id)
        if not context:
            return {
                "history": [],
                "entities": {},
                "has_context": False
            }
        
        recent_messages = context.get_last_messages(max_messages)
        
        return {
            "history": recent_messages,
            "entities": dict(context.entities),
            "has_context": len(recent_messages) > 0,
            "conversation_summary": context.get_context_summary()
        }
    
    def clear_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Clear a conversation's history"""
        context = self.get_conversation(conversation_id)
        if not context:
            return {"status": "error", "message": "Conversation not found"}
        
        context.clear()
        logger.info(f"Cleared conversation: {conversation_id}")
        
        return {
            "status": "success",
            "conversation_id": conversation_id
        }
    
    def delete_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Delete a conversation entirely"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Deleted conversation: {conversation_id}")
            return {"status": "success"}
        
        return {"status": "error", "message": "Conversation not found"}
    
    def _cleanup_old_conversations(self, keep: int = 50):
        """Remove oldest conversations when limit is reached"""
        if len(self.conversations) <= keep:
            return
        
        # Sort by last update time
        sorted_convs = sorted(
            self.conversations.items(),
            key=lambda x: x[1].updated_at
        )
        
        # Remove oldest
        to_remove = len(self.conversations) - keep
        for conv_id, _ in sorted_convs[:to_remove]:
            del self.conversations[conv_id]
        
        logger.info(f"Cleaned up {to_remove} old conversations")
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Get summary of all active conversations"""
        return [
            context.get_context_summary()
            for context in self.conversations.values()
        ]
    
    def save_conversation(
        self,
        conversation_id: str,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Save a conversation to disk.
        
        Args:
            conversation_id: Conversation to save
            file_path: Path to save to
            
        Returns:
            Save status
        """
        context = self.get_conversation(conversation_id)
        if not context:
            return {"status": "error", "message": "Conversation not found"}
        
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "conversation_id": context.conversation_id,
                "messages": list(context.messages),
                "entities": dict(context.entities),
                "metadata": context.metadata,
                "created_at": context.created_at.isoformat(),
                "updated_at": context.updated_at.isoformat()
            }
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved conversation to {file_path}")
            
            return {
                "status": "success",
                "file_path": file_path,
                "num_messages": len(context.messages)
            }
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return {"status": "error", "message": str(e)}
    
    def load_conversation(self, file_path: str) -> Dict[str, Any]:
        """
        Load a conversation from disk.
        
        Args:
            file_path: Path to load from
            
        Returns:
            Load status
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            conversation_id = data["conversation_id"]
            context = self.create_conversation(conversation_id)
            
            # Restore messages
            for msg in data["messages"]:
                context.messages.append(msg)
            
            # Restore entities
            context.entities = defaultdict(list, data["entities"])
            context.metadata = data.get("metadata", {})
            
            logger.info(f"Loaded conversation from {file_path}")
            
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "num_messages": len(context.messages)
            }
            
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the context manager"""
        return {
            "num_conversations": len(self.conversations),
            "max_conversations": self.max_conversations,
            "default_context_window": self.default_context_window,
            "total_messages": sum(
                len(c.messages) for c in self.conversations.values()
            )
        }


# Singleton instance
_context_manager_instance = None


def get_context_manager_service() -> ContextManagerService:
    """Get or create the singleton context manager service"""
    global _context_manager_instance
    if _context_manager_instance is None:
        _context_manager_instance = ContextManagerService()
    return _context_manager_instance
