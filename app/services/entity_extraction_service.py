"""
Entity Extraction Service (NER)

Extracts named entities from text using spaCy.
Supports custom entity types and domain-specific extraction.
"""

from typing import Dict, List, Any, Optional, Set
import re
from collections import defaultdict
from ..core.logger import get_logger

logger = get_logger(__name__)

# Try to import spaCy
try:
    import spacy
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Install with: pip install spacy")


class EntityExtractionService:
    """Service for extracting named entities from text"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the entity extraction service.
        
        Args:
            model_name: spaCy model to use (default: en_core_web_sm)
        """
        self.model_name = model_name
        self.nlp = None
        self.custom_entities: Dict[str, List[str]] = {}
        
        if SPACY_AVAILABLE:
            try:
                logger.info(f"Loading spaCy model: {model_name}")
                self.nlp = spacy.load(model_name)
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning(f"spaCy model '{model_name}' not found. Download with: python -m spacy download {model_name}")
            except Exception as e:
                logger.error(f"Error loading spaCy model: {e}")
        else:
            logger.warning("Entity extraction unavailable - spaCy not installed")
    
    def is_available(self) -> bool:
        """Check if entity extraction is available"""
        return self.nlp is not None
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with entities by type
        """
        if not self.is_available():
            return {
                "entities": [],
                "entities_by_type": {},
                "error": "Entity extraction not available"
            }
        
        try:
            # Process text
            doc = self.nlp(text)
            
            # Extract entities
            entities = []
            entities_by_type = defaultdict(list)
            
            for ent in doc.ents:
                entity_data = {
                    "text": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                entities.append(entity_data)
                entities_by_type[ent.label_].append(ent.text)
            
            # Extract custom entities
            custom_found = self._extract_custom_entities(text)
            for entity_type, values in custom_found.items():
                for value in values:
                    entities.append({
                        "text": value,
                        "type": entity_type,
                        "start": text.find(value),
                        "end": text.find(value) + len(value)
                    })
                    entities_by_type[entity_type].extend([value])
            
            # Remove duplicates from entities_by_type
            entities_by_type = {
                k: list(set(v)) for k, v in entities_by_type.items()
            }
            
            return {
                "entities": entities,
                "entities_by_type": dict(entities_by_type),
                "num_entities": len(entities)
            }
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {
                "entities": [],
                "entities_by_type": {},
                "error": str(e)
            }
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Extract important keywords/phrases using POS tagging.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of keywords with metadata
        """
        if not self.is_available():
            return []
        
        try:
            doc = self.nlp(text)
            
            # Extract noun phrases and important tokens
            keywords = []
            seen = set()
            
            # Noun phrases
            for chunk in doc.noun_chunks:
                if chunk.text.lower() not in seen:
                    keywords.append({
                        "text": chunk.text,
                        "type": "noun_phrase",
                        "pos": "NOUN"
                    })
                    seen.add(chunk.text.lower())
            
            # Important single tokens
            for token in doc:
                if (token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] and
                    not token.is_stop and
                    len(token.text) > 2 and
                    token.text.lower() not in seen):
                    keywords.append({
                        "text": token.text,
                        "type": "token",
                        "pos": token.pos_
                    })
                    seen.add(token.text.lower())
            
            return keywords[:top_n]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def add_custom_entities(
        self,
        entity_type: str,
        entities: List[str]
    ) -> None:
        """
        Add custom entities for domain-specific extraction.
        
        Args:
            entity_type: Type/category of entities (e.g., "PRODUCT", "STORE")
            entities: List of entity values to recognize
        """
        if entity_type not in self.custom_entities:
            self.custom_entities[entity_type] = []
        
        self.custom_entities[entity_type].extend(entities)
        # Remove duplicates
        self.custom_entities[entity_type] = list(set(self.custom_entities[entity_type]))
        
        logger.info(f"Added {len(entities)} custom entities for type '{entity_type}'")
    
    def _extract_custom_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract custom domain-specific entities"""
        found = defaultdict(list)
        
        text_lower = text.lower()
        
        for entity_type, entity_list in self.custom_entities.items():
            for entity in entity_list:
                if entity.lower() in text_lower:
                    found[entity_type].append(entity)
        
        return dict(found)
    
    def extract_dates(self, text: str) -> List[str]:
        """Extract date mentions from text"""
        if not self.is_available():
            return []
        
        try:
            doc = self.nlp(text)
            dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
            return dates
        except Exception as e:
            logger.error(f"Error extracting dates: {e}")
            return []
    
    def extract_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Extract numeric values from text"""
        if not self.is_available():
            return []
        
        try:
            doc = self.nlp(text)
            numbers = []
            
            for ent in doc.ents:
                if ent.label_ in ["MONEY", "PERCENT", "QUANTITY", "CARDINAL"]:
                    numbers.append({
                        "text": ent.text,
                        "type": ent.label_,
                        "value": self._parse_number(ent.text)
                    })
            
            return numbers
            
        except Exception as e:
            logger.error(f"Error extracting numbers: {e}")
            return []
    
    def _parse_number(self, text: str) -> Optional[float]:
        """Try to parse a number from text"""
        try:
            # Remove currency symbols and percent signs
            cleaned = re.sub(r'[R$%,]', '', text)
            return float(cleaned.strip())
        except:
            return None
    
    def batch_extract(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Extract entities from multiple texts.
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of extraction results
        """
        return [self.extract_entities(text) for text in texts]
    
    def get_entity_types(self) -> Dict[str, List[str]]:
        """Get available entity types"""
        standard_types = [
            "PERSON",      # People
            "ORG",         # Organizations
            "GPE",         # Countries, cities
            "DATE",        # Dates
            "TIME",        # Times
            "MONEY",       # Monetary values
            "PERCENT",     # Percentages
            "PRODUCT",     # Products
            "EVENT",       # Events
            "LOC",         # Locations
        ] if self.is_available() else []
        
        return {
            "standard": standard_types,
            "custom": list(self.custom_entities.keys())
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the entity extraction service"""
        if not self.is_available():
            return {
                "status": "unavailable",
                "message": "spaCy not installed or model not loaded"
            }
        
        return {
            "status": "available",
            "model": self.model_name,
            "custom_entity_types": list(self.custom_entities.keys()),
            "num_custom_entities": sum(len(v) for v in self.custom_entities.values())
        }


# Singleton instance
_entity_extraction_instance = None


def get_entity_extraction_service(model_name: str = "en_core_web_sm") -> EntityExtractionService:
    """Get or create the singleton entity extraction service"""
    global _entity_extraction_instance
    if _entity_extraction_instance is None:
        _entity_extraction_instance = EntityExtractionService(model_name)
    return _entity_extraction_instance
