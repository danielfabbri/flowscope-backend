"""
Response Generation Step Service
Generates responses for pipeline processing
"""
import pandas as pd
import json
import re
from typing import Dict, Any, Optional, List
from app.services.base_service import BasePipelineService
from app.core.logger import get_logger

logger = get_logger(__name__)


class ResponseGenerationStepService(BasePipelineService):
    """
    Pipeline step that generates responses using templates or LLM
    
    Input: DataFrame with intent, entities, and context columns
    Output: DataFrame with added response column
    """
    
    def __init__(self):
        self.default_templates = {
            'greeting': 'Olá! Como posso ajudá-lo hoje?',
            'faq': 'Deixe-me buscar isso para você...',
            'complaint': 'Sinto muito pelo problema. Vou resolver isso.',
            'thanks': 'De nada! Posso ajudar com mais alguma coisa?',
            'unknown': 'Não entendi. Pode reformular a pergunta?'
        }
        logger.info("ResponseGenerationStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute response generation on DataFrame
        
        Args:
            data: DataFrame with intent, entities, context columns
            config: Configuration with:
                - response_mode: 'template' or 'llm'
                - intent_column: column name with intent
                - entity_column: column name with entities (JSON)
                - context_column: column name with context (JSON)
                - output_column: name for response output column
                - templates: dict of templates by intent (JSON string)
                - llm_provider: 'ollama', 'openai', 'anthropic'
                - llm_model: model name
                - system_prompt: system prompt for LLM
        
        Returns:
            DataFrame with added response column
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for response generation")
        
        response_mode = config.get('response_mode', 'template')
        intent_column = config.get('intent_column', 'intent')
        entity_column = config.get('entity_column', 'entities')
        context_column = config.get('context_column', 'conversation_context')
        output_column = config.get('output_column', 'response')
        templates_str = config.get('templates', '')
        
        # Parse templates
        templates = self.default_templates.copy()
        if templates_str:
            try:
                if isinstance(templates_str, str):
                    custom_templates = json.loads(templates_str)
                else:
                    custom_templates = templates_str
                templates.update(custom_templates)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse custom templates, using defaults")
        
        logger.info(f"Generating responses for {len(data)} messages using mode: {response_mode}")
        
        if response_mode == 'template':
            responses = self._generate_template_responses(
                data, intent_column, entity_column, context_column, templates
            )
        elif response_mode == 'llm':
            responses = self._generate_llm_responses(
                data, intent_column, entity_column, context_column, config
            )
        else:
            raise ValueError(f"Unknown response_mode: {response_mode}")
        
        # Add new column to DataFrame
        result = data.copy()
        result[output_column] = responses
        
        logger.info(f"Response generation complete")
        
        return result
    
    def _generate_template_responses(
        self,
        data: pd.DataFrame,
        intent_column: str,
        entity_column: str,
        context_column: str,
        templates: Dict[str, str]
    ) -> list[str]:
        """Generate responses using templates"""
        responses = []
        
        for idx, row in data.iterrows():
            # Get intent
            intent = row.get(intent_column, 'unknown') if intent_column in data.columns else 'unknown'
            
            # Get template for intent
            template = templates.get(intent, templates.get('unknown', 'Como posso ajudar?'))
            
            # Get entities if available
            entities = {}
            if entity_column in data.columns:
                try:
                    entities = json.loads(str(row[entity_column]))
                except:
                    pass
            
            # Get context if available
            context = {}
            if context_column in data.columns:
                try:
                    context = json.loads(str(row[context_column]))
                except:
                    pass
            
            # Replace placeholders in template
            response = self._fill_template(template, entities, context)
            
            responses.append(response)
        
        return responses
    
    def _fill_template(self, template: str, entities: Dict, context: Dict) -> str:
        """Fill template with entities and context"""
        response = template
        
        # Replace {entity_type} with first entity of that type
        for entity_type, entity_values in entities.items():
            placeholder = f'{{{entity_type}}}'
            if placeholder in response:
                if isinstance(entity_values, list) and entity_values:
                    response = response.replace(placeholder, str(entity_values[0]))
                else:
                    response = response.replace(placeholder, str(entity_values))
        
        # Replace {context} with history
        if '{context}' in response and 'history' in context:
            response = response.replace('{context}', context['history'])
        
        # Replace any remaining placeholders with empty string
        response = re.sub(r'\{[^}]+\}', '', response)
        
        return response.strip()
    
    def _generate_llm_responses(
        self,
        data: pd.DataFrame,
        intent_column: str,
        entity_column: str,
        context_column: str,
        config: Dict[str, Any]
    ) -> list[str]:
        """Generate responses using LLM (placeholder for future implementation)"""
        logger.warning("LLM response generation not yet implemented. Using template fallback.")
        
        # Fallback to template mode
        return self._generate_template_responses(
            data, intent_column, entity_column, context_column, self.default_templates
        )


def get_response_generation_step_service() -> ResponseGenerationStepService:
    """Get singleton instance"""
    if not hasattr(get_response_generation_step_service, '_instance'):
        get_response_generation_step_service._instance = ResponseGenerationStepService()
    return get_response_generation_step_service._instance
