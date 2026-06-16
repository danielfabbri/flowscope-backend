"""
Serviço para treinar e salvar modelos conversacionais completos.
Um modelo conversacional encapsula: intent classifier + knowledge base + response logic.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from app.services.intent_classification_step_service import IntentClassificationStepService
from app.services.simple_kb_search_service import simple_kb_search_service
from app.services.intent_conditioned_generation_service import IntentConditionedGenerationService
from app.services.template_response_service import TemplateResponseService
from app.services.astrology_entity_service import AstrologyEntityService
from app.services.seq2seq_generation_service import Seq2SeqGenerationService


class ConversationalModelService:
    """Gerencia modelos conversacionais completos."""
    
    def __init__(self):
        backend_dir = Path(__file__).parent.parent.parent
        self.models_dir = backend_dir / "data" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Services para os componentes internos
        self.intent_service = IntentClassificationStepService()
        self.search_service = simple_kb_search_service  # CHANGED: Using simple TF-IDF search
        self.generation_service = IntentConditionedGenerationService()
        
        # NEW: Template-based generation with entity extraction
        # KB is in project root, not backend folder
        project_root = backend_dir.parent
        kb_path = project_root / "data" / "pipeline_data" / "astrology_knowledge_massive.csv"
        self.template_service = TemplateResponseService(kb_path=str(kb_path))
        self.entity_service = AstrologyEntityService()
        
        # NEW: Level 2 - LSTM Seq2Seq generation (LAZY LOADING)
        self.seq2seq_service = None  # Will be loaded on-demand when first used
    
    def save_conversational_model(
        self,
        model_name: str,
        intent_model_name: str,
        kb_name: str,
        generation_model_name: Optional[str] = None,
        min_confidence: float = 0.4,
        top_k: int = 3,
        description: str = "",
        default_responses: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Salva um modelo conversacional completo.
        
        Args:
            model_name: Nome do modelo conversacional (ex: "space_explorer_chat")
            intent_model_name: Nome do modelo de classificação de intenção (ex: "space_explorer")
            kb_name: Nome da knowledge base (ex: "space_explorer_kb")
            generation_model_name: Nome do modelo de geração de texto (opcional)
            min_confidence: Confiança mínima para classificação
            top_k: Número de documentos para busca semântica
            description: Descrição do modelo
            default_responses: Respostas padrão por intenção
        """
        if default_responses is None:
            default_responses = {}
        
        # Configuração do modelo conversacional
        config = {
            "model_name": model_name,
            "model_type": "conversational_chatbot",
            "created_at": datetime.now().isoformat(),
            "description": description,
            "components": {
                "intent_classifier": {
                    "model_name": intent_model_name,
                    "min_confidence": min_confidence
                },
                "knowledge_base": {
                    "kb_name": kb_name,
                    "top_k": top_k
                },
                "text_generation": {
                    "model_name": generation_model_name,
                    "enabled": generation_model_name is not None
                }
            },
            "default_responses": default_responses,
            "features": ["message"],  # Input esperado
            "target": "response"  # Output
        }
        
        # Salvar configuração
        config_path = self.models_dir / f"{model_name}_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "model_name": model_name,
            "model_type": "conversational_chatbot",
            "config_path": str(config_path),
            "components": {
                "intent_classifier": intent_model_name,
                "knowledge_base": kb_name,
                "text_generation": generation_model_name
            }
        }
    
    def load_conversational_model(self, model_name: str) -> Dict[str, Any]:
        """Carrega a configuração de um modelo conversacional."""
        config_path = self.models_dir / f"{model_name}_config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Modelo conversacional '{model_name}' não encontrado")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return config
    
    def chat(
        self,
        model_name: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Processa uma mensagem usando o modelo conversacional.
        
        Args:
            model_name: Nome do modelo conversacional
            message: Mensagem do usuário
            context: Contexto adicional (opcional)
        
        Returns:
            Dict com resposta, intenção classificada e documentos relevantes
        """
        # Carregar configuração
        config = self.load_conversational_model(model_name)
        
        intent_component = config["components"]["intent_classifier"]
        kb_component = config["components"]["knowledge_base"]
        generation_component = config["components"].get("text_generation", {})
        
        # Load generation model if available
        generation_model_name = generation_component.get("model_name")
        print(f"[DEBUG-CHAT] generation_component: {generation_component}")
        print(f"[DEBUG-CHAT] generation_model_name: {generation_model_name}")
        print(f"[DEBUG-CHAT] enabled: {generation_component.get('enabled', False)}")
        
        # Special handling for LSTM models (already loaded in __init__)
        if generation_model_name == "seq2seq_lstm":
            print(f"[DEBUG-CHAT] Using Level 2 LSTM Seq2Seq (pre-loaded in service)")
            # LSTM model is already loaded in self.seq2seq_service, don't try to load again
        elif generation_model_name and generation_component.get("enabled", False):
            try:
                print(f"[DEBUG-CHAT] Loading N-gram generation model: {generation_model_name}")
                self.generation_service.load_model(generation_model_name)
                print(f"[DEBUG-CHAT] N-gram generation model loaded successfully!")
            except Exception as e:
                print(f"[WARNING] Could not load generation model '{generation_model_name}': {e}")
                import traceback
                traceback.print_exc()
                generation_model_name = None
        else:
            print(f"[DEBUG-CHAT] Skipping generation model (not enabled or name missing)")
            generation_model_name = None
        
        # 1. Classificar intenção
        import pandas as pd
        data = pd.DataFrame([{"message": message}])
        
        intent_config = {
            "text_column": "message",
            "model_name": intent_component["model_name"],
            "min_confidence": intent_component["min_confidence"]
        }
        
        print(f"[DEBUG] Calling intent_service.execute() with:")
        print(f"  - data shape: {data.shape}")
        print(f"  - config: {intent_config}")
        print(f"  - method signature: {self.intent_service.execute.__code__.co_varnames[:3]}")
        
        intent_result = self.intent_service.execute(
            data=data,
            config=intent_config
        )
        
        predicted_intent = intent_result["intent"].iloc[0] if len(intent_result) > 0 else "unknown"
        confidence = intent_result["intent_confidence"].iloc[0] if len(intent_result) > 0 else 0.0
        
        # 2. Buscar na knowledge base
        search_config = {
            "kb_name": kb_component["kb_name"],
            "query_column": "message",
            "top_k": kb_component["top_k"]
        }
        
        search_result = self.search_service.execute(
            data=data,
            config=search_config
        )
        
        search_results = search_result["search_results"].iloc[0] if len(search_result) > 0 else []
        
        # 3. Gerar resposta
        response = self._generate_response(
            intent=predicted_intent,
            confidence=confidence,
            search_results=search_results,
            default_responses=config.get("default_responses", {}),
            message=message,
            generation_model_name=generation_model_name,
            config=config
        )
        
        return {
            "response": response,
            "intent": predicted_intent,
            "confidence": float(confidence),
            "context_documents": search_results[:3] if search_results else [],
            "model_name": model_name
        }
    
    def _generate_response(
        self,
        intent: str,
        confidence: float,
        search_results: list,
        default_responses: Dict[str, str],
        message: str,
        generation_model_name: Optional[str] = None,
        config: Optional[Dict] = None
    ) -> str:
        """Gera resposta baseada na intenção e contexto (TEMPLATE-FIRST strategy)."""
        
        print(f"[DEBUG-RESPONSE] Intent: {intent}, Confidence: {confidence:.2%}, Has KB results: {len(search_results) > 0 if search_results else False}")
        print(f"[DEBUG-RESPONSE] Generation model: {generation_model_name}")
        
        # Extract entities first (used in multiple priorities)
        entities = self.entity_service.extract(message)
        print(f"[DEBUG-RESPONSE] Extracted entities: zodiac={entities.zodiac_sign}, topic={entities.topic}, date={entities.date}")
        
        # PRIORITY -1: Check for general knowledge questions (no specific entity)
        if config and "general_knowledge" in config:
            general_response = self._check_general_knowledge(message, config["general_knowledge"])
            if general_response:
                print(f"[DEBUG-RESPONSE] ✅ General knowledge response: {general_response[:80]}...")
                return general_response
        
        # PRIORITY 0: Try Level 2 LSTM Seq2Seq generation FIRST (if configured)
        if generation_model_name == "seq2seq_lstm" and entities and entities.has_zodiac():
            try:
                print(f"[DEBUG-RESPONSE] 🧠 Attempting Level 2 LSTM Seq2Seq generation...")
                
                # Get structured facts from template_service KB
                facts = {}
                signo = entities.zodiac_sign
                topic = entities.topic or "caracteristicas"
                
                # Debug: Check KB availability
                print(f"[DEBUG-KB] template_service exists: {self.template_service is not None}")
                print(f"[DEBUG-KB] has kb_data attr: {hasattr(self.template_service, 'kb_data') if self.template_service else False}")
                if self.template_service and hasattr(self.template_service, 'kb_data'):
                    print(f"[DEBUG-KB] kb_data keys (signos): {list(self.template_service.kb_data.keys())}")
                    if signo in self.template_service.kb_data:
                        print(f"[DEBUG-KB] {signo} topics: {list(self.template_service.kb_data[signo].keys())}")
                        if topic in self.template_service.kb_data[signo]:
                            facts = self.template_service.kb_data[signo][topic].copy()
                            print(f"[DEBUG-RESPONSE] ✅ Loaded {len(facts)} facts from KB: {facts}")
                        else:
                            print(f"[DEBUG-KB] ⚠️ Topic '{topic}' not found in {signo} data")
                    else:
                        print(f"[DEBUG-KB] ⚠️ Signo '{signo}' not found in KB")
                
                # Fallback: if no structured facts, create generic placeholder
                if not facts:
                    print(f"[DEBUG-RESPONSE] ⚠️ No KB data for {signo}/{topic}, using fallback")
                    facts = {topic: "dados não encontrados"}
                
                print(f"[DEBUG-RESPONSE] LSTM input: signo={entities.zodiac_sign}, topic={topic}, facts={facts}")
                
                # Lazy load LSTM models on first use
                if self.seq2seq_service is None:
                    print("[INFO] 🔄 Lazy loading Level 2 LSTM Seq2Seq models...")
                    self.seq2seq_service = Seq2SeqGenerationService()
                    self.seq2seq_service.load_models()
                    print("[INFO] ✅ Level 2 LSTM Seq2Seq models loaded successfully")
                
                # Get generation config
                gen_config = config.get("generation_config", {})
                temperature = gen_config.get("temperature", 0.8)
                
                # Generate using LSTM
                lstm_response = self.seq2seq_service.generate_from_facts(
                    signo=entities.zodiac_sign,
                    topic=entities.topic or "caracteristicas",
                    facts=facts,
                    temperature=temperature
                )
                
                if lstm_response and len(lstm_response) > 10:
                    print(f"[DEBUG-RESPONSE] ✅ Level 2 LSTM response: {lstm_response[:100]}...")
                    # Add context that this is experimental
                    return f"{lstm_response}\n\n🧠 _Resposta gerada por IA neural (Level 2 - experimental)_"
                else:
                    print(f"[DEBUG-RESPONSE] ⚠️ LSTM generated text too short or None")
            except Exception as e:
                print(f"[DEBUG-RESPONSE] ❌ Level 2 LSTM generation failed: {e}")
                import traceback
                traceback.print_exc()
        
        # PRIORITY 1: Try template-based generation as fallback (most accurate)
        try:
            if entities.has_zodiac() and entities.has_topic():
                template_response = self.template_service.generate(intent, message, entities)
                if template_response:
                    print(f"[DEBUG-RESPONSE] ✅ Template response (fallback): {template_response[:100]}...")
                    return template_response
                else:
                    print(f"[DEBUG-RESPONSE] ⚠️ Template service returned None")
            else:
                print(f"[DEBUG-RESPONSE] ⚠️ Missing entities for template generation")
        except Exception as e:
            print(f"[DEBUG-RESPONSE] ❌ Template generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # PRIORITY 2: TRY N-GRAM GENERATIVE MODEL (fallback - only if has context)
        # Only use n-gram if we have a specific zodiac sign (to avoid nonsense)
        if generation_model_name and entities and entities.has_zodiac():
            try:
                # Extract context from search results
                context_docs = []
                if search_results:
                    for doc in search_results[:3]:
                        if isinstance(doc, dict):
                            if "answer" in doc or "resposta" in doc:
                                context_docs.append(doc.get("answer", doc.get("resposta", "")))
                            elif "content" in doc:
                                context_docs.append(doc["content"][:200])
                
                print(f"[DEBUG-RESPONSE] Checking if has model for intent '{intent}'...")
                # Attempt generation
                if self.generation_service.has_model_for_intent(intent):
                    print(f"[DEBUG-RESPONSE] ✅ Has model! Generating response...")
                    generated = self.generation_service.generate_response(
                        intent=intent,
                        seed=None,
                        max_length=50,
                        temperature=0.9,  # Increased for more variety
                        context_docs=context_docs
                    )
                    
                    if generated and len(generated) > 10:
                        print(f"[DEBUG-RESPONSE] ✅ Generated response: {generated[:100]}...")
                        return generated
                    else:
                        print(f"[DEBUG-RESPONSE] ⚠️ Generated text too short: {generated}")
                else:
                    print(f"[DEBUG-RESPONSE] ❌ No generation model for intent: {intent}")
            except Exception as e:
                print(f"[DEBUG-RESPONSE] ❌ Generation failed: {e}")
                import traceback
                traceback.print_exc()
        elif not entities or not entities.has_zodiac():
            print(f"[DEBUG-RESPONSE] ⚠️ Skipping n-gram generation (no specific zodiac sign)")
        else:
            print(f"[DEBUG-RESPONSE] ⚠️ No generation model name configured")
        
        # PRIORITY 3: Default responses for greetings/thanks (exact match needed)
        if intent in default_responses:
            responses = default_responses[intent]
            if isinstance(responses, list):
                import random
                selected = random.choice(responses)
                print(f"[DEBUG-RESPONSE] Using default response for intent '{intent}': {selected}")
                return selected
            print(f"[DEBUG-RESPONSE] Using default response for intent '{intent}': {responses}")
            return responses
        
        # PRIORITY 4: Use KB documents as fallback
        if search_results and len(search_results) > 0:
            top_doc = search_results[0]
            
            if isinstance(top_doc, dict):
                score = top_doc.get("score", 0)
                
                # KB com formato Q&A (pergunta/resposta)
                if "answer" in top_doc or "resposta" in top_doc:
                    response_text = top_doc.get("answer", top_doc.get("resposta", ""))
                    
                    print(f"[DEBUG-RESPONSE] Using KB response as fallback (score: {score:.3f}): {response_text[:80]}...")
                    
                    # Se confiança alta E score alto, retorna direto
                    if confidence >= 0.5 and score > 0.15:
                        return response_text
                    
                    # Se confiança média ou score médio, adiciona contexto
                    if confidence >= 0.3 and score > 0.1:
                        return f"{response_text}\n\n💡 _Essa é minha melhor resposta com {confidence*100:.1f}% de confiança. Se quiser mais informações, pergunte de outra forma!_"
                
                # KB com formato documento genérico
                elif "content" in top_doc:
                    context = top_doc["content"][:400]
                    return f"Baseado no que sei: {context}"
        
        # PRIORITY 5: Generic low-confidence message
        if confidence < 0.5:
            return f"Hmm, não tenho muita certeza sobre '{message}'. Pode reformular a pergunta?"
        
        # LAST RESORT: Acknowledge intent
        return f"Entendi que você quer saber sobre {intent}."
    
    def _check_general_knowledge(self, message: str, general_knowledge: Dict[str, List[str]]) -> Optional[str]:
        """
        Verifica se a pergunta é sobre conhecimento geral de astrologia (sem entidade específica)
        
        Args:
            message: Mensagem do usuário
            general_knowledge: Dicionário com respostas gerais
            
        Returns:
            Resposta apropriada ou None se não for pergunta geral
        """
        import random
        
        message_lower = message.lower()
        
        # Patterns para perguntas gerais
        patterns = {
            'o_que_e_astrologia': [
                'o que é astrologia', 'o que e astrologia', 'explique astrologia',
                'me explique astrologia', 'o que você sabe sobre astrologia',
                'fale sobre astrologia', 'defina astrologia'
            ],
            'quantos_signos': [
                'quantos signos', 'quantos são os signos', 'quais são os signos',
                'quais os signos', 'liste os signos', 'todos os signos'
            ],
            'o_que_e_ascendente': [
                'o que é ascendente', 'o que e ascendente', 'explique ascendente',
                'o que é o ascendente', 'significado de ascendente', 'para que serve ascendente'
            ],
            'o_que_sao_signos': [
                'o que são signos', 'o que sao signos', 'o que são os signos',
                'explique os signos', 'me explique signos', 'significado de signos'
            ],
            'o_que_e_mapa_astral': [
                'o que é mapa astral', 'o que e mapa astral', 'explique mapa astral',
                'o que é carta natal', 'significado de mapa astral'
            ]
        }
        
        # Verificar se algum pattern match
        for key, patterns_list in patterns.items():
            for pattern in patterns_list:
                if pattern in message_lower:
                    if key in general_knowledge:
                        responses = general_knowledge[key]
                        return random.choice(responses) if isinstance(responses, list) else responses
        
        return None
    
    def list_conversational_models(self) -> list:
        """Lista todos os modelos conversacionais salvos."""
        models = []
        
        for config_file in self.models_dir.glob("*_config.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                if config.get("model_type") == "conversational_chatbot":
                    models.append({
                        "name": config["model_name"],
                        "type": config["model_type"],
                        "description": config.get("description", ""),
                        "created_at": config.get("created_at", ""),
                        "components": config.get("components", {})
                    })
            except Exception as e:
                print(f"Erro ao ler {config_file}: {e}")
                continue
        
        return models
