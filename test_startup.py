"""
Teste mínimo de startup para identificar travamento
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("1. Importando intent_classification_step_service...")
from app.services.intent_classification_step_service import IntentClassificationStepService
print("✅ intent_classification_step_service importado")

print("2. Importando simple_kb_search_service...")
from app.services.simple_kb_search_service import SimpleKnowledgeBaseSearchService
print("✅ simple_kb_search_service importado")

print("3. Importando intent_conditioned_generation_service...")
from app.services.intent_conditioned_generation_service import IntentConditionedGenerationService
print("✅ intent_conditioned_generation_service importado")

print("4. Importando template_response_service...")
from app.services.template_response_service import TemplateResponseService
print("✅ template_response_service importado")

print("5. Criando TemplateResponseService...")
kb_path = Path(__file__).parent.parent / "data" / "pipeline_data" / "astrology_knowledge_massive.csv"
template_service = TemplateResponseService(kb_path=str(kb_path))
print("✅ TemplateResponseService criado")

print("6. Importando astrology_entity_service...")
from app.services.astrology_entity_service import AstrologyEntityService
print("✅ astrology_entity_service importado")

print("7. Criando AstrologyEntityService...")
entity_service = AstrologyEntityService()
print("✅ AstrologyEntityService criado")

print("\n🎉 Todos os serviços inicializados com sucesso!")
