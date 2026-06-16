"""
Script para associar modelos existentes às soluções baseado em padrões de nome.
"""

import json
from pathlib import Path

# Mapeamento de modelos para soluções baseado em nomes e contexto
MODEL_TO_SOLUTION = {
    # Astrobot solution (c01650b1-ff0c-40a4-ac27-3c2433109a09)
    'astronomy_bot.joblib': 'c01650b1-ff0c-40a4-ac27-3c2433109a09',
    'astronomy_bot_gen_intgen.joblib': 'c01650b1-ff0c-40a4-ac27-3c2433109a09',
    
    # Atendimento Loja RAG solution (0861a2c3-e8a2-470a-93f3-a5e40aaa6a7a)
    'atendente_loja_rag.pkl': '0861a2c3-e8a2-470a-93f3-a5e40aaa6a7a',
    
    # Análise de Sentimentos solution (3d916817-896c-4fbd-b75f-e0752bc29487)
    'review_generator_ngram.pkl': '3d916817-896c-4fbd-b75f-e0752bc29487',
    'review_generator_ngram_ngram.pkl': '3d916817-896c-4fbd-b75f-e0752bc29487',
    
    # Chatbots & Conversação solution (d3a7f6d5-9604-444f-b745-b2ad7e28416b)
    'space_explorer.joblib': 'd3a7f6d5-9604-444f-b745-b2ad7e28416b',
    'space_explorer_gen_intgen.joblib': 'd3a7f6d5-9604-444f-b745-b2ad7e28416b',
    'space_explorer_kb.npz': 'd3a7f6d5-9604-444f-b745-b2ad7e28416b',
    
    # Seguros & Predição solution (procurar ID)
    'car_insurance_predictor.pkl': None,  # Será associado depois
    'car_insurance_predictor_v1.pkl': None,
    
    # Habla Chatbot solution (procurar ID)
    'Habla.pkl': None,
    'Habla_Simple.pkl': None,
    'Habla_metadata.json': None,
    'Habla_Simple_metadata.json': None,
    
    # Outros modelos
    'Climatempo.pkl': None,
    'Terapeuta.pkl': None,
    'Terapeuta_v1.pkl': None,
    'Terapeuta_v2.pkl': None,
    'Terapeuta_v3.pkl': None,
    'test_churn_model.pkl': None,
    'test_churn_model_v1.pkl': None,
}

def main():
    models_dir = Path(__file__).parent / "data" / "models"
    solutions_file = Path(__file__).parent / "data" / "solutions.json"
    
    # Carregar soluções
    with open(solutions_file, 'r', encoding='utf-8') as f:
        solutions_data = json.load(f)
    
    solutions = solutions_data['solutions']
    
    # Criar mapeamento de nome de solução para ID
    solution_name_to_id = {}
    for sol in solutions:
        solution_name_to_id[sol['name'].lower()] = sol['id']
    
    print("Soluções disponíveis:")
    for sol in solutions:
        print(f"  - {sol['name']} ({sol['id']})")
    print()
    
    # Atualizar mapeamento com IDs corretos
    MODEL_TO_SOLUTION['car_insurance_predictor.pkl'] = solution_name_to_id.get('🚗 seguros & predição')
    MODEL_TO_SOLUTION['car_insurance_predictor_v1.pkl'] = solution_name_to_id.get('🚗 seguros & predição')
    MODEL_TO_SOLUTION['Habla.pkl'] = solution_name_to_id.get('🗣️ habla chatbot')
    MODEL_TO_SOLUTION['Habla_Simple.pkl'] = solution_name_to_id.get('🗣️ habla chatbot')
    
    # Processar cada modelo
    updated_count = 0
    for model_file in models_dir.glob("*"):
        if model_file.suffix not in ['.pkl', '.joblib', '.npz']:
            continue
        
        model_name = model_file.name
        solution_id = MODEL_TO_SOLUTION.get(model_name)
        
        if not solution_id:
            print(f"⚠️  Modelo sem solução definida: {model_name}")
            continue
        
        # Carregar ou criar metadata
        metadata_file = models_dir / f"{model_file.stem}_metadata.json"
        metadata = {}
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except:
                pass
        
        # Adicionar solution_id ao metadata
        metadata['solution_id'] = solution_id
        
        # Salvar metadata atualizado
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        solution_name = next((s['name'] for s in solutions if s['id'] == solution_id), 'Unknown')
        print(f"✅ {model_name} → {solution_name}")
        updated_count += 1
    
    print(f"\n✨ {updated_count} modelos associados a soluções!")

if __name__ == '__main__':
    main()
