#!/usr/bin/env python
import sys
import json
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')
from app.pipeline.storage import PipelineStorage

# Inicializar storage
storage = PipelineStorage('./data')
solutions_file = Path('./data/solutions.json')

# Carregar soluções existentes
if solutions_file.exists():
    with open(solutions_file, 'r', encoding='utf-8') as f:
        solutions_data = json.load(f)
else:
    solutions_data = {'solutions': []}

# Definir soluções a criar
new_solutions = [
    {
        'name': '🏪 Atendimento Loja RAG',
        'description': 'Sistemas RAG para atendimento automatizado em lojas com base de conhecimento',
        'icon': '🏪',
        'color': 'blue',
        'category': 'ai',
        'keywords': ['rag', 'atendente', 'loja', 'conhecimento']
    },
    {
        'name': '💬 Análise de Sentimentos',
        'description': 'Modelos de análise de sentimento em reviews e comentários',
        'icon': '💬',
        'color': 'purple',
        'category': 'nlp',
        'keywords': ['review', 'sentiment', 'sentimento', 'análise']
    },
    {
        'name': '🤖 Chatbots & Conversação',
        'description': 'Chatbots conversacionais com classificação de intenções e resposta contextual',
        'icon': '🤖',
        'color': 'green',
        'category': 'ai',
        'keywords': ['chatbot', 'space', 'conversação', 'chat']
    },
    {
        'name': '👥 Segmentação de Clientes',
        'description': 'Análise e segmentação de clientes com Machine Learning',
        'icon': '👥',
        'color': 'orange',
        'category': 'ml',
        'keywords': ['customer', 'cliente', 'segment', 'segmentação']
    },
    {
        'name': '🍦 Previsão de Vendas',
        'description': 'Modelos preditivos para vendas de sorvetes baseado em temperatura',
        'icon': '🍦',
        'color': 'pink',
        'category': 'ml',
        'keywords': ['ice cream', 'sorvete', 'vendas', 'previsão']
    },
    {
        'name': '🚗 Seguros & Predição',
        'description': 'Modelos de Machine Learning para seguros e avaliação de risco',
        'icon': '🚗',
        'color': 'indigo',
        'category': 'ml',
        'keywords': ['insurance', 'seguros', 'risco', 'predição']
    },
    {
        'name': '🧪 Experimentos ML',
        'description': 'Experimentos e testes de modelos de Machine Learning (Classificação, Regressão, Clustering)',
        'icon': '🧪',
        'color': 'teal',
        'category': 'ml',
        'keywords': ['classification', 'regression', 'clustering', 'test', 'teste']
    },
    {
        'name': '🗣️ Habla Chatbot',
        'description': 'Chatbot Habla com múltiplos testes de temperatura e treinamento',
        'icon': '🗣️',
        'color': 'yellow',
        'category': 'ai',
        'keywords': ['habla', 'botalk', 'temperatura', 'graus']
    }
]

# Criar soluções
created_solutions = {}
for sol_def in new_solutions:
    solution_id = str(uuid.uuid4())
    solution = {
        'id': solution_id,
        'name': sol_def['name'],
        'description': sol_def['description'],
        'icon': sol_def['icon'],
        'color': sol_def['color'],
        'category': sol_def['category'],
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'pipelines': []
    }
    solutions_data['solutions'].append(solution)
    created_solutions[tuple(sol_def['keywords'])] = solution_id
    print(f'✅ Solução criada: {sol_def["name"]} (ID: {solution_id[:8]}...)')

# Mapear pipelines para soluções
pipes = storage.list_all_pipelines()
no_solution = [p for p in pipes if not p.get('solution_id')]

mapping = {}
for p in no_solution:
    name = p['name'].lower()
    pipeline_id = p['id']
    
    # Encontrar solução apropriada
    matched_solution_id = None
    for keywords, solution_id in created_solutions.items():
        if any(kw in name for kw in keywords):
            matched_solution_id = solution_id
            break
    
    if matched_solution_id:
        mapping[pipeline_id] = matched_solution_id

print(f'\n📊 Mapeamento: {len(mapping)} pipelines de {len(no_solution)} serão associadas a soluções')

# Atualizar configs das pipelines
updated_count = 0
for pipeline_id, solution_id in mapping.items():
    config = storage.get_pipeline_config(pipeline_id)
    if config:
        config['solution_id'] = solution_id
        storage.save_pipeline_config(pipeline_id, config)
        
        # Adicionar pipeline à solução
        for sol in solutions_data['solutions']:
            if sol['id'] == solution_id and pipeline_id not in sol['pipelines']:
                sol['pipelines'].append(pipeline_id)
                sol['updated_at'] = datetime.now().isoformat()
        
        updated_count += 1
        print(f'  ✓ Pipeline {pipeline_id[:8]}... → Solução {solution_id[:8]}...')

# Salvar solutions.json
with open(solutions_file, 'w', encoding='utf-8') as f:
    json.dump(solutions_data, f, indent=2, ensure_ascii=False)

print(f'\n✅ CONCLUÍDO!')
print(f'   - {len(new_solutions)} novas soluções criadas')
print(f'   - {updated_count} pipelines atualizadas')
print(f'   - {len(no_solution) - updated_count} pipelines não mapeadas (ficarão como "Sem Solução")')
