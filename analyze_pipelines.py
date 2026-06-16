#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
from app.pipeline.storage import PipelineStorage
import json

s = PipelineStorage('./data')
pipes = s.list_all_pipelines()

# Pipelines sem solução
no_solution = [p for p in pipes if not p.get('solution_id')]

print(f'📋 Total de pipelines: {len(pipes)}')
print(f'❌ Pipelines sem solução: {len(no_solution)}')
print(f'✅ Pipelines com solução: {len(pipes) - len(no_solution)}')
print()

# Agrupar por padrões
groups = {}
for p in no_solution:
    name = p['name'].lower()
    
    # Detectar categoria pelo nome
    if 'rag' in name or 'conhecimento' in name or 'atendente' in name:
        category = 'rag'
    elif 'space' in name or 'chatbot' in name or 'chat' in name:
        category = 'chatbot'
    elif 'review' in name or 'sentiment' in name or 'sentimento' in name:
        category = 'sentiment'
    elif 'text gen' in name or 'geração' in name or 'ngram' in name:
        category = 'text_generation'
    elif 'insurance' in name or 'seguros' in name:
        category = 'insurance'
    elif 'customer' in name or 'cliente' in name or 'segment' in name:
        category = 'customer'
    elif 'ice cream' in name or 'sorvete' in name:
        category = 'icecream'
    elif 'ml' in name or 'machine learning' in name or 'classifier' in name:
        category = 'ml'
    elif 'nlp' in name or 'processamento' in name:
        category = 'nlp'
    else:
        category = 'other'
    
    if category not in groups:
        groups[category] = []
    groups[category].append(p)

print('📁 Agrupamento sugerido:')
print()
for category, pipelines in sorted(groups.items()):
    print(f'{category.upper()} ({len(pipelines)} pipelines):')
    for p in pipelines:
        print(f'  - {p["name"][:70]} (ID: {p["id"][:8]}...)')
    print()

# Salvar detalhes em JSON
with open('pipeline_groups.json', 'w', encoding='utf-8') as f:
    json.dump({
        'total': len(pipes),
        'no_solution': len(no_solution),
        'groups': {k: [{'id': p['id'], 'name': p['name']} for p in v] for k, v in groups.items()}
    }, f, indent=2, ensure_ascii=False)

print('✅ Detalhes salvos em pipeline_groups.json')
