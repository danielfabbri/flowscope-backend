#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
from app.pipeline.storage import PipelineStorage
import json

s = PipelineStorage('./data')
pipes = s.list_all_pipelines()

# Contar pipelines por status
total = len(pipes)
with_solution = [p for p in pipes if p.get('solution_id')]
without_solution = [p for p in pipes if not p.get('solution_id')]

print('=' * 60)
print('✅ RESUMO FINAL - ORGANIZAÇÃO DE PIPELINES')
print('=' * 60)
print()
print(f'📊 Total de pipelines: {total}')
print(f'✅ Pipelines organizadas: {len(with_solution)} ({len(with_solution)/total*100:.1f}%)')
print(f'❌ Pipelines sem solução: {len(without_solution)} ({len(without_solution)/total*100:.1f}%)')
print()

# Carregar soluções
with open('data/solutions.json', 'r', encoding='utf-8') as f:
    solutions_data = json.load(f)

# Contar soluções com e sem pipelines
solutions_with_pipes = [s for s in solutions_data['solutions'] if len(s['pipelines']) > 0]
solutions_empty = [s for s in solutions_data['solutions'] if len(s['pipelines']) == 0]

print(f'📁 Total de soluções: {len(solutions_data["solutions"])}')
print(f'   - Com pipelines: {len(solutions_with_pipes)}')
print(f'   - Vazias: {len(solutions_empty)}')
print()

print('📋 SOLUÇÕES ATIVAS (com pipelines):')
print('=' * 60)
for s in sorted(solutions_with_pipes, key=lambda x: len(x['pipelines']), reverse=True):
    print(f'  {s["icon"]} {s["name"]}: {len(s["pipelines"])} pipelines')
print()

if without_solution:
    print('⚠️  PIPELINES SEM SOLUÇÃO:')
    print('=' * 60)
    for p in without_solution:
        print(f'  - {p["name"][:60]} (ID: {p["id"][:8]}...)')
    print()
else:
    print('🎉 SUCESSO! Todas as pipelines estão organizadas em soluções!')
    print()

print('=' * 60)
print('✅ Organização concluída com sucesso!')
print('=' * 60)
