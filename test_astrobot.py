#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
from app.pipeline.storage import PipelineStorage
import json

s = PipelineStorage('./data')
pipes = s.list_all_pipelines()
astro = [p for p in pipes if 'astronomy' in str(p.get('name', '')).lower() or p.get('solution_id') == 'c01650b1-ff0c-40a4-ac27-3c2433109a09']

print(f'✅ Total de pipelines: {len(pipes)}')
print(f'✅ Pipelines do Astrobot: {len(astro)}')

if astro:
    print('\n📋 Pipelines encontradas:')
    for p in astro:
        print(f"  - {p['name']}")
        print(f"    ID: {p['id']}")
        print(f"    Solution ID: {p.get('solution_id')}")
        print()
else:
    print('❌ Nenhuma pipeline do Astrobot encontrada')
