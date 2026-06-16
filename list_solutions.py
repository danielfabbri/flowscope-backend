#!/usr/bin/env python
import json

with open('data/solutions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'✅ Total de soluções: {len(data["solutions"])}')
print()
for s in data['solutions']:
    print(f'  📁 {s["name"]} ({len(s["pipelines"])} pipelines)')
