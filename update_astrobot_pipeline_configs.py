"""
Script para adicionar solution_id aos configs das pipelines existentes do Astrobot.
"""

import json
from pathlib import Path

# Paths
SOLUTIONS_FILE = Path(__file__).parent / "data" / "solutions.json"
PIPELINES_DIR = Path(__file__).parent.parent / "data" / "pipelines"

def load_solutions():
    """Load solutions from JSON file."""
    if SOLUTIONS_FILE.exists():
        with open(SOLUTIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"solutions": [], "version": "1.0"}

def update_pipeline_config(pipeline_id, solution_id):
    """Add solution_id to pipeline config."""
    config_file = PIPELINES_DIR / f"{pipeline_id}_config.json"
    
    if not config_file.exists():
        print(f"   ⚠️  Config não encontrado: {pipeline_id}")
        return False
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Add solution_id
        config['solution_id'] = solution_id
        
        # Save updated config
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"   ❌ Erro ao atualizar {pipeline_id}: {e}")
        return False

def main():
    print("=" * 80)
    print("🔧 ATUALIZAR CONFIGS - Adicionar solution_id às Pipelines do Astrobot")
    print("=" * 80)
    
    # Load solutions
    solutions_data = load_solutions()
    
    # Find Astrobot solution
    astrobot = None
    for solution in solutions_data["solutions"]:
        if solution["name"].lower() == "astrobot":
            astrobot = solution
            break
    
    if not astrobot:
        print("❌ Solução 'Astrobot' não encontrada!")
        return
    
    print(f"\n✅ Solução encontrada: {astrobot['name']}")
    print(f"   ID: {astrobot['id']}")
    print(f"   Pipelines no array: {len(astrobot.get('pipelines', []))}")
    
    # Update each pipeline config
    success_count = 0
    for pipeline_id in astrobot.get('pipelines', []):
        print(f"\n🔄 Atualizando pipeline: {pipeline_id}")
        if update_pipeline_config(pipeline_id, astrobot['id']):
            print(f"   ✅ Config atualizado com solution_id")
            success_count += 1
        else:
            print(f"   ❌ Falhou")
    
    print("\n" + "=" * 80)
    print(f"✅ CONCLUÍDO! {success_count}/{len(astrobot.get('pipelines', []))} pipelines atualizadas")
    print("=" * 80)
    print("\n🌐 Recarregue: http://localhost:5178/solutions")
    print("   → Click em 'Astrobot' para ver as pipelines!")

if __name__ == "__main__":
    main()
