"""
Script para criar e adicionar as pipelines do Astronomy Bot à solução Astrobot.

Este script cria as 3 pipelines principais usadas no astronomy_bot_chatbot:
1. Treinar Classificador de Intenções - Astronomia
2. Indexar Base de Conhecimento - Astronomia  
3. Treinar Gerador de Respostas - Astronomia (5-grams)
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

# Paths
SOLUTIONS_FILE = Path(__file__).parent / "data" / "solutions.json"
PIPELINES_DIR = Path(__file__).parent.parent / "data" / "pipelines"

def load_solutions():
    """Load solutions from JSON file."""
    if SOLUTIONS_FILE.exists():
        with open(SOLUTIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"solutions": [], "version": "1.0", "last_updated": datetime.now().isoformat()}

def save_solutions(data):
    """Save solutions to JSON file."""
    data["last_updated"] = datetime.now().isoformat()
    with open(SOLUTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def find_solution_by_name(solutions_data, name):
    """Find solution by name."""
    for solution in solutions_data["solutions"]:
        if solution["name"].lower() == name.lower():
            return solution
    return None

def create_pipeline_config(pipeline_id, name, description, steps, solution_id=None):
    """Create pipeline configuration."""
    config = {
        "name": name,
        "description": description,
        "steps": steps
    }
    
    # Add solution_id if provided
    if solution_id:
        config["solution_id"] = solution_id
    
    status = {
        "pipeline_id": pipeline_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "config": config,
        "execution_history": []
    }
    
    return config, status

def save_pipeline(pipeline_id, config, status):
    """Save pipeline config and status files."""
    PIPELINES_DIR.mkdir(parents=True, exist_ok=True)
    
    config_file = PIPELINES_DIR / f"{pipeline_id}_config.json"
    status_file = PIPELINES_DIR / f"{pipeline_id}_status.json"
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    with open(status_file, 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2, ensure_ascii=False)

def add_pipeline_to_solution(solutions_data, solution_id, pipeline_id):
    """Add pipeline to solution."""
    for solution in solutions_data["solutions"]:
        if solution["id"] == solution_id:
            if pipeline_id not in solution["pipelines"]:
                solution["pipelines"].append(pipeline_id)
                solution["updated_at"] = datetime.now().isoformat()
            break

def main():
    print("=" * 80)
    print("🌟 ASTRONOMY BOT - Criando Pipelines na Solução Astrobot")
    print("=" * 80)
    
    # Load solutions
    solutions_data = load_solutions()
    
    # Find Astrobot solution
    astrobot = find_solution_by_name(solutions_data, "Astrobot")
    if not astrobot:
        print("❌ Solução 'Astrobot' não encontrada!")
        print("   Crie a solução primeiro via interface.")
        return
    
    print(f"\n✅ Solução encontrada: {astrobot['name']}")
    print(f"   ID: {astrobot['id']}")
    print(f"   Pipelines atuais: {len(astrobot['pipelines'])}")
    
    # Pipeline 1: Intent Classifier Training
    print("\n📊 Criando Pipeline 1: Treinar Classificador de Intenções...")
    pipeline1_id = str(uuid.uuid4())
    pipeline1_steps = [
        {
            "name": "Ingestão de Dados",
            "type": "ingestion",
            "enabled": True,
            "config": {
                "ingestion_type": "generated",
                "data_template": "custom",
                "num_rows": 85,
                "description": "85 perguntas anotadas com intenções: question_blackhole, question_cosmos, question_planet, question_star, question_galaxy"
            }
        },
        {
            "name": "Preparação de Dados",
            "type": "data_preparation",
            "enabled": True,
            "config": {
                "train_test_split": True,
                "test_size": 0.2,
                "random_state": 42,
                "handle_missing": "drop"
            }
        },
        {
            "name": "Treinamento ML - Intent Classifier",
            "type": "ml_training",
            "enabled": True,
            "config": {
                "ml_task": "classification",
                "algorithm": "naive_bayes",
                "target_column": "intent",
                "text_column": "text",
                "use_tfidf": True,
                "max_features": 5000
            }
        },
        {
            "name": "Salvar Modelo - astronomy_intent_classifier",
            "type": "model_persistence",
            "enabled": True,
            "config": {
                "operation": "save",
                "model_name": "astronomy_intent_classifier",
                "format": "joblib"
            }
        }
    ]
    
    config1, status1 = create_pipeline_config(
        pipeline1_id,
        "🧠 Treinar Classificador de Intenções - Astronomia",
        "Treina modelo Naive Bayes para classificar perguntas sobre astronomia em 5 intenções: buraco negro, cosmos, planetas, estrelas e galáxias (85 exemplos robustos)",
        pipeline1_steps,
        astrobot['id']  # Add solution_id
    )
    save_pipeline(pipeline1_id, config1, status1)
    add_pipeline_to_solution(solutions_data, astrobot['id'], pipeline1_id)
    print(f"   ✅ Criada: {config1['name']}")
    print(f"      ID: {pipeline1_id}")
    
    # Pipeline 2: Knowledge Base RAG Indexing
    print("\n📚 Criando Pipeline 2: Indexar Base de Conhecimento...")
    pipeline2_id = str(uuid.uuid4())
    pipeline2_steps = [
        {
            "name": "Ingestão de Documentos",
            "type": "ingestion",
            "enabled": True,
            "config": {
                "ingestion_type": "file_upload",
                "file_format": "txt",
                "description": "Arquivos de texto sobre astronomia, física espacial, cosmologia"
            }
        },
        {
            "name": "Pré-processamento de Texto",
            "type": "text_preprocessing",
            "enabled": True,
            "config": {
                "lowercase": True,
                "remove_punctuation": False,
                "remove_stopwords": True,
                "language": "portuguese"
            }
        },
        {
            "name": "Chunking de Texto",
            "type": "text_chunking",
            "enabled": True,
            "config": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "method": "recursive"
            }
        },
        {
            "name": "Gerar Embeddings",
            "type": "generate_embeddings",
            "enabled": True,
            "config": {
                "embedding_model": "tfidf",
                "max_features": 1000
            }
        },
        {
            "name": "Criar Índice RAG",
            "type": "rag_index",
            "enabled": True,
            "config": {
                "index_name": "astronomy_knowledge_base",
                "similarity_metric": "cosine"
            }
        }
    ]
    
    config2, status2 = create_pipeline_config(
        pipeline2_id,
        "📖 Indexar Base de Conhecimento - Astronomia",
        "Processa documentos sobre astronomia, cria chunks semânticos e indexa para busca RAG (Retrieval Augmented Generation) usando TF-IDF embeddings",
        pipeline2_steps,
        astrobot['id']  # Add solution_id
    )
    save_pipeline(pipeline2_id, config2, status2)
    add_pipeline_to_solution(solutions_data, astrobot['id'], pipeline2_id)
    print(f"   ✅ Criada: {config2['name']}")
    print(f"      ID: {pipeline2_id}")
    
    # Pipeline 3: Response Generator Training (5-grams)
    print("\n💬 Criando Pipeline 3: Treinar Gerador de Respostas...")
    pipeline3_id = str(uuid.uuid4())
    pipeline3_steps = [
        {
            "name": "Ingestão de Respostas Exemplo",
            "type": "ingestion",
            "enabled": True,
            "config": {
                "ingestion_type": "generated",
                "data_template": "custom",
                "description": "Corpus de respostas exemplo para cada intenção (blackhole, cosmos, planet, star, galaxy)"
            }
        },
        {
            "name": "Treinamento N-gram - 5-grams",
            "type": "ngram_training",
            "enabled": True,
            "config": {
                "text_column": "response_text",
                "intent_column": "intent",
                "n": 5,
                "enable_kneser_ney": True,
                "smoothing_discount": 0.75,
                "min_frequency": 1
            }
        },
        {
            "name": "Geração de Respostas",
            "type": "ngram_generation",
            "enabled": True,
            "config": {
                "max_length": 150,
                "temperature": 0.8,
                "enable_grammar_postprocessing": True,
                "language": "portuguese"
            }
        },
        {
            "name": "Salvar Modelo - astronomy_response_generator",
            "type": "model_persistence",
            "enabled": True,
            "config": {
                "operation": "save",
                "model_name": "astronomy_bot_chatbot",
                "format": "pickle"
            }
        }
    ]
    
    config3, status3 = create_pipeline_config(
        pipeline3_id,
        "✨ Treinar Gerador de Respostas - Astronomia (5-grams)",
        "Treina modelo de linguagem 5-gram com Kneser-Ney smoothing e pós-processamento de gramática portuguesa para gerar respostas naturais condicionadas por intenção",
        pipeline3_steps,
        astrobot['id']  # Add solution_id
    )
    save_pipeline(pipeline3_id, config3, status3)
    add_pipeline_to_solution(solutions_data, astrobot['id'], pipeline3_id)
    print(f"   ✅ Criada: {config3['name']}")
    print(f"      ID: {pipeline3_id}")
    
    # Save updated solutions
    save_solutions(solutions_data)
    
    print("\n" + "=" * 80)
    print("✅ CONCLUÍDO! Pipelines adicionadas à solução Astrobot")
    print("=" * 80)
    print(f"\n📊 Total de Pipelines na Solução: {len(astrobot['pipelines']) + 3}")
    print("\nPipelines criadas:")
    print(f"  1. {config1['name']}")
    print(f"  2. {config2['name']}")
    print(f"  3. {config3['name']}")
    print("\n🌐 Acesse: http://localhost:5178/solutions")
    print("   → Click em 'Astrobot' para ver todas as pipelines!")
    print("\n💡 Essas 3 pipelines representam o processo completo usado")
    print("   para treinar o modelo astronomy_bot_chatbot com:")
    print("   • Intent Classification (Naive Bayes)")
    print("   • RAG Knowledge Base (TF-IDF)")
    print("   • 5-gram Response Generator (Kneser-Ney + Grammar)")

if __name__ == "__main__":
    main()
