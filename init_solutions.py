"""
Initialize default solutions for organizing pipelines
Run this script to create default solution templates
"""
from app.pipeline.storage import storage
from app.core.logger import logger


def initialize_default_solutions():
    """Create default solutions if they don't exist."""
    
    # Check if solutions already exist
    existing_solutions = storage.list_all_solutions()
    if existing_solutions:
        logger.info(f"✅ Solutions already exist ({len(existing_solutions)} found). Skipping initialization.")
        return
    
    logger.info("🎯 Creating default solutions...")
    
    default_solutions = [
        {
            "name": "🤖 Chatbot Conversacional",
            "description": "Pipeline completo para criar um chatbot com classificação de intenções, geração de respostas e base de conhecimento RAG",
            "icon": "bot",
            "color": "#10b981",
            "category": "ai"
        },
        {
            "name": "📝 Geração de Texto",
            "description": "Treinamento de modelos de linguagem N-gram para geração automática de texto baseado em corpus",
            "icon": "sparkles",
            "color": "#8b5cf6",
            "category": "nlp"
        },
        {
            "name": "💬 Análise de Sentimentos",
            "description": "Pipeline para análise de sentimentos em reviews, comentários e textos com classificação ML",
            "icon": "message-square",
            "color": "#f59e0b",
            "category": "nlp"
        },
        {
            "name": "🔍 Análise Exploratória de Dados",
            "description": "Ferramentas para análise, limpeza, profiling e visualização de datasets",
            "icon": "search",
            "color": "#3b82f6",
            "category": "data"
        },
        {
            "name": "🧠 Machine Learning - Classificação",
            "description": "Pipelines para treinar modelos de classificação (Random Forest, Naive Bayes, etc.)",
            "icon": "brain",
            "color": "#ec4899",
            "category": "ml"
        },
        {
            "name": "📊 Machine Learning - Regressão",
            "description": "Pipelines para treinar modelos de regressão e predição de valores contínuos",
            "icon": "trending-up",
            "color": "#06b6d4",
            "category": "ml"
        },
        {
            "name": "🎯 NLP - Processamento de Linguagem",
            "description": "Ferramentas de NLP: tokenização, stemming, POS tagging, NER, embeddings",
            "icon": "zap",
            "color": "#f97316",
            "category": "nlp"
        },
        {
            "name": "🗄️ ETL & Preparação de Dados",
            "description": "Ingestão, transformação e preparação de dados de múltiplas fontes",
            "icon": "database",
            "color": "#6366f1",
            "category": "data"
        }
    ]
    
    created_count = 0
    for solution_data in default_solutions:
        try:
            solution_id = storage.create_solution(solution_data)
            logger.info(f"✅ Created solution: {solution_data['name']} (ID: {solution_id})")
            created_count += 1
        except Exception as e:
            logger.error(f"❌ Failed to create solution '{solution_data['name']}': {e}")
    
    logger.info(f"\n🎉 Successfully created {created_count}/{len(default_solutions)} default solutions!")
    
    # List all solutions
    solutions = storage.list_all_solutions()
    logger.info("\n📁 Available Solutions:")
    for solution in solutions:
        logger.info(f"  - {solution['name']} ({solution['category']})")
    
    logger.info("\n💡 Next steps:")
    logger.info("  1. Start creating pipelines and assign them to solutions")
    logger.info("  2. Use the Solutions view in the frontend to organize your pipelines")
    logger.info("  3. Each solution can contain multiple related pipelines")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("   FLOWSCOPE - INITIALIZE DEFAULT SOLUTIONS")
    logger.info("=" * 60)
    logger.info("")
    
    initialize_default_solutions()
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("   DONE!")
    logger.info("=" * 60)
