"""API routes for model management."""
# Updated: Support for .pkl and .joblib files
from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
import joblib
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel

router = APIRouter()

# Use absolute path relative to backend directory
_backend_dir = Path(__file__).parent.parent.parent
MODELS_DIR = _backend_dir / "data" / "models"


class PredictionRequest(BaseModel):
    """Request body for model prediction."""
    inputs: Dict[str, Any]  # Feature name -> value mapping


class TextPredictionRequest(BaseModel):
    """Request body for text-based prediction."""
    text: str  # Raw text to analyze


@router.get("/models")
async def list_models() -> Dict[str, Any]:
    """List all saved models with their metadata."""
    try:
        if not MODELS_DIR.exists():
            return {"models": [], "count": 0}
        
        models = []
        
        # Find all .pkl and .joblib files
        import itertools
        model_files = itertools.chain(MODELS_DIR.glob("*.pkl"), MODELS_DIR.glob("*.joblib"))
        
        for model_file in model_files:
            model_name = model_file.stem
            metadata_file = MODELS_DIR / f"{model_name}_metadata.json"
            
            model_info = {
                "name": model_name,
                "file_path": str(model_file),
                "created_at": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                "size_bytes": model_file.stat().st_size,
            }
            
            # Detect special model types by naming convention
            if model_name.endswith("_rag"):
                model_info.update({
                    "type": "rag_qa",
                    "algorithm": "TF-IDF + Cosine Similarity",
                    "target": "Q&A Retrieval",
                    "features": ["question", "top_k", "min_similarity"],
                    "n_features": 3,
                    "category": "language_model"
                })
            elif model_name.endswith("_ngram") or "_ngram" in model_name:
                model_info.update({
                    "type": "ngram_text_generation",
                    "algorithm": "N-gram Language Model",
                    "target": "Text Generation",
                    "features": ["prompt", "max_length", "temperature", "top_k"],
                    "n_features": 4,
                    "category": "language_model"
                })
            else:
                # Default to intent classification for .joblib files
                model_info.update({
                    "type": "intent_classification",
                    "algorithm": "Naive Bayes + TF-IDF",
                    "target": "Intent Classification",
                    "features": ["text"],
                    "n_features": 1,
                    "category": "language_model"
                })
            
            # Load metadata if exists (may override above)
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        model_info.update({
                            "type": metadata.get("model_type", model_info.get("type", "unknown")),
                            "algorithm": metadata.get("algorithm", model_info.get("algorithm", "unknown")),
                            "target": metadata.get("target_column", model_info.get("target", "unknown")),
                            "features": metadata.get("feature_columns", model_info.get("features", [])),
                            "n_features": metadata.get("n_features", model_info.get("n_features", 0)),
                            "r2_score": metadata.get("metrics", {}).get("r2_score"),
                            "trained_at": metadata.get("training_info", {}).get("trained_at"),
                            "total_rows": metadata.get("training_info", {}).get("total_rows"),
                            "category": metadata.get("category", model_info.get("category", "ml_model"))
                        })
                except Exception as e:
                    print(f"Error loading metadata for {model_name}: {e}")
            
            models.append(model_info)
        
        # Incluir modelos conversacionais (_config.json)
        for config_file in MODELS_DIR.glob("*_config.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                if config.get("model_type") == "conversational_chatbot":
                    model_info = {
                        "name": config["model_name"],
                        "type": "conversational_chatbot",
                        "algorithm": "Intent Classification + Semantic Search + Response Generation",
                        "target": "Conversational Response",
                        "features": config.get("features", ["message"]),
                        "n_features": len(config.get("features", ["message"])),
                        "category": "chatbot",
                        "created_at": config.get("created_at", ""),
                        "size_bytes": config_file.stat().st_size,
                        "file_path": str(config_file),
                        "description": config.get("description", ""),
                        "components": config.get("components", {})
                    }
                    models.append(model_info)
            except Exception as e:
                print(f"Error loading conversational config {config_file}: {e}")
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return {
            "models": models,
            "count": len(models)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/models/{model_name}")
async def get_model_details(model_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    try:
        model_file = MODELS_DIR / f"{model_name}.pkl"
        metadata_file = MODELS_DIR / f"{model_name}_metadata.json"
        
        if not model_file.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        model_info = {
            "name": model_name,
            "file_path": str(model_file),
            "created_at": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
            "size_bytes": model_file.stat().st_size,
        }
        
        # Load full metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                model_info["metadata"] = metadata
        else:
            model_info["metadata"] = None
        
        return model_info
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model details: {str(e)}")


@router.delete("/models/{model_name}")
async def delete_model(model_name: str) -> Dict[str, str]:
    """Delete a model and its metadata."""
    try:
        model_file = MODELS_DIR / f"{model_name}.pkl"
        metadata_file = MODELS_DIR / f"{model_name}_metadata.json"
        
        if not model_file.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        # Delete files
        model_file.unlink()
        if metadata_file.exists():
            metadata_file.unlink()
        
        return {
            "message": f"Model '{model_name}' deleted successfully",
            "deleted_files": [str(model_file), str(metadata_file) if metadata_file.exists() else None]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@router.get("/models/{model_name}/schema")
async def get_model_schema(model_name: str) -> Dict[str, Any]:
    """Get the expected input schema (features) for a model."""
    try:
        # Check if this is a conversational chatbot model (_config.json)
        config_file = MODELS_DIR / f"{model_name}_config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if config.get("model_type") == "conversational_chatbot":
                return {
                    "model_name": model_name,
                    "model_type": "conversational_chatbot",
                    "description": config.get("description", ""),
                    "features": config.get("features", ["message"]),
                    "target": config.get("target", "response"),
                    "input_type": "text",
                    "components": config.get("components", {}),
                    "parameters": {
                        "message": {
                            "type": "string",
                            "description": "Mensagem do usuário"
                        }
                    }
                }
        
        # Check if this is an N-gram model
        if model_name.endswith("_ngram"):
            model_file = MODELS_DIR / f"{model_name}.pkl"
            if not model_file.exists():
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
            
            # N-gram models have a different schema (text generation)
            return {
                "model_name": model_name,
                "model_type": "ngram_text_generation",
                "input_type": "text",
                "parameters": {
                    "prompt": {
                        "type": "string",
                        "description": "Starting text (can be empty for random start)",
                        "default": ""
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum words to generate",
                        "default": 50,
                        "min": 5,
                        "max": 200
                    },
                    "temperature": {
                        "type": "float",
                        "description": "Randomness (0.5=conservative, 1.0=balanced, 2.0=creative)",
                        "default": 1.0,
                        "min": 0.1,
                        "max": 2.0
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Sample from top-k most likely words",
                        "default": 40
                    }
                },
                "example_usage": {
                    "prompt": "This product is",
                    "max_length": 30,
                    "temperature": 1.0
                }
            }
        
        # Check if this is a RAG model
        if model_name.endswith("_rag"):
            model_file = MODELS_DIR / f"{model_name}.pkl"
            if not model_file.exists():
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
            
            # RAG models for Q&A
            return {
                "model_name": model_name,
                "model_type": "rag_qa",
                "input_type": "question",
                "parameters": {
                    "question": {
                        "type": "string",
                        "description": "Your question",
                        "required": True
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of contexts to consider",
                        "default": 3,
                        "min": 1,
                        "max": 10
                    },
                    "min_similarity": {
                        "type": "float",
                        "description": "Minimum similarity threshold",
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0
                    }
                },
                "example_usage": {
                    "question": "Qual a cor da camisa?",
                    "top_k": 3,
                    "min_similarity": 0.1
                }
            }
        
        # Regular ML model - load metadata
        metadata_file = MODELS_DIR / f"{model_name}_metadata.json"
        
        if not metadata_file.exists():
            raise HTTPException(status_code=404, detail=f"Model metadata for '{model_name}' not found")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        features = metadata.get("feature_columns", [])
        target = metadata.get("target_column", "")
        model_type = metadata.get("model_type", "unknown")
        
        return {
            "model_name": model_name,
            "model_type": model_type,
            "target": target,
            "features": features,
            "feature_count": len(features),
            "example_input": {feature: 0.0 for feature in features}
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model schema: {str(e)}")


@router.post("/models/{model_name}/predict")
async def predict(model_name: str, request: PredictionRequest) -> Dict[str, Any]:
    """Make predictions using a saved model."""
    try:
        model_file = MODELS_DIR / f"{model_name}.pkl"
        metadata_file = MODELS_DIR / f"{model_name}_metadata.json"
        
        if not model_file.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        # Load model
        model = joblib.load(model_file)
        
        # Load metadata to get feature columns
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        expected_features = metadata.get("feature_columns", [])
        model_type = metadata.get("model_type", "unknown")
        target_column = metadata.get("target_column", "prediction")
        
        # Validate input features
        if not expected_features:
            raise HTTPException(
                status_code=400, 
                detail="Model metadata missing feature columns. Cannot validate input."
            )
        
        missing_features = [f for f in expected_features if f not in request.inputs]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}. Expected: {expected_features}"
            )
        
        # Create DataFrame with features in correct order
        input_data = pd.DataFrame([request.inputs])[expected_features]
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # For classification, get probabilities if available
        probabilities = None
        if model_type == "classification" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)
            probabilities = proba[0].tolist()
        
        result = {
            "model_name": model_name,
            "model_type": model_type,
            "prediction": prediction[0].tolist() if hasattr(prediction[0], 'tolist') else float(prediction[0]),
            "target": target_column,
            "inputs": request.inputs,
            "timestamp": datetime.now().isoformat()
        }
        
        if probabilities:
            result["probabilities"] = probabilities
            if hasattr(model, "classes_"):
                result["classes"] = model.classes_.tolist()
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/models/{model_name}/predict_text")
async def predict_from_text(model_name: str, request: TextPredictionRequest) -> Dict[str, Any]:
    """
    Make predictions from raw text input.
    
    This endpoint processes raw text through NLP pipeline (normalization + sentiment analysis)
    and uses the trained model to predict sentiment.
    
    Example:
        POST /models/Terapeuta_v1/predict_text
        Body: {"text": "I loved this product!"}
        
    Returns:
        Prediction result with sentiment classification
    """
    try:
        from app.services.text_prediction_service import text_prediction_service
        
        result = text_prediction_service.predict_from_text(model_name, request.text)
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text prediction failed: {str(e)}")


class RAGAskRequest(BaseModel):
    """Request body for RAG Q&A."""
    question: str
    top_k: int = 3
    min_similarity: float = 0.1


@router.post("/models/{model_name}/ask")
async def ask_rag_model(model_name: str, request: RAGAskRequest) -> Dict[str, Any]:
    """
    Ask a question to a RAG model.
    
    This endpoint loads a trained RAG model and answers the question
    based on its knowledge base.
    
    Example:
        POST /models/atendente_loja_rag/ask
        Body: {
            "question": "Qual a cor da camisa?",
            "top_k": 3,
            "min_similarity": 0.1
        }
        
    Returns:
        Answer with confidence and contexts used
    """
    try:
        import pickle
        from app.services.rag_service import RAGService
        
        # Load RAG model
        model_file = MODELS_DIR / f"{model_name}.pkl"
        if not model_file.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        # Reconstruct RAG
        rag = RAGService()
        rag.knowledge_base = model_data['knowledge_base']
        rag.vectorizer = model_data['vectorizer']
        rag.tfidf_matrix = model_data['tfidf_matrix']
        rag.current_knowledge_name = model_data['model_name']
        
        # Answer question
        result = rag.answer(
            question=request.question,
            top_k=request.top_k,
            min_similarity=request.min_similarity
        )
        
        return {
            "model_name": model_name,
            "question": request.question,
            "answer": result['answer'],
            "confidence": result['confidence'],
            "contexts_found": result['retrieved_docs'],
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

