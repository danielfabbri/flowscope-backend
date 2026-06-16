"""API routes for text generation."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import logging

from app.services.text_generation_service import text_generation_service

router = APIRouter(prefix="/generate", tags=["text-generation"])
logger = logging.getLogger(__name__)


class TrainRequest(BaseModel):
    """Request to train a new text generation model."""
    csv_path: str = Field(..., description="Path to CSV file with training data")
    text_column: str = Field(..., description="Column containing text")
    model_name: str = Field(default="review_generator", description="Name for the model")
    n: int = Field(default=3, ge=2, le=5, description="N-gram size (2-5)")


class GenerateRequest(BaseModel):
    """Request to generate text."""
    model_name: Optional[str] = Field(None, description="Model to use (if not current)")
    prompt: str = Field(default="", description="Starting text (empty for random)")
    max_length: int = Field(default=50, ge=5, le=200, description="Maximum words to generate")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Randomness (0.5=safe, 2.0=creative)")
    top_k: Optional[int] = Field(default=40, description="Sample from top-k words only")
    num_samples: int = Field(default=1, ge=1, le=5, description="Number of samples to generate")


@router.post("/train")
async def train_model(request: TrainRequest):
    """
    Train a new n-gram text generation model.
    
    This learns word patterns from your text data and can generate new text.
    
    Example:
        POST /generate/train
        {
            "csv_path": "data/use_cases/nlp/product_reviews_train.csv",
            "text_column": "review_text",
            "model_name": "review_generator",
            "n": 3
        }
    """
    try:
        logger.info(f"Training model: {request.model_name}")
        
        result = text_generation_service.train_from_csv(
            csv_path=request.csv_path,
            text_column=request.text_column,
            model_name=request.model_name,
            n=request.n
        )
        
        return {
            "status": "success",
            "message": f"Model '{request.model_name}' trained successfully",
            "statistics": result
        }
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/text")
async def generate_text(request: GenerateRequest):
    """
    Generate text using a trained n-gram model.
    
    The model predicts the next word based on the previous n-1 words,
    creating text that follows patterns learned from training data.
    
    Parameters:
    - prompt: Starting text (empty = random start)
    - max_length: How many words to generate
    - temperature: Controls randomness
        - 0.5 = Conservative, picks likely words
        - 1.0 = Balanced
        - 2.0 = Creative, more unexpected words
    - top_k: Only consider top-k most likely words
    - num_samples: Generate multiple variations
    
    Example:
        POST /generate/text
        {
            "prompt": "This product is",
            "max_length": 30,
            "temperature": 0.8,
            "num_samples": 3
        }
    """
    try:
        # Load model if specified
        if request.model_name:
            text_generation_service.load_model(request.model_name)
        
        result = text_generation_service.generate_text(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            num_samples=request.num_samples
        )
        
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.get("/models")
async def list_generation_models():
    """List available text generation models."""
    try:
        models_dir = text_generation_service.models_dir
        models = []
        
        for model_file in models_dir.glob("*_ngram.pkl"):
            model_name = model_file.stem.replace("_ngram", "")
            models.append({
                "name": model_name,
                "path": str(model_file),
                "size_bytes": model_file.stat().st_size
            })
        
        return {
            "models": models,
            "current_model": text_generation_service.current_model_name
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
