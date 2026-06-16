from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logger import logger
from app.routes import pipeline, chat, output, models, generation, ai_chat, solutions


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        version="1.0.0",
        description="FlowScope AI - Data Pipeline Simulation Platform"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins in development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(pipeline.router)
    app.include_router(solutions.router)
    app.include_router(chat.router)
    app.include_router(ai_chat.router)
    app.include_router(output.router)
    app.include_router(models.router)
    app.include_router(generation.router)
    
    @app.get("/")
    async def root():
        return {
            "app": settings.app_name,
            "version": "1.0.0",
            "status": "running"
        }
    
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    logger.info(f"{settings.app_name} started")
    
    return app


app = create_app()
