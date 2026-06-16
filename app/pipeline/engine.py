import uuid
from typing import Dict, Any, List
from datetime import datetime
import time
import pandas as pd

from app.core.logger import log_pipeline_step, logger
from app.pipeline.storage import storage
from app.schemas.pipeline import PipelineConfig, PipelineStatus


class PipelineEngine:
    """Core pipeline execution engine."""
    
    def __init__(self):
        self.pipelines: Dict[str, Dict] = {}
    
    def create_pipeline(self, config: PipelineConfig) -> str:
        """Create a new pipeline."""
        pipeline_id = str(uuid.uuid4())
        
        pipeline_data = {
            "id": pipeline_id,
            "config": config.dict(),
            "status": "created",
            "current_step": None,
            "steps_completed": [],
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
        }
        
        self.pipelines[pipeline_id] = pipeline_data
        
        # Save to storage
        storage.save_pipeline_config(pipeline_id, config.dict())
        storage.save_pipeline_status(pipeline_id, pipeline_data)
        
        logger.info(f"Pipeline {pipeline_id} created: {config.name}")
        
        return pipeline_id
    
    def get_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get pipeline status."""
        if pipeline_id in self.pipelines:
            return self.pipelines[pipeline_id]
        
        # Try loading from storage
        status = storage.get_pipeline_status(pipeline_id)
        if status:
            self.pipelines[pipeline_id] = status
            return status
        
        raise ValueError(f"Pipeline {pipeline_id} not found")
    
    def execute_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """Execute pipeline steps."""
        try:
            # Get pipeline data
            pipeline_data = self.get_status(pipeline_id)
            config = pipeline_data["config"]
            
            # Reset and update status for re-execution
            pipeline_data["status"] = "running"
            pipeline_data["started_at"] = datetime.now().isoformat()
            pipeline_data["completed_at"] = None
            pipeline_data["current_step"] = None
            pipeline_data["steps_completed"] = []
            pipeline_data["error"] = None
            storage.save_pipeline_status(pipeline_id, pipeline_data)
            
            log_pipeline_step(pipeline_id, "START", "RUNNING", "Pipeline execution started")
            
            # Import services dynamically to avoid circular imports
            from app.services.ingestion_service import IngestionService
            from app.services.profiling_service import ProfilingService
            from app.services.column_selection_service import ColumnSelectionService
            from app.services.cleaning_service import CleaningService
            from app.services.transformation_service import TransformationService
            from app.services.feature_service import FeatureService
            from app.services.ml_service import MLService
            from app.services.filter_service import FilterService
            from app.services.enrichment_service import EnrichmentService
            from app.services.sorting_service import SortingService
            from app.services.output_service import OutputService
            from app.services.model_training_service import ModelTrainingService
            from app.services.model_loading_service import ModelLoadingService
            from app.services.model_inference_service import ModelInferenceService
            from app.services.user_input_service import UserInputService
            
            # Import NLP services
            from app.services.text_normalization_service import TextNormalizationService
            from app.services.tokenization_service import TokenizationService
            from app.services.stopwords_removal_service import StopWordsRemovalService
            from app.services.stemming_lemmatization_service import StemmingLemmatizationService
            from app.services.ngrams_service import NgramsService
            from app.services.text_vectorization_service import TextVectorizationService
            from app.services.sentiment_analysis_service import SentimentAnalysisService
            from app.services.ngram_training_service import NgramTrainingService, NgramGenerationService
            from app.services.rag_training_service import RAGTrainingService, RAGAnswerService
            
            # Import AI Conversational services
            from app.services.intent_training_step_service import get_intent_training_step_service
            from app.services.knowledge_indexing_step_service import get_knowledge_indexing_step_service
            from app.services.intent_classification_step_service import get_intent_classification_step_service
            from app.services.entity_extraction_step_service import get_entity_extraction_step_service
            from app.services.semantic_search_step_service import get_semantic_search_step_service
            from app.services.context_manager_step_service import get_context_manager_step_service
            from app.services.response_generation_step_service import get_response_generation_step_service
            
            # Import Advanced AI services
            from app.services.dialogue_state_tracking_step_service import get_dialogue_state_tracking_step_service
            from app.services.coreference_resolution_step_service import get_coreference_resolution_step_service
            from app.services.answer_reranking_step_service import get_answer_reranking_step_service
            from app.services.extractive_qa_step_service import get_extractive_qa_step_service
            from app.services.paraphrase_detection_step_service import get_paraphrase_detection_step_service
            from app.services.response_selection_step_service import get_response_selection_step_service
            from app.services.slot_filling_step_service import get_slot_filling_step_service
            from app.services.topic_tracking_step_service import get_topic_tracking_step_service
            from app.services.query_expansion_step_service import get_query_expansion_step_service
            from app.services.active_learning_step_service import get_active_learning_step_service
            from app.services.intent_disambiguation_step_service import get_intent_disambiguation_step_service
            from app.services.relation_extraction_step_service import get_relation_extraction_step_service
            from app.services.anomaly_detection_step_service import get_anomaly_detection_step_service
            from app.services.answer_fusion_step_service import get_answer_fusion_step_service
            from app.services.personalization_step_service import get_personalization_step_service
            
            logger.info("✅ All AI Conversational services imported successfully")
            
            # Service mapping
            service_map = {
                "ingestion": IngestionService(),
                "profiling": ProfilingService(),
                "column_selection": ColumnSelectionService(),
                "cleaning": CleaningService(),
                "transformation": TransformationService(),
                "feature_engineering": FeatureService(),
                "ml_modeling": MLService(),
                "row_filtering": FilterService(),
                "data_enrichment": EnrichmentService(),
                "sorting": SortingService(),
                "output": OutputService(),
                "model_training": ModelTrainingService(),
                "model_loading": ModelLoadingService(),
                "model_inference": ModelInferenceService(),
                "user_input": UserInputService(),
                # NLP services
                "text_normalization": TextNormalizationService(),
                "tokenization": TokenizationService(),
                "stopwords_removal": StopWordsRemovalService(),
                "stemming_lemmatization": StemmingLemmatizationService(),
                "ngrams": NgramsService(),
                "text_vectorization": TextVectorizationService(),
                "sentiment_analysis": SentimentAnalysisService(),
                # Text Generation services
                "ngram_training": NgramTrainingService(),
                "ngram_generation": NgramGenerationService(),
                # RAG services
                "rag_training": RAGTrainingService(),
                "rag_answer": RAGAnswerService(),
                # AI Conversational services
                "intent_training": get_intent_training_step_service(),
                "knowledge_indexing": get_knowledge_indexing_step_service(),
                "intent_classification": get_intent_classification_step_service(),
                "entity_extraction": get_entity_extraction_step_service(),
                "semantic_search": get_semantic_search_step_service(),
                "context_manager": get_context_manager_step_service(),
                "response_generation": get_response_generation_step_service(),
                # Advanced AI services
                "dialogue_state_tracking": get_dialogue_state_tracking_step_service(),
                "coreference_resolution": get_coreference_resolution_step_service(),
                "answer_reranking": get_answer_reranking_step_service(),
                "extractive_qa": get_extractive_qa_step_service(),
                "paraphrase_detection": get_paraphrase_detection_step_service(),
                "response_selection": get_response_selection_step_service(),
                "slot_filling": get_slot_filling_step_service(),
                "topic_tracking": get_topic_tracking_step_service(),
                "query_expansion": get_query_expansion_step_service(),
                "active_learning": get_active_learning_step_service(),
                "intent_disambiguation": get_intent_disambiguation_step_service(),
                "relation_extraction": get_relation_extraction_step_service(),
                "anomaly_detection": get_anomaly_detection_step_service(),
                "answer_fusion": get_answer_fusion_step_service(),
                "personalization": get_personalization_step_service(),
            }
            
            logger.info(f"📋 Service map keys: {list(service_map.keys())}")
            logger.info(f"🔍 Looking for AI services: intent_training={('intent_training' in service_map)}, knowledge_indexing={('knowledge_indexing' in service_map)}")
            
            # Execute each step
            data = None
            for index, step in enumerate(config["steps"]):
                if not step["enabled"]:
                    continue
                
                step_name = step["name"]
                step_type = step["type"]
                step_config = step.get("config", {})
                
                # Create unique step identifier to handle duplicate names
                step_key = f"{step_name} #{index + 1}"
                
                logger.info(f"🔍 Processing step {index + 1}: name='{step_name}', type='{step_type}'")
                
                log_pipeline_step(pipeline_id, step_name, "RUNNING")
                pipeline_data["current_step"] = step_name
                storage.save_pipeline_status(pipeline_id, pipeline_data)
                
                # Delay to allow frontend to capture the running status
                time.sleep(1.0)
                
                # Get service for this step type
                logger.info(f"🔎 Looking up service for type: '{step_type}'")
                logger.info(f"📋 Available types: {list(service_map.keys())}")
                
                service = service_map.get(step_type)
                if not service:
                    logger.error(f"❌ Service not found for type: '{step_type}'")
                    logger.error(f"❌ Available services: {list(service_map.keys())}")
                    raise ValueError(f"Unknown step type: {step_type}")
                
                # Execute step
                data = service.execute(data, step_config)
                
                # Save stage data with unique key
                storage.save_stage_data(pipeline_id, step_key, data)
                
                # Update completion
                pipeline_data["steps_completed"].append(step_name)
                pipeline_data["current_step"] = None
                storage.save_pipeline_status(pipeline_id, pipeline_data)
                log_pipeline_step(pipeline_id, step_name, "COMPLETED")
                
                # Delay after completion to allow frontend to capture
                time.sleep(0.8)
            
            # Mark as completed
            pipeline_data["status"] = "completed"
            pipeline_data["completed_at"] = datetime.now().isoformat()
            pipeline_data["current_step"] = None
            storage.save_pipeline_status(pipeline_id, pipeline_data)
            
            log_pipeline_step(pipeline_id, "END", "COMPLETED", "Pipeline execution completed")
            
            return pipeline_data
            
        except Exception as e:
            logger.error(f"Pipeline {pipeline_id} failed: {str(e)}")
            pipeline_data["status"] = "failed"
            pipeline_data["error"] = str(e)
            pipeline_data["completed_at"] = datetime.now().isoformat()
            storage.save_pipeline_status(pipeline_id, pipeline_data)
            raise
    
    def get_stage_data(self, pipeline_id: str, stage: str):
        """Get data for a specific stage."""
        return storage.get_stage_data(pipeline_id, stage)
    
    def list_stages(self, pipeline_id: str) -> List[str]:
        """List all stages for a pipeline."""
        return storage.list_stages(pipeline_id)
    
    def list_all_pipelines(self) -> List[Dict[str, Any]]:
        """List all saved pipelines."""
        return storage.list_all_pipelines()
    
    def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline."""
        # Remove from memory
        if pipeline_id in self.pipelines:
            del self.pipelines[pipeline_id]
        
        # Delete from storage
        return storage.delete_pipeline(pipeline_id)
    
    def update_pipeline(self, pipeline_id: str, config: PipelineConfig) -> bool:
        """Update pipeline configuration."""
        # Check if pipeline exists
        status = self.get_status(pipeline_id)
        
        # Update config in storage
        success = storage.update_pipeline_config(pipeline_id, config.dict())
        
        # Update in memory
        if success and pipeline_id in self.pipelines:
            self.pipelines[pipeline_id]["config"] = config.dict()
        
        return success


# Global engine instance
engine = PipelineEngine()
