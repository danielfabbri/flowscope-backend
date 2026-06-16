import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class PipelineStorage:
    """Simple file-based storage for pipeline data."""
    
    def __init__(self, base_path: str = "./data"):
        self.base_path = Path(base_path)
        self.pipelines_path = self.base_path / "pipelines"
        self.data_path = self.base_path / "pipeline_data"
        
        # Create directories
        self.pipelines_path.mkdir(parents=True, exist_ok=True)
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def save_pipeline_config(self, pipeline_id: str, config: Dict[str, Any]):
        """Save pipeline configuration."""
        file_path = self.pipelines_path / f"{pipeline_id}_config.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, default=str, ensure_ascii=False)
    
    def get_pipeline_config(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline configuration."""
        file_path = self.pipelines_path / f"{pipeline_id}_config.json"
        if not file_path.exists():
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            from app.core.logger import logger
            logger.error(f"Failed to parse config file {file_path}: {e}")
            return None
        except Exception as e:
            from app.core.logger import logger
            logger.error(f"Error reading config file {file_path}: {e}")
            return None
    
    def save_pipeline_status(self, pipeline_id: str, status: Dict[str, Any]):
        """Save pipeline status."""
        file_path = self.pipelines_path / f"{pipeline_id}_status.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, default=str, ensure_ascii=False)
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline status."""
        file_path = self.pipelines_path / f"{pipeline_id}_status.json"
        if not file_path.exists():
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            from app.core.logger import logger
            logger.error(f"Failed to parse status file {file_path}: {e}")
            return None
        except Exception as e:
            from app.core.logger import logger
            logger.error(f"Error reading status file {file_path}: {e}")
            return None
    
    def save_stage_data(self, pipeline_id: str, stage: str, data: Any):
        """Save data for a specific stage."""
        import pandas as pd
        
        stage_path = self.data_path / pipeline_id
        stage_path.mkdir(parents=True, exist_ok=True)
        
        # Sanitize stage name for filesystem (remove/replace invalid characters)
        safe_stage = stage.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        file_path = stage_path / f"{safe_stage}.csv"
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        else:
            # Save as JSON if not a DataFrame
            with open(stage_path / f"{safe_stage}.json", 'w') as f:
                json.dump(data, f, indent=2, default=str)
    
    def get_stage_data(self, pipeline_id: str, stage: str) -> Optional[Any]:
        """Get data for a specific stage."""
        import pandas as pd
        
        stage_path = self.data_path / pipeline_id
        
        # Sanitize stage name for filesystem
        safe_stage = stage.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        csv_path = stage_path / f"{safe_stage}.csv"
        json_path = stage_path / f"{safe_stage}.json"
        
        if csv_path.exists():
            return pd.read_csv(csv_path)
        elif json_path.exists():
            with open(json_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def list_stages(self, pipeline_id: str) -> list:
        """List all available stages for a pipeline."""
        stage_path = self.data_path / pipeline_id
        if not stage_path.exists():
            return []
        
        stages = []
        for file in stage_path.glob("*.csv"):
            stages.append(file.stem)
        for file in stage_path.glob("*.json"):
            if file.stem not in stages:
                stages.append(file.stem)
        
        return sorted(stages)
    
    def list_all_pipelines(self) -> list:
        """List all saved pipelines."""
        pipelines = []
        
        # Find all config files (primary source of truth)
        for file in self.pipelines_path.glob("*_config.json"):
            pipeline_id = file.stem.replace("_config", "")
            
            # Load config
            config = self.get_pipeline_config(pipeline_id)
            if not config:
                continue
            
            # Load status if exists
            status = self.get_pipeline_status(pipeline_id)
            
            pipelines.append({
                "id": pipeline_id,
                "name": config.get("name", "Unnamed"),
                "status": status.get("status", "created") if status else "created",
                "created_at": config.get("created_at") or (status.get("created_at") if status else None),
                "completed_at": status.get("completed_at") if status else None,
                "steps": config.get("steps", []),
                "solution_id": config.get("solution_id")  # Include solution_id
            })
        
        # Sort by creation date (newest first), handling None values
        pipelines.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        
        return pipelines
    
    def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline and all its data."""
        import shutil
        
        # Delete config file
        config_path = self.pipelines_path / f"{pipeline_id}_config.json"
        if config_path.exists():
            config_path.unlink()
        
        # Delete status file
        status_path = self.pipelines_path / f"{pipeline_id}_status.json"
        if status_path.exists():
            status_path.unlink()
        
        # Delete data folder
        data_path = self.data_path / pipeline_id
        if data_path.exists():
            shutil.rmtree(data_path)
        
        return True
    
    def update_pipeline_config(self, pipeline_id: str, config: Dict[str, Any]) -> bool:
        """Update pipeline configuration."""
        # Check if pipeline exists
        if not self.get_pipeline_status(pipeline_id):
            return False
        
        # Save updated config
        self.save_pipeline_config(pipeline_id, config)
        
        # Clean up orphan stages (stages that were removed from config)
        self.clean_orphan_stages(pipeline_id, config)
        
        return True
    
    def clean_orphan_stages(self, pipeline_id: str, config: Dict[str, Any]):
        """Delete stage data files that are no longer in the pipeline config."""
        # Get current stages from config with unique keys
        current_stages = set()
        for index, step in enumerate(config.get("steps", [])):
            # Stage key includes index to handle duplicates
            stage_name = step.get("name", step.get("type", "unknown"))
            stage_key = f"{stage_name} #{index + 1}"
            # Sanitize for filesystem
            safe_stage = stage_key.replace('/', '_').replace('\\', '_').replace(':', '_')
            current_stages.add(safe_stage)
        
        # Get saved stages from disk
        saved_stages = self.list_stages(pipeline_id)
        
        # Delete orphan stages
        stage_path = self.data_path / pipeline_id
        for stage in saved_stages:
            if stage not in current_stages:
                # Delete CSV file
                csv_path = stage_path / f"{stage}.csv"
                if csv_path.exists():
                    csv_path.unlink()
                    from app.core.logger import logger
                    logger.info(f"Deleted orphan stage data: {pipeline_id}/{stage}.csv")
                
                # Delete JSON file
                json_path = stage_path / f"{stage}.json"
                if json_path.exists():
                    json_path.unlink()
                    from app.core.logger import logger
                    logger.info(f"Deleted orphan stage data: {pipeline_id}/{stage}.json")
    
    # ==================== SOLUTIONS MANAGEMENT ====================
    
    def get_solutions_file_path(self) -> Path:
        """Get path to solutions JSON file."""
        return self.base_path / "solutions.json"
    
    def load_solutions(self) -> Dict[str, Any]:
        """Load all solutions from file."""
        solutions_path = self.get_solutions_file_path()
        if not solutions_path.exists():
            # Return default solutions if file doesn't exist
            return self._get_default_solutions()
        
        try:
            with open(solutions_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            from app.core.logger import logger
            logger.error(f"Failed to load solutions: {e}")
            return self._get_default_solutions()
    
    def save_solutions(self, solutions: Dict[str, Any]):
        """Save all solutions to file."""
        solutions_path = self.get_solutions_file_path()
        with open(solutions_path, 'w', encoding='utf-8') as f:
            json.dump(solutions, f, indent=2, ensure_ascii=False, default=str)
    
    def _get_default_solutions(self) -> Dict[str, Any]:
        """Get default solutions structure."""
        return {
            "solutions": [],
            "version": "1.0",
            "last_updated": datetime.now().isoformat()
        }
    
    def create_solution(self, solution_data: Dict[str, Any]) -> str:
        """Create a new solution."""
        import uuid
        
        solutions_data = self.load_solutions()
        
        solution_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        new_solution = {
            "id": solution_id,
            "name": solution_data["name"],
            "description": solution_data["description"],
            "icon": solution_data.get("icon", "folder"),
            "color": solution_data.get("color", "#6366f1"),
            "category": solution_data.get("category", "general"),
            "created_at": now,
            "updated_at": now,
            "pipelines": []
        }
        
        solutions_data["solutions"].append(new_solution)
        solutions_data["last_updated"] = now
        
        self.save_solutions(solutions_data)
        
        return solution_id
    
    def get_solution(self, solution_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific solution by ID."""
        solutions_data = self.load_solutions()
        
        for solution in solutions_data["solutions"]:
            if solution["id"] == solution_id:
                return solution
        
        return None
    
    def update_solution(self, solution_id: str, updates: Dict[str, Any]) -> bool:
        """Update a solution."""
        solutions_data = self.load_solutions()
        
        for solution in solutions_data["solutions"]:
            if solution["id"] == solution_id:
                # Update fields
                for key, value in updates.items():
                    if value is not None and key != "id":
                        solution[key] = value
                
                solution["updated_at"] = datetime.now().isoformat()
                solutions_data["last_updated"] = datetime.now().isoformat()
                
                self.save_solutions(solutions_data)
                return True
        
        return False
    
    def delete_solution(self, solution_id: str) -> bool:
        """Delete a solution (pipelines remain but lose solution_id)."""
        solutions_data = self.load_solutions()
        
        initial_count = len(solutions_data["solutions"])
        solutions_data["solutions"] = [
            s for s in solutions_data["solutions"] if s["id"] != solution_id
        ]
        
        if len(solutions_data["solutions"]) < initial_count:
            solutions_data["last_updated"] = datetime.now().isoformat()
            self.save_solutions(solutions_data)
            
            # Remove solution_id from all pipelines
            self._remove_solution_from_pipelines(solution_id)
            
            return True
        
        return False
    
    def list_all_solutions(self) -> list:
        """List all solutions with pipeline counts."""
        solutions_data = self.load_solutions()
        pipelines = self.list_all_pipelines()
        
        # Count pipelines per solution
        pipeline_counts = {}
        for pipeline in pipelines:
            solution_id = pipeline.get("solution_id")
            if solution_id:
                pipeline_counts[solution_id] = pipeline_counts.get(solution_id, 0) + 1
        
        # Enrich solutions with pipeline counts
        enriched_solutions = []
        for solution in solutions_data["solutions"]:
            solution_copy = solution.copy()
            solution_copy["pipeline_count"] = pipeline_counts.get(solution["id"], 0)
            enriched_solutions.append(solution_copy)
        
        return enriched_solutions
    
    def add_pipeline_to_solution(self, pipeline_id: str, solution_id: str) -> bool:
        """Add a pipeline to a solution."""
        # Verify solution exists
        solution = self.get_solution(solution_id)
        if not solution:
            return False
        
        # Get pipeline config
        config = self.get_pipeline_config(pipeline_id)
        if not config:
            return False
        
        # Add solution_id to pipeline config
        config["solution_id"] = solution_id
        config["updated_at"] = datetime.now().isoformat()
        self.save_pipeline_config(pipeline_id, config)
        
        # Add pipeline to solution's pipeline list if not already there
        if pipeline_id not in solution.get("pipelines", []):
            solution.setdefault("pipelines", []).append(pipeline_id)
            self.update_solution(solution_id, {"pipelines": solution["pipelines"]})
        
        return True
    
    def remove_pipeline_from_solution(self, pipeline_id: str) -> bool:
        """Remove a pipeline from its solution."""
        config = self.get_pipeline_config(pipeline_id)
        if not config:
            return False
        
        solution_id = config.get("solution_id")
        if not solution_id:
            return True  # Already not in a solution
        
        # Remove solution_id from pipeline
        config.pop("solution_id", None)
        config["updated_at"] = datetime.now().isoformat()
        self.save_pipeline_config(pipeline_id, config)
        
        # Remove from solution's pipeline list
        solution = self.get_solution(solution_id)
        if solution and pipeline_id in solution.get("pipelines", []):
            solution["pipelines"].remove(pipeline_id)
            self.update_solution(solution_id, {"pipelines": solution["pipelines"]})
        
        return True
    
    def _remove_solution_from_pipelines(self, solution_id: str):
        """Remove solution_id from all pipelines when solution is deleted."""
        pipelines = self.list_all_pipelines()
        
        for pipeline in pipelines:
            if pipeline.get("solution_id") == solution_id:
                self.remove_pipeline_from_solution(pipeline["id"])
    
    def get_pipelines_by_solution(self, solution_id: str) -> list:
        """Get all pipelines belonging to a solution."""
        all_pipelines = self.list_all_pipelines()
        
        return [
            p for p in all_pipelines 
            if p.get("solution_id") == solution_id
        ]


# Global storage instance
# Use absolute path relative to backend directory
from pathlib import Path as PathLib
_backend_dir = PathLib(__file__).parent.parent.parent
storage = PipelineStorage(base_path=str(_backend_dir / "data"))
