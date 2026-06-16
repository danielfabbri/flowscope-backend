"""
Solutions API routes - Group multiple pipelines into solutions
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional, List
import os
import shutil
from pathlib import Path
from datetime import datetime

from app.schemas.solution import (
    SolutionCreate,
    SolutionUpdate,
    SolutionResponse,
    SolutionListResponse
)
from app.pipeline.storage import storage
from app.core.logger import logger

router = APIRouter(prefix="/solutions", tags=["solutions"])


@router.get("/list", response_model=SolutionListResponse)
async def list_solutions():
    """List all solutions with pipeline counts."""
    try:
        solutions = storage.list_all_solutions()
        
        return SolutionListResponse(
            solutions=[
                SolutionResponse(
                    id=s["id"],
                    name=s["name"],
                    description=s["description"],
                    icon=s["icon"],
                    color=s["color"],
                    category=s["category"],
                    created_at=s["created_at"],
                    updated_at=s["updated_at"],
                    pipeline_count=s["pipeline_count"],
                    pipelines=s.get("pipelines", [])
                )
                for s in solutions
            ],
            total=len(solutions)
        )
    except Exception as e:
        logger.error(f"Failed to list solutions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create", response_model=SolutionResponse)
async def create_solution(request: SolutionCreate):
    """Create a new solution."""
    try:
        solution_id = storage.create_solution(request.dict())
        solution = storage.get_solution(solution_id)
        
        if not solution:
            raise HTTPException(status_code=500, detail="Failed to create solution")
        
        return SolutionResponse(
            id=solution["id"],
            name=solution["name"],
            description=solution["description"],
            icon=solution["icon"],
            color=solution["color"],
            category=solution["category"],
            created_at=solution["created_at"],
            updated_at=solution["updated_at"],
            pipeline_count=0,
            pipelines=[]
        )
    except Exception as e:
        logger.error(f"Failed to create solution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{solution_id}", response_model=SolutionResponse)
async def get_solution(solution_id: str):
    """Get a specific solution by ID."""
    try:
        solution = storage.get_solution(solution_id)
        
        if not solution:
            raise HTTPException(status_code=404, detail="Solution not found")
        
        # Get pipeline count
        pipelines = storage.get_pipelines_by_solution(solution_id)
        
        return SolutionResponse(
            id=solution["id"],
            name=solution["name"],
            description=solution["description"],
            icon=solution["icon"],
            color=solution["color"],
            category=solution["category"],
            created_at=solution["created_at"],
            updated_at=solution["updated_at"],
            pipeline_count=len(pipelines),
            pipelines=solution.get("pipelines", [])
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get solution {solution_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{solution_id}", response_model=SolutionResponse)
async def update_solution(solution_id: str, request: SolutionUpdate):
    """Update a solution."""
    try:
        updates = {k: v for k, v in request.dict().items() if v is not None}
        
        success = storage.update_solution(solution_id, updates)
        
        if not success:
            raise HTTPException(status_code=404, detail="Solution not found")
        
        solution = storage.get_solution(solution_id)
        pipelines = storage.get_pipelines_by_solution(solution_id)
        
        return SolutionResponse(
            id=solution["id"],
            name=solution["name"],
            description=solution["description"],
            icon=solution["icon"],
            color=solution["color"],
            category=solution["category"],
            created_at=solution["created_at"],
            updated_at=solution["updated_at"],
            pipeline_count=len(pipelines),
            pipelines=solution.get("pipelines", [])
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update solution {solution_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{solution_id}")
async def delete_solution(solution_id: str):
    """Delete a solution (pipelines remain but lose solution_id)."""
    try:
        success = storage.delete_solution(solution_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Solution not found")
        
        return {"message": f"Solution {solution_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete solution {solution_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{solution_id}/pipelines")
async def get_solution_pipelines(solution_id: str):
    """Get all pipelines belonging to a solution."""
    try:
        # Verify solution exists
        solution = storage.get_solution(solution_id)
        if not solution:
            raise HTTPException(status_code=404, detail="Solution not found")
        
        pipelines = storage.get_pipelines_by_solution(solution_id)
        
        return {
            "solution_id": solution_id,
            "pipelines": pipelines,
            "total": len(pipelines)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pipelines for solution {solution_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{solution_id}/pipelines/{pipeline_id}")
async def add_pipeline_to_solution(solution_id: str, pipeline_id: str):
    """Add a pipeline to a solution."""
    try:
        success = storage.add_pipeline_to_solution(pipeline_id, solution_id)
        
        if not success:
            raise HTTPException(
                status_code=404, 
                detail="Solution or pipeline not found"
            )
        
        return {
            "message": f"Pipeline {pipeline_id} added to solution {solution_id}",
            "solution_id": solution_id,
            "pipeline_id": pipeline_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add pipeline to solution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{solution_id}/models")
async def get_solution_models(solution_id: str):
    """Get all models belonging to a solution based on pipeline outputs and metadata."""
    try:
        from pathlib import Path
        import json
        from datetime import datetime
        
        # Verify solution exists
        solution = storage.get_solution(solution_id)
        if not solution:
            raise HTTPException(status_code=404, detail="Solution not found")
        
        # Get all pipelines in this solution
        pipelines = storage.get_pipelines_by_solution(solution_id)
        pipeline_ids = [p.get('id') for p in pipelines]
        
        # Get models directory
        models_dir = Path(__file__).parent.parent.parent / "data" / "models"
        if not models_dir.exists():
            return {
                "solution_id": solution_id,
                "solution_name": solution.get('name', 'Unknown'),
                "models": [],
                "total": 0
            }
        
        # Collect all model names from pipeline configs
        model_names_from_config = set()
        for pipeline in pipelines:
            # Check if pipeline has model_persistence step
            for step in pipeline.get('steps', []):
                if step.get('type') == 'model_persistence':
                    model_name = step.get('config', {}).get('model_name')
                    if model_name:
                        model_names_from_config.add(model_name)
        
        # List all models and match them
        models = []
        for model_file in models_dir.glob("*"):
            if model_file.suffix not in ['.pkl', '.joblib', '.npz']:
                continue
            
            model_name = model_file.stem
            belongs_to_solution = False
            
            # Method 1: Check metadata file for solution_id or pipeline_id
            metadata_file = models_dir / f"{model_name}_metadata.json"
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Check if metadata contains solution_id
                    if metadata.get('solution_id') == solution_id:
                        belongs_to_solution = True
                    
                    # Check if metadata contains pipeline_id from this solution
                    if metadata.get('pipeline_id') in pipeline_ids:
                        belongs_to_solution = True
                        
                except:
                    pass
            
            # Method 2: Check if model name matches any from config
            if model_name in model_names_from_config:
                belongs_to_solution = True
            
            # Method 3: Check partial matches (flexible matching)
            if not belongs_to_solution and model_names_from_config:
                for config_name in model_names_from_config:
                    if config_name in model_name or model_name in config_name:
                        belongs_to_solution = True
                        break
            
            if not belongs_to_solution:
                continue
            
            model_info = {
                "name": model_name,
                "file_path": str(model_file),
                "file_type": model_file.suffix[1:],  # Remove the dot
                "size_bytes": model_file.stat().st_size,
                "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                "type": metadata.get("model_type", "unknown"),
                "algorithm": metadata.get("algorithm", "unknown"),
                "target": metadata.get("target_column", "unknown"),
                "metrics": metadata.get("metrics", {}),
                "training_info": metadata.get("training_info", {}),
                "solution_id": metadata.get("solution_id"),
                "pipeline_id": metadata.get("pipeline_id")
            }
            
            models.append(model_info)
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x['created_at'], reverse=True)
        
        return {
            "solution_id": solution_id,
            "solution_name": solution.get('name', 'Unknown'),
            "models": models,
            "total": len(models)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get models for solution {solution_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{solution_id}/pipelines/{pipeline_id}")
async def remove_pipeline_from_solution(solution_id: str, pipeline_id: str):
    """Remove a pipeline from a solution."""
    try:
        success = storage.remove_pipeline_from_solution(pipeline_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        return {
            "message": f"Pipeline {pipeline_id} removed from solution {solution_id}",
            "solution_id": solution_id,
            "pipeline_id": pipeline_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove pipeline from solution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{solution_id}/files")
async def get_solution_files(solution_id: str):
    """Get all data files uploaded to a solution."""
    try:
        # Verify solution exists
        solution = storage.get_solution(solution_id)
        if not solution:
            raise HTTPException(status_code=404, detail="Solution not found")
        
        # Create solution files directory if it doesn't exist
        solution_files_dir = Path(f"data/solutions/{solution_id}/files")
        solution_files_dir.mkdir(parents=True, exist_ok=True)
        
        # List all files in the directory
        files = []
        for file_path in solution_files_dir.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "size": stat.st_size,
                    "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "path": str(file_path)
                })
        
        # Sort by upload date (newest first)
        files.sort(key=lambda x: x['uploaded_at'], reverse=True)
        
        return {
            "solution_id": solution_id,
            "files": files,
            "total": len(files)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get files for solution {solution_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{solution_id}/files/upload")
async def upload_solution_files(solution_id: str, files: List[UploadFile] = File(...)):
    """Upload data files to a solution."""
    try:
        # Verify solution exists
        solution = storage.get_solution(solution_id)
        if not solution:
            raise HTTPException(status_code=404, detail="Solution not found")
        
        # Create solution files directory if it doesn't exist
        solution_files_dir = Path(f"data/solutions/{solution_id}/files")
        solution_files_dir.mkdir(parents=True, exist_ok=True)
        
        uploaded_files = []
        for file in files:
            # Save file
            file_path = solution_files_dir / file.filename
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            stat = file_path.stat()
            uploaded_files.append({
                "name": file.filename,
                "size": stat.st_size,
                "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "path": str(file_path)
            })
            
            logger.info(f"Uploaded file {file.filename} to solution {solution_id}")
        
        return {
            "message": f"{len(uploaded_files)} file(s) uploaded successfully",
            "solution_id": solution_id,
            "files": uploaded_files
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload files to solution {solution_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{solution_id}/files/{file_name}")
async def delete_solution_file(solution_id: str, file_name: str):
    """Delete a data file from a solution."""
    try:
        # Verify solution exists
        solution = storage.get_solution(solution_id)
        if not solution:
            raise HTTPException(status_code=404, detail="Solution not found")
        
        # Get file path
        file_path = Path(f"data/solutions/{solution_id}/files/{file_name}")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete file
        file_path.unlink()
        logger.info(f"Deleted file {file_name} from solution {solution_id}")
        
        return {
            "message": f"File {file_name} deleted successfully",
            "solution_id": solution_id,
            "file_name": file_name
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete file from solution {solution_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{solution_id}/files/{file_name}/content")
async def get_file_content(solution_id: str, file_name: str):
    """Get the content of a file from a solution."""
    try:
        # Verify solution exists
        solution = storage.get_solution(solution_id)
        if not solution:
            raise HTTPException(status_code=404, detail="Solution not found")
        
        # Get file path
        file_path = Path(f"data/solutions/{solution_id}/files/{file_name}")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine file type and read content
        file_extension = file_path.suffix.lower()
        
        # Read file based on type
        if file_extension in ['.csv', '.txt']:
            # Read as text
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # For CSV, parse into rows for preview
            if file_extension == '.csv':
                lines = content.split('\n')
                # Limit to first 1000 rows for performance
                preview_lines = lines[:1000]
                return {
                    "file_name": file_name,
                    "file_type": "csv",
                    "content": content,
                    "preview_lines": preview_lines,
                    "total_lines": len(lines),
                    "truncated": len(lines) > 1000
                }
            else:
                return {
                    "file_name": file_name,
                    "file_type": "text",
                    "content": content,
                    "total_lines": len(content.split('\n'))
                }
        
        elif file_extension == '.json':
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            return {
                "file_name": file_name,
                "file_type": "json",
                "content": content
            }
        
        elif file_extension in ['.xlsx', '.xls']:
            try:
                import pandas as pd
                df = pd.read_excel(file_path)
                # Limit to first 1000 rows
                preview_df = df.head(1000)
                return {
                    "file_name": file_name,
                    "file_type": "excel",
                    "columns": df.columns.tolist(),
                    "data": preview_df.to_dict('records'),
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "truncated": len(df) > 1000
                }
            except ImportError:
                raise HTTPException(
                    status_code=500, 
                    detail="pandas or openpyxl not installed. Cannot read Excel files."
                )
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read file content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
