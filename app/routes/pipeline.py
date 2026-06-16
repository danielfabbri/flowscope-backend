from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import numpy as np

from app.schemas.pipeline import (
    PipelineCreate,
    PipelineResponse,
    PipelineStatus,
)
from app.pipeline.engine import engine
from app.core.logger import logger

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.get("/list")
async def list_pipelines():
    """List all pipelines."""
    try:
        pipelines = engine.list_all_pipelines()
        return {
            "pipelines": pipelines,
            "total": len(pipelines)
        }
    except Exception as e:
        logger.error(f"Failed to list pipelines: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create", response_model=PipelineResponse)
async def create_pipeline(request: PipelineCreate):
    """Create a new pipeline."""
    try:
        pipeline_id = engine.create_pipeline(request.config)
        
        return PipelineResponse(
            pipeline_id=pipeline_id,
            config=request.config,
            status="created"
        )
    except Exception as e:
        logger.error(f"Failed to create pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/{pipeline_id}")
async def run_pipeline(pipeline_id: str, background_tasks: BackgroundTasks):
    """Execute a pipeline."""
    try:
        # Verify pipeline exists
        status = engine.get_status(pipeline_id)
        
        # Run in background
        background_tasks.add_task(engine.execute_pipeline, pipeline_id)
        
        return {
            "pipeline_id": pipeline_id,
            "status": "running",
            "message": "Pipeline execution started"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to run pipeline {pipeline_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{pipeline_id}/status")
async def get_pipeline_status(pipeline_id: str):
    """Get pipeline execution status."""
    try:
        status = engine.get_status(pipeline_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get status for pipeline {pipeline_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{pipeline_id}/stages")
async def list_pipeline_stages(pipeline_id: str):
    """List all available stages for a pipeline."""
    try:
        stages = engine.list_stages(pipeline_id)
        return {"pipeline_id": pipeline_id, "stages": stages}
    except Exception as e:
        logger.error(f"Failed to list stages for pipeline {pipeline_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Delete a pipeline."""
    try:
        success = engine.delete_pipeline(pipeline_id)
        if success:
            return {"message": f"Pipeline {pipeline_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Pipeline not found")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete pipeline {pipeline_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{pipeline_id}")
async def update_pipeline(pipeline_id: str, request: PipelineCreate):
    """Update a pipeline configuration."""
    try:
        success = engine.update_pipeline(pipeline_id, request.config)
        if success:
            return {
                "pipeline_id": pipeline_id,
                "config": request.config.dict(),
                "message": "Pipeline updated successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Pipeline not found")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update pipeline {pipeline_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{pipeline_id}/stages/{stage}/columns")
async def get_stage_columns(pipeline_id: str, stage: str):
    """Get column names for a specific pipeline stage."""
    try:
        data = engine.get_stage_data(pipeline_id, stage)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"Stage '{stage}' not found")
        
        # Get columns from DataFrame
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return {
                "pipeline_id": pipeline_id,
                "stage": stage,
                "columns": list(data.columns),
                "total_rows": len(data)
            }
        
        return {
            "pipeline_id": pipeline_id,
            "stage": stage,
            "columns": []
        }
    except Exception as e:
        logger.error(f"Failed to get columns for stage: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{pipeline_id}/stages/{stage}/download")
async def download_stage_data(pipeline_id: str, stage: str, format: str = 'csv'):
    """Download stage data in specified format (csv, json, parquet)."""
    from fastapi.responses import Response
    import io
    
    try:
        data = engine.get_stage_data(pipeline_id, stage)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"Stage '{stage}' not found")
        
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            if format == 'csv':
                content = data.to_csv(index=False)
                media_type = "text/csv"
                filename = f"{pipeline_id}_{stage}.csv"
            elif format == 'json':
                content = data.to_json(orient='records', indent=2)
                media_type = "application/json"
                filename = f"{pipeline_id}_{stage}.json"
            elif format == 'parquet':
                buffer = io.BytesIO()
                data.to_parquet(buffer, index=False)
                content = buffer.getvalue()
                media_type = "application/octet-stream"
                filename = f"{pipeline_id}_{stage}.parquet"
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
            
            return Response(
                content=content,
                media_type=media_type,
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )
        
        raise HTTPException(status_code=404, detail="No data available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download stage data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get columns for pipeline {pipeline_id}, stage {stage}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-source")
async def test_data_source(request: Dict[str, Any]):
    """Test a data source and return available columns."""
    try:
        import pandas as pd
        
        ingestion_type = request.get('ingestion_type')
        config = request.get('config', {})
        
        df = None
        
        if ingestion_type == 'file_upload':
            file_path = config.get('file_path')
            file_format = config.get('file_format', 'csv')
            
            if not file_path:
                raise HTTPException(status_code=400, detail="file_path is required")
            
            # Read file based on format
            if file_format == 'csv':
                df = pd.read_csv(file_path)
            elif file_format == 'json':
                df = pd.read_json(file_path)
            elif file_format == 'excel':
                df = pd.read_excel(file_path)
            elif file_format == 'parquet':
                df = pd.read_parquet(file_path)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_format}")
        
        elif ingestion_type == 'database':
            # Database connection logic would go here
            raise HTTPException(status_code=501, detail="Database testing not yet implemented")
        
        elif ingestion_type == 'http_pull':
            # HTTP API testing logic would go here
            raise HTTPException(status_code=501, detail="HTTP Pull testing not yet implemented")
        
        elif ingestion_type == 'generated':
            # For generated data, return sample structure
            data_type = config.get('data_type', 'sales')
            num_rows = min(int(config.get('num_rows', 100)), 100)  # Limit to 100 for testing
            
            # Generate sample data based on type
            if data_type == 'sales':
                df = pd.DataFrame({
                    'order_id': range(1, num_rows + 1),
                    'customer_id': range(1, num_rows + 1),
                    'product': ['Product A'] * num_rows,
                    'quantity': [1] * num_rows,
                    'price': [10.0] * num_rows,
                    'date': pd.date_range('2024-01-01', periods=num_rows)
                })
            elif data_type == 'customers':
                df = pd.DataFrame({
                    'customer_id': range(1, num_rows + 1),
                    'name': ['Customer ' + str(i) for i in range(1, num_rows + 1)],
                    'email': [f'customer{i}@example.com' for i in range(1, num_rows + 1)],
                    'age': [25] * num_rows,
                    'city': ['City A'] * num_rows
                })
            elif data_type == 'timeseries':
                df = pd.DataFrame({
                    'timestamp': pd.date_range('2024-01-01', periods=num_rows, freq='H'),
                    'value': [0.0] * num_rows,
                    'category': ['A'] * num_rows
                })
            else:
                df = pd.DataFrame({
                    'id': range(1, num_rows + 1),
                    'value': [0] * num_rows
                })
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported ingestion type: {ingestion_type}")
        
        if df is not None:
            # Replace NaN/Inf values with None for JSON compatibility
            sample_df = df.head(5).replace([np.nan, np.inf, -np.inf], None)
            return {
                "success": True,
                "columns": list(df.columns),
                "total_rows": len(df),
                "sample_data": sample_df.to_dict(orient='records'),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
        
        raise HTTPException(status_code=500, detail="Failed to read data source")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test data source: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{pipeline_id}/data/{stage:path}")
async def get_stage_data(pipeline_id: str, stage: str, limit: int = 100):
    """Get data for a specific pipeline stage."""
    try:
        data = engine.get_stage_data(pipeline_id, stage)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"Stage '{stage}' not found")
        
        # Convert DataFrame to dict if needed
        import pandas as pd
        import numpy as np
        if isinstance(data, pd.DataFrame):
            # Limit rows for response size
            limited_data = data.head(limit)
            # Replace NaN/Inf values with None for JSON compatibility
            limited_data = limited_data.replace([np.nan, np.inf, -np.inf], None)
            return {
                "pipeline_id": pipeline_id,
                "stage": stage,
                "total_rows": len(data),
                "returned_rows": len(limited_data),
                "columns": list(data.columns),
                "data": limited_data.to_dict(orient="records")
            }
        
        return {
            "pipeline_id": pipeline_id,
            "stage": stage,
            "data": data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get data for pipeline {pipeline_id}, stage {stage}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
