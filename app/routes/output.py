from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
import io

from app.pipeline.engine import engine
from app.core.logger import logger

router = APIRouter(prefix="/output", tags=["output"])


@router.get("/{pipeline_id}/{format}")
async def get_output_data(pipeline_id: str, format: str):
    """
    Get output data for a pipeline in the specified format.
    
    This endpoint looks for the last Output step in the pipeline and returns its data.
    Supported formats: csv, json, parquet
    
    URLs follow the pattern:
    - /output/{pipeline_id}/csv
    - /output/{pipeline_id}/json
    - /output/{pipeline_id}/parquet
    """
    try:
        # Get pipeline status to find output stages
        status = engine.get_status(pipeline_id)
        
        if not status:
            raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
        
        # Find the last Output step
        config = status.get('config', {})
        steps = config.get('steps', [])
        
        output_stage = None
        for index, step in enumerate(steps):
            if step.get('type') == 'output':
                step_key = f"{step.get('name', 'Output')} #{index + 1}"
                output_stage = step_key
        
        if not output_stage:
            raise HTTPException(
                status_code=404, 
                detail="No Output step found in this pipeline. Add an Output step to enable data export."
            )
        
        # Get the output stage data
        data = engine.get_stage_data(pipeline_id, output_stage)
        
        if data is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Output data not available. Make sure you've run the pipeline at least once."
            )
        
        import pandas as pd
        if not isinstance(data, pd.DataFrame):
            raise HTTPException(status_code=500, detail="Invalid data format")
        
        # Format data based on requested format
        if format == 'csv':
            content = data.to_csv(index=False)
            media_type = "text/csv"
            
        elif format == 'json':
            content = data.to_json(orient='records', indent=2)
            media_type = "application/json"
            
        elif format == 'parquet':
            buffer = io.BytesIO()
            data.to_parquet(buffer, index=False)
            content = buffer.getvalue()
            media_type = "application/octet-stream"
            
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported format: {format}. Supported formats: csv, json, parquet"
            )
        
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"inline; filename=output_{pipeline_id}.{format}",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get output data for pipeline {pipeline_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
