from typing import Dict, Any
import pandas as pd
import logging
import os

from app.services.base_service import BasePipelineService


class EnrichmentService(BasePipelineService):
    """Service for enriching data by joining with external sources."""
    
    def execute(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Enrich data by joining with an external data source.
        
        Config:
        - source_type: Type of source (file, database, api) - currently only 'file' supported
        - source_path: Path to the data file (CSV or JSON)
        - join_key_left: Column name in current DataFrame to join on
        - join_key_right: Column name in source data to join on
        - join_type: Type of join (inner, left, right, outer)
        - columns_to_add: Comma-separated list of columns to add (empty = all columns)
        """
        logger = logging.getLogger(__name__)
        
        source_type = config.get("source_type", "file")
        source_path = config.get("source_path", "").strip()
        join_key_left = config.get("join_key_left", "").strip()
        join_key_right = config.get("join_key_right", "").strip()
        join_type = config.get("join_type", "left")
        columns_to_add_str = config.get("columns_to_add", "").strip()
        
        logger.info(f"Data Enrichment - Source: {source_path}")
        logger.info(f"Join: {join_key_left} (current) = {join_key_right} (source), type={join_type}")
        logger.info(f"Initial row count: {len(data)}")
        
        # Validate inputs
        if not source_path:
            raise ValueError("source_path is required")
        
        if not join_key_left:
            raise ValueError("join_key_left is required")
        
        if join_key_left not in data.columns:
            raise ValueError(f"Join key '{join_key_left}' not found in current data. Available: {list(data.columns)}")
        
        # Load source data
        source_data = self._load_source_data(source_type, source_path)
        logger.info(f"Loaded {len(source_data)} rows from source with columns: {list(source_data.columns)}")
        
        # Use same key name if not specified
        if not join_key_right:
            join_key_right = join_key_left
        
        if join_key_right not in source_data.columns:
            raise ValueError(f"Join key '{join_key_right}' not found in source data. Available: {list(source_data.columns)}")
        
        # Determine which columns to add
        if columns_to_add_str:
            columns_to_add = [col.strip() for col in columns_to_add_str.split(',') if col.strip()]
            # Always include the join key from source
            if join_key_right not in columns_to_add:
                columns_to_add.append(join_key_right)
            
            # Filter source data to only include specified columns
            missing_cols = [col for col in columns_to_add if col not in source_data.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in source: {missing_cols}. Available: {list(source_data.columns)}")
            
            source_data = source_data[columns_to_add]
            logger.info(f"Adding specific columns: {columns_to_add}")
        else:
            logger.info(f"Adding all columns from source: {list(source_data.columns)}")
        
        # Perform join
        enriched_data = pd.merge(
            data,
            source_data,
            left_on=join_key_left,
            right_on=join_key_right,
            how=join_type,
            suffixes=('', '_source')
        )
        
        logger.info(f"Enriched row count: {len(enriched_data)}")
        logger.info(f"New columns added: {[col for col in enriched_data.columns if col not in data.columns]}")
        
        if len(enriched_data) == 0:
            logger.warning("⚠️ Join resulted in ZERO rows! Check if join keys match.")
        
        return enriched_data
    
    def _load_source_data(self, source_type: str, source_path: str) -> pd.DataFrame:
        """Load data from external source."""
        logger = logging.getLogger(__name__)
        
        if source_type != "file":
            raise ValueError(f"Source type '{source_type}' not supported yet. Use 'file'.")
        
        # Check if file exists
        if not os.path.exists(source_path):
            raise ValueError(f"Source file not found: {source_path}")
        
        # Detect file format and load
        file_ext = os.path.splitext(source_path)[1].lower()
        
        if file_ext == '.csv':
            logger.info(f"Loading CSV file: {source_path}")
            return pd.read_csv(source_path)
        elif file_ext == '.json':
            logger.info(f"Loading JSON file: {source_path}")
            return pd.read_json(source_path)
        elif file_ext in ['.xlsx', '.xls']:
            logger.info(f"Loading Excel file: {source_path}")
            return pd.read_excel(source_path)
        elif file_ext == '.parquet':
            logger.info(f"Loading Parquet file: {source_path}")
            return pd.read_parquet(source_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported: .csv, .json, .xlsx, .parquet")
