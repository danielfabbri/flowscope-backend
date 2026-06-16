from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests
from pathlib import Path

from app.services.base_service import BasePipelineService
from app.core.logger import logger


class IngestionService(BasePipelineService):
    """Data ingestion service - supports multiple ingestion types."""
    
    def __init__(self):
        super().__init__()
        # Backend directory for resolving relative paths
        self.backend_dir = Path(__file__).parent.parent.parent
    
    def _resolve_file_path(self, file_path: str) -> Path:
        """Resolve file path relative to backend directory if not absolute."""
        path = Path(file_path)
        if path.is_absolute():
            return path
        # Try relative to backend directory first
        backend_path = self.backend_dir / file_path
        if backend_path.exists():
            return backend_path
        # Return original path if not found (will error later with clear message)
        return path
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Execute data ingestion based on type.
        
        Supported ingestion types:
            - generated: Generate synthetic data
            - webhook: Receive data from external systems (passive)
            - http_pull: Pull data from external APIs (active)
            - file_upload: Load data from files
            - database: Query database
            - stream: Connect to streaming platforms
        """
        ingestion_type = config.get("ingestion_type", "generated")
        
        logger.info(f"Executing {ingestion_type} ingestion")
        
        if ingestion_type == "generated":
            return self._execute_generated(config)
        elif ingestion_type == "webhook":
            return self._execute_webhook(config)
        elif ingestion_type == "http_pull":
            return self._execute_http_pull(config)
        elif ingestion_type == "file_upload":
            return self._execute_file_upload(config)
        elif ingestion_type == "database":
            return self._execute_database(config)
        elif ingestion_type == "stream":
            return self._execute_stream(config)
        else:
            raise ValueError(f"Unknown ingestion type: {ingestion_type}")
    
    def _execute_generated(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate synthetic time-series data.
        
        Config options:
            - num_rows: number of rows to generate (default: 10000)
            - start_date: start date for time series
            - frequency: time frequency (1H, 1D, etc.)
            - data_template: template type (sensor, sales, user_events, etc.)
        """
        num_rows = config.get("num_rows", 10000)
        start_date = config.get("start_date", "2024-01-01")
        frequency = config.get("frequency", "1H")
        template = config.get("data_template", "sensor")
        
        # Generate timestamps
        start = pd.to_datetime(start_date)
        timestamps = pd.date_range(start=start, periods=num_rows, freq=frequency)
        
        # Generate data based on template
        if template == "sensor":
            df = self._generate_sensor_data(timestamps)
        elif template == "sales":
            df = self._generate_sales_data(timestamps)
        elif template == "user_events":
            df = self._generate_user_events_data(timestamps)
        elif template == "iot":
            df = self._generate_iot_data(timestamps)
        elif template == "financial":
            df = self._generate_financial_data(timestamps)
        else:
            df = self._generate_sensor_data(timestamps)  # Default
        
        return df
    
    def _generate_sensor_data(self, timestamps) -> pd.DataFrame:
        """Generate sensor/IoT data."""
        num_rows = len(timestamps)
        t = np.arange(num_rows)
        
        # Generate base signal with trend and seasonality
        trend = 0.01 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 24)  # Daily seasonality
        noise = np.random.normal(0, 2, num_rows)
        values = 100 + trend + seasonal + noise
        
        # Add anomalies (2%)
        anomaly_flags = np.zeros(num_rows, dtype=bool)
        num_anomalies = int(num_rows * 0.02)
        anomaly_indices = np.random.choice(num_rows, num_anomalies, replace=False)
        for idx in anomaly_indices:
            values[idx] += np.random.choice([-1, 1]) * np.random.uniform(20, 40)
            anomaly_flags[idx] = True
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "value": values,
            "is_anomaly": anomaly_flags,
            "sensor_id": np.random.choice(["sensor_1", "sensor_2", "sensor_3"], num_rows),
            "location": np.random.choice(["site_A", "site_B", "site_C"], num_rows),
        })
    
    def _generate_sales_data(self, timestamps) -> pd.DataFrame:
        """Generate sales/e-commerce data."""
        num_rows = len(timestamps)
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "product_id": np.random.choice([f"P{i:03d}" for i in range(1, 51)], num_rows),
            "category": np.random.choice(["Electronics", "Clothing", "Food", "Books"], num_rows),
            "quantity": np.random.randint(1, 10, num_rows),
            "price": np.random.uniform(10, 500, num_rows).round(2),
            "customer_id": np.random.choice([f"C{i:04d}" for i in range(1, 1001)], num_rows),
            "region": np.random.choice(["North", "South", "East", "West"], num_rows),
        })
    
    def _generate_user_events_data(self, timestamps) -> pd.DataFrame:
        """Generate user events/clickstream data."""
        num_rows = len(timestamps)
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "user_id": np.random.choice([f"U{i:05d}" for i in range(1, 10001)], num_rows),
            "event_type": np.random.choice(["page_view", "click", "purchase", "signup", "logout"], num_rows),
            "page": np.random.choice(["/home", "/products", "/cart", "/checkout", "/profile"], num_rows),
            "session_id": np.random.choice([f"S{i:08d}" for i in range(1, num_rows//5)], num_rows),
            "device": np.random.choice(["mobile", "desktop", "tablet"], num_rows),
        })
    
    def _generate_iot_data(self, timestamps) -> pd.DataFrame:
        """Generate IoT device data."""
        num_rows = len(timestamps)
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "device_id": np.random.choice([f"D{i:04d}" for i in range(1, 101)], num_rows),
            "temperature": np.random.uniform(15, 35, num_rows).round(1),
            "humidity": np.random.uniform(30, 80, num_rows).round(1),
            "battery_level": np.random.uniform(0, 100, num_rows).round(1),
            "signal_strength": np.random.uniform(-90, -30, num_rows).round(0),
            "status": np.random.choice(["online", "offline", "warning"], num_rows, p=[0.85, 0.10, 0.05]),
        })
    
    def _generate_financial_data(self, timestamps) -> pd.DataFrame:
        """Generate financial transaction data."""
        num_rows = len(timestamps)
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "transaction_id": [f"T{i:010d}" for i in range(num_rows)],
            "amount": np.random.uniform(10, 10000, num_rows).round(2),
            "transaction_type": np.random.choice(["debit", "credit", "transfer"], num_rows),
            "account_id": np.random.choice([f"A{i:06d}" for i in range(1, 5001)], num_rows),
            "merchant": np.random.choice(["Amazon", "Walmart", "Target", "Local Store", "Restaurant"], num_rows),
            "category": np.random.choice(["Shopping", "Food", "Transport", "Entertainment", "Bills"], num_rows),
        })
    
    def _execute_webhook(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Retrieve data received via webhook.
        
        Note: Webhook data is received asynchronously via API endpoint.
        This method retrieves already received data from storage.
        """
        # In a real implementation, this would retrieve data from a queue/storage
        # where webhook POST requests are stored
        logger.warning("Webhook ingestion: returning empty DataFrame. Data should be received via webhook endpoint.")
        
        return pd.DataFrame({
            "timestamp": [datetime.now()],
            "message": ["Webhook configured - waiting for data"],
            "status": ["waiting"]
        })
    
    def _execute_http_pull(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Pull data from external HTTP API."""
        url = config.get("http_url", "")
        method = config.get("http_method", "GET")
        auth_type = config.get("http_auth_type", "none")
        api_key = config.get("http_api_key", "")
        
        if not url:
            raise ValueError("http_url is required for HTTP Pull ingestion")
        
        # Setup headers
        headers = {"Content-Type": "application/json"}
        if auth_type == "api_key":
            headers["X-API-Key"] = api_key
        elif auth_type == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        
        try:
            # Make HTTP request
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=30)
            else:
                response = requests.post(url, headers=headers, timeout=30)
            
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Check for common data keys
                for key in ["data", "results", "items", "records"]:
                    if key in data and isinstance(data[key], list):
                        df = pd.DataFrame(data[key])
                        break
                else:
                    # Single record
                    df = pd.DataFrame([data])
            else:
                raise ValueError(f"Unexpected response format: {type(data)}")
            
            logger.info(f"HTTP Pull: retrieved {len(df)} records from {url}")
            return df
            
        except Exception as e:
            logger.error(f"HTTP Pull failed: {str(e)}")
            raise ValueError(f"Failed to pull data from {url}: {str(e)}")
    
    def _execute_file_upload(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Load data from file."""
        file_path = config.get("file_path", "")
        file_format = config.get("file_format", "csv")
        
        if not file_path:
            raise ValueError("file_path is required for File Upload ingestion")
        
        # Resolve path relative to backend if needed
        path = self._resolve_file_path(file_path)
        if not path.exists():
            raise ValueError(f"File not found: {file_path} (resolved to: {path})")
        
        try:
            if file_format == "csv":
                df = pd.read_csv(path)
            elif file_format == "json":
                df = pd.read_json(path)
            elif file_format == "parquet":
                df = pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"File Upload: loaded {len(df)} records from {path}")
            return df
            
        except Exception as e:
            logger.error(f"File Upload failed: {str(e)}")
            raise ValueError(f"Failed to load file {path}: {str(e)}")
    
    def _execute_database(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Query data from database."""
        db_type = config.get("db_type", "postgresql")
        host = config.get("db_host", "")
        port = config.get("db_port", 5432)
        database = config.get("db_name", "")
        user = config.get("db_user", "")
        password = config.get("db_password", "")
        query = config.get("db_query", "")
        
        if not all([host, database, user, query]):
            raise ValueError("db_host, db_name, db_user, and db_query are required")
        
        try:
            # Build connection string based on DB type
            if db_type in ["postgresql", "mysql", "sqlserver"]:
                # SQL databases - requires sqlalchemy
                try:
                    from sqlalchemy import create_engine
                except ImportError:
                    raise ValueError("sqlalchemy is required for SQL database connections. Install with: pip install sqlalchemy")
                
                if db_type == "postgresql":
                    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"
                elif db_type == "mysql":
                    conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
                elif db_type == "sqlserver":
                    conn_str = f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
                
                engine = create_engine(conn_str)
                df = pd.read_sql(query, engine)
                
            elif db_type == "mongodb":
                # MongoDB - requires pymongo
                try:
                    from pymongo import MongoClient
                except ImportError:
                    raise ValueError("pymongo is required for MongoDB connections. Install with: pip install pymongo")
                
                client = MongoClient(f"mongodb://{user}:{password}@{host}:{port}")
                db = client[database]
                collection = db[query]  # In MongoDB, query is the collection name
                data = list(collection.find().limit(10000))
                df = pd.DataFrame(data)
                
            elif db_type == "cosmosdb":
                # Azure Cosmos DB - requires azure-cosmos
                logger.warning("Cosmos DB support is limited. Using MongoDB API.")
                try:
                    from pymongo import MongoClient
                except ImportError:
                    raise ValueError("pymongo is required for Cosmos DB (MongoDB API)")
                
                client = MongoClient(host)  # host should be full connection string for Cosmos DB
                db = client[database]
                collection = db[query]
                data = list(collection.find().limit(10000))
                df = pd.DataFrame(data)
            
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            logger.info(f"Database: queried {len(df)} records from {db_type}")
            return df
            
        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise ValueError(f"Failed to query database: {str(e)}")
    
    def _execute_stream(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Connect to streaming platform and consume messages."""
        stream_type = config.get("stream_type", "kafka")
        brokers = config.get("stream_brokers", "")
        topic = config.get("stream_topic", "")
        group_id = config.get("stream_group_id", "flowscope-consumer")
        
        if not all([brokers, topic]):
            raise ValueError("stream_brokers and stream_topic are required")
        
        try:
            if stream_type == "kafka":
                # Apache Kafka - requires kafka-python
                try:
                    from kafka import KafkaConsumer
                except ImportError:
                    raise ValueError("kafka-python is required for Kafka. Install with: pip install kafka-python")
                
                consumer = KafkaConsumer(
                    topic,
                    bootstrap_servers=brokers.split(','),
                    group_id=group_id,
                    auto_offset_reset='earliest',
                    max_poll_records=1000,
                    consumer_timeout_ms=5000  # 5 second timeout
                )
                
                messages = []
                for message in consumer:
                    try:
                        data = json.loads(message.value.decode('utf-8'))
                        messages.append(data)
                    except:
                        messages.append({"raw_value": message.value.decode('utf-8', errors='ignore')})
                
                consumer.close()
                
                if not messages:
                    logger.warning("No messages consumed from Kafka")
                    return pd.DataFrame({"message": ["No data available in topic"]})
                
                df = pd.DataFrame(messages)
                
            elif stream_type == "eventhub":
                # Azure Event Hub - requires azure-eventhub
                logger.warning("Event Hub support requires azure-eventhub package")
                raise ValueError("Event Hub ingestion not fully implemented. Install azure-eventhub package.")
                
            elif stream_type == "kinesis":
                # AWS Kinesis - requires boto3
                logger.warning("Kinesis support requires boto3 package")
                raise ValueError("Kinesis ingestion not fully implemented. Install boto3 package.")
                
            else:
                raise ValueError(f"Unsupported stream type: {stream_type}")
            
            logger.info(f"Stream: consumed {len(df)} messages from {stream_type}")
            return df
            
        except Exception as e:
            logger.error(f"Stream consumption failed: {str(e)}")
            raise ValueError(f"Failed to consume from stream: {str(e)}")

