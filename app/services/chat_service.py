from typing import Dict, Any
import pandas as pd

from app.pipeline.storage import storage


class ChatService:
    """Simple chat service for querying pipeline data."""
    
    def __init__(self):
        self.intent_map = {
            "anomaly": self._handle_anomaly_query,
            "trend": self._handle_trend_query,
            "summary": self._handle_summary_query,
            "transformation": self._handle_transformation_query,
            "forecast": self._handle_forecast_query,
        }
    
    def query(self, pipeline_id: str, query_text: str) -> Dict[str, Any]:
        """Process chat query and return response."""
        query_lower = query_text.lower()
        
        # Simple intent detection
        intent = self._detect_intent(query_lower)
        
        # Get handler
        handler = self.intent_map.get(intent, self._handle_default)
        
        # Execute handler
        return handler(pipeline_id, query_text)
    
    def _detect_intent(self, query: str) -> str:
        """Simple keyword-based intent detection."""
        if any(word in query for word in ["anomaly", "anomalies", "outlier", "unusual"]):
            return "anomaly"
        elif any(word in query for word in ["trend", "pattern", "over time"]):
            return "trend"
        elif any(word in query for word in ["summary", "overview", "stats", "statistics"]):
            return "summary"
        elif any(word in query for word in ["transform", "transformation", "change", "process"]):
            return "transformation"
        elif any(word in query for word in ["forecast", "prediction", "predict", "future"]):
            return "forecast"
        
        return "default"
    
    def _handle_anomaly_query(self, pipeline_id: str, query: str) -> Dict[str, Any]:
        """Handle anomaly-related queries."""
        # Get latest stage data
        stages = storage.list_stages(pipeline_id)
        if not stages:
            return {"answer": "No data available yet. Please run the pipeline first."}
        
        # Try to get data from ML stage
        data = None
        for stage in reversed(stages):
            data = storage.get_stage_data(pipeline_id, stage)
            if isinstance(data, pd.DataFrame) and "anomaly_detected" in data.columns:
                break
        
        if data is None or not isinstance(data, pd.DataFrame):
            return {"answer": "No anomaly detection data available."}
        
        # Count anomalies
        if "anomaly_detected" in data.columns:
            num_anomalies = data["anomaly_detected"].sum()
            total = len(data)
            pct = (num_anomalies / total * 100) if total > 0 else 0
            
            response = {
                "answer": f"Detected {num_anomalies} anomalies out of {total} records ({pct:.2f}%).",
                "data": {
                    "num_anomalies": int(num_anomalies),
                    "total_records": int(total),
                    "percentage": round(pct, 2)
                }
            }
            
            # Include sample anomalies
            if num_anomalies > 0:
                anomalies = data[data["anomaly_detected"] == 1].head(5)
                response["data"]["samples"] = anomalies[["timestamp", "value"]].to_dict("records") if "timestamp" in anomalies.columns else anomalies.head(5).to_dict("records")
            
            return response
        
        return {"answer": "No anomaly information found in the data."}
    
    def _handle_trend_query(self, pipeline_id: str, query: str) -> Dict[str, Any]:
        """Handle trend-related queries."""
        stages = storage.list_stages(pipeline_id)
        if not stages:
            return {"answer": "No data available yet."}
        
        # Get latest data
        data = storage.get_stage_data(pipeline_id, stages[-1])
        
        if not isinstance(data, pd.DataFrame) or "value" not in data.columns:
            return {"answer": "No value data available for trend analysis."}
        
        # Calculate basic trend statistics
        first_val = data["value"].iloc[0]
        last_val = data["value"].iloc[-1]
        mean_val = data["value"].mean()
        std_val = data["value"].std()
        
        trend_direction = "increasing" if last_val > first_val else "decreasing"
        
        return {
            "answer": f"The data shows a {trend_direction} trend. Mean: {mean_val:.2f}, Std: {std_val:.2f}",
            "data": {
                "first_value": round(first_val, 2),
                "last_value": round(last_val, 2),
                "mean": round(mean_val, 2),
                "std": round(std_val, 2),
                "trend": trend_direction
            }
        }
    
    def _handle_summary_query(self, pipeline_id: str, query: str) -> Dict[str, Any]:
        """Handle summary queries."""
        stages = storage.list_stages(pipeline_id)
        
        if not stages:
            return {"answer": "No data available yet."}
        
        # Get latest data
        data = storage.get_stage_data(pipeline_id, stages[-1])
        
        if not isinstance(data, pd.DataFrame):
            return {"answer": "Unable to summarize data."}
        
        summary = {
            "num_records": len(data),
            "num_columns": len(data.columns),
            "columns": list(data.columns),
            "stages_completed": stages
        }
        
        return {
            "answer": f"Pipeline has {len(stages)} stages completed with {len(data)} records and {len(data.columns)} columns.",
            "data": summary
        }
    
    def _handle_transformation_query(self, pipeline_id: str, query: str) -> Dict[str, Any]:
        """Handle transformation queries."""
        stages = storage.list_stages(pipeline_id)
        
        return {
            "answer": f"Pipeline completed {len(stages)} transformation stages: {', '.join(stages)}",
            "data": {"stages": stages}
        }
    
    def _handle_forecast_query(self, pipeline_id: str, query: str) -> Dict[str, Any]:
        """Handle forecast queries."""
        stages = storage.list_stages(pipeline_id)
        if not stages:
            return {"answer": "No data available yet."}
        
        # Get latest data
        data = storage.get_stage_data(pipeline_id, stages[-1])
        
        if not isinstance(data, pd.DataFrame) or "forecast" not in data.columns:
            return {"answer": "No forecast data available. Make sure ML modeling is enabled."}
        
        # Get last few forecast values
        forecast_data = data[["timestamp", "value", "forecast"]].tail(10).to_dict("records") if "timestamp" in data.columns else []
        
        return {
            "answer": "Forecast data is available in the ML output.",
            "data": {"forecast_samples": forecast_data}
        }
    
    def _handle_default(self, pipeline_id: str, query: str) -> Dict[str, Any]:
        """Default handler."""
        return {
            "answer": "I can help you with: anomalies, trends, summaries, transformations, and forecasts. Try asking about those!"
        }


# Global chat service instance
chat_service = ChatService()
