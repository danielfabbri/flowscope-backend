import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("flowscope")


def get_logger(name: str = "flowscope"):
    """Get or create a logger instance."""
    return logging.getLogger(name)


def log_pipeline_step(pipeline_id: str, step_name: str, status: str, message: str = ""):
    """Log pipeline step execution."""
    logger.info(f"[Pipeline {pipeline_id}] {step_name} - {status}: {message}")
