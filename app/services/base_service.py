from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class BasePipelineService(ABC):
    """Base class for all pipeline services."""
    
    @abstractmethod
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """Execute the service logic.
        
        Args:
            data: Input DataFrame (None for first step)
            config: Step-specific configuration
            
        Returns:
            Processed DataFrame
        """
        pass
