"""
Base strategy class for all trading strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must implement the generate_signal method.
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.parameters = {}
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> float:
        """
        Generate a trading signal based on the provided data.
        
        Args:
            data: OHLCV price data as pandas DataFrame with columns:
                  ['open', 'high', 'low', 'close', 'volume']
                  
        Returns:
            Float signal between -1.0 and 1.0:
            - Positive values indicate buy signals (strength proportional to value)
            - Negative values indicate sell/short signals  
            - 0.0 indicates no signal
            - Values closer to ±1.0 indicate stronger conviction
        """
        pass
    
    def set_parameters(self, **kwargs):
        """Set strategy parameters."""
        self.parameters.update(kwargs)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return self.parameters.copy()
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the input data has required columns and sufficient length.
        
        Args:
            data: Input price data
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['close']  # Minimum requirement
        
        if data.empty:
            return False
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False
        
        # Check for sufficient data points
        if len(data) < 2:
            return False
        
        return True
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data before signal generation.
        Override this method for custom preprocessing.
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Preprocessed data
        """
        return data.copy()
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.parameters})"
