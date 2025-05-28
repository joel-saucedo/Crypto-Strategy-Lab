"""
Base data provider interface for consistent data fetching across sources.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass

@dataclass
class DataRequest:
    """Standard data request structure."""
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    timeframe: str = '1h'
    asset_type: str = 'crypto'  # crypto, stock, forex, commodity
    metadata: Dict[str, Any] = None

@dataclass  
class DataResponse:
    """Standard data response structure."""
    data: pd.DataFrame
    metadata: Dict[str, Any]
    source: str
    quality_score: float
    warnings: List[str] = None

class BaseDataProvider(ABC):
    """
    Abstract base class for all data providers.
    Ensures consistent interface across Yahoo Finance, FMP, exchanges, etc.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.source_name = self.__class__.__name__.replace('Provider', '').lower()
    
    @abstractmethod
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """
        Fetch OHLCV data for given request.
        
        Args:
            request: DataRequest with symbols, dates, timeframe
            
        Returns:
            DataResponse with standardized OHLCV data
        """
        pass
    
    @abstractmethod
    def validate_symbols(self, symbols: List[str], asset_type: str) -> Tuple[List[str], List[str]]:
        """
        Validate symbols for this provider.
        
        Returns:
            Tuple of (valid_symbols, invalid_symbols)
        """
        pass
    
    @abstractmethod
    def get_available_timeframes(self) -> List[str]:
        """Get supported timeframes for this provider."""
        pass
    
    def normalize_symbol(self, symbol: str, asset_type: str) -> str:
        """Normalize symbol format for this provider."""
        return symbol.upper()
    
    def calculate_quality_score(self, data: pd.DataFrame) -> float:
        """
        Calculate data quality score (0-1).
        
        Factors:
        - Missing data percentage
        - OHLC validity
        - Volume consistency
        - Timestamp gaps
        """
        if data.empty:
            return 0.0
            
        score = 1.0
        
        # Penalize missing data
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        score -= missing_pct * 0.3
        
        # Check OHLC validity
        if 'high' in data.columns and 'low' in data.columns:
            invalid_ohlc = (data['high'] < data['low']).sum()
            score -= (invalid_ohlc / len(data)) * 0.2
        
        # Check for reasonable volume (if available)
        if 'volume' in data.columns:
            zero_volume_pct = (data['volume'] == 0).sum() / len(data)
            score -= zero_volume_pct * 0.1
        
        return max(0.0, min(1.0, score))
    
    def standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and format."""
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
            'Date': 'timestamp',
            'Datetime': 'timestamp'
        }
        
        # Rename columns
        data = data.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                if col == 'volume':
                    data[col] = 0  # Default volume to 0 if missing
                else:
                    raise ValueError(f"Required column '{col}' missing from data")
        
        # Ensure timestamp is datetime
        if 'timestamp' not in data.columns and data.index.name in ['Date', 'Datetime']:
            data = data.reset_index()
            data = data.rename(columns={data.columns[0]: 'timestamp'})
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data
