"""
Yahoo Finance data provider using yfinance library.
"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import logging

from .base_provider import BaseDataProvider, DataRequest, DataResponse

logger = logging.getLogger(__name__)

class YFinanceProvider(BaseDataProvider):
    """
    Yahoo Finance data provider for stocks, crypto, forex, and other assets.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.source_name = 'yfinance'
        
        # Timeframe mapping
        self.timeframe_mapping = {
            '1m': '1m',
            '2m': '2m', 
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '60m': '1h',
            '1h': '1h',
            '1d': '1d',
            '5d': '5d',
            '1wk': '1wk',
            '1mo': '1mo',
            '3mo': '3mo'
        }
    
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """
        Fetch data from Yahoo Finance.
        
        Args:
            request: DataRequest with symbols and parameters
            
        Returns:
            DataResponse with OHLCV data
        """
        warnings = []
        
        # Validate timeframe
        yf_timeframe = self.timeframe_mapping.get(request.timeframe)
        if not yf_timeframe:
            raise ValueError(f"Unsupported timeframe: {request.timeframe}")
        
        # Normalize symbols for Yahoo Finance
        normalized_symbols = [self.normalize_symbol(symbol, request.asset_type) 
                            for symbol in request.symbols]
        
        try:
            # Use yfinance to fetch data
            if len(normalized_symbols) == 1:
                ticker = yf.Ticker(normalized_symbols[0])
                data = ticker.history(
                    start=request.start_date,
                    end=request.end_date + timedelta(days=1),  # Include end date
                    interval=yf_timeframe,
                    auto_adjust=True,
                    prepost=True
                )
                
                if data.empty:
                    warnings.append(f"No data returned for {normalized_symbols[0]}")
                    return DataResponse(
                        data=pd.DataFrame(),
                        metadata={'request': request, 'symbol_count': 0},
                        source=self.source_name,
                        quality_score=0.0,
                        warnings=warnings
                    )
                
                # Add symbol column
                data['symbol'] = normalized_symbols[0]
                
            else:
                # Download multiple symbols
                data = yf.download(
                    normalized_symbols,
                    start=request.start_date,
                    end=request.end_date + timedelta(days=1),
                    interval=yf_timeframe,
                    auto_adjust=True,
                    prepost=True,
                    group_by='ticker'
                )
                
                if data.empty:
                    warnings.append("No data returned for any symbols")
                    return DataResponse(
                        data=pd.DataFrame(),
                        metadata={'request': request, 'symbol_count': 0},
                        source=self.source_name,
                        quality_score=0.0,
                        warnings=warnings
                    )
                
                # Reshape multi-symbol data
                data = self._reshape_multi_symbol_data(data, normalized_symbols)
            
            # Standardize the data format
            data = self.standardize_columns(data)
            
            # Add metadata columns
            data['exchange'] = 'yahoo'
            data['timeframe'] = request.timeframe
            data['source'] = self.source_name
            
            # Calculate quality score
            quality_score = self.calculate_quality_score(data)
            
            # Prepare metadata
            metadata = {
                'request': request,
                'symbol_count': len(data['symbol'].unique()) if 'symbol' in data.columns else 1,
                'date_range': {
                    'start': data['timestamp'].min() if 'timestamp' in data.columns else None,
                    'end': data['timestamp'].max() if 'timestamp' in data.columns else None
                },
                'total_records': len(data),
                'timeframe_used': yf_timeframe
            }
            
            return DataResponse(
                data=data,
                metadata=metadata,
                source=self.source_name,
                quality_score=quality_score,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance: {e}")
            warnings.append(f"Fetch error: {str(e)}")
            
            return DataResponse(
                data=pd.DataFrame(),
                metadata={'request': request, 'error': str(e)},
                source=self.source_name,
                quality_score=0.0,
                warnings=warnings
            )
    
    def _reshape_multi_symbol_data(self, data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Reshape multi-symbol data from yfinance into long format."""
        reshaped_data = []
        
        for symbol in symbols:
            if symbol in data.columns.get_level_values(1):
                symbol_data = data.xs(symbol, level=1, axis=1)
                symbol_data = symbol_data.dropna()
                
                if not symbol_data.empty:
                    symbol_data = symbol_data.reset_index()
                    symbol_data['symbol'] = symbol
                    reshaped_data.append(symbol_data)
        
        if reshaped_data:
            return pd.concat(reshaped_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def validate_symbols(self, symbols: List[str], asset_type: str) -> Tuple[List[str], List[str]]:
        """
        Validate symbols for Yahoo Finance.
        
        Args:
            symbols: List of symbols to validate
            asset_type: Type of asset (crypto, stock, etc.)
            
        Returns:
            Tuple of (valid_symbols, invalid_symbols)
        """
        valid_symbols = []
        invalid_symbols = []
        
        for symbol in symbols:
            normalized = self.normalize_symbol(symbol, asset_type)
            
            try:
                # Quick test fetch to validate symbol
                ticker = yf.Ticker(normalized)
                info = ticker.info
                
                # Check if we got valid data
                if info and 'symbol' in info:
                    valid_symbols.append(normalized)
                else:
                    invalid_symbols.append(symbol)
                    
            except Exception:
                invalid_symbols.append(symbol)
        
        return valid_symbols, invalid_symbols
    
    def normalize_symbol(self, symbol: str, asset_type: str) -> str:
        """
        Normalize symbol for Yahoo Finance format.
        
        Args:
            symbol: Raw symbol
            asset_type: Type of asset
            
        Returns:
            Normalized symbol for Yahoo Finance
        """
        symbol = symbol.upper().strip()
        
        if asset_type == 'crypto':
            # Convert crypto symbols to Yahoo Finance format
            crypto_mapping = {
                'BTC': 'BTC-USD',
                'ETH': 'ETH-USD', 
                'ADA': 'ADA-USD',
                'SOL': 'SOL-USD',
                'MATIC': 'MATIC-USD',
                'AVAX': 'AVAX-USD',
                'DOT': 'DOT-USD',
                'LINK': 'LINK-USD',
                'UNI': 'UNI-USD',
                'LTC': 'LTC-USD',
                'BCH': 'BCH-USD',
                'XRP': 'XRP-USD',
                'DOGE': 'DOGE-USD',
                'SHIB': 'SHIB-USD'
            }
            
            # If it's a base symbol, convert it
            if symbol in crypto_mapping:
                return crypto_mapping[symbol]
            
            # If it doesn't end with -USD, add it
            if not symbol.endswith('-USD') and not symbol.endswith('USD'):
                if len(symbol) <= 5:  # Likely a crypto base symbol
                    return f"{symbol}-USD"
        
        return symbol
    
    def get_available_timeframes(self) -> List[str]:
        """Get supported timeframes."""
        return list(self.timeframe_mapping.keys())
