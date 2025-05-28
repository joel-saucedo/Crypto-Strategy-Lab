"""
Financial Modeling Prep (FMP) data provider.
"""

import aiohttp
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import logging
import os

from .base_provider import BaseDataProvider, DataRequest, DataResponse

logger = logging.getLogger(__name__)

class FMPProvider(BaseDataProvider):
    """
    Financial Modeling Prep API data provider.
    Supports stocks, forex, crypto, and commodities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.source_name = 'fmp'
        self.api_key = self.config.get('api_key') or os.getenv('FMP_API_KEY')
        self.base_url = "https://financialmodelingprep.com/api/v3"
        
        if not self.api_key:
            logger.warning("FMP API key not found. Set FMP_API_KEY environment variable.")
        
        # Timeframe mapping for FMP
        self.timeframe_mapping = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1hour',
            '4h': '4hour',
            '1d': '1day'
        }
    
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """
        Fetch data from Financial Modeling Prep.
        
        Args:
            request: DataRequest with symbols and parameters
            
        Returns:
            DataResponse with OHLCV data
        """
        if not self.api_key:
            return DataResponse(
                data=pd.DataFrame(),
                metadata={'error': 'FMP API key not available'},
                source=self.source_name,
                quality_score=0.0,
                warnings=['FMP API key required']
            )
        
        warnings = []
        all_data = []
        
        # Validate timeframe
        fmp_timeframe = self.timeframe_mapping.get(request.timeframe)
        if not fmp_timeframe:
            warnings.append(f"Unsupported timeframe: {request.timeframe}, using 1day")
            fmp_timeframe = '1day'
        
        async with aiohttp.ClientSession() as session:
            for symbol in request.symbols:
                normalized_symbol = self.normalize_symbol(symbol, request.asset_type)
                
                try:
                    symbol_data = await self._fetch_symbol_data(
                        session, normalized_symbol, request, fmp_timeframe
                    )
                    
                    if not symbol_data.empty:
                        symbol_data['symbol'] = normalized_symbol
                        all_data.append(symbol_data)
                    else:
                        warnings.append(f"No data returned for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error fetching {symbol} from FMP: {e}")
                    warnings.append(f"Error fetching {symbol}: {str(e)}")
        
        # Combine all data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = self.standardize_columns(combined_data)
            
            # Add metadata columns
            combined_data['exchange'] = 'fmp'
            combined_data['timeframe'] = request.timeframe
            combined_data['source'] = self.source_name
        else:
            combined_data = pd.DataFrame()
        
        # Calculate quality score
        quality_score = self.calculate_quality_score(combined_data)
        
        # Prepare metadata
        metadata = {
            'request': request,
            'symbol_count': len(combined_data['symbol'].unique()) if 'symbol' in combined_data.columns else 0,
            'date_range': {
                'start': combined_data['timestamp'].min() if 'timestamp' in combined_data.columns else None,
                'end': combined_data['timestamp'].max() if 'timestamp' in combined_data.columns else None
            },
            'total_records': len(combined_data),
            'timeframe_used': fmp_timeframe
        }
        
        return DataResponse(
            data=combined_data,
            metadata=metadata,
            source=self.source_name,
            quality_score=quality_score,
            warnings=warnings
        )
    
    async def _fetch_symbol_data(
        self, 
        session: aiohttp.ClientSession,
        symbol: str, 
        request: DataRequest,
        fmp_timeframe: str
    ) -> pd.DataFrame:
        """Fetch data for a single symbol."""
        
        # Choose appropriate endpoint based on timeframe and asset type
        if fmp_timeframe == '1day':
            endpoint = f"{self.base_url}/historical-price-full/{symbol}"
            params = {
                'apikey': self.api_key,
                'from': request.start_date.strftime('%Y-%m-%d'),
                'to': request.end_date.strftime('%Y-%m-%d')
            }
        else:
            # Intraday data
            endpoint = f"{self.base_url}/historical-chart/{fmp_timeframe}/{symbol}"
            params = {
                'apikey': self.api_key,
                'from': request.start_date.strftime('%Y-%m-%d'),
                'to': request.end_date.strftime('%Y-%m-%d')
            }
        
        async with session.get(endpoint, params=params) as response:
            if response.status != 200:
                raise Exception(f"FMP API error: {response.status}")
            
            data = await response.json()
        
        # Parse response based on endpoint
        if fmp_timeframe == '1day' and 'historical' in data:
            df = pd.DataFrame(data['historical'])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return pd.DataFrame()
        
        if df.empty:
            return df
        
        # Standardize column names
        column_mapping = {
            'date': 'timestamp',
            'datetime': 'timestamp'
        }
        df = df.rename(columns=column_mapping)
        
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def validate_symbols(self, symbols: List[str], asset_type: str) -> Tuple[List[str], List[str]]:
        """
        Validate symbols for FMP.
        
        Note: This is a simplified validation. In practice, you'd want to
        call FMP's symbol search API to validate.
        """
        valid_symbols = []
        invalid_symbols = []
        
        for symbol in symbols:
            normalized = self.normalize_symbol(symbol, asset_type)
            
            # Basic validation - check if symbol looks reasonable
            if len(normalized) >= 1 and normalized.replace('-', '').replace('.', '').isalnum():
                valid_symbols.append(normalized)
            else:
                invalid_symbols.append(symbol)
        
        return valid_symbols, invalid_symbols
    
    def normalize_symbol(self, symbol: str, asset_type: str) -> str:
        """
        Normalize symbol for FMP format.
        
        Args:
            symbol: Raw symbol
            asset_type: Type of asset
            
        Returns:
            Normalized symbol for FMP
        """
        symbol = symbol.upper().strip()
        
        if asset_type == 'crypto':
            # FMP crypto symbols typically end with USD
            crypto_mapping = {
                'BTC': 'BTCUSD',
                'ETH': 'ETHUSD',
                'ADA': 'ADAUSD',
                'SOL': 'SOLUSD',
                'MATIC': 'MATICUSD',
                'AVAX': 'AVAXUSD',
                'DOT': 'DOTUSD',
                'LINK': 'LINKUSD',
                'UNI': 'UNIUSD',
                'LTC': 'LTCUSD',
                'BCH': 'BCHUSD',
                'XRP': 'XRPUSD',
                'DOGE': 'DOGEUSD'
            }
            
            if symbol in crypto_mapping:
                return crypto_mapping[symbol]
            
            # If it doesn't end with USD, add it
            if not symbol.endswith('USD') and len(symbol) <= 5:
                return f"{symbol}USD"
        
        elif asset_type == 'forex':
            # FMP forex pairs are typically like EURUSD, GBPUSD
            if len(symbol) == 6:
                return symbol
            elif '/' in symbol:
                return symbol.replace('/', '')
        
        return symbol
    
    def get_available_timeframes(self) -> List[str]:
        """Get supported timeframes."""
        return list(self.timeframe_mapping.keys())
