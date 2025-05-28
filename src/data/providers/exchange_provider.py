"""
Exchange-based data provider using existing fetcher.py.
Wraps the current exchange APIs for consistency with the new provider framework.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import asyncio
import logging

from .base_provider import BaseDataProvider, DataRequest, DataResponse
from ..fetcher import DataFetcher

logger = logging.getLogger(__name__)

class ExchangeProvider(BaseDataProvider):
    """
    Exchange data provider that wraps the existing DataFetcher.
    Supports Binance, Coinbase, Kraken, etc.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.source_name = 'exchange'
        self.config_path = config.get('config_path', 'config/data_sources.yaml')
        
        # Timeframe mapping for exchanges
        self.timeframe_mapping = {
            '1m': '1m',
            '3m': '3m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h', 
            '4h': '4h',
            '6h': '6h',
            '8h': '8h',
            '12h': '12h',
            '1d': '1d',
            '3d': '3d',
            '1w': '1w',
            '1M': '1M'
        }
    
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """
        Fetch data from cryptocurrency exchanges.
        
        Args:
            request: DataRequest with symbols and parameters
            
        Returns:
            DataResponse with OHLCV data
        """
        warnings = []
        
        # Validate timeframe
        exchange_timeframe = self.timeframe_mapping.get(request.timeframe)
        if not exchange_timeframe:
            warnings.append(f"Unsupported timeframe: {request.timeframe}, using 1h")
            exchange_timeframe = '1h'
        
        try:
            async with DataFetcher(self.config_path) as fetcher:
                all_data = []
                
                for symbol in request.symbols:
                    normalized_symbol = self.normalize_symbol(symbol, request.asset_type)
                    
                    try:
                        # Try different exchanges in order of preference
                        exchanges = ['binance', 'coinbase', 'kraken']
                        symbol_data = None
                        
                        for exchange in exchanges:
                            try:
                                if exchange == 'binance':
                                    symbol_data = await fetcher.fetch_binance_klines(
                                        symbol=normalized_symbol,
                                        interval=exchange_timeframe,
                                        start_time=request.start_date,
                                        end_time=request.end_date
                                    )
                                elif exchange == 'coinbase':
                                    # Convert timeframe to seconds for Coinbase
                                    granularity = self._timeframe_to_seconds(exchange_timeframe)
                                    symbol_data = await fetcher.fetch_coinbase_candles(
                                        symbol=normalized_symbol,
                                        granularity=granularity,
                                        start_time=request.start_date,
                                        end_time=request.end_date
                                    )
                                
                                if symbol_data is not None and not symbol_data.empty:
                                    symbol_data['symbol'] = normalized_symbol
                                    symbol_data['source_exchange'] = exchange
                                    all_data.append(symbol_data)
                                    break
                                    
                            except Exception as e:
                                logger.debug(f"Failed to fetch {symbol} from {exchange}: {e}")
                                continue
                        
                        if symbol_data is None or symbol_data.empty:
                            warnings.append(f"No data available for {symbol} from any exchange")
                            
                    except Exception as e:
                        logger.error(f"Error fetching {symbol}: {e}")
                        warnings.append(f"Error fetching {symbol}: {str(e)}")
                
                # Combine all data
                if all_data:
                    combined_data = pd.concat(all_data, ignore_index=True)
                    combined_data = self.standardize_columns(combined_data)
                    
                    # Add metadata columns
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
                    'exchanges_used': combined_data['source_exchange'].unique().tolist() if 'source_exchange' in combined_data.columns else []
                }
                
                return DataResponse(
                    data=combined_data,
                    metadata=metadata,
                    source=self.source_name,
                    quality_score=quality_score,
                    warnings=warnings
                )
                
        except Exception as e:
            logger.error(f"Error in ExchangeProvider: {e}")
            return DataResponse(
                data=pd.DataFrame(),
                metadata={'error': str(e)},
                source=self.source_name,
                quality_score=0.0,
                warnings=[f"Provider error: {str(e)}"]
            )
    
    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert timeframe to seconds for Coinbase API."""
        mapping = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '6h': 21600,
            '1d': 86400
        }
        return mapping.get(timeframe, 3600)  # Default to 1 hour
    
    def validate_symbols(self, symbols: List[str], asset_type: str) -> Tuple[List[str], List[str]]:
        """
        Validate symbols for exchange APIs.
        
        Args:
            symbols: List of symbols to validate
            asset_type: Type of asset (should be 'crypto' for exchanges)
            
        Returns:
            Tuple of (valid_symbols, invalid_symbols)
        """
        valid_symbols = []
        invalid_symbols = []
        
        for symbol in symbols:
            normalized = self.normalize_symbol(symbol, asset_type)
            
            # Basic validation for crypto pairs
            if asset_type == 'crypto':
                # Should be in format like BTCUSDT, ETH-USD, etc.
                if (len(normalized) >= 6 and 
                    (normalized.endswith('USDT') or normalized.endswith('USD') or 
                     '-USD' in normalized or normalized.endswith('BTC') or normalized.endswith('ETH'))):
                    valid_symbols.append(normalized)
                else:
                    invalid_symbols.append(symbol)
            else:
                # Non-crypto assets not supported by exchange provider
                invalid_symbols.append(symbol)
        
        return valid_symbols, invalid_symbols
    
    def normalize_symbol(self, symbol: str, asset_type: str) -> str:
        """
        Normalize symbol for exchange APIs.
        
        Args:
            symbol: Raw symbol
            asset_type: Type of asset
            
        Returns:
            Normalized symbol for exchanges (Binance format preferred)
        """
        symbol = symbol.upper().strip()
        
        if asset_type == 'crypto':
            # Convert to Binance-style format (BTCUSDT)
            crypto_mapping = {
                'BTC': 'BTCUSDT',
                'ETH': 'ETHUSDT',
                'ADA': 'ADAUSDT',
                'SOL': 'SOLUSDT',
                'MATIC': 'MATICUSDT',
                'AVAX': 'AVAXUSDT',
                'DOT': 'DOTUSDT',
                'LINK': 'LINKUSDT',
                'UNI': 'UNIUSDT',
                'LTC': 'LTCUSDT',
                'BCH': 'BCHUSDT',
                'XRP': 'XRPUSDT',
                'DOGE': 'DOGEUSDT'
            }
            
            # Handle different input formats
            if symbol in crypto_mapping:
                return crypto_mapping[symbol]
            
            # Handle formats like BTC-USD, BTC/USD
            if '-' in symbol or '/' in symbol:
                base = symbol.replace('-USD', '').replace('/USD', '').replace('-USDT', '').replace('/USDT', '')
                if base in crypto_mapping:
                    return crypto_mapping[base]
                # Convert to USDT pair
                return f"{base}USDT"
            
            # If it doesn't end with USDT/USD, add USDT
            if not symbol.endswith('USDT') and not symbol.endswith('USD') and len(symbol) <= 5:
                return f"{symbol}USDT"
        
        return symbol
    
    def get_available_timeframes(self) -> List[str]:
        """Get supported timeframes."""
        return list(self.timeframe_mapping.keys())
