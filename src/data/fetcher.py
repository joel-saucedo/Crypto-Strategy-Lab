"""
Data fetcher for multiple cryptocurrency exchanges.
Implements comprehensive data quality checks and normalization.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Multi-exchange cryptocurrency data fetcher with quality control.
    """
    
    def __init__(self, config_path: str = "config/data_sources.yaml"):
        """Initialize fetcher with exchange configurations."""
        self.config_path = config_path
        self.load_config()
        self.session = None
        
    def load_config(self) -> None:
        """Load exchange configurations from YAML."""
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
            
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.exchanges = self.config.get('exchanges', {})
        self.symbols = self.config.get('symbols', [])
        self.timeframes = self.config.get('timeframes', ['1h'])
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    async def fetch_binance_klines(
        self, 
        symbol: str, 
        interval: str = '1h',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance API.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1m', '5m', '1h', '1d', etc.)
            start_time: Start timestamp
            end_time: End timestamp  
            limit: Max number of klines (default 1000)
            
        Returns:
            DataFrame with OHLCV data
        """
        base_url = self.exchanges['binance']['base_url']
        endpoint = f"{base_url}/api/v3/klines"
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
            
        async with self.session.get(endpoint, params=params) as response:
            if response.status != 200:
                raise Exception(f"Binance API error: {response.status}")
                
            data = await response.json()
            
        # Convert to DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                  'close_time', 'quote_asset_volume', 'number_of_trades',
                  'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
                  
        df = pd.DataFrame(data, columns=columns)
        
        # Clean and format
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # Add metadata
        df['exchange'] = 'binance'
        df['symbol'] = symbol
        df['timeframe'] = interval
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                  'exchange', 'symbol', 'timeframe']].copy()
    
    async def fetch_coinbase_candles(
        self,
        symbol: str,
        granularity: int = 3600,  # 1 hour in seconds
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Coinbase Pro API.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            granularity: Timeframe in seconds
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            DataFrame with OHLCV data
        """
        base_url = self.exchanges['coinbase']['base_url']
        endpoint = f"{base_url}/products/{symbol}/candles"
        
        params = {'granularity': granularity}
        
        if start_time:
            params['start'] = start_time.isoformat()
        if end_time:
            params['end'] = end_time.isoformat()
            
        async with self.session.get(endpoint, params=params) as response:
            if response.status != 200:
                raise Exception(f"Coinbase API error: {response.status}")
                
            data = await response.json()
            
        # Convert to DataFrame
        columns = ['timestamp', 'low', 'high', 'open', 'close', 'volume']
        df = pd.DataFrame(data, columns=columns)
        
        # Clean and format
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add metadata
        df['exchange'] = 'coinbase'
        df['symbol'] = symbol
        df['timeframe'] = f"{granularity}s"
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume',
                  'exchange', 'symbol', 'timeframe']].copy()
    
    def validate_ohlcv_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Comprehensive OHLCV data validation and quality assessment.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Tuple of (cleaned_df, quality_metrics)
        """
        initial_len = len(df)
        quality_metrics = {
            'initial_rows': initial_len,
            'timestamp_gaps': 0,
            'price_anomalies': 0,
            'volume_anomalies': 0,
            'ohlc_violations': 0,
            'outlier_returns': 0
        }
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # Check timestamp gaps
        if len(df) > 1:
            expected_freq = pd.infer_freq(df['timestamp'])
            if expected_freq:
                full_range = pd.date_range(
                    start=df['timestamp'].min(),
                    end=df['timestamp'].max(),
                    freq=expected_freq
                )
                quality_metrics['timestamp_gaps'] = len(full_range) - len(df)
        
        # OHLC relationship validation
        ohlc_valid = (
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close']) &
            (df['high'] >= df['low'])
        )
        quality_metrics['ohlc_violations'] = (~ohlc_valid).sum()
        
        # Remove OHLC violations
        df = df[ohlc_valid].copy()
        
        # Price anomaly detection (extreme price jumps)
        df['returns'] = df['close'].pct_change()
        return_threshold = df['returns'].std() * 5  # 5-sigma threshold
        
        extreme_returns = df['returns'].abs() > return_threshold
        quality_metrics['outlier_returns'] = extreme_returns.sum()
        
        # Volume anomaly detection
        if 'volume' in df.columns:
            volume_median = df['volume'].median()
            volume_threshold = volume_median * 100  # 100x median volume
            
            extreme_volume = df['volume'] > volume_threshold
            quality_metrics['volume_anomalies'] = extreme_volume.sum()
        
        # Remove extreme outliers
        df = df[~extreme_returns].copy()
        
        quality_metrics['final_rows'] = len(df)
        quality_metrics['data_quality_score'] = len(df) / initial_len if initial_len > 0 else 0
        
        return df, quality_metrics
    
    async def fetch_historical_data(
        self,
        symbol: str,
        exchange: str = 'binance',
        timeframe: str = '1h',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_days: int = 365
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch historical data with automatic pagination and quality control.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange name ('binance' or 'coinbase')
            timeframe: Data timeframe
            start_date: Start date (default: max_days ago)
            end_date: End date (default: now)
            max_days: Maximum lookback period
            
        Returns:
            Tuple of (data_df, quality_metrics)
        """
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=max_days)
            
        all_data = []
        current_start = start_date
        batch_size = timedelta(days=30)  # Fetch in 30-day chunks
        
        logger.info(f"Fetching {symbol} data from {exchange} ({timeframe})")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        while current_start < end_date:
            current_end = min(current_start + batch_size, end_date)
            
            try:
                if exchange == 'binance':
                    # Convert timeframe to Binance format
                    interval_map = {'1m': '1m', '5m': '5m', '1h': '1h', '1d': '1d'}
                    interval = interval_map.get(timeframe, '1h')
                    
                    batch_data = await self.fetch_binance_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=current_start,
                        end_time=current_end
                    )
                    
                elif exchange == 'coinbase':
                    # Convert timeframe to Coinbase granularity
                    granularity_map = {'1m': 60, '5m': 300, '1h': 3600, '1d': 86400}
                    granularity = granularity_map.get(timeframe, 3600)
                    
                    # Convert symbol format for Coinbase
                    if 'USDT' in symbol:
                        cb_symbol = symbol.replace('USDT', '-USD')
                    else:
                        cb_symbol = symbol
                        
                    batch_data = await self.fetch_coinbase_candles(
                        symbol=cb_symbol,
                        granularity=granularity,
                        start_time=current_start,
                        end_time=current_end
                    )
                    
                else:
                    raise ValueError(f"Unsupported exchange: {exchange}")
                
                if len(batch_data) > 0:
                    all_data.append(batch_data)
                    
                await asyncio.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Failed to fetch batch {current_start}-{current_end}: {e}")
                
            current_start = current_end
        
        if not all_data:
            raise Exception(f"No data fetched for {symbol} from {exchange}")
            
        # Combine all batches
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Validate and clean data
        clean_df, quality_metrics = self.validate_ohlcv_data(combined_df)
        
        logger.info(f"Fetched {len(clean_df)} records with quality score: {quality_metrics['data_quality_score']:.3f}")
        
        return clean_df, quality_metrics
    
    async def fetch_multiple_symbols(
        self,
        symbols: Optional[List[str]] = None,
        exchange: str = 'binance',
        timeframe: str = '1h',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Fetch data for multiple symbols concurrently.
        
        Args:
            symbols: List of symbols (default: from config)
            exchange: Exchange name
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping symbol to (data_df, quality_metrics)
        """
        if symbols is None:
            symbols = self.symbols
            
        tasks = []
        for symbol in symbols:
            task = self.fetch_historical_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            tasks.append((symbol, task))
            
        results = {}
        for symbol, task in tasks:
            try:
                data, metrics = await task
                results[symbol] = (data, metrics)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                
        return results
    
    def save_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str = "data/raw"
    ) -> str:
        """
        Save data to parquet format with proper directory structure.
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading pair symbol
            exchange: Exchange name
            timeframe: Data timeframe
            data_dir: Base data directory
            
        Returns:
            Path to saved file
        """
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{symbol}_{exchange}_{timeframe}.parquet"
        filepath = data_path / filename
        
        # Add metadata columns if missing
        if 'exchange' not in data.columns:
            data['exchange'] = exchange
        if 'symbol' not in data.columns:
            data['symbol'] = symbol
        if 'timeframe' not in data.columns:
            data['timeframe'] = timeframe
            
        data.to_parquet(filepath, index=False)
        logger.info(f"Saved {len(data)} records to {filepath}")
        
        return str(filepath)
    
    def load_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str = "data/raw"
    ) -> Optional[pd.DataFrame]:
        """
        Load previously saved data.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            timeframe: Data timeframe
            data_dir: Base data directory
            
        Returns:
            OHLCV DataFrame or None if file doesn't exist
        """
        data_path = Path(data_dir)
        filename = f"{symbol}_{exchange}_{timeframe}.parquet"
        filepath = data_path / filename
        
        if filepath.exists():
            return pd.read_parquet(filepath)
        else:
            return None
