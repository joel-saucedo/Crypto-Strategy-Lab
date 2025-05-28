"""
Unified data manager that coordinates multiple data providers and handles
data fetching, caching, and fallback strategies.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle

try:
    from .providers.provider_factory import ProviderFactory
    from .providers.base_provider import DataRequest, DataResponse, BaseDataProvider
except ImportError:
    # Handle direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.providers.provider_factory import ProviderFactory
    from data.providers.base_provider import DataRequest, DataResponse, BaseDataProvider

logger = logging.getLogger(__name__)

class DataManager:
    """
    Unified data manager that handles multiple providers, caching, and fallbacks.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.providers: Dict[str, BaseDataProvider] = {}
        self.cache_dir = Path(self.config.get('cache_dir', 'data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = self.config.get('enable_cache', True)
        
        # Provider priority order for fallbacks
        self.provider_priority = self.config.get('provider_priority', ['yahoo', 'fmp', 'exchange'])
        
        # Initialize providers
        self._setup_providers()
    
    def _setup_providers(self):
        """Initialize configured data providers."""
        provider_configs = self.config.get('providers', {})
        
        for provider_name in self.provider_priority:
            try:
                provider_config = provider_configs.get(provider_name, {})
                provider = ProviderFactory.create_provider(provider_name, provider_config)
                self.providers[provider_name] = provider
                logger.info(f"Initialized provider: {provider_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize provider {provider_name}: {e}")
    
    async def fetch_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        timeframe: str = '1h',
        asset_type: str = 'crypto',
        preferred_provider: str = None,
        enable_fallback: bool = True
    ) -> pd.DataFrame:
        """
        Fetch data with automatic fallback between providers.
        
        Args:
            symbols: Symbol or list of symbols to fetch
            start_date: Start date for data
            end_date: End date for data (defaults to now)
            timeframe: Data timeframe (1m, 5m, 1h, 1d, etc.)
            asset_type: Type of asset (crypto, stock, forex)
            preferred_provider: Preferred provider to try first
            enable_fallback: Whether to try fallback providers
            
        Returns:
            Combined DataFrame with OHLCV data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Check cache first
        cache_key = self._get_cache_key(symbols, start_date, end_date, timeframe, asset_type)
        if self.cache_enabled:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Loaded data from cache for {symbols}")
                return cached_data
        
        # Create data request
        request = DataRequest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            asset_type=asset_type
        )
        
        # Determine provider order
        provider_order = self._get_provider_order(preferred_provider)
        
        # Try providers in order
        best_response = None
        best_score = 0
        
        for provider_name in provider_order:
            if provider_name not in self.providers:
                continue
                
            try:
                provider = self.providers[provider_name]
                response = await provider.fetch_data(request)
                
                if response.data is not None and not response.data.empty:
                    logger.info(f"Successfully fetched data from {provider_name} "
                               f"(quality: {response.quality_score:.2f})")
                    
                    # Use best quality response or first successful one
                    if response.quality_score > best_score or best_response is None:
                        best_response = response
                        best_score = response.quality_score
                    
                    # If quality is excellent and fallback disabled, stop here
                    if response.quality_score >= 0.9 and not enable_fallback:
                        break
                        
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue
        
        if best_response is None or best_response.data.empty:
            raise RuntimeError(f"Failed to fetch data for {symbols} from any provider")
        
        # Cache the result
        if self.cache_enabled and best_response.quality_score >= 0.7:
            self._save_to_cache(cache_key, best_response.data)
        
        return best_response.data
    
    def _get_provider_order(self, preferred_provider: str = None) -> List[str]:
        """Get ordered list of providers to try."""
        if preferred_provider and preferred_provider in self.providers:
            order = [preferred_provider]
            order.extend([p for p in self.provider_priority if p != preferred_provider])
            return order
        return self.provider_priority.copy()
    
    def _get_cache_key(self, symbols: List[str], start_date: datetime, 
                      end_date: datetime, timeframe: str, asset_type: str) -> str:
        """Generate cache key for data request."""
        symbols_str = "_".join(sorted(symbols))
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        return f"{symbols_str}_{start_str}_{end_str}_{timeframe}_{asset_type}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and not expired."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # Check if cache is expired (default 1 hour for intraday, 1 day for daily+)
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        max_age = timedelta(hours=1) if any(tf in cache_key for tf in ['1m', '5m', '15m', '30m', '1h']) else timedelta(days=1)
        
        if cache_age > max_age:
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def clear_cache(self):
        """Clear all cached data."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache cleared")
    
    async def get_available_symbols(self, asset_type: str = 'crypto', 
                                   provider: str = None) -> List[str]:
        """Get list of available symbols from provider."""
        provider_name = provider or self.provider_priority[0]
        if provider_name not in self.providers:
            return []
        
        try:
            provider = self.providers[provider_name]
            if hasattr(provider, 'get_available_symbols'):
                return await provider.get_available_symbols(asset_type)
        except Exception as e:
            logger.warning(f"Failed to get symbols from {provider_name}: {e}")
        
        return []
    
    def get_provider_status(self) -> Dict[str, bool]:
        """Check which providers are available and working."""
        status = {}
        for name, provider in self.providers.items():
            try:
                # Simple health check - could be enhanced
                status[name] = provider is not None
            except:
                status[name] = False
        return status
