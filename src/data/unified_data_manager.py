"""
Unified Data Management System - Consolidates all data providers and management
functionality into a single, comprehensive interface.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle
import json
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

try:
    from .providers.provider_factory import ProviderFactory
    from .providers.base_provider import DataRequest, DataResponse, BaseDataProvider
    from .data_manager import DataManager
    from .fetcher import DataFetcher
    from .preprocessor import DataPreprocessor
    from ..utils.data_utils import validate_ohlcv_data, detect_gaps
except ImportError:
    # Handle direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.providers.provider_factory import ProviderFactory
    from data.providers.base_provider import DataRequest, DataResponse, BaseDataProvider
    from data.data_manager import DataManager
    from data.fetcher import DataFetcher
    from data.preprocessor import DataPreprocessor
    from utils.data_utils import validate_ohlcv_data, detect_gaps

logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics."""
    completeness: float  # % of expected data points
    consistency: float   # Data consistency score
    accuracy: float      # Price accuracy score
    freshness: float     # How recent the data is
    overall_score: float # Combined quality score
    gaps_detected: int   # Number of data gaps
    anomalies_detected: int  # Number of anomalies
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class UnifiedDataResponse:
    """Enhanced data response with comprehensive metadata."""
    data: pd.DataFrame
    quality_metrics: DataQualityMetrics
    provider_used: str
    providers_attempted: List[str]
    fetch_time: float
    cache_hit: bool
    metadata: Dict[str, Any]

class UnifiedDataManager:
    """
    Unified data manager that consolidates all data management functionality.
    Replaces scattered data management across multiple files.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.data_manager = DataManager(self.config.get('data_manager', {}))
        self.fetcher = DataFetcher(self.config.get('fetcher_config_path', 'config/data_sources.yaml'))
        self.preprocessor = DataPreprocessor()
        
        # Caching and performance
        self.cache_dir = Path(self.config.get('cache_dir', 'data/cache/unified'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = self.config.get('enable_cache', True)
        self.max_cache_age_hours = self.config.get('max_cache_age_hours', 24)
        
        # Provider management
        self.provider_priority = self.config.get('provider_priority', ['yahoo', 'fmp', 'exchange'])
        self.fallback_enabled = self.config.get('enable_fallback', True)
        self.parallel_fetch = self.config.get('parallel_fetch', False)
        
        # Quality thresholds
        self.min_quality_score = self.config.get('min_quality_score', 0.7)
        self.max_gap_percentage = self.config.get('max_gap_percentage', 0.05)
        
        # Performance tracking
        self._fetch_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'provider_successes': {name: 0 for name in self.provider_priority},
            'provider_failures': {name: 0 for name in self.provider_priority},
            'avg_fetch_time': 0.0
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'cache_dir': 'data/cache/unified',
            'enable_cache': True,
            'max_cache_age_hours': 24,
            'provider_priority': ['yahoo', 'fmp', 'exchange'],
            'enable_fallback': True,
            'parallel_fetch': False,
            'min_quality_score': 0.7,
            'max_gap_percentage': 0.05,
            'data_manager': {
                'enable_cache': True,
                'cache_dir': 'data/cache/providers'
            },
            'fetcher_config_path': 'config/data_sources.yaml'
        }
    
    async def fetch_unified_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        timeframe: str = '1h',
        asset_type: str = 'crypto',
        preferred_provider: str = None,
        quality_threshold: float = None,
        preprocess: bool = True,
        validate_quality: bool = True
    ) -> UnifiedDataResponse:
        """
        Fetch data with comprehensive quality assessment and fallback handling.
        
        Args:
            symbols: Symbol or list of symbols to fetch
            start_date: Start date for data
            end_date: End date for data (defaults to now)
            timeframe: Data timeframe (1m, 5m, 1h, 1d, etc.)
            asset_type: Type of asset (crypto, stock, forex)
            preferred_provider: Preferred provider to try first
            quality_threshold: Minimum quality score (overrides config)
            preprocess: Whether to preprocess the data
            validate_quality: Whether to perform quality validation
            
        Returns:
            UnifiedDataResponse with comprehensive metadata
        """
        start_time = time.time()
        self._fetch_stats['total_requests'] += 1
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        quality_threshold = quality_threshold or self.min_quality_score
        
        # Check cache first
        cache_key = self._get_cache_key(symbols, start_date, end_date, timeframe, asset_type)
        cached_response = await self._load_from_cache(cache_key)
        if cached_response:
            self._fetch_stats['cache_hits'] += 1
            logger.info(f"Loaded data from unified cache for {symbols}")
            return cached_response
        
        # Determine fetch strategy
        if self.parallel_fetch and len(self.provider_priority) > 1:
            response = await self._fetch_parallel(symbols, start_date, end_date, 
                                               timeframe, asset_type, preferred_provider)
        else:
            response = await self._fetch_sequential(symbols, start_date, end_date,
                                                  timeframe, asset_type, preferred_provider)
        
        # Quality validation
        if validate_quality:
            response.quality_metrics = self._assess_data_quality(response.data, symbols, 
                                                               start_date, end_date, timeframe)
            
            if response.quality_metrics.overall_score < quality_threshold:
                logger.warning(f"Data quality below threshold: {response.quality_metrics.overall_score:.2f} < {quality_threshold}")
        
        # Preprocessing
        if preprocess and not response.data.empty:
            response.data = self.preprocessor.process_data(response.data)
            response.metadata['preprocessed'] = True
        
        # Update performance stats
        fetch_time = time.time() - start_time
        response.fetch_time = fetch_time
        self._update_performance_stats(fetch_time)
        
        # Cache the result
        if (self.cache_enabled and 
            response.quality_metrics.overall_score >= quality_threshold and
            not response.data.empty):
            await self._save_to_cache(cache_key, response)
        
        return response
    
    async def _fetch_sequential(self, symbols: List[str], start_date: datetime,
                              end_date: datetime, timeframe: str, asset_type: str,
                              preferred_provider: str = None) -> UnifiedDataResponse:
        """Fetch data sequentially from providers."""
        providers_attempted = []
        best_response = None
        best_score = 0
        
        # Determine provider order
        provider_order = self._get_provider_order(preferred_provider)
        
        for provider_name in provider_order:
            providers_attempted.append(provider_name)
            
            try:
                # Try DataManager first
                data = await self.data_manager.fetch_data(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe,
                    asset_type=asset_type,
                    preferred_provider=provider_name,
                    enable_fallback=False
                )
                
                if data is not None and not data.empty:
                    quality_metrics = self._assess_data_quality(data, symbols, start_date, end_date, timeframe)
                    
                    response = UnifiedDataResponse(
                        data=data,
                        quality_metrics=quality_metrics,
                        provider_used=provider_name,
                        providers_attempted=providers_attempted.copy(),
                        fetch_time=0.0,  # Will be set later
                        cache_hit=False,
                        metadata={'source': 'data_manager', 'provider': provider_name}
                    )
                    
                    self._fetch_stats['provider_successes'][provider_name] += 1
                    
                    if quality_metrics.overall_score > best_score or best_response is None:
                        best_response = response
                        best_score = quality_metrics.overall_score
                    
                    if quality_metrics.overall_score >= 0.9 and not self.fallback_enabled:
                        break
                        
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                self._fetch_stats['provider_failures'][provider_name] += 1
                
                # Try DataFetcher as fallback
                try:
                    async with self.fetcher as fetcher:
                        data = await fetcher.fetch_multiple_symbols(
                            symbols, start_date, end_date, timeframe
                        )
                    
                    if data and not data.empty:
                        quality_metrics = self._assess_data_quality(data, symbols, start_date, end_date, timeframe)
                        
                        response = UnifiedDataResponse(
                            data=data,
                            quality_metrics=quality_metrics,
                            provider_used=f"{provider_name}_fetcher",
                            providers_attempted=providers_attempted.copy(),
                            fetch_time=0.0,
                            cache_hit=False,
                            metadata={'source': 'fetcher', 'provider': provider_name}
                        )
                        
                        if quality_metrics.overall_score > best_score or best_response is None:
                            best_response = response
                            best_score = quality_metrics.overall_score
                            
                except Exception as e2:
                    logger.warning(f"Fetcher fallback for {provider_name} also failed: {e2}")
                    continue
        
        if best_response is None:
            # Create empty response
            best_response = UnifiedDataResponse(
                data=pd.DataFrame(),
                quality_metrics=DataQualityMetrics(0, 0, 0, 0, 0, 0, 0),
                provider_used="none",
                providers_attempted=providers_attempted,
                fetch_time=0.0,
                cache_hit=False,
                metadata={'error': 'All providers failed'}
            )
        
        return best_response
    
    async def _fetch_parallel(self, symbols: List[str], start_date: datetime,
                            end_date: datetime, timeframe: str, asset_type: str,
                            preferred_provider: str = None) -> UnifiedDataResponse:
        """Fetch data in parallel from multiple providers."""
        provider_order = self._get_provider_order(preferred_provider)
        
        async def fetch_from_provider(provider_name: str) -> Optional[UnifiedDataResponse]:
            try:
                data = await self.data_manager.fetch_data(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe,
                    asset_type=asset_type,
                    preferred_provider=provider_name,
                    enable_fallback=False
                )
                
                if data is not None and not data.empty:
                    quality_metrics = self._assess_data_quality(data, symbols, start_date, end_date, timeframe)
                    
                    return UnifiedDataResponse(
                        data=data,
                        quality_metrics=quality_metrics,
                        provider_used=provider_name,
                        providers_attempted=[provider_name],
                        fetch_time=0.0,
                        cache_hit=False,
                        metadata={'source': 'data_manager', 'provider': provider_name}
                    )
            except Exception as e:
                logger.warning(f"Parallel fetch from {provider_name} failed: {e}")
                return None
        
        # Launch parallel fetches
        tasks = [fetch_from_provider(provider) for provider in provider_order[:3]]  # Limit parallel requests
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Find best response
        best_response = None
        best_score = 0
        providers_attempted = []
        
        for i, response in enumerate(responses):
            provider_name = provider_order[i]
            providers_attempted.append(provider_name)
            
            if isinstance(response, UnifiedDataResponse):
                if response.quality_metrics.overall_score > best_score or best_response is None:
                    best_response = response
                    best_score = response.quality_metrics.overall_score
                    best_response.providers_attempted = providers_attempted.copy()
        
        if best_response is None:
            # Fallback to sequential
            return await self._fetch_sequential(symbols, start_date, end_date, 
                                              timeframe, asset_type, preferred_provider)
        
        return best_response
    
    def _assess_data_quality(self, data: pd.DataFrame, symbols: List[str],
                           start_date: datetime, end_date: datetime, 
                           timeframe: str) -> DataQualityMetrics:
        """Comprehensive data quality assessment."""
        if data.empty:
            return DataQualityMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Completeness: percentage of expected data points
        expected_periods = self._calculate_expected_periods(start_date, end_date, timeframe)
        actual_periods = len(data)
        completeness = min(actual_periods / expected_periods, 1.0) if expected_periods > 0 else 0
        
        # Consistency: check for data anomalies
        consistency_score = 1.0
        anomalies_detected = 0
        
        try:
            # Check for negative prices
            if 'close' in data.columns and (data['close'] <= 0).any():
                consistency_score -= 0.2
                anomalies_detected += (data['close'] <= 0).sum()
            
            # Check for extreme price movements (>50% in one period)
            if 'close' in data.columns and len(data) > 1:
                price_changes = data['close'].pct_change().abs()
                extreme_moves = (price_changes > 0.5).sum()
                if extreme_moves > 0:
                    consistency_score -= min(extreme_moves * 0.1, 0.3)
                    anomalies_detected += extreme_moves
            
            # Check OHLCV consistency
            ohlcv_issues = validate_ohlcv_data(data)
            if ohlcv_issues['has_issues']:
                consistency_score -= 0.1
                anomalies_detected += len(ohlcv_issues['issues'])
        except Exception as e:
            logger.warning(f"Error in consistency check: {e}")
            consistency_score = 0.5
        
        # Accuracy: based on data validation
        accuracy_score = 0.9 if not data.empty else 0
        
        # Freshness: how recent is the data
        if not data.empty and 'timestamp' in data.columns:
            latest_data_time = pd.to_datetime(data['timestamp'].iloc[-1])
            time_diff = datetime.now() - latest_data_time
            freshness = max(0, 1 - (time_diff.total_seconds() / (24 * 3600)))  # Decay over 24 hours
        else:
            freshness = 0.5
        
        # Gap detection
        gaps_detected = 0
        try:
            gap_info = detect_gaps(data, timeframe)
            gaps_detected = len(gap_info.get('gaps', []))
            if gaps_detected > 0:
                completeness *= (1 - min(gaps_detected * 0.1, 0.5))
        except Exception as e:
            logger.warning(f"Error in gap detection: {e}")
        
        # Overall score (weighted average)
        weights = {'completeness': 0.3, 'consistency': 0.3, 'accuracy': 0.2, 'freshness': 0.2}
        overall_score = (
            completeness * weights['completeness'] +
            consistency_score * weights['consistency'] +
            accuracy_score * weights['accuracy'] +
            freshness * weights['freshness']
        )
        
        return DataQualityMetrics(
            completeness=completeness,
            consistency=consistency_score,
            accuracy=accuracy_score,
            freshness=freshness,
            overall_score=overall_score,
            gaps_detected=gaps_detected,
            anomalies_detected=anomalies_detected
        )
    
    def _calculate_expected_periods(self, start_date: datetime, end_date: datetime, timeframe: str) -> int:
        """Calculate expected number of data periods."""
        time_diff = end_date - start_date
        
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440, '1w': 10080
        }
        
        minutes = timeframe_minutes.get(timeframe, 60)
        total_minutes = time_diff.total_seconds() / 60
        
        return int(total_minutes / minutes)
    
    def _get_provider_order(self, preferred_provider: str = None) -> List[str]:
        """Get provider order with preferred provider first."""
        if preferred_provider and preferred_provider in self.provider_priority:
            order = [preferred_provider]
            order.extend([p for p in self.provider_priority if p != preferred_provider])
            return order
        return self.provider_priority.copy()
    
    def _get_cache_key(self, symbols: List[str], start_date: datetime,
                      end_date: datetime, timeframe: str, asset_type: str) -> str:
        """Generate cache key for data request."""
        symbols_str = "_".join(sorted(symbols))
        start_str = start_date.strftime('%Y%m%d_%H%M')
        end_str = end_date.strftime('%Y%m%d_%H%M')
        return f"unified_{symbols_str}_{start_str}_{end_str}_{timeframe}_{asset_type}"
    
    async def _load_from_cache(self, cache_key: str) -> Optional[UnifiedDataResponse]:
        """Load data from unified cache."""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        metadata_file = self.cache_dir / f"{cache_key}_meta.json"
        
        if not cache_file.exists() or not metadata_file.exists():
            return None
        
        try:
            # Check cache age
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age > (self.max_cache_age_hours * 3600):
                return None
            
            # Load data and metadata
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Reconstruct response
            quality_metrics = DataQualityMetrics(**metadata['quality_metrics'])
            
            response = UnifiedDataResponse(
                data=data,
                quality_metrics=quality_metrics,
                provider_used=metadata['provider_used'],
                providers_attempted=metadata['providers_attempted'],
                fetch_time=metadata['fetch_time'],
                cache_hit=True,
                metadata=metadata['metadata']
            )
            
            return response
            
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None
    
    async def _save_to_cache(self, cache_key: str, response: UnifiedDataResponse):
        """Save data to unified cache."""
        if not self.cache_enabled:
            return
        
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            metadata_file = self.cache_dir / f"{cache_key}_meta.json"
            
            # Save data
            with open(cache_file, 'wb') as f:
                pickle.dump(response.data, f)
            
            # Save metadata
            metadata = {
                'quality_metrics': response.quality_metrics.to_dict(),
                'provider_used': response.provider_used,
                'providers_attempted': response.providers_attempted,
                'fetch_time': response.fetch_time,
                'metadata': response.metadata,
                'cached_at': time.time()
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def _update_performance_stats(self, fetch_time: float):
        """Update performance statistics."""
        total_requests = self._fetch_stats['total_requests']
        current_avg = self._fetch_stats['avg_fetch_time']
        
        # Update running average
        self._fetch_stats['avg_fetch_time'] = (
            (current_avg * (total_requests - 1) + fetch_time) / total_requests
        )
    
    # ============================================================================
    # ADDITIONAL UNIFIED METHODS
    # ============================================================================
    
    async def get_available_symbols(self, asset_type: str = 'crypto') -> Dict[str, List[str]]:
        """Get available symbols from all providers."""
        symbols_by_provider = {}
        
        for provider_name in self.provider_priority:
            try:
                symbols = await self.data_manager.get_available_symbols(asset_type, provider_name)
                if symbols:
                    symbols_by_provider[provider_name] = symbols
            except Exception as e:
                logger.warning(f"Failed to get symbols from {provider_name}: {e}")
        
        return symbols_by_provider
    
    async def validate_symbols(self, symbols: List[str], asset_type: str = 'crypto') -> Dict[str, bool]:
        """Validate if symbols are available across providers."""
        available_symbols = await self.get_available_symbols(asset_type)
        all_symbols = set()
        for provider_symbols in available_symbols.values():
            all_symbols.update(provider_symbols)
        
        return {symbol: symbol in all_symbols for symbol in symbols}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get unified data manager performance statistics."""
        return {
            'fetch_stats': self._fetch_stats.copy(),
            'cache_stats': {
                'enabled': self.cache_enabled,
                'cache_hit_rate': (self._fetch_stats['cache_hits'] / 
                                 max(self._fetch_stats['total_requests'], 1)) * 100,
                'cache_dir_size': sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
            },
            'provider_stats': {
                'priority_order': self.provider_priority,
                'success_rates': {
                    name: (successes / max(successes + failures, 1)) * 100
                    for name, successes in self._fetch_stats['provider_successes'].items()
                    for failures in [self._fetch_stats['provider_failures'][name]]
                }
            }
        }
    
    async def clear_cache(self, older_than_hours: Optional[int] = None):
        """Clear unified cache."""
        if not self.cache_dir.exists():
            return
        
        cutoff_time = time.time() - (older_than_hours * 3600 if older_than_hours else 0)
        
        for cache_file in self.cache_dir.rglob('*'):
            if cache_file.is_file():
                if older_than_hours is None or cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
        
        logger.info(f"Cache cleared (older than {older_than_hours} hours)" if older_than_hours else "Cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all data sources."""
        health_status = {
            'overall_status': 'healthy',
            'providers': {},
            'cache': {
                'enabled': self.cache_enabled,
                'accessible': self.cache_dir.exists() and self.cache_dir.is_dir()
            },
            'performance': self.get_performance_stats()
        }
        
        # Test each provider
        test_symbols = ['BTCUSD', 'ETHUSD']
        test_start = datetime.now() - timedelta(days=1)
        test_end = datetime.now()
        
        for provider_name in self.provider_priority:
            try:
                response = await self.fetch_unified_data(
                    symbols=test_symbols[:1],  # Test with one symbol
                    start_date=test_start,
                    end_date=test_end,
                    timeframe='1h',
                    preferred_provider=provider_name,
                    validate_quality=True
                )
                
                health_status['providers'][provider_name] = {
                    'status': 'healthy' if not response.data.empty else 'no_data',
                    'quality_score': response.quality_metrics.overall_score,
                    'fetch_time': response.fetch_time
                }
                
            except Exception as e:
                health_status['providers'][provider_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Determine overall status
        healthy_providers = sum(1 for p in health_status['providers'].values() 
                              if p.get('status') == 'healthy')
        
        if healthy_providers == 0:
            health_status['overall_status'] = 'critical'
        elif healthy_providers < len(self.provider_priority) / 2:
            health_status['overall_status'] = 'degraded'
        
        return health_status

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global instance for easy access
_unified_manager = None

def get_unified_manager(config: Dict[str, Any] = None) -> UnifiedDataManager:
    """Get global unified data manager instance."""
    global _unified_manager
    if _unified_manager is None:
        _unified_manager = UnifiedDataManager(config)
    return _unified_manager

async def fetch_data_unified(symbols: Union[str, List[str]], 
                           start_date: Union[str, datetime],
                           end_date: Union[str, datetime] = None,
                           **kwargs) -> pd.DataFrame:
    """Convenience function for unified data fetching."""
    manager = get_unified_manager()
    response = await manager.fetch_unified_data(symbols, start_date, end_date, **kwargs)
    return response.data

# Export key classes and functions
__all__ = [
    'UnifiedDataManager',
    'UnifiedDataResponse', 
    'DataQualityMetrics',
    'get_unified_manager',
    'fetch_data_unified'
]
