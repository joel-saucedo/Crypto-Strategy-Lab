"""
Factory for creating data providers and managing provider configurations.
"""

from typing import Dict, Type, Any, List
from .base_provider import BaseDataProvider, DataRequest, DataResponse
from .yfinance_provider import YFinanceProvider
from .fmp_provider import FMPProvider
from .exchange_provider import ExchangeProvider

class ProviderFactory:
    """Factory for creating and managing data providers."""
    
    _providers: Dict[str, Type[BaseDataProvider]] = {
        'yahoo': YFinanceProvider,
        'yfinance': YFinanceProvider,
        'fmp': FMPProvider,
        'financial_modeling_prep': FMPProvider,
        'exchange': ExchangeProvider,
        'binance': ExchangeProvider,
        'coinbase': ExchangeProvider,
    }
    
    @classmethod
    def create_provider(cls, provider_name: str, config: Dict[str, Any] = None) -> BaseDataProvider:
        """
        Create a data provider instance.
        
        Args:
            provider_name: Name of provider (yahoo, fmp, exchange)
            config: Provider-specific configuration
            
        Returns:
            Configured provider instance
        """
        provider_name = provider_name.lower()
        
        if provider_name not in cls._providers:
            available = ', '.join(cls._providers.keys())
            raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")
        
        provider_class = cls._providers[provider_name]
        return provider_class(config or {})
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available provider names."""
        return list(cls._providers.keys())
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseDataProvider]):
        """Register a new provider class."""
        cls._providers[name.lower()] = provider_class
