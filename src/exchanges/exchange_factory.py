"""
Exchange Factory

This module provides a factory pattern for creating exchange adapters,
enabling easy switching between different exchanges and trading modes.
"""

from typing import Dict, Type, Optional, Any
from .base_exchange import BaseExchange, TradingMode
import logging
import os
from pathlib import Path


class ExchangeFactory:
    """
    Factory class for creating exchange adapters
    
    This factory manages the registration and creation of different exchange
    adapters, providing a unified interface for instantiating exchanges.
    """
    
    _exchanges: Dict[str, Type[BaseExchange]] = {}
    _logger = logging.getLogger(__name__)
    
    @classmethod
    def register_exchange(cls, name: str, exchange_class: Type[BaseExchange]) -> None:
        """
        Register an exchange adapter class
        
        Args:
            name: Name of the exchange (e.g., 'bybit', 'alpaca')
            exchange_class: Exchange adapter class
        """
        cls._exchanges[name.lower()] = exchange_class
        cls._logger.info(f"Registered exchange adapter: {name}")
    
    @classmethod
    def get_available_exchanges(cls) -> list[str]:
        """
        Get list of available exchange names
        
        Returns:
            List of registered exchange names
        """
        return list(cls._exchanges.keys())
    
    @classmethod
    def create_exchange(cls, 
                       exchange_name: str,
                       api_key: Optional[str] = None,
                       api_secret: Optional[str] = None,
                       mode: TradingMode = TradingMode.PAPER,
                       **kwargs) -> BaseExchange:
        """
        Create an exchange adapter instance
        
        Args:
            exchange_name: Name of the exchange
            api_key: API key (if not provided, will try to load from environment)
            api_secret: API secret (if not provided, will try to load from environment)
            mode: Trading mode
            **kwargs: Additional exchange-specific parameters
        
        Returns:
            Exchange adapter instance
        
        Raises:
            ValueError: If exchange is not registered or credentials are missing
        """
        exchange_name = exchange_name.lower()
        
        if exchange_name not in cls._exchanges:
            available = ', '.join(cls._exchanges.keys())
            raise ValueError(f"Exchange '{exchange_name}' not registered. Available: {available}")
        
        # Load credentials from environment if not provided
        if api_key is None:
            api_key = cls._get_env_var(exchange_name, 'API_KEY')
        if api_secret is None:
            api_secret = cls._get_env_var(exchange_name, 'API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError(f"Missing API credentials for {exchange_name}. "
                           f"Provide api_key/api_secret or set environment variables.")
        
        exchange_class = cls._exchanges[exchange_name]
        
        try:
            return exchange_class(
                api_key=api_key,
                api_secret=api_secret,
                mode=mode,
                **kwargs
            )
        except Exception as e:
            cls._logger.error(f"Failed to create {exchange_name} exchange: {e}")
            raise
    
    @classmethod
    def _get_env_var(cls, exchange_name: str, var_type: str) -> Optional[str]:
        """
        Get environment variable for exchange credentials
        
        Args:
            exchange_name: Name of the exchange
            var_type: Type of variable ('API_KEY' or 'API_SECRET')
        
        Returns:
            Environment variable value or None
        """
        # Try multiple naming conventions
        var_names = [
            f"{exchange_name.upper()}_{var_type}",
            f"{exchange_name.upper()}_API_{var_type.split('_')[1]}",
            f"{exchange_name}_{var_type}".lower(),
        ]
        
        for var_name in var_names:
            value = os.getenv(var_name)
            if value:
                return value
        
        return None
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> BaseExchange:
        """
        Create exchange adapter from configuration dictionary
        
        Args:
            config: Configuration dictionary containing exchange settings
        
        Returns:
            Exchange adapter instance
        
        Example config:
        {
            "exchange": "bybit",
            "mode": "testnet",
            "api_key": "your_key",
            "api_secret": "your_secret",
            "additional_params": {}
        }
        """
        exchange_name = config.get('exchange')
        if not exchange_name:
            raise ValueError("Missing 'exchange' in configuration")
        
        mode_str = config.get('mode', 'paper')
        try:
            mode = TradingMode(mode_str)
        except ValueError:
            raise ValueError(f"Invalid trading mode: {mode_str}")
        
        return cls.create_exchange(
            exchange_name=exchange_name,
            api_key=config.get('api_key'),
            api_secret=config.get('api_secret'),
            mode=mode,
            **config.get('additional_params', {})
        )
    
    @classmethod
    def load_config_from_file(cls, config_path: str) -> Dict[str, Any]:
        """
        Load exchange configuration from file
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
        
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix.lower() == '.json':
            import json
            with open(config_path, 'r') as f:
                return json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required to load YAML configuration files")
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


# Auto-registration of available exchanges
def _auto_register_exchanges():
    """Auto-register available exchange adapters"""
    try:
        from .bybit_adapter import BybitAdapter
        ExchangeFactory.register_exchange('bybit', BybitAdapter)
    except ImportError:
        pass
    
    try:
        from .alpaca_adapter import AlpacaAdapter
        ExchangeFactory.register_exchange('alpaca', AlpacaAdapter)
    except ImportError:
        pass


# Register exchanges on module import
_auto_register_exchanges()
