"""
Configuration management utilities.
"""

import yaml
import json
import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import warnings


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    return config or {}


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration against a schema.
    
    Args:
        config: Configuration to validate
        schema: Schema definition
        
    Returns:
        Validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required fields
    required_fields = schema.get('required', [])
    for field in required_fields:
        if field not in config:
            results['valid'] = False
            results['errors'].append(f"Missing required field: {field}")
    
    # Check field types
    field_types = schema.get('types', {})
    for field, expected_type in field_types.items():
        if field in config:
            actual_value = config[field]
            if not isinstance(actual_value, expected_type):
                results['errors'].append(f"Field '{field}' should be {expected_type.__name__}, got {type(actual_value).__name__}")
                results['valid'] = False
    
    # Check allowed values
    allowed_values = schema.get('allowed_values', {})
    for field, allowed in allowed_values.items():
        if field in config:
            if config[field] not in allowed:
                results['errors'].append(f"Field '{field}' value '{config[field]}' not in allowed values: {allowed}")
                results['valid'] = False
    
    # Check numeric ranges
    ranges = schema.get('ranges', {})
    for field, (min_val, max_val) in ranges.items():
        if field in config:
            value = config[field]
            if isinstance(value, (int, float)):
                if value < min_val or value > max_val:
                    results['errors'].append(f"Field '{field}' value {value} outside allowed range [{min_val}, {max_val}]")
                    results['valid'] = False
    
    return results


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration
    """
    merged = {}
    
    for config in configs:
        if isinstance(config, dict):
            merged = _deep_merge(merged, config)
    
    return merged


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        update: Dictionary to merge in
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., 'database.host')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., 'database.host')
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def load_strategy_config(strategy_name: str, config_dir: str = "config/strategies") -> Dict[str, Any]:
    """
    Load strategy-specific configuration.
    
    Args:
        strategy_name: Name of the strategy
        config_dir: Directory containing strategy configs
        
    Returns:
        Strategy configuration
    """
    config_path = Path(config_dir) / f"{strategy_name}.yaml"
    
    if not config_path.exists():
        # Try with .yml extension
        config_path = Path(config_dir) / f"{strategy_name}.yml"
    
    if not config_path.exists():
        warnings.warn(f"Strategy configuration not found: {strategy_name}")
        return {}
    
    return load_config(config_path)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for the backtesting framework.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'backtest': {
            'initial_capital': 100000.0,
            'commission': 0.001,
            'slippage': 0.0005,
            'max_positions': 10,
            'position_sizing': {
                'method': 'fixed_fractional',
                'risk_per_trade': 0.02,
                'max_position_size': 0.1
            }
        },
        'data': {
            'source': 'local',
            'timeframe': '1d',
            'start_date': '2020-01-01',
            'end_date': None,
            'symbols': ['BTCUSD']
        },
        'validation': {
            'enable_monte_carlo': True,
            'monte_carlo_runs': 1000,
            'min_trades': 30,
            'min_dsr': 0.95,
            'outlier_threshold': 3.0
        },
        'risk_management': {
            'max_drawdown_limit': 0.2,
            'stop_loss': 0.05,
            'take_profit': 0.1,
            'position_timeout': 30
        },
        'logging': {
            'level': 'INFO',
            'file': 'logs/backtest.log',
            'console': True
        }
    }


def validate_strategy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate strategy configuration.
    
    Args:
        config: Strategy configuration
        
    Returns:
        Validation results
    """
    schema = {
        'required': ['name', 'parameters'],
        'types': {
            'name': str,
            'parameters': dict,
            'description': str
        },
        'allowed_values': {},
        'ranges': {}
    }
    
    return validate_config(config, schema)


def create_config_template(config_type: str) -> Dict[str, Any]:
    """
    Create a configuration template for different components.
    
    Args:
        config_type: Type of configuration ('strategy', 'backtest', 'data')
        
    Returns:
        Configuration template
    """
    templates = {
        'strategy': {
            'name': 'my_strategy',
            'description': 'Strategy description',
            'parameters': {
                'lookback_period': 20,
                'threshold': 0.02
            },
            'risk_management': {
                'stop_loss': 0.05,
                'take_profit': 0.1
            }
        },
        'backtest': {
            'initial_capital': 100000.0,
            'commission': 0.001,
            'slippage': 0.0005,
            'start_date': '2020-01-01',
            'end_date': '2023-12-31',
            'benchmark': 'BTCUSD'
        },
        'data': {
            'source': 'local',
            'symbols': ['BTCUSD'],
            'timeframe': '1d',
            'data_dir': 'data/raw',
            'cache_dir': 'data/cache'
        }
    }
    
    return templates.get(config_type, {})


def expand_config_variables(config: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand variables in configuration using template substitution.
    
    Args:
        config: Configuration with variables to expand
        variables: Dictionary of variable values
        
    Returns:
        Configuration with expanded variables
    """
    def expand_value(value):
        if isinstance(value, str):
            for var_name, var_value in variables.items():
                placeholder = f"${{{var_name}}}"
                if placeholder in value:
                    value = value.replace(placeholder, str(var_value))
        elif isinstance(value, dict):
            return {k: expand_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [expand_value(item) for item in value]
        return value
    
    return expand_value(config)
