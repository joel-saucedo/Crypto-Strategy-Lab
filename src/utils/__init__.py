"""
Utility functions for the Crypto Strategy Lab.

This module provides essential utility functions for:
- Mathematical calculations and statistical functions
- Data validation and preprocessing helpers
- Configuration management utilities
- Performance optimization tools
- Risk management helpers
"""

from .math_utils import *
from .data_utils import *
from .config_utils import *
from .risk_utils import *
from .performance_utils import *
from .feature_engineering import FeatureEngine

__all__ = [
    # Math utilities
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'rolling_correlation',
    'exponential_moving_average',
    'bollinger_bands',
    
    # Data utilities
    'validate_dataframe',
    'clean_ohlcv_data',
    'resample_data',
    'fill_missing_values',
    'detect_outliers',
    
    # Config utilities
    'load_config',
    'validate_config',
    'merge_configs',
    'get_config_value',
    
    # Risk utilities
    'calculate_var',
    'calculate_cvar',
    'kelly_criterion',
    'position_sizing',
    'risk_parity_weights',
    
    # Performance utilities
    'calculate_portfolio_metrics',
    'create_performance_report',
    'benchmark_comparison',
    
    # Feature Engineering
    'FeatureEngine',
]
