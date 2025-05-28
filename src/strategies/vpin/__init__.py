"""
VPIN (Volume-Synchronized Order-Flow Imbalance) Strategy

A trading strategy based on detecting toxic order flow through volume-synchronized
probability of informed trading. Identifies periods of high order flow imbalance
that typically precede adverse price movements.

Example usage:
    from src.strategies.vpin import VPINStrategy
    
    config = {
        'lookback': 50,
        'buckets': 50,
        'vpin_history_length': 252,
        'toxic_threshold_pct': 95,
        'benign_threshold_pct': 10,
        'ema_smoothing': 0.1,
        'position_scale_factor': 0.5
    }
    
    strategy = VPINStrategy(config)
    signals = strategy.generate_signal(ohlcv_data)
"""

from .signal import VPINStrategy

__all__ = ['VPINStrategy']
__version__ = '1.0.0'
__author__ = 'Crypto Strategy Lab'
__description__ = 'VPIN Order-Flow Imbalance Trading Strategy'
