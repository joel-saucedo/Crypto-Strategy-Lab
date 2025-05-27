"""
True-Range Divergence Mean-Reversion Strategy Module

This module implements Strategy #12 from the 12 orthogonal trading edges compendium.
It exploits divergences between True Range volatility and price momentum to identify
mean-reversion opportunities.

Key Features:
- True Range volatility analysis with gap detection
- Price momentum divergence identification  
- Mean-reversion signal generation with timing
- Volatility regime filtering
- Comprehensive statistical validation

Mathematical Foundation:
The strategy identifies periods where volatility (measured by True Range) and price
momentum show divergent patterns, indicating potential exhaustion and reversal.

Usage:
    from strategies.true_range_divergence.strategy import TrueRangeDivergenceSignal
    
    strategy = TrueRangeDivergenceSignal()
    signals = strategy.generate(ohlcv_data)
"""

from .strategy import TrueRangeDivergenceSignal

__all__ = ['TrueRangeDivergenceSignal']
__version__ = '1.0.0'
__author__ = 'Crypto Strategy Lab'
__description__ = 'True-Range Divergence Mean-Reversion Trading Strategy'
