"""
Backtesting package for strategy evaluation.
"""

from .portfolio_manager import PortfolioManager, Position, Trade
from ..core.backtest_engine import MultiStrategyOrchestrator as BacktestEngine, BacktestConfig, BacktestResult

__all__ = [
    'PortfolioManager',
    'Position',
    'Trade', 
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult'
]
