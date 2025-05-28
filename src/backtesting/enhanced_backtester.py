"""
Enhanced backtesting engine supporting multiple strategies and assets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
import asyncio
import logging
from pathlib import Path

try:
    from .portfolio_manager import PortfolioManager, Trade
    from ..data.data_manager import DataManager
    from ..strategies.base_strategy import BaseStrategy
except ImportError:
    # Handle direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backtesting.portfolio_manager import PortfolioManager, Trade
    from data.data_manager import DataManager
    from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class BacktestConfig:
    """Configuration for backtesting."""
    
    def __init__(self,
                 start_date: Union[str, datetime],
                 end_date: Union[str, datetime],
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001,
                 timeframe: str = '1h',
                 benchmark_symbol: str = 'BTC-USD',
                 enable_short_selling: bool = True,
                 max_position_size: float = 0.1,
                 max_total_exposure: float = 0.8):
        
        self.start_date = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        self.end_date = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.timeframe = timeframe
        self.benchmark_symbol = benchmark_symbol
        self.enable_short_selling = enable_short_selling
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure

class BacktestResult:
    """Results from a backtest run."""
    
    def __init__(self, portfolio_manager: PortfolioManager, 
                 benchmark_data: pd.DataFrame = None,
                 strategy_performances: Dict[str, Dict] = None):
        self.portfolio_manager = portfolio_manager
        self.benchmark_data = benchmark_data
        self.strategy_performances = strategy_performances or {}
        
        # Calculate metrics
        self.metrics = portfolio_manager.get_performance_metrics()
        self.trades = portfolio_manager.closed_trades
        self.portfolio_history = pd.DataFrame(portfolio_manager.portfolio_history)
        
    def get_strategy_breakdown(self) -> Dict[str, Dict]:
        """Get performance breakdown by strategy."""
        if not self.trades:
            return {}
        
        strategy_stats = {}
        
        for strategy_id in set(trade.strategy_id for trade in self.trades):
            strategy_trades = [t for t in self.trades if t.strategy_id == strategy_id]
            
            if strategy_trades:
                strategy_df = pd.DataFrame([{
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'symbol': trade.symbol
                } for trade in strategy_trades])
                
                winning_trades = strategy_df[strategy_df['pnl'] > 0]
                losing_trades = strategy_df[strategy_df['pnl'] < 0]
                
                strategy_stats[strategy_id] = {
                    'total_trades': len(strategy_trades),
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades),
                    'win_rate': len(winning_trades) / len(strategy_trades) * 100,
                    'total_pnl': strategy_df['pnl'].sum(),
                    'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
                    'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
                }
        
        return strategy_stats
    
    def get_symbol_breakdown(self) -> Dict[str, Dict]:
        """Get performance breakdown by symbol."""
        if not self.trades:
            return {}
        
        symbol_stats = {}
        
        for symbol in set(trade.symbol for trade in self.trades):
            symbol_trades = [t for t in self.trades if t.symbol == symbol]
            
            if symbol_trades:
                symbol_df = pd.DataFrame([{
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'strategy_id': trade.strategy_id
                } for trade in symbol_trades])
                
                symbol_stats[symbol] = {
                    'total_trades': len(symbol_trades),
                    'total_pnl': symbol_df['pnl'].sum(),
                    'strategies_used': symbol_df['strategy_id'].nunique(),
                    'avg_pnl_per_trade': symbol_df['pnl'].mean()
                }
        
        return symbol_stats

class EnhancedBacktester:
    """
    Enhanced backtesting engine that supports:
    - Multiple strategies running simultaneously
    - Multiple assets per strategy
    - Portfolio management with risk controls
    - Detailed performance analytics
    """
    
    def __init__(self, data_manager: DataManager = None):
        self.data_manager = data_manager or DataManager()
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_symbols: Dict[str, List[str]] = {}
        
    def add_strategy(self, strategy: BaseStrategy, symbols: List[str], strategy_id: str = None):
        """
        Add a strategy to the backtest.
        
        Args:
            strategy: Strategy instance
            symbols: List of symbols this strategy will trade
            strategy_id: Unique identifier for the strategy
        """
        if strategy_id is None:
            strategy_id = strategy.__class__.__name__
        
        # Ensure unique strategy ID
        original_id = strategy_id
        counter = 1
        while strategy_id in self.strategies:
            strategy_id = f"{original_id}_{counter}"
            counter += 1
        
        self.strategies[strategy_id] = strategy
        self.strategy_symbols[strategy_id] = symbols
        
        logger.info(f"Added strategy '{strategy_id}' for symbols: {symbols}")
    
    async def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """
        Run the backtest with all configured strategies.
        
        Args:
            config: Backtest configuration
            
        Returns:
            BacktestResult with performance metrics and data
        """
        logger.info(f"Starting backtest from {config.start_date} to {config.end_date}")
        
        # Collect all unique symbols
        all_symbols = set()
        for symbols in self.strategy_symbols.values():
            all_symbols.update(symbols)
        all_symbols = list(all_symbols)
        
        if config.benchmark_symbol and config.benchmark_symbol not in all_symbols:
            all_symbols.append(config.benchmark_symbol)
        
        # Fetch data for all symbols
        logger.info(f"Fetching data for {len(all_symbols)} symbols")
        data = await self.data_manager.fetch_data(
            symbols=all_symbols,
            start_date=config.start_date,
            end_date=config.end_date,
            timeframe=config.timeframe
        )
        
        # Initialize portfolio manager
        portfolio = PortfolioManager(
            initial_capital=config.initial_capital,
            commission_rate=config.commission_rate
        )
        portfolio.max_position_size = config.max_position_size
        portfolio.max_total_exposure = config.max_total_exposure
        
        # Prepare strategy data
        strategy_data = {}
        for strategy_id, symbols in self.strategy_symbols.items():
            strategy_data[strategy_id] = self._prepare_strategy_data(data, symbols)
        
        # Run backtest simulation
        await self._run_simulation(data, strategy_data, portfolio, config)
        
        # Get benchmark data if specified
        benchmark_data = None
        if config.benchmark_symbol and config.benchmark_symbol in data.columns:
            benchmark_data = data[config.benchmark_symbol].copy()
        
        # Calculate individual strategy performances
        strategy_performances = {}
        for strategy_id in self.strategies.keys():
            strategy_trades = [t for t in portfolio.closed_trades if t.strategy_id == strategy_id]
            if strategy_trades:
                strategy_pnl = sum(t.pnl for t in strategy_trades)
                strategy_performances[strategy_id] = {
                    'total_pnl': strategy_pnl,
                    'trade_count': len(strategy_trades),
                    'win_rate': len([t for t in strategy_trades if t.pnl > 0]) / len(strategy_trades) * 100
                }
        
        return BacktestResult(
            portfolio_manager=portfolio,
            benchmark_data=benchmark_data,
            strategy_performances=strategy_performances
        )
    
    def _prepare_strategy_data(self, full_data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Prepare data subset for a specific strategy."""
        # Get columns that match the symbols (handling multi-level columns)
        if isinstance(full_data.columns, pd.MultiIndex):
            strategy_columns = []
            for symbol in symbols:
                symbol_columns = [col for col in full_data.columns if col[1] == symbol]
                strategy_columns.extend(symbol_columns)
            return full_data[strategy_columns].copy()
        else:
            # Simple column structure
            available_symbols = [col for col in symbols if col in full_data.columns]
            return full_data[available_symbols].copy()
    
    async def _run_simulation(self, data: pd.DataFrame, strategy_data: Dict[str, pd.DataFrame],
                            portfolio: PortfolioManager, config: BacktestConfig):
        """Run the actual backtest simulation."""
        
        # Get timestamps from data
        timestamps = data.index
        
        logger.info(f"Running simulation over {len(timestamps)} time periods")
        
        for i, timestamp in enumerate(timestamps):
            
            # Get current prices for all symbols
            current_prices = {}
            if isinstance(data.columns, pd.MultiIndex):
                for col in data.columns:
                    if col[0] == 'close':  # Assuming OHLCV structure
                        symbol = col[1]
                        price = data.iloc[i][col]
                        if not pd.isna(price):
                            current_prices[symbol] = price
            else:
                # Simple column structure - assume prices are in the data
                for col in data.columns:
                    price = data.iloc[i][col]
                    if not pd.isna(price):
                        current_prices[col] = price
            
            # Update portfolio history
            portfolio.update_portfolio_history(timestamp, current_prices)
            
            # Run each strategy
            for strategy_id, strategy in self.strategies.items():
                symbols = self.strategy_symbols[strategy_id]
                strategy_df = strategy_data[strategy_id]
                
                # Get current row data for this strategy
                current_data = strategy_df.iloc[:i+1] if i+1 <= len(strategy_df) else strategy_df
                
                if len(current_data) < 2:  # Need at least 2 periods for signals
                    continue
                
                # Generate signals for each symbol this strategy trades
                for symbol in symbols:
                    if symbol not in current_prices:
                        continue
                    
                    try:
                        # Get symbol-specific data
                        if isinstance(strategy_df.columns, pd.MultiIndex):
                            symbol_data = self._extract_symbol_data(current_data, symbol)
                        else:
                            symbol_data = current_data[[symbol]].copy()
                            symbol_data.columns = ['close']
                        
                        if symbol_data.empty or len(symbol_data) < 2:
                            continue
                        
                        # Generate signal
                        signal = strategy.generate_signal(symbol_data)
                        
                        if signal != 0:  # Non-zero signal
                            self._execute_signal(
                                portfolio, strategy_id, symbol, signal,
                                current_prices[symbol], timestamp, config
                            )
                    
                    except Exception as e:
                        logger.warning(f"Error processing {strategy_id}/{symbol} at {timestamp}: {e}")
                        continue
            
            # Progress logging
            if i % max(1, len(timestamps) // 10) == 0:
                progress = (i / len(timestamps)) * 100
                logger.info(f"Progress: {progress:.1f}% - Portfolio value: ${portfolio.get_portfolio_value(current_prices):,.2f}")
        
        # Close all remaining positions at the end
        final_prices = {}
        if isinstance(data.columns, pd.MultiIndex):
            for col in data.columns:
                if col[0] == 'close':
                    symbol = col[1]
                    price = data.iloc[-1][col]
                    if not pd.isna(price):
                        final_prices[symbol] = price
        else:
            for col in data.columns:
                price = data.iloc[-1][col]
                if not pd.isna(price):
                    final_prices[col] = price
        
        portfolio.close_all_positions(final_prices, timestamps[-1])
    
    def _extract_symbol_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Extract OHLCV data for a specific symbol from multi-level columns."""
        symbol_cols = [col for col in data.columns if col[1] == symbol]
        if not symbol_cols:
            return pd.DataFrame()
        
        symbol_data = data[symbol_cols].copy()
        # Rename columns to remove symbol level
        symbol_data.columns = [col[0] for col in symbol_data.columns]
        
        return symbol_data
    
    def _execute_signal(self, portfolio: PortfolioManager, strategy_id: str, symbol: str,
                       signal: float, price: float, timestamp: datetime, config: BacktestConfig):
        """Execute a trading signal."""
        
        # Calculate position size based on signal strength and portfolio value
        portfolio_value = portfolio.get_portfolio_value()
        max_position_value = portfolio_value * config.max_position_size
        
        # Simple position sizing: signal strength * max position size
        target_value = abs(signal) * max_position_value
        position_size = target_value / price
        
        # Apply signal direction
        if signal < 0:
            if not config.enable_short_selling:
                return  # Skip short signals if not enabled
            position_size = -position_size
        
        # Try to open the position
        portfolio.open_position(
            symbol=symbol,
            strategy_id=strategy_id,
            size=position_size,
            price=price,
            timestamp=timestamp,
            metadata={'signal_strength': signal}
        )
