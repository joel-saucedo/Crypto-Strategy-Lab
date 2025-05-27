"""
Core backtesting engine for Crypto Strategy Lab.
Enforces the master recipe validation framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import yaml

@dataclass
class BacktestConfig:
    initial_capital: float = 100000
    fees: Dict[str, float] = None
    slippage: Dict[str, float] = None
    
class BacktestEngine:
    """
    Unified backtesting engine that enforces the blueprint.
    All strategies use this same engine for consistency.
    """
    
    def __init__(self, config_path: str = "config/base.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.reset()
        
    def reset(self):
        """Reset engine state for new backtest."""
        self.equity_curve = []
        self.trades = []
        self.positions = {}
        self.cash = self.config['backtest']['initial_capital']
        
    def run_strategy(self, 
                    prices: pd.DataFrame, 
                    signals: pd.Series, 
                    strategy_name: str) -> Dict[str, Any]:
        """
        Run a strategy through the backtest engine.
        
        Args:
            prices: OHLCV data
            signals: Signal vector {-1, 0, 1}
            strategy_name: Name for reporting
            
        Returns:
            Dictionary with performance metrics
        """
        self.reset()
        
        for date, signal in signals.items():
            if date not in prices.index:
                continue
                
            price = prices.loc[date, 'close']
            
            # Execute position changes
            current_pos = self.positions.get(strategy_name, 0)
            target_pos = self._calculate_position_size(signal, price)
            
            if target_pos != current_pos:
                self._execute_trade(strategy_name, target_pos - current_pos, price, date)
                
            # Update equity curve
            portfolio_value = self._calculate_portfolio_value(prices.loc[date])
            self.equity_curve.append({
                'date': date,
                'equity': portfolio_value,
                'position': self.positions.get(strategy_name, 0),
                'signal': signal
            })
            
        return self._calculate_metrics()
        
    def _calculate_position_size(self, signal: int, price: float) -> float:
        """Calculate position size based on signal and risk management."""
        if signal == 0:
            return 0
            
        # Simple fixed fractional sizing for now
        max_position = self.config['position_limits']['max_position_size']
        return signal * max_position
        
    def _execute_trade(self, strategy: str, size_change: float, price: float, date):
        """Execute a trade with fees and slippage."""
        if abs(size_change) < 1e-6:
            return
            
        # Apply fees
        fee_rate = self.config['fees']['taker'] if abs(size_change) > 0 else 0
        trade_value = abs(size_change) * price
        fees = trade_value * fee_rate
        
        # Apply slippage
        slippage_rate = self.config['slippage']['linear']
        slippage_cost = trade_value * slippage_rate
        
        # Update positions and cash
        total_cost = trade_value + fees + slippage_cost
        if size_change > 0:  # Buy
            self.cash -= total_cost
        else:  # Sell
            self.cash += trade_value - fees - slippage_cost
            
        self.positions[strategy] = self.positions.get(strategy, 0) + size_change
        
        # Record trade
        self.trades.append({
            'date': date,
            'strategy': strategy,
            'size': size_change,
            'price': price,
            'fees': fees,
            'slippage': slippage_cost
        })
        
    def _calculate_portfolio_value(self, price_row: pd.Series) -> float:
        """Calculate total portfolio value."""
        position_value = sum(
            pos * price_row['close'] 
            for pos in self.positions.values()
        )
        return self.cash + position_value
        
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics required by blueprint."""
        if not self.equity_curve:
            return {}
            
        df = pd.DataFrame(self.equity_curve).set_index('date')
        returns = df['equity'].pct_change(fill_method=None).dropna()
        
        if len(returns) == 0:
            return {}
            
        # Core metrics required by DSR gate
        metrics = {
            'total_return': (df['equity'].iloc[-1] / df['equity'].iloc[0]) - 1,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(df['equity']),
            'volatility': returns.std() * np.sqrt(252),
            'num_trades': len(self.trades),
        }
        
        # Calculate DSR (placeholder - full implementation needed)
        metrics['dsr'] = self._calculate_dsr(returns)
        
        return metrics
        
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        return drawdown.min()
        
    def _calculate_dsr(self, returns: pd.Series) -> float:
        """Calculate Deflated Sharpe Ratio (simplified version)."""
        if len(returns) < 30:
            return 0
            
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Simplified DSR calculation
        # Full implementation requires multiple testing adjustment
        n_trials = 1000  # Placeholder for actual number of trials
        adjustment = np.sqrt(2 * np.log(n_trials))
        
        dsr = sharpe / adjustment if adjustment > 0 else 0
        return max(0, dsr)  # DSR should be non-negative
