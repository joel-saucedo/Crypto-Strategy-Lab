"""
Unified Backtesting Engine - Comprehensive backtesting framework with advanced validation,
multi-strategy orchestration, and sophisticated position sizing capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import warnings
import logging
from pathlib import Path
import asyncio
import concurrent.futures
from dataclasses import dataclass, field
import json
from enum import Enum
from abc import ABC, abstractmethod
from scipy import stats

# Import consolidated utilities to replace duplicates
try:
    from ..utils.consolidation_utils import (
        calculate_sharpe_ratio_optimized,
        calculate_max_drawdown_optimized,
        calculate_comprehensive_metrics
    )
except ImportError:
    from utils.consolidation_utils import (
        calculate_sharpe_ratio_optimized,
        calculate_max_drawdown_optimized,
        calculate_comprehensive_metrics
    )
from numba import jit, prange

# Set up logging
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION AND DATA STRUCTURES
# Unified configuration, position, and trade representations
# ============================================================================

@dataclass
class BacktestConfig:
    """Unified configuration for all backtesting operations."""
    # Basic settings
    initial_capital: float = 100000
    start_date: Union[str, datetime] = None
    end_date: Union[str, datetime] = None
    
    # Trading costs
    fees: Dict[str, float] = field(default_factory=lambda: {
        'taker': 0.001,
        'maker': 0.0005
    })
    slippage: Dict[str, float] = field(default_factory=lambda: {
        'linear': 0.0005,
        'sqrt': 0.0001
    })
    
    # Risk management
    max_position_size: float = 0.1
    max_total_exposure: float = 0.8
    enable_short_selling: bool = True
    
    # Validation requirements
    min_dsr: float = 0.95
    min_psr: float = 0.95
    monte_carlo_trials: int = 10000
    
    # Performance
    timeframe: str = '1d'
    benchmark_symbol: str = 'BTC-USD'

@dataclass
class Position:
    """Unified position representation."""
    symbol: str
    strategy_id: str
    size: float
    entry_price: float
    entry_time: datetime
    position_type: str  # 'long' or 'short'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return abs(self.size) * self.entry_price
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL."""
        if self.position_type == 'long':
            return self.size * (current_price - self.entry_price)
        else:
            return self.size * (self.entry_price - current_price)

@dataclass
class Trade:
    """Unified trade representation."""
    symbol: str
    strategy_id: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    position_type: str
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> timedelta:
        """Trade duration."""
        return self.exit_time - self.entry_time
    
    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0

# ============================================================================
# POSITION SIZING STRATEGIES
# Comprehensive position sizing framework with multiple algorithms
# ============================================================================

class PositionSizingType(Enum):
    """Available position sizing methods."""
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_TARGETING = "volatility_targeting"
    REGIME_AWARE = "regime_aware"
    BAYESIAN_OPTIMAL = "bayesian_optimal"

@dataclass
class PositionSizingConfig:
    """Configuration for position sizing strategies."""
    method: PositionSizingType = PositionSizingType.FIXED_FRACTIONAL
    
    # Fixed Fractional
    fixed_fraction: float = 0.02
    
    # Kelly Criterion
    kelly_lookback: int = 252
    kelly_fraction: float = 0.25  # Fractional Kelly
    kelly_min: float = 0.01
    kelly_max: float = 0.10
    
    # Volatility Targeting
    target_volatility: float = 0.15
    vol_lookback: int = 21
    vol_adjustment_factor: float = 1.0
    
    # Regime Aware
    regime_lookback: int = 63
    high_vol_multiplier: float = 0.5
    low_vol_multiplier: float = 1.5
    crisis_multiplier: float = 0.25
    
    # Bayesian/RL
    learning_rate: float = 0.01
    exploration_factor: float = 0.05
    memory_decay: float = 0.95

class PositionSizingEngine:
    """
    Comprehensive position sizing engine supporting multiple strategies.
    """
    
    def __init__(self, config: PositionSizingConfig = None):
        self.config = config or PositionSizingConfig()
        
        # Historical tracking for adaptive methods
        self.trade_history: List[Dict] = []
        self.market_regimes: List[str] = []
        self.volatility_history: List[float] = []
        
        # Bayesian learning components
        self.success_rates: Dict[str, float] = {}
        self.risk_estimates: Dict[str, float] = {}
        
    def calculate_position_size(self, 
                              signal_strength: float,
                              current_price: float,
                              portfolio_value: float,
                              symbol: str,
                              market_data: pd.DataFrame = None,
                              strategy_id: str = None) -> float:
        """
        Calculate optimal position size based on configured method.
        
        Args:
            signal_strength: Strategy signal strength (-1 to 1)
            current_price: Current asset price
            portfolio_value: Current portfolio value
            symbol: Asset symbol
            market_data: Historical market data for calculations
            strategy_id: Strategy identifier for tracking
            
        Returns:
            Position size (positive for long, negative for short)
        """
        
        if abs(signal_strength) < 1e-6:
            return 0.0
        
        # Calculate base position size
        if self.config.method == PositionSizingType.FIXED_FRACTIONAL:
            base_size = self._fixed_fractional_sizing(portfolio_value, current_price)
            
        elif self.config.method == PositionSizingType.KELLY_CRITERION:
            base_size = self._kelly_criterion_sizing(
                portfolio_value, current_price, market_data, symbol
            )
            
        elif self.config.method == PositionSizingType.VOLATILITY_TARGETING:
            base_size = self._volatility_targeting_sizing(
                portfolio_value, current_price, market_data
            )
            
        elif self.config.method == PositionSizingType.REGIME_AWARE:
            base_size = self._regime_aware_sizing(
                portfolio_value, current_price, market_data
            )
            
        elif self.config.method == PositionSizingType.BAYESIAN_OPTIMAL:
            base_size = self._bayesian_optimal_sizing(
                portfolio_value, current_price, signal_strength, strategy_id, symbol
            )
            
        else:
            base_size = self._fixed_fractional_sizing(portfolio_value, current_price)
        
        # Apply signal strength scaling
        scaled_size = base_size * abs(signal_strength)
        
        # Apply signal direction
        return scaled_size if signal_strength > 0 else -scaled_size
    
    def _fixed_fractional_sizing(self, portfolio_value: float, current_price: float) -> float:
        """Simple fixed fractional position sizing."""
        position_value = portfolio_value * self.config.fixed_fraction
        return position_value / current_price
    
    def _kelly_criterion_sizing(self, 
                               portfolio_value: float, 
                               current_price: float,
                               market_data: pd.DataFrame,
                               symbol: str) -> float:
        """Kelly Criterion position sizing based on historical performance."""
        
        if market_data is None or len(market_data) < self.config.kelly_lookback:
            return self._fixed_fractional_sizing(portfolio_value, current_price)
        
        try:
            # Calculate historical returns
            returns = market_data['close'].pct_change().dropna()
            recent_returns = returns.tail(self.config.kelly_lookback)
            
            if len(recent_returns) < 50:  # Minimum data requirement
                return self._fixed_fractional_sizing(portfolio_value, current_price)
            
            # Kelly calculation: f = (bp - q) / b
            # where b = odds, p = prob of win, q = prob of loss
            winning_returns = recent_returns[recent_returns > 0]
            losing_returns = recent_returns[recent_returns < 0]
            
            if len(winning_returns) == 0 or len(losing_returns) == 0:
                return self._fixed_fractional_sizing(portfolio_value, current_price)
            
            # Probabilities
            p_win = len(winning_returns) / len(recent_returns)
            p_loss = 1 - p_win
            
            # Average returns
            avg_win = winning_returns.mean()
            avg_loss = abs(losing_returns.mean())
            
            # Kelly fraction
            if avg_loss > 0:
                kelly_f = (p_win * avg_win - p_loss * avg_loss) / avg_win
            else:
                kelly_f = 0
            
            # Apply fractional Kelly and bounds
            kelly_f = kelly_f * self.config.kelly_fraction
            kelly_f = np.clip(kelly_f, self.config.kelly_min, self.config.kelly_max)
            
            position_value = portfolio_value * kelly_f
            return position_value / current_price
            
        except Exception as e:
            logger.warning(f"Kelly calculation failed for {symbol}: {e}")
            return self._fixed_fractional_sizing(portfolio_value, current_price)
    
    def _volatility_targeting_sizing(self, 
                                   portfolio_value: float,
                                   current_price: float,
                                   market_data: pd.DataFrame) -> float:
        """Volatility targeting position sizing."""
        
        if market_data is None or len(market_data) < self.config.vol_lookback:
            return self._fixed_fractional_sizing(portfolio_value, current_price)
        
        try:
            # Calculate recent volatility
            returns = market_data['close'].pct_change().dropna()
            recent_vol = returns.tail(self.config.vol_lookback).std() * np.sqrt(252)
            
            if recent_vol <= 0:
                return self._fixed_fractional_sizing(portfolio_value, current_price)
            
            # Scale position inversely with volatility
            vol_scalar = (self.config.target_volatility / recent_vol) * self.config.vol_adjustment_factor
            vol_scalar = np.clip(vol_scalar, 0.1, 3.0)  # Reasonable bounds
            
            base_fraction = self.config.fixed_fraction * vol_scalar
            position_value = portfolio_value * base_fraction
            
            return position_value / current_price
            
        except Exception as e:
            logger.warning(f"Volatility targeting failed: {e}")
            return self._fixed_fractional_sizing(portfolio_value, current_price)
    
    def _regime_aware_sizing(self,
                           portfolio_value: float,
                           current_price: float,
                           market_data: pd.DataFrame) -> float:
        """Regime-aware position sizing that adapts to market conditions."""
        
        if market_data is None or len(market_data) < self.config.regime_lookback:
            return self._fixed_fractional_sizing(portfolio_value, current_price)
        
        try:
            # Detect current market regime
            returns = market_data['close'].pct_change().dropna()
            recent_returns = returns.tail(self.config.regime_lookback)
            
            # Volatility regime
            recent_vol = recent_returns.std() * np.sqrt(252)
            historical_vol = returns.std() * np.sqrt(252)
            
            # Trend regime
            prices = market_data['close'].tail(self.config.regime_lookback)
            trend_strength = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
            
            # Crisis detection (large drawdowns)
            rolling_max = prices.rolling(21).max()
            drawdown = (prices - rolling_max) / rolling_max
            max_recent_dd = drawdown.min()
            
            # Determine regime multiplier
            multiplier = 1.0
            
            # Volatility adjustment
            if recent_vol > historical_vol * 1.5:
                multiplier *= self.config.high_vol_multiplier
            elif recent_vol < historical_vol * 0.7:
                multiplier *= self.config.low_vol_multiplier
            
            # Crisis adjustment
            if max_recent_dd < -0.15:  # 15% drawdown
                multiplier *= self.config.crisis_multiplier
            
            # Trend adjustment
            if abs(trend_strength) > 0.2:  # Strong trend
                multiplier *= 1.2
            
            # Apply regime adjustment
            adjusted_fraction = self.config.fixed_fraction * multiplier
            adjusted_fraction = np.clip(adjusted_fraction, 0.005, 0.20)  # Safety bounds
            
            position_value = portfolio_value * adjusted_fraction
            return position_value / current_price
            
        except Exception as e:
            logger.warning(f"Regime-aware sizing failed: {e}")
            return self._fixed_fractional_sizing(portfolio_value, current_price)
    
    def _bayesian_optimal_sizing(self,
                               portfolio_value: float,
                               current_price: float,
                               signal_strength: float,
                               strategy_id: str,
                               symbol: str) -> float:
        """Bayesian optimal position sizing with reinforcement learning."""
        
        strategy_key = f"{strategy_id}_{symbol}"
        
        # Initialize if new strategy-symbol combination
        if strategy_key not in self.success_rates:
            self.success_rates[strategy_key] = 0.5  # Neutral prior
            self.risk_estimates[strategy_key] = 0.02  # Conservative start
        
        # Get current estimates
        success_rate = self.success_rates[strategy_key]
        risk_estimate = self.risk_estimates[strategy_key]
        
        # Bayesian position sizing: consider both success rate and risk
        # Thompson Sampling approach for exploration-exploitation
        
        # Sample from Beta distribution for success rate
        alpha = success_rate * 100 + 1  # Prior strength
        beta = (1 - success_rate) * 100 + 1
        sampled_success_rate = np.random.beta(alpha, beta)
        
        # Exploration factor
        exploration_bonus = self.config.exploration_factor * np.random.normal(0, 1)
        
        # Calculate optimal fraction
        optimal_fraction = (
            sampled_success_rate * abs(signal_strength) * 
            (1 - risk_estimate) + exploration_bonus
        )
        
        # Bounds checking
        optimal_fraction = np.clip(optimal_fraction, 0.005, 0.15)
        
        position_value = portfolio_value * optimal_fraction
        return position_value / current_price
    
    def update_performance(self, 
                         strategy_id: str,
                         symbol: str,
                         signal_strength: float,
                         pnl: float,
                         position_size: float):
        """Update performance tracking for adaptive sizing methods."""
        
        strategy_key = f"{strategy_id}_{symbol}"
        
        # Record trade
        trade_record = {
            'strategy_key': strategy_key,
            'signal_strength': signal_strength,
            'pnl': pnl,
            'position_size': position_size,
            'timestamp': datetime.now(),
            'success': pnl > 0
        }
        
        self.trade_history.append(trade_record)
        
        # Update Bayesian estimates
        if strategy_key in self.success_rates:
            # Update success rate using exponential moving average
            new_success = 1.0 if pnl > 0 else 0.0
            self.success_rates[strategy_key] = (
                self.config.memory_decay * self.success_rates[strategy_key] +
                (1 - self.config.memory_decay) * new_success
            )
            
            # Update risk estimate based on actual vs expected performance
            expected_pnl = abs(signal_strength) * position_size * 0.001  # Rough estimate
            risk_factor = abs(pnl - expected_pnl) / max(abs(expected_pnl), 1e-6)
            
            self.risk_estimates[strategy_key] = (
                self.config.memory_decay * self.risk_estimates[strategy_key] +
                (1 - self.config.memory_decay) * risk_factor
            )
        
        # Cleanup old records (keep last 1000)
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
    
    def get_sizing_summary(self) -> Dict[str, Any]:
        """Get summary of position sizing performance."""
        if not self.trade_history:
            return {'status': 'No trades recorded'}
        
        recent_trades = self.trade_history[-100:]  # Last 100 trades
        
        total_trades = len(recent_trades)
        winning_trades = sum(1 for trade in recent_trades if trade['success'])
        
        avg_pnl = np.mean([trade['pnl'] for trade in recent_trades])
        avg_position_size = np.mean([abs(trade['position_size']) for trade in recent_trades])
        
        return {
            'method': self.config.method.value,
            'total_recent_trades': total_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'average_pnl': avg_pnl,
            'average_position_size': avg_position_size,
            'success_rates': dict(self.success_rates),
            'risk_estimates': dict(self.risk_estimates),
            'active_strategies': len(self.success_rates)
        }

# ============================================================================
# PORTFOLIO MANAGEMENT
# Unified portfolio management for handling positions, trades, and risk controls
# ============================================================================

class PortfolioManager:
    """
    Unified portfolio management combining the best of existing implementations.
    Handles positions, trades, and risk controls across multiple strategies.
    """
    
    def __init__(self, initial_capital: float = 100000, commission_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        
        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}  # key: f"{strategy_id}_{symbol}"
        self.closed_trades: List[Trade] = []
        self.portfolio_history: List[Dict] = []
        
        # Risk controls
        self.max_position_size = 0.1
        self.max_total_exposure = 0.8
        
        # Performance tracking
        self._equity_curve = []
        self._daily_returns = []
        
    def get_portfolio_value(self, current_prices: Dict[str, float] = None) -> float:
        """Calculate total portfolio value."""
        if current_prices is None:
            current_prices = {}
            
        cash_value = self.cash
        position_value = 0
        
        for pos_key, position in self.positions.items():
            if position.symbol in current_prices:
                position_value += position.calculate_pnl(current_prices[position.symbol])
                position_value += position.market_value
            else:
                # Use entry price if current price not available
                position_value += position.market_value
                
        return cash_value + position_value
    
    def get_total_exposure(self, current_prices: Dict[str, float] = None) -> float:
        """Calculate total portfolio exposure."""
        if not self.positions or current_prices is None:
            return 0
            
        portfolio_value = self.get_portfolio_value(current_prices)
        if portfolio_value <= 0:
            return 0
            
        total_exposure = 0
        for position in self.positions.values():
            if position.symbol in current_prices:
                exposure = abs(position.size) * current_prices[position.symbol]
                total_exposure += exposure
                
        return total_exposure / portfolio_value
    
    def can_open_position(self, symbol: str, size: float, price: float, 
                         current_prices: Dict[str, float] = None) -> bool:
        """Check if position can be opened given risk constraints."""
        portfolio_value = self.get_portfolio_value(current_prices or {})
        
        # Check individual position size limit
        position_value = abs(size) * price
        position_ratio = position_value / portfolio_value
        
        if position_ratio > self.max_position_size:
            return False
            
        # Check total exposure limit
        current_exposure = self.get_total_exposure(current_prices or {})
        new_exposure = position_value / portfolio_value
        
        if current_exposure + new_exposure > self.max_total_exposure:
            return False
            
        # Check sufficient cash
        cash_needed = position_value * (1 + self.commission_rate)
        if self.cash < cash_needed:
            return False
            
        return True
    
    def open_position(self, symbol: str, strategy_id: str, size: float, 
                     price: float, timestamp: datetime, 
                     current_prices: Dict[str, float] = None,
                     metadata: Dict[str, Any] = None) -> bool:
        """Open a new position with risk checks."""
        if not self.can_open_position(symbol, size, price, current_prices):
            logger.warning(f"Position rejected: {strategy_id}/{symbol} size={size} price={price}")
            return False
            
        position_key = f"{strategy_id}_{symbol}"
        
        # Close existing position if any
        if position_key in self.positions:
            self.close_position(position_key, price, timestamp)
            
        # Calculate costs
        position_value = abs(size) * price
        commission = position_value * self.commission_rate
        
        # Create position
        position_type = 'long' if size > 0 else 'short'
        position = Position(
            symbol=symbol,
            strategy_id=strategy_id,
            size=size,
            entry_price=price,
            entry_time=timestamp,
            position_type=position_type,
            metadata=metadata or {}
        )
        
        # Update cash
        self.cash -= commission
        if size > 0:  # Long position
            self.cash -= position_value
        else:  # Short position - receive cash but owe shares
            self.cash += position_value
            
        self.positions[position_key] = position
        
        logger.debug(f"Opened {position_type} position: {strategy_id}/{symbol} "
                    f"size={size:.4f} price=${price:.2f}")
        return True
    
    def close_position(self, position_key: str, price: float, 
                      timestamp: datetime, partial_size: float = None) -> bool:
        """Close a position (full or partial)."""
        if position_key not in self.positions:
            return False
            
        position = self.positions[position_key]
        close_size = partial_size or position.size
        
        # Calculate PnL
        pnl = position.calculate_pnl(price)
        if partial_size:
            pnl = pnl * (partial_size / position.size)
            
        # Calculate commission
        close_value = abs(close_size) * price
        commission = close_value * self.commission_rate
        
        # Calculate percentage return
        entry_value = abs(close_size) * position.entry_price
        pnl_pct = pnl / entry_value if entry_value > 0 else 0
        
        # Create trade record
        trade = Trade(
            symbol=position.symbol,
            strategy_id=position.strategy_id,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=price,
            size=close_size,
            position_type=position.position_type,
            pnl=pnl - commission,
            pnl_pct=pnl_pct,
            commission=commission,
            slippage=0,  # Simplified for now
            metadata=position.metadata
        )
        
        self.closed_trades.append(trade)
        
        # Update cash
        net_pnl = pnl - commission
        self.cash += net_pnl
        
        if position.position_type == 'long':
            self.cash += close_value
        else:  # Short position
            self.cash -= close_value
            
        # Update or remove position
        if partial_size and abs(partial_size) < abs(position.size):
            # Partial close - update position size
            position.size -= close_size
        else:
            # Full close - remove position
            del self.positions[position_key]
            
        logger.debug(f"Closed position: {position.strategy_id}/{position.symbol} "
                    f"PnL=${net_pnl:.2f} ({pnl_pct*100:.2f}%)")
        return True
    
    def close_all_positions(self, current_prices: Dict[str, float], timestamp: datetime):
        """Close all open positions at market prices."""
        position_keys = list(self.positions.keys())
        for position_key in position_keys:
            position = self.positions[position_key]
            if position.symbol in current_prices:
                self.close_position(position_key, current_prices[position.symbol], timestamp)
    
    def update_portfolio_history(self, timestamp: datetime, current_prices: Dict[str, float]):
        """Update portfolio history for performance tracking."""
        portfolio_value = self.get_portfolio_value(current_prices)
        
        # Calculate daily return
        if self.portfolio_history:
            prev_value = self.portfolio_history[-1]['portfolio_value']
            daily_return = (portfolio_value / prev_value) - 1 if prev_value > 0 else 0
            self._daily_returns.append(daily_return)
        
        self.portfolio_history.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash,
            'num_positions': len(self.positions),
            'exposure': self.get_total_exposure(current_prices)
        })
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not self.portfolio_history or not self._daily_returns:
            return {}
            
        returns = np.array(self._daily_returns)
        portfolio_values = [h['portfolio_value'] for h in self.portfolio_history]
        
        # Basic metrics using consolidated utilities
        total_return = (portfolio_values[-1] / self.initial_capital) - 1
        annualized_return = np.mean(returns) * 252
        volatility = np.std(returns) * np.sqrt(252)
        
        # Use consolidated Sharpe ratio calculation
        sharpe_ratio = calculate_sharpe_ratio_optimized(returns, 0.0, 252)
        
        # Use consolidated max drawdown calculation
        max_drawdown, _, _ = calculate_max_drawdown_optimized(np.array(portfolio_values))
        
        # Calmar ratio with consolidated utility import
        from ..utils.consolidation_utils import calculate_calmar_ratio_optimized
        calmar_ratio = calculate_calmar_ratio_optimized(annualized_return, max_drawdown)
        
        # Trade statistics
        winning_trades = [t for t in self.closed_trades if t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl <= 0]
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'total_trades': len(self.closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.closed_trades) if self.closed_trades else 0,
            'avg_win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
            'profit_factor': (sum(t.pnl for t in winning_trades) / 
                             abs(sum(t.pnl for t in losing_trades))) if losing_trades else float('inf'),
            'final_portfolio_value': portfolio_values[-1]
        }
        
        return metrics

# ============================================================================
# MONTE CARLO VALIDATION FRAMEWORK
# Implements comprehensive statistical validation with Numba acceleration
# ============================================================================

@jit(nopython=True, parallel=True)
def _cumprod_numba(returns: np.ndarray) -> np.ndarray:
    """Fast cumulative product calculation for equity curves."""
    result = np.zeros(len(returns))
    result[0] = 1.0 + returns[0]
    for i in range(1, len(returns)):
        result[i] = result[i-1] * (1.0 + returns[i])
    return result

@jit(nopython=True, parallel=True)
def _bootstrap_sample_numba(returns: np.ndarray, n_samples: int) -> np.ndarray:
    """Generate bootstrap samples efficiently."""
    n_obs = len(returns)
    indices = np.random.randint(0, n_obs, n_samples)
    return returns[indices]

@jit(nopython=True, parallel=True)
def _block_bootstrap_numba(returns: np.ndarray, block_size: int, n_samples: int) -> np.ndarray:
    """Generate block bootstrap samples to preserve autocorrelation."""
    n_obs = len(returns)
    n_blocks = (n_samples + block_size - 1) // block_size
    
    # Pre-generate all blocks in parallel
    blocks = np.zeros((n_blocks, block_size))
    
    for i in prange(n_blocks):
        # Random starting position for each block
        start_idx = np.random.randint(0, max(1, n_obs - block_size + 1))
        end_idx = min(start_idx + block_size, n_obs)
        
        # Fill the block
        for j in range(block_size):
            if start_idx + j < end_idx:
                blocks[i, j] = returns[start_idx + j]
            else:
                # If block is shorter than block_size, repeat the last value
                blocks[i, j] = returns[end_idx - 1] if end_idx > start_idx else 0.0
    
    # Concatenate blocks sequentially to avoid race conditions
    result = np.zeros(n_samples)
    pos = 0
    
    for i in range(n_blocks):
        remaining = n_samples - pos
        if remaining <= 0:
            break
            
        copy_len = min(block_size, remaining)
        for j in range(copy_len):
            result[pos + j] = blocks[i, j]
        
        pos += copy_len
    
    return result[:n_samples]

@jit(nopython=True, parallel=True)
def _stationary_bootstrap_numba(returns: np.ndarray, avg_block_size: float, n_samples: int) -> np.ndarray:
    """Stationary bootstrap with geometric block sizes."""
    n_obs = len(returns)
    p = 1.0 / avg_block_size  # Probability of ending a block
    
    result = np.zeros(n_samples)
    pos = 0
    
    while pos < n_samples:
        # Random starting position
        start_idx = np.random.randint(0, n_obs)
        
        # Generate geometric block length
        block_len = 1
        while np.random.random() > p and block_len < n_obs and pos + block_len < n_samples:
            block_len += 1
        
        # Copy block (with wraparound if needed)
        for i in range(min(block_len, n_samples - pos)):
            idx = (start_idx + i) % n_obs
            result[pos + i] = returns[idx]
        
        pos += block_len
    
    return result[:n_samples]

@jit(nopython=True, parallel=True)
def _calculate_sharpe_numba(returns: np.ndarray) -> float:
    """Fast Sharpe ratio calculation."""
    if len(returns) == 0:
        return 0.0
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    
    if std_ret == 0:
        return 0.0
    
    return mean_ret / std_ret * np.sqrt(252.0)

@jit(nopython=True, parallel=True) 
def _analyze_simulations_numba(simulations: np.ndarray) -> Tuple[float, float, float, float]:
    """Analyze Monte Carlo simulation results efficiently."""
    n_sims, n_periods = simulations.shape
    
    # Calculate final returns for each simulation
    final_returns = np.zeros(n_sims)
    sharpe_ratios = np.zeros(n_sims)
    max_drawdowns = np.zeros(n_sims)
    
    for i in prange(n_sims):
        sim_returns = simulations[i, :]
        
        # Final cumulative return
        cum_prod = 1.0
        for j in range(n_periods):
            cum_prod *= (1.0 + sim_returns[j])
        final_returns[i] = cum_prod - 1.0
        
        # Sharpe ratio
        sharpe_ratios[i] = _calculate_sharpe_numba(sim_returns)
        
        # Maximum drawdown
        equity_curve = _cumprod_numba(sim_returns)
        running_max = equity_curve[0]
        max_dd = 0.0
        
        for j in range(1, n_periods):
            if equity_curve[j] > running_max:
                running_max = equity_curve[j]
            
            drawdown = (running_max - equity_curve[j]) / running_max
            if drawdown > max_dd:
                max_dd = drawdown
                
        max_drawdowns[i] = max_dd
    
    return (np.mean(final_returns), np.std(final_returns), 
            np.mean(sharpe_ratios), np.mean(max_drawdowns))

class MonteCarloValidator:
    """
    Comprehensive Monte Carlo validation framework implementing:
    - Bootstrap confidence intervals
    - Block bootstrap for autocorrelation preservation
    - Stationary bootstrap
    - Permutation tests
    - White Reality Check
    - Hansen's SPA test
    - DSR and PSR validation
    """
    
    def __init__(self, min_dsr: float = 0.95, min_psr: float = 0.95):
        self.min_dsr = min_dsr
        self.min_psr = min_psr
        
    def validate_strategy_comprehensive(self, returns: pd.Series, 
                                      n_trials: int = 10000,
                                      benchmark_returns: pd.Series = None) -> Dict[str, Any]:
        """
        Comprehensive Monte Carlo validation with all statistical tests.
        
        Returns validation results with pass/fail decision based on:
        - DSR ≥ 0.95 
        - PSR ≥ 0.95
        - Bootstrap confidence intervals
        - Permutation test significance
        - Reality check p-values
        """
        if len(returns) < 30:
            return {'validation_passed': False, 'error': 'Insufficient data (need ≥30 observations)'}
        
        returns_array = returns.values
        
        results = {
            'basic_metrics': self._calculate_basic_metrics(returns),
            'dsr_analysis': self._calculate_dsr_advanced(returns, n_trials),
            'psr_analysis': self._calculate_psr_advanced(returns),
            'bootstrap_analysis': self._bootstrap_validation(returns_array, n_trials),
            'permutation_tests': self._permutation_tests(returns_array, n_trials),
            'reality_check': self._white_reality_check(returns_array, n_trials),
            'monte_carlo_simulation': self._monte_carlo_with_replacement(returns_array, n_trials),
            'regime_analysis': self._regime_detection_analysis(returns)
        }
        
        # Overall validation decision
        results['validation_passed'] = self._evaluate_comprehensive_validation(results)
        results['validation_summary'] = self._create_validation_summary(results)
        
        return results
    
    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Enhanced basic metrics calculation."""
        if returns.std() == 0:
            return {'sharpe_ratio': 0, 'total_return': 0, 'volatility': 0, 'max_drawdown': 0}
        
        # Convert to numpy for performance
        ret_array = returns.values
        
        metrics = {
            'total_return': np.prod(1 + ret_array) - 1,
            'annualized_return': np.mean(ret_array) * 252,
            'volatility': np.std(ret_array) * np.sqrt(252),
            'sharpe_ratio': _calculate_sharpe_numba(ret_array),
            'max_drawdown': self._calculate_max_drawdown_fast(ret_array),
            'calmar_ratio': 0,
            'win_rate': np.mean(ret_array > 0),
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'var_95': float(np.percentile(ret_array, 5)),
            'cvar_95': float(np.mean(ret_array[ret_array <= np.percentile(ret_array, 5)])),
            'sortino_ratio': self._calculate_sortino_ratio(ret_array)
        }
        
        # Calculate Calmar ratio
        if metrics['max_drawdown'] != 0:
            metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
            
        return metrics
    
    def _calculate_max_drawdown_fast(self, returns: np.ndarray) -> float:
        """Fast maximum drawdown calculation."""
        equity_curve = _cumprod_numba(returns)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        return float(np.max(drawdowns))
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns) * np.sqrt(252)
        annualized_return = np.mean(returns) * 252
        
        return annualized_return / downside_deviation if downside_deviation > 0 else 0
    
    def _calculate_dsr_advanced(self, returns: pd.Series, n_trials: int = 10000) -> Dict[str, float]:
        """
        Advanced Deflated Sharpe Ratio calculation with multiple testing adjustment.
        
        DSR accounts for the multiple testing bias inherent in strategy selection.
        Formula: DSR = (SR - E[max SR under H0]) / std[max SR under H0]
        """
        returns_array = returns.values
        
        if np.std(returns_array) == 0:
            return {'dsr': 0, 'dsr_probability': 0, 'sharpe_ratio': 0}
        
        observed_sharpe = _calculate_sharpe_numba(returns_array)
        n_obs = len(returns_array)
        
        # Expected maximum Sharpe ratio under null hypothesis (multiple strategies tested)
        # Uses asymptotic approximation for large n_trials
        if n_trials > 100:
            gamma = 0.5772156649015329  # Euler-Mascheroni constant
            expected_max_sr = (np.sqrt(2 * np.log(n_trials)) - 
                             (np.log(np.log(n_trials)) + np.log(4 * np.pi)) / 
                             (2 * np.sqrt(2 * np.log(n_trials))))
        else:
            # For small n_trials, use simulation
            expected_max_sr = self._simulate_expected_max_sharpe(n_trials, n_obs)
        
        # Standard deviation of maximum Sharpe ratio
        std_max_sr = 1 / np.sqrt(2 * np.log(n_trials)) if n_trials > 1 else 1
        
        # Deflated Sharpe Ratio
        dsr = (observed_sharpe - expected_max_sr) / std_max_sr
        
        # Convert to probability scale
        dsr_probability = float(stats.norm.cdf(dsr))
        
        # Haircut factor (reduction in Sharpe ratio due to multiple testing)
        haircut_sr = observed_sharpe - expected_max_sr
        
        return {
            'dsr': float(dsr),
            'dsr_probability': dsr_probability,
            'sharpe_ratio': float(observed_sharpe),
            'expected_max_sr': float(expected_max_sr),
            'std_max_sr': float(std_max_sr),
            'haircut_sharpe': float(haircut_sr),
            'trials_adjustment': n_trials,
            'passes_dsr_gate': dsr_probability >= self.min_dsr
        }
    
    def _simulate_expected_max_sharpe(self, n_trials: int, n_obs: int) -> float:
        """Simulate expected maximum Sharpe ratio for small n_trials."""
        max_sharpes = []
        
        for _ in range(100):  # Monte Carlo estimation
            trial_sharpes = []
            for _ in range(n_trials):
                random_returns = np.random.normal(0, 1, n_obs)
                sharpe = _calculate_sharpe_numba(random_returns)
                trial_sharpes.append(sharpe)
            max_sharpes.append(np.max(trial_sharpes))
            
        return np.mean(max_sharpes)
    
    def _calculate_psr_advanced(self, returns: pd.Series, benchmark_sr: float = 0) -> Dict[str, float]:
        """
        Advanced Probabilistic Sharpe Ratio with higher moment corrections.
        
        PSR = P[SR* > SR_benchmark] where SR* is the true underlying Sharpe ratio
        Accounts for estimation error, skewness, and kurtosis.
        """
        returns_array = returns.values
        
        if np.std(returns_array) == 0:
            return {'psr': 0, 'psr_confidence': 0}
        
        n_obs = len(returns_array)
        observed_sharpe = _calculate_sharpe_numba(returns_array)
        
        # Higher moment calculations
        skew = float(stats.skew(returns_array))
        kurt = float(stats.kurtosis(returns_array, fisher=True))  # Excess kurtosis
        
        # Standard error with higher moment corrections (Opdyke 2007)
        # SE_SR ≈ sqrt((1 + 0.5*SR² - γ₁*SR + (γ₂-3)/4*SR²) / T)
        sr_variance = (1 + 0.5 * observed_sharpe**2 - 
                      skew * observed_sharpe + 
                      (kurt / 4) * observed_sharpe**2) / n_obs
        
        sr_std_error = np.sqrt(max(sr_variance, 1e-8))  # Avoid numerical issues
        
        # PSR calculation
        if sr_std_error > 0:
            psr_statistic = (observed_sharpe - benchmark_sr) / sr_std_error
            psr = float(stats.norm.cdf(psr_statistic))
        else:
            psr = 1.0 if observed_sharpe > benchmark_sr else 0.0
            psr_statistic = float('inf') if observed_sharpe > benchmark_sr else float('-inf')
        
        # Alternative PSR using bootstrapped distribution
        bootstrap_psr = self._bootstrap_psr(returns_array, benchmark_sr)
        
        return {
            'psr': psr,
            'psr_statistic': float(psr_statistic),
            'sharpe_ratio': float(observed_sharpe),
            'sharpe_std_error': float(sr_std_error),
            'skewness': skew,
            'kurtosis': kurt,
            'bootstrap_psr': bootstrap_psr,
            'benchmark_sr': benchmark_sr,
            'passes_psr_gate': psr >= self.min_psr
        }
    
    def _bootstrap_psr(self, returns: np.ndarray, benchmark_sr: float, n_bootstrap: int = 1000) -> float:
        """Calculate PSR using bootstrap distribution."""
        bootstrap_sharpes = []
        
        for _ in range(n_bootstrap):
            boot_sample = _bootstrap_sample_numba(returns, len(returns))
            boot_sharpe = _calculate_sharpe_numba(boot_sample)
            bootstrap_sharpes.append(boot_sharpe)
        
        # PSR = fraction of bootstrap Sharpes > benchmark
        bootstrap_sharpes = np.array(bootstrap_sharpes)
        return float(np.mean(bootstrap_sharpes >= benchmark_sr))
    
    def _bootstrap_validation(self, returns: np.ndarray, n_bootstrap: int = 10000) -> Dict[str, Any]:
        """
        Comprehensive bootstrap validation with multiple resampling methods.
        
        Implements:
        1. Standard bootstrap (IID resampling)
        2. Block bootstrap (preserves autocorrelation)
        3. Stationary bootstrap (random block lengths)
        """
        results = {}
        
        # Standard bootstrap
        results['standard'] = self._standard_bootstrap(returns, n_bootstrap)
        
        # Block bootstrap with optimal block size
        optimal_block_size = self._calculate_optimal_block_size(returns)
        results['block'] = self._block_bootstrap(returns, optimal_block_size, n_bootstrap)
        
        # Stationary bootstrap
        results['stationary'] = self._stationary_bootstrap(returns, optimal_block_size, n_bootstrap)
        
        # Confidence intervals comparison
        results['confidence_intervals'] = self._compare_bootstrap_confidence(results)
        
        return results
    
    def _standard_bootstrap(self, returns: np.ndarray, n_bootstrap: int) -> Dict[str, float]:
        """Standard bootstrap resampling."""
        bootstrap_sharpes = []
        bootstrap_returns = []
        
        for _ in range(n_bootstrap):
            boot_sample = _bootstrap_sample_numba(returns, len(returns))
            boot_sharpe = _calculate_sharpe_numba(boot_sample)
            boot_return = np.prod(1 + boot_sample) - 1
            
            bootstrap_sharpes.append(boot_sharpe)
            bootstrap_returns.append(boot_return)
        
        return {
            'mean_sharpe': float(np.mean(bootstrap_sharpes)),
            'std_sharpe': float(np.std(bootstrap_sharpes)),
            'ci_lower_sharpe': float(np.percentile(bootstrap_sharpes, 2.5)),
            'ci_upper_sharpe': float(np.percentile(bootstrap_sharpes, 97.5)),
            'mean_return': float(np.mean(bootstrap_returns)),
            'ci_lower_return': float(np.percentile(bootstrap_returns, 2.5)),
            'ci_upper_return': float(np.percentile(bootstrap_returns, 97.5))
        }
    
    def _block_bootstrap(self, returns: np.ndarray, block_size: int, n_bootstrap: int) -> Dict[str, float]:
        """Block bootstrap preserving autocorrelation structure."""
        bootstrap_sharpes = []
        
        for _ in range(n_bootstrap):
            boot_sample = _block_bootstrap_numba(returns, block_size, len(returns))
            boot_sharpe = _calculate_sharpe_numba(boot_sample)
            bootstrap_sharpes.append(boot_sharpe)
        
        return {
            'mean_sharpe': float(np.mean(bootstrap_sharpes)),
            'std_sharpe': float(np.std(bootstrap_sharpes)),
            'ci_lower_sharpe': float(np.percentile(bootstrap_sharpes, 2.5)),
            'ci_upper_sharpe': float(np.percentile(bootstrap_sharpes, 97.5)),
            'block_size': block_size
        }
    
    def _stationary_bootstrap(self, returns: np.ndarray, avg_block_size: float, n_bootstrap: int) -> Dict[str, float]:
        """Stationary bootstrap with geometric block lengths."""
        bootstrap_sharpes = []
        
        for _ in range(n_bootstrap):
            boot_sample = _stationary_bootstrap_numba(returns, avg_block_size, len(returns))
            boot_sharpe = _calculate_sharpe_numba(boot_sample)
            bootstrap_sharpes.append(boot_sharpe)
        
        return {
            'mean_sharpe': float(np.mean(bootstrap_sharpes)),
            'std_sharpe': float(np.std(bootstrap_sharpes)),
            'ci_lower_sharpe': float(np.percentile(bootstrap_sharpes, 2.5)),
            'ci_upper_sharpe': float(np.percentile(bootstrap_sharpes, 97.5)),
            'avg_block_size': avg_block_size
        }
    
    def _calculate_optimal_block_size(self, returns: np.ndarray) -> int:
        """Calculate optimal block size for block bootstrap."""
        # Use rule of thumb: optimal block size ≈ T^(1/3)
        n_obs = len(returns)
        optimal_size = max(1, int(n_obs**(1/3)))
        
        # Adjust based on autocorrelation structure
        autocorr = self._calculate_autocorrelation_length(returns)
        adjusted_size = max(optimal_size, min(autocorr * 2, n_obs // 4))
        
        return int(adjusted_size)
    
    def _calculate_autocorrelation_length(self, returns: np.ndarray) -> int:
        """Estimate autocorrelation length."""
        try:
            # Calculate autocorrelation function
            n = len(returns)
            autocorr = np.correlate(returns - np.mean(returns), 
                                  returns - np.mean(returns), mode='full')
            autocorr = autocorr[n-1:]
            autocorr = autocorr / autocorr[0]
            
            # Find first lag where autocorr falls below 1/e
            threshold = 1.0 / np.e
            significant_lags = np.where(autocorr > threshold)[0]
            
            if len(significant_lags) > 1:
                return int(significant_lags[-1])
            else:
                return 1
        except:
            return 1  # Fallback
    
    def _compare_bootstrap_confidence(self, bootstrap_results: Dict) -> Dict[str, Any]:
        """Compare confidence intervals from different bootstrap methods."""
        methods = ['standard', 'block', 'stationary']
        comparison = {}
        
        for method in methods:
            if method in bootstrap_results:
                result = bootstrap_results[method]
                comparison[f'{method}_width'] = result['ci_upper_sharpe'] - result['ci_lower_sharpe']
                comparison[f'{method}_center'] = (result['ci_upper_sharpe'] + result['ci_lower_sharpe']) / 2
        
        # Consensus interval (intersection of all methods)
        if all(method in bootstrap_results for method in methods):
            lower_bounds = [bootstrap_results[m]['ci_lower_sharpe'] for m in methods]
            upper_bounds = [bootstrap_results[m]['ci_upper_sharpe'] for m in methods]
            
            comparison['consensus_lower'] = max(lower_bounds)
            comparison['consensus_upper'] = min(upper_bounds)
            comparison['consensus_width'] = max(0, comparison['consensus_upper'] - comparison['consensus_lower'])
        
        return comparison
    
    def _permutation_tests(self, returns: np.ndarray, n_permutations: int = 10000) -> Dict[str, Any]:
        """
        Comprehensive permutation tests for return significance.
        
        Tests:
        1. Sign permutation (randomize return signs)
        2. Block permutation (randomize blocks to preserve autocorrelation)
        3. Circular permutation (rotate time series)
        4. Residual permutation (permute residuals from fitted model)
        """
        results = {}
        
        # Observed statistics
        observed_mean = np.mean(returns)
        observed_sharpe = _calculate_sharpe_numba(returns)
        observed_total_return = np.prod(1 + returns) - 1
        
        # 1. Sign permutation test
        results['sign_permutation'] = self._sign_permutation_test(
            returns, observed_mean, observed_sharpe, n_permutations)
        
        # 2. Block permutation test  
        results['block_permutation'] = self._block_permutation_test(
            returns, observed_sharpe, n_permutations)
        
        # 3. Circular permutation test
        results['circular_permutation'] = self._circular_permutation_test(
            returns, observed_sharpe, n_permutations)
        
        # 4. Residual permutation test (if autocorrelation detected)
        if self._has_significant_autocorrelation(returns):
            results['residual_permutation'] = self._residual_permutation_test(
                returns, observed_sharpe, n_permutations)
        
        # Combined p-value using Fisher's method
        p_values = [results[test]['p_value'] for test in results if 'p_value' in results[test]]
        if len(p_values) > 1:
            fisher_stat = -2 * np.sum(np.log(np.clip(p_values, 1e-10, 1)))
            combined_p = 1 - stats.chi2.cdf(fisher_stat, 2 * len(p_values))
            results['combined_p_value'] = float(combined_p)
        
        return results
    
    def _sign_permutation_test(self, returns: np.ndarray, observed_mean: float, 
                              observed_sharpe: float, n_permutations: int) -> Dict[str, float]:
        """Test significance by randomizing return signs."""
        null_means = []
        null_sharpes = []
        
        abs_returns = np.abs(returns)
        
        for _ in range(n_permutations):
            # Random signs
            signs = np.random.choice([-1, 1], size=len(returns))
            permuted_returns = abs_returns * signs
            
            null_means.append(np.mean(permuted_returns))
            null_sharpes.append(_calculate_sharpe_numba(permuted_returns))
        
        # P-values (two-tailed)
        p_value_mean = np.mean(np.abs(null_means) >= abs(observed_mean))
        p_value_sharpe = np.mean(np.abs(null_sharpes) >= abs(observed_sharpe))
        
        return {
            'p_value': float(p_value_mean),
            'p_value_sharpe': float(p_value_sharpe),
            'null_mean_sharpe': float(np.mean(null_sharpes)),
            'null_std_sharpe': float(np.std(null_sharpes))
        }
    
    def _block_permutation_test(self, returns: np.ndarray, observed_sharpe: float, 
                               n_permutations: int) -> Dict[str, float]:
        """Block permutation preserving local autocorrelation."""
        block_size = self._calculate_optimal_block_size(returns)
        null_sharpes = []
        
        for _ in range(n_permutations):
            # Create blocks
            n_blocks = len(returns) // block_size
            blocks = [returns[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
            
            # Add remainder as a block
            remainder = len(returns) % block_size
            if remainder > 0:
                blocks.append(returns[-remainder:])
            
            # Shuffle blocks
            np.random.shuffle(blocks)
            permuted_returns = np.concatenate(blocks)[:len(returns)]
            
            null_sharpes.append(_calculate_sharpe_numba(permuted_returns))
        
        p_value = np.mean(np.abs(null_sharpes) >= abs(observed_sharpe))
        
        return {
            'p_value': float(p_value),
            'block_size': block_size,
            'null_mean_sharpe': float(np.mean(null_sharpes)),
            'null_std_sharpe': float(np.std(null_sharpes))
        }
    
    def _circular_permutation_test(self, returns: np.ndarray, observed_sharpe: float,
                                  n_permutations: int) -> Dict[str, float]:
        """Circular permutation (rotation) test."""
        null_sharpes = []
        n_obs = len(returns)
        
        for _ in range(n_permutations):
            # Random rotation
            shift = np.random.randint(1, n_obs)
            rotated_returns = np.roll(returns, shift)
            null_sharpes.append(_calculate_sharpe_numba(rotated_returns))
        
        p_value = np.mean(np.abs(null_sharpes) >= abs(observed_sharpe))
        
        return {
            'p_value': float(p_value),
            'null_mean_sharpe': float(np.mean(null_sharpes)),
            'null_std_sharpe': float(np.std(null_sharpes))
        }
    
    def _residual_permutation_test(self, returns: np.ndarray, observed_sharpe: float,
                                  n_permutations: int) -> Dict[str, float]:
        """Permutation test on AR model residuals."""
        try:
            # Fit simple AR(1) model
            y = returns[1:]
            x = returns[:-1]
            
            # Simple linear regression
            beta = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else 0
            alpha = np.mean(y) - beta * np.mean(x)
            
            # Calculate residuals
            predicted = alpha + beta * x
            residuals = y - predicted
            
            null_sharpes = []
            
            for _ in range(n_permutations):
                # Permute residuals
                perm_residuals = np.random.permutation(residuals)
                
                # Reconstruct series
                reconstructed = np.zeros(len(returns))
                reconstructed[0] = returns[0]  # Keep first observation
                
                for i in range(1, len(returns)):
                    reconstructed[i] = alpha + beta * reconstructed[i-1] + perm_residuals[i-1]
                
                null_sharpes.append(_calculate_sharpe_numba(reconstructed))
            
            p_value = np.mean(np.abs(null_sharpes) >= abs(observed_sharpe))
            
            return {
                'p_value': float(p_value),
                'ar_coefficient': float(beta),
                'null_mean_sharpe': float(np.mean(null_sharpes)),
                'null_std_sharpe': float(np.std(null_sharpes))
            }
        
        except Exception as e:
            logger.warning(f"Residual permutation test failed: {e}")
            return {'p_value': 1.0, 'error': str(e)}
    
    def _has_significant_autocorrelation(self, returns: np.ndarray, max_lag: int = 10) -> bool:
        """Check for significant autocorrelation using Ljung-Box test."""
        try:
            from scipy.stats import jarque_bera
            
            # Simple autocorrelation check
            n = len(returns)
            if n < 20:
                return False
            
            # Calculate first-order autocorrelation
            demeaned = returns - np.mean(returns)
            autocorr_1 = np.corrcoef(demeaned[:-1], demeaned[1:])[0, 1]
            
            # Significance threshold (approximately)
            threshold = 2.0 / np.sqrt(n)
            
            return abs(autocorr_1) > threshold
        
        except:
            return False
    
    def _white_reality_check(self, returns: np.ndarray, n_bootstrap: int = 10000) -> Dict[str, float]:
        """
        White's Reality Check for Data Snooping.
        
        Tests the null hypothesis that the best strategy has no superior performance
        compared to a benchmark (typically zero return).
        """
        observed_sharpe = _calculate_sharpe_numba(returns)
        
        # Bootstrap under null hypothesis (zero mean)
        null_sharpes = []
        demeaned_returns = returns - np.mean(returns)  # Remove drift for null
        
        for _ in range(n_bootstrap):
            # Bootstrap sample from demeaned returns
            boot_sample = _bootstrap_sample_numba(demeaned_returns, len(returns))
            boot_sharpe = _calculate_sharpe_numba(boot_sample)
            null_sharpes.append(boot_sharpe)
        
        # P-value: fraction of bootstrap Sharpes >= observed
        p_value = np.mean(np.array(null_sharpes) >= observed_sharpe)
        
        # Superior Predictive Ability (SPA) adjustment
        max_null_sharpes = []
        for _ in range(1000):  # Simulate multiple strategies
            n_strategies = np.random.randint(10, 1000)  # Random number of strategies tested
            strategy_sharpes = []
            
            for _ in range(n_strategies):
                boot_sample = _bootstrap_sample_numba(demeaned_returns, len(returns))
                strategy_sharpes.append(_calculate_sharpe_numba(boot_sample))
            
            max_null_sharpes.append(np.max(strategy_sharpes))
        
        # SPA p-value
        spa_p_value = np.mean(np.array(max_null_sharpes) >= observed_sharpe)
        
        return {
            'reality_check_p_value': float(p_value),
            'spa_p_value': float(spa_p_value),
            'observed_sharpe': float(observed_sharpe),
            'max_null_sharpe_mean': float(np.mean(max_null_sharpes)),
            'passes_reality_check': p_value < 0.05,
            'passes_spa_test': spa_p_value < 0.05
        }
    
    def _monte_carlo_with_replacement(self, returns: np.ndarray, n_simulations: int = 10000) -> Dict[str, Any]:
        """
        Monte Carlo simulation with replacement to generate synthetic equity curves.
        
        Creates thousands of alternative equity curves by resampling returns
        to assess the robustness of the observed performance.
        """
        n_periods = len(returns)
        
        # Pre-allocate simulation array
        simulations = np.zeros((n_simulations, n_periods))
        
        # Generate simulations
        for i in range(n_simulations):
            simulations[i, :] = _bootstrap_sample_numba(returns, n_periods)
        
        # Analyze simulations using Numba
        mean_final_return, std_final_return, mean_sharpe, mean_max_dd = _analyze_simulations_numba(simulations)
        
        # Calculate percentiles for final returns
        final_returns = np.zeros(n_simulations)
        sharpe_ratios = np.zeros(n_simulations)
        max_drawdowns = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            sim_returns = simulations[i, :]
            
            # Final return
            final_returns[i] = np.prod(1 + sim_returns) - 1
            
            # Sharpe ratio
            sharpe_ratios[i] = _calculate_sharpe_numba(sim_returns)
            
            # Maximum drawdown
            equity_curve = _cumprod_numba(sim_returns)
            max_dd = self._calculate_max_drawdown_fast(sim_returns)
            max_drawdowns[i] = max_dd
        
        # Observed metrics
        observed_final_return = np.prod(1 + returns) - 1
        observed_sharpe = _calculate_sharpe_numba(returns)
        observed_max_dd = self._calculate_max_drawdown_fast(returns)
        
        # Percentile ranks (what percentile is the observed performance?)
        return_percentile = np.mean(final_returns <= observed_final_return) * 100
        sharpe_percentile = np.mean(sharpe_ratios <= observed_sharpe) * 100
        dd_percentile = np.mean(max_drawdowns >= observed_max_dd) * 100  # Lower DD is better
        
        return {
            'n_simulations': n_simulations,
            'observed_final_return': float(observed_final_return),
            'observed_sharpe': float(observed_sharpe),
            'observed_max_drawdown': float(observed_max_dd),
            'simulated_returns': {
                'mean': float(mean_final_return),
                'std': float(std_final_return),
                'percentile_5': float(np.percentile(final_returns, 5)),
                'percentile_25': float(np.percentile(final_returns, 25)),
                'percentile_50': float(np.percentile(final_returns, 50)),
                'percentile_75': float(np.percentile(final_returns, 75)),
                'percentile_95': float(np.percentile(final_returns, 95))
            },
            'simulated_sharpes': {
                'mean': float(mean_sharpe),
                'std': float(np.std(sharpe_ratios)),
                'percentile_5': float(np.percentile(sharpe_ratios, 5)),
                'percentile_25': float(np.percentile(sharpe_ratios, 25)),
                'percentile_50': float(np.percentile(sharpe_ratios, 50)),
                'percentile_75': float(np.percentile(sharpe_ratios, 75)),
                'percentile_95': float(np.percentile(sharpe_ratios, 95))
            },
            'percentile_ranks': {
                'return_percentile': float(return_percentile),
                'sharpe_percentile': float(sharpe_percentile),
                'drawdown_percentile': float(dd_percentile)
            },
            'probability_estimates': {
                'prob_positive_return': float(np.mean(final_returns > 0)),
                'prob_sharpe_gt_1': float(np.mean(sharpe_ratios > 1.0)),
                'prob_max_dd_lt_20pct': float(np.mean(max_drawdowns < 0.2))
            }
        }
    
    def _regime_detection_analysis(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Regime detection analysis to test strategy performance across different market conditions.
        
        Identifies:
        1. Bull/Bear market regimes
        2. High/Low volatility regimes  
        3. Trending/Ranging regimes
        4. Crisis periods
        """
        returns_array = returns.values
        
        results = {}
        
        # 1. Bull/Bear regime detection (based on cumulative returns)
        results['bull_bear'] = self._detect_bull_bear_regimes(returns)
        
        # 2. Volatility regime detection
        results['volatility'] = self._detect_volatility_regimes(returns)
        
        # 3. Trend regime detection
        results['trend'] = self._detect_trend_regimes(returns)
        
        # 4. Crisis detection (extreme drawdown periods)
        results['crisis'] = self._detect_crisis_periods(returns)
        
        # 5. Overall regime consistency
        results['consistency'] = self._analyze_regime_consistency(results)
        
        return results
    
    def _detect_bull_bear_regimes(self, returns: pd.Series) -> Dict[str, Any]:
        """Detect bull and bear market regimes."""
        # Calculate rolling cumulative returns (6-month periods)
        window = min(126, len(returns) // 4)  # ~6 months or 1/4 of data
        if window < 20:
            return {'error': 'Insufficient data for regime detection'}
        
        rolling_returns = returns.rolling(window=window).apply(lambda x: np.prod(1 + x) - 1)
        
        # Define regimes
        bull_threshold = 0.1  # 10% gain over period
        bear_threshold = -0.1  # 10% loss over period
        
        bull_periods = rolling_returns > bull_threshold
        bear_periods = rolling_returns < bear_threshold
        neutral_periods = ~(bull_periods | bear_periods)
        
        # Calculate performance in each regime
        regimes = {
            'bull': self._calculate_regime_performance(returns[bull_periods]),
            'bear': self._calculate_regime_performance(returns[bear_periods]),
            'neutral': self._calculate_regime_performance(returns[neutral_periods])
        }
        
        # Add regime statistics
        regimes['regime_distribution'] = {
            'bull_periods': int(bull_periods.sum()),
            'bear_periods': int(bear_periods.sum()), 
            'neutral_periods': int(neutral_periods.sum()),
            'bull_percentage': float(bull_periods.mean() * 100),
            'bear_percentage': float(bear_periods.mean() * 100),
            'neutral_percentage': float(neutral_periods.mean() * 100)
        }
        
        return regimes
    
    def _detect_volatility_regimes(self, returns: pd.Series) -> Dict[str, Any]:
        """Detect high and low volatility regimes."""
        # Calculate rolling volatility
        window = min(63, len(returns) // 4)  # ~3 months
        if window < 10:
            return {'error': 'Insufficient data for volatility regime detection'}
        
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        median_vol = rolling_vol.median()
        
        high_vol_periods = rolling_vol > median_vol
        low_vol_periods = rolling_vol <= median_vol
        
        regimes = {
            'high_volatility': self._calculate_regime_performance(returns[high_vol_periods]),
            'low_volatility': self._calculate_regime_performance(returns[low_vol_periods])
        }
        
        regimes['volatility_stats'] = {
            'median_volatility': float(median_vol),
            'high_vol_periods': int(high_vol_periods.sum()),
            'low_vol_periods': int(low_vol_periods.sum())
        }
        
        return regimes
    
    def _detect_trend_regimes(self, returns: pd.Series) -> Dict[str, Any]:
        """Detect trending vs ranging market regimes."""
        # Use price series if available, otherwise cumulative returns
        if hasattr(returns, 'index'):
            prices = (1 + returns).cumprod()
        else:
            prices = pd.Series((1 + returns).cumprod())
        
        # Calculate trend strength using linear regression slope
        window = min(63, len(returns) // 4)
        if window < 10:
            return {'error': 'Insufficient data for trend regime detection'}
        
        def trend_strength(price_window):
            if len(price_window) < 5:
                return 0
            x = np.arange(len(price_window))
            slope, _, r_value, _, _ = stats.linregress(x, np.log(price_window))
            return abs(slope) * (r_value ** 2)  # Weighted by R²
        
        rolling_trend = prices.rolling(window=window).apply(trend_strength)
        median_trend = rolling_trend.median()
        
        trending_periods = rolling_trend > median_trend
        ranging_periods = rolling_trend <= median_trend
        
        regimes = {
            'trending': self._calculate_regime_performance(returns[trending_periods]),
            'ranging': self._calculate_regime_performance(returns[ranging_periods])
        }
        
        regimes['trend_stats'] = {
            'median_trend_strength': float(median_trend),
            'trending_periods': int(trending_periods.sum()),
            'ranging_periods': int(ranging_periods.sum())
        }
        
        return regimes
    
    def _detect_crisis_periods(self, returns: pd.Series) -> Dict[str, Any]:
        """Detect crisis periods based on extreme drawdowns."""
        # Calculate running drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Define crisis as drawdown > 15%
        crisis_threshold = -0.15
        crisis_periods = drawdown < crisis_threshold
        normal_periods = ~crisis_periods
        
        regimes = {
            'crisis': self._calculate_regime_performance(returns[crisis_periods]),
            'normal': self._calculate_regime_performance(returns[normal_periods])
        }
        
        regimes['crisis_stats'] = {
            'crisis_periods': int(crisis_periods.sum()),
            'normal_periods': int(normal_periods.sum()),
            'crisis_percentage': float(crisis_periods.mean() * 100),
            'max_drawdown_in_crisis': float(drawdown[crisis_periods].min()) if crisis_periods.any() else 0
        }
        
        return regimes
    
    def _calculate_regime_performance(self, regime_returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics for a specific regime."""
        if len(regime_returns) == 0:
            return {'count': 0, 'mean_return': 0, 'sharpe_ratio': 0, 'volatility': 0, 'win_rate': 0}
        
        regime_array = regime_returns.values
        
        return {
            'count': len(regime_returns),
            'mean_return': float(np.mean(regime_array)),
            'total_return': float(np.prod(1 + regime_array) - 1),
            'sharpe_ratio': float(_calculate_sharpe_numba(regime_array)),
            'volatility': float(np.std(regime_array) * np.sqrt(252)),
            'win_rate': float(np.mean(regime_array > 0)),
            'max_drawdown': float(self._calculate_max_drawdown_fast(regime_array))
        }
    
    def _analyze_regime_consistency(self, regime_results: Dict) -> Dict[str, Any]:
        """Analyze consistency of performance across regimes."""
        consistency_metrics = {}
        
        # Extract Sharpe ratios across regimes where available
        sharpe_ratios = []
        regime_names = []
        
        for regime_type, regime_data in regime_results.items():
            if isinstance(regime_data, dict) and not regime_data.get('error'):
                for regime_name, metrics in regime_data.items():
                    if isinstance(metrics, dict) and 'sharpe_ratio' in metrics and metrics['count'] > 10:
                        sharpe_ratios.append(metrics['sharpe_ratio'])
                        regime_names.append(f"{regime_type}_{regime_name}")
        
        if len(sharpe_ratios) > 1:
            consistency_metrics['sharpe_ratio_std'] = float(np.std(sharpe_ratios))
            consistency_metrics['sharpe_ratio_range'] = float(max(sharpe_ratios) - min(sharpe_ratios))
            consistency_metrics['min_sharpe'] = float(min(sharpe_ratios))
            consistency_metrics['max_sharpe'] = float(max(sharpe_ratios))
            consistency_metrics['negative_sharpe_regimes'] = sum(1 for sr in sharpe_ratios if sr < 0)
            consistency_metrics['positive_sharpe_regimes'] = sum(1 for sr in sharpe_ratios if sr > 0)
            consistency_metrics['consistency_score'] = 1.0 - (np.std(sharpe_ratios) / (abs(np.mean(sharpe_ratios)) + 0.01))
        
        return consistency_metrics
    
    def _evaluate_comprehensive_validation(self, results: Dict[str, Any]) -> bool:
        """
        Evaluate whether strategy passes comprehensive validation.
        
        Requires:
        1. DSR ≥ 0.95
        2. PSR ≥ 0.95  
        3. Bootstrap confidence intervals exclude zero
        4. Permutation tests show significance
        5. Passes reality check
        6. Consistent performance across regimes
        """
        validation_checks = []
        
        # 1. DSR requirement
        dsr_analysis = results.get('dsr_analysis', {})
        dsr_pass = dsr_analysis.get('passes_dsr_gate', False)
        validation_checks.append(('DSR ≥ 0.95', dsr_pass))
        
        # 2. PSR requirement
        psr_analysis = results.get('psr_analysis', {})
        psr_pass = psr_analysis.get('passes_psr_gate', False)
        validation_checks.append(('PSR ≥ 0.95', psr_pass))
        
        # 3. Bootstrap significance
        bootstrap_analysis = results.get('bootstrap_analysis', {})
        bootstrap_pass = False
        if 'standard' in bootstrap_analysis:
            ci_lower = bootstrap_analysis['standard'].get('ci_lower_sharpe', -999)
            bootstrap_pass = ci_lower > 0
        validation_checks.append(('Bootstrap CI > 0', bootstrap_pass))
        
        # 4. Permutation test significance
        permutation_tests = results.get('permutation_tests', {})
        permutation_pass = False
        if 'combined_p_value' in permutation_tests:
            p_value = permutation_tests['combined_p_value']
            permutation_pass = p_value < 0.05
        elif 'sign_permutation' in permutation_tests:
            p_value = permutation_tests['sign_permutation'].get('p_value', 1)
            permutation_pass = p_value < 0.05
        validation_checks.append(('Permutation test p < 0.05', permutation_pass))
        
        # 5. Reality check
        reality_check = results.get('reality_check', {})
        reality_pass = reality_check.get('passes_reality_check', False)
        validation_checks.append(('Reality check', reality_pass))
        
        # 6. Regime consistency (optional but recommended)
        regime_analysis = results.get('regime_analysis', {})
        regime_pass = True  # Default to pass for now
        if 'consistency' in regime_analysis:
            consistency = regime_analysis['consistency']
            negative_regimes = consistency.get('negative_sharpe_regimes', 0)
            total_regimes = consistency.get('positive_sharpe_regimes', 0) + negative_regimes
            if total_regimes > 0:
                regime_pass = (negative_regimes / total_regimes) < 0.5  # Less than 50% negative regimes
        validation_checks.append(('Regime consistency', regime_pass))
        
        # Store individual check results
        results['validation_checks'] = validation_checks
        
        # Overall pass: all critical checks must pass
        critical_checks = [dsr_pass, psr_pass, bootstrap_pass]  # Core requirements
        overall_pass = all(critical_checks)
        
        # Store pass rate
        total_pass = sum(check[1] for check in validation_checks)
        results['validation_pass_rate'] = total_pass / len(validation_checks)
        
        return overall_pass
    
    def _create_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive validation summary."""
        summary = {
            'overall_result': 'PASS' if results.get('validation_passed', False) else 'FAIL',
            'pass_rate': f"{results.get('validation_pass_rate', 0)*100:.1f}%",
            'critical_metrics': {},
            'warnings': [],
            'recommendations': []
        }
        
        # Extract critical metrics
        if 'dsr_analysis' in results:
            dsr = results['dsr_analysis']
            summary['critical_metrics']['DSR'] = f"{dsr.get('dsr_probability', 0):.3f}"
            summary['critical_metrics']['Sharpe'] = f"{dsr.get('sharpe_ratio', 0):.3f}"
            
        if 'psr_analysis' in results:
            psr = results['psr_analysis']
            summary['critical_metrics']['PSR'] = f"{psr.get('psr', 0):.3f}"
            
        if 'basic_metrics' in results:
            metrics = results['basic_metrics']
            summary['critical_metrics']['Total Return'] = f"{metrics.get('total_return', 0)*100:.1f}%"
            summary['critical_metrics']['Max Drawdown'] = f"{metrics.get('max_drawdown', 0)*100:.1f}%"
            summary['critical_metrics']['Win Rate'] = f"{metrics.get('win_rate', 0)*100:.1f}%"
        
        # Add warnings
        if results.get('dsr_analysis', {}).get('dsr_probability', 0) < 0.9:
            summary['warnings'].append("DSR below 0.90 - high risk of false discovery")
            
        if results.get('psr_analysis', {}).get('psr', 0) < 0.9:
            summary['warnings'].append("PSR below 0.90 - low confidence in positive Sharpe ratio")
            
        if results.get('basic_metrics', {}).get('max_drawdown', 0) < -0.3:
            summary['warnings'].append("Maximum drawdown exceeds 30%")
            
        # Add recommendations
        if not results.get('validation_passed', False):
            summary['recommendations'].append("Strategy requires additional development before deployment")
            summary['recommendations'].append("Consider ensemble methods or risk management overlays")
            
        if len(summary['warnings']) == 0:
            summary['recommendations'].append("Strategy passes all validation criteria")
            summary['recommendations'].append("Consider gradual position scaling in live deployment")
        
        return summary

# ============================================================================
# MULTI-STRATEGY ORCHESTRATION ENGINE
# Enhanced backtesting with multiple strategies and portfolio-level management
# ============================================================================

class BacktestResult:
    """Results from a backtest run with comprehensive analytics."""
    
    def __init__(self, 
                 portfolio_manager: PortfolioManager, 
                 strategies: Dict[str, Any] = None,
                 benchmark_data: pd.DataFrame = None,
                 validation_results: Dict[str, Any] = None):
        self.portfolio_manager = portfolio_manager
        self.strategies = strategies or {}
        self.benchmark_data = benchmark_data
        self.validation_results = validation_results or {}
        
        # Calculate comprehensive metrics
        self.metrics = portfolio_manager.get_performance_metrics()
        self.trades = portfolio_manager.closed_trades
        self.portfolio_history = pd.DataFrame(portfolio_manager.portfolio_history) if portfolio_manager.portfolio_history else pd.DataFrame()
        
    def get_strategy_breakdown(self) -> Dict[str, Dict]:
        """Get performance breakdown by strategy."""
        if not self.trades:
            return {}
        
        strategy_stats = {}
        
        for strategy_id in set(trade.strategy_id for trade in self.trades):
            strategy_trades = [t for t in self.trades if t.strategy_id == strategy_id]
            
            if strategy_trades:
                trade_data = pd.DataFrame([{
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'symbol': trade.symbol,
                    'size': trade.size
                } for trade in strategy_trades])
                
                winning_trades = trade_data[trade_data['pnl'] > 0]
                losing_trades = trade_data[trade_data['pnl'] < 0]
                
                strategy_stats[strategy_id] = {
                    'total_trades': len(strategy_trades),
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades),
                    'win_rate': len(winning_trades) / len(strategy_trades) * 100 if strategy_trades else 0,
                    'total_pnl': trade_data['pnl'].sum(),
                    'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
                    'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
                    'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else float('inf'),
                    'symbols_traded': trade_data['symbol'].nunique()
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
                trade_data = pd.DataFrame([{
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'strategy_id': trade.strategy_id
                } for trade in symbol_trades])
                
                symbol_stats[symbol] = {
                    'total_trades': len(symbol_trades),
                    'total_pnl': trade_data['pnl'].sum(),
                    'strategies_used': trade_data['strategy_id'].nunique(),
                    'avg_pnl_per_trade': trade_data['pnl'].mean(),
                    'best_strategy': trade_data.groupby('strategy_id')['pnl'].sum().idxmax() if len(trade_data) > 0 else None
                }
        
        return symbol_stats

class MultiStrategyOrchestrator:
    """
    Enhanced backtesting orchestrator supporting multiple strategies simultaneously.
    Consolidates functionality from enhanced_backtester.py into the unified framework.
    """
    
    def __init__(self, data_manager=None):
        """
        Initialize the multi-strategy orchestrator.
        
        Args:
            data_manager: Data manager instance for fetching market data
        """
        # Import data manager dynamically to avoid circular imports
        if data_manager is None:
            try:
                from ..data.data_manager import DataManager
                self.data_manager = DataManager()
            except ImportError:
                logger.warning("DataManager not available, will need external data source")
                self.data_manager = None
        else:
            self.data_manager = data_manager
            
        self.strategies: Dict[str, Any] = {}
        self.strategy_symbols: Dict[str, List[str]] = {}
        self.strategy_configs: Dict[str, Dict] = {}
        
    def add_strategy(self, 
                    strategy: Any, 
                    symbols: List[str], 
                    strategy_id: str = None,
                    config: Dict[str, Any] = None) -> str:
        """
        Add a strategy to the orchestrator.
        
        Args:
            strategy: Strategy instance with generate_signal method
            symbols: List of symbols this strategy will trade
            strategy_id: Unique identifier (auto-generated if None)
            config: Strategy-specific configuration
            
        Returns:
            The final strategy_id used
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
        self.strategy_configs[strategy_id] = config or {}
        
        logger.info(f"Added strategy '{strategy_id}' for symbols: {symbols}")
        return strategy_id
    
    async def run_backtest(self, 
                          config: BacktestConfig,
                          data: pd.DataFrame = None,
                          validate: bool = True) -> BacktestResult:
        """
        Run multi-strategy backtest with comprehensive orchestration.
        
        Args:
            config: Backtest configuration
            data: Pre-loaded market data (optional)
            validate: Whether to run validation
            
        Returns:
            BacktestResult with comprehensive analytics
        """
        logger.info(f"Starting multi-strategy backtest with {len(self.strategies)} strategies")
        
        # Prepare data
        if data is None:
            if self.data_manager is None:
                raise ValueError("No data provided and no data manager available")
            data = await self._fetch_market_data(config)
        
        # Initialize portfolio manager
        portfolio = PortfolioManager(
            initial_capital=config.initial_capital,
            commission_rate=config.fees.get('taker', 0.001)
        )
        portfolio.max_position_size = config.max_position_size
        portfolio.max_total_exposure = config.max_total_exposure
        
        # Initialize position sizing engine
        position_sizing_config = PositionSizingConfig(
            method=getattr(config, 'position_sizing_method', PositionSizingType.FIXED_FRACTIONAL),
            fixed_fraction=config.max_position_size,
            target_volatility=getattr(config, 'target_volatility', 0.15),
            memory_decay=getattr(config, 'memory_decay', 0.95)
        )
        position_sizer = PositionSizingEngine(position_sizing_config)
        
        # Prepare strategy-specific data
        strategy_data = {}
        for strategy_id, symbols in self.strategy_symbols.items():
            strategy_data[strategy_id] = self._prepare_strategy_data(data, symbols)
        
        # Run simulation
        await self._run_multi_strategy_simulation(
            data, strategy_data, portfolio, position_sizer, config
        )
        
        # Run validation if requested
        validation_results = None
        if validate and portfolio.closed_trades:
            # Calculate portfolio returns for validation
            portfolio_returns = self._calculate_portfolio_returns(portfolio.portfolio_history)
            
            # Run comprehensive validation
            validator = MonteCarloValidator(
                min_dsr=config.min_dsr,
                min_psr=config.min_psr
            )
            validation_results = validator.validate_strategy_with_advanced_layers(
                portfolio_returns, 
                config.monte_carlo_trials
            )
        
        # Create comprehensive result
        result = BacktestResult(
            portfolio_manager=portfolio,
            strategies=dict(self.strategies),
            benchmark_data=self._extract_benchmark_data(data, config),
            validation_results=validation_results
        )
        
        logger.info(f"Multi-strategy backtest completed. Final value: ${result.metrics.get('final_portfolio_value', 0):,.2f}")
        
        return result
    
    async def _fetch_market_data(self, config: BacktestConfig) -> pd.DataFrame:
        """Fetch market data for all required symbols."""
        # Collect all unique symbols
        all_symbols = set()
        for symbols in self.strategy_symbols.values():
            all_symbols.update(symbols)
        all_symbols = list(all_symbols)
        
        logger.info(f"Fetching data for {len(all_symbols)} symbols")
        
        return await self.data_manager.fetch_data(
            symbols=all_symbols,
            start_date=config.start_date,
            end_date=config.end_date,
            timeframe=getattr(config, 'timeframe', '1h')
        )
    
    def _prepare_strategy_data(self, full_data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Prepare data subset for a specific strategy."""
        if isinstance(full_data.columns, pd.MultiIndex):
            # Multi-level columns (e.g., OHLCV data)
            strategy_columns = []
            for symbol in symbols:
                symbol_columns = [col for col in full_data.columns if len(col) > 1 and col[1] == symbol]
                strategy_columns.extend(symbol_columns)
            return full_data[strategy_columns].copy() if strategy_columns else pd.DataFrame()
        else:
            # Simple column structure
            available_symbols = [col for col in symbols if col in full_data.columns]
            return full_data[available_symbols].copy() if available_symbols else pd.DataFrame()
    
    async def _run_multi_strategy_simulation(self,
                                           data: pd.DataFrame,
                                           strategy_data: Dict[str, pd.DataFrame],
                                           portfolio: PortfolioManager,
                                           position_sizer: PositionSizingEngine,
                                           config: BacktestConfig):
        """Run the multi-strategy simulation loop."""
        timestamps = data.index
        total_periods = len(timestamps)
        
        logger.info(f"Running simulation over {total_periods} time periods")
        
        for i, timestamp in enumerate(timestamps):
            try:
                # Extract current prices
                current_prices = self._extract_current_prices(data, i)
                
                if not current_prices:
                    continue
                
                # Update portfolio history
                portfolio.update_portfolio_history(timestamp, current_prices)
                
                # Process each strategy
                for strategy_id, strategy in self.strategies.items():
                    symbols = self.strategy_symbols[strategy_id]
                    strategy_df = strategy_data.get(strategy_id, pd.DataFrame())
                    
                    if strategy_df.empty:
                        continue
                    
                    # Get historical data up to current point
                    current_data = strategy_df.iloc[:i+1] if i+1 <= len(strategy_df) else strategy_df
                    
                    if len(current_data) < 2:  # Need minimum data for signals
                        continue
                    
                    # Generate signals for each symbol
                    for symbol in symbols:
                        if symbol not in current_prices:
                            continue
                        
                        try:
                            # Extract symbol-specific data
                            symbol_data = self._extract_symbol_data(current_data, symbol)
                            
                            if symbol_data.empty or len(symbol_data) < 2:
                                continue
                            
                            # Generate trading signal
                            signal = strategy.generate_signal(symbol_data)
                            
                            if signal != 0:  # Non-zero signal
                                self._execute_strategy_signal(
                                    portfolio, position_sizer, strategy_id, symbol,
                                    signal, current_prices[symbol], timestamp, config
                                )
                        
                        except Exception as e:
                            logger.warning(f"Error processing {strategy_id}/{symbol} at {timestamp}: {e}")
                            continue
                
                # Progress logging
                if i % max(1, total_periods // 20) == 0:
                    progress = (i / total_periods) * 100
                    portfolio_value = portfolio.get_portfolio_value(current_prices)
                    logger.info(f"Progress: {progress:.1f}% - Portfolio: ${portfolio_value:,.2f}")
            
            except Exception as e:
                logger.error(f"Error at timestamp {timestamp}: {e}")
                continue
        
        # Close all remaining positions
        final_prices = self._extract_current_prices(data, -1)
        if final_prices:
            portfolio.close_all_positions(final_prices, timestamps[-1])
    
    def _extract_current_prices(self, data: pd.DataFrame, index: int) -> Dict[str, float]:
        """Extract current prices for all symbols at given index."""
        current_prices = {}
        
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-level columns
            for col in data.columns:
                if len(col) > 1 and col[0] == 'close':
                    symbol = col[1]
                    price = data.iloc[index][col]
                    if not pd.isna(price):
                        current_prices[symbol] = float(price)
        else:
            # Simple columns - assume they are price data
            for col in data.columns:
                price = data.iloc[index][col]
                if not pd.isna(price):
                    current_prices[col] = float(price)
        
        return current_prices
    
    def _extract_symbol_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Extract OHLCV data for a specific symbol."""
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-level columns - extract OHLCV for symbol
            symbol_columns = [col for col in data.columns if len(col) > 1 and col[1] == symbol]
            if symbol_columns:
                symbol_data = data[symbol_columns].copy()
                # Flatten column names (remove symbol level)
                symbol_data.columns = [col[0] for col in symbol_data.columns]
                return symbol_data
        else:
            # Simple structure - return the symbol column as 'close'
            if symbol in data.columns:
                symbol_data = pd.DataFrame()
                symbol_data['close'] = data[symbol]
                return symbol_data
        
        return pd.DataFrame()
    
    def _execute_strategy_signal(self,
                               portfolio: PortfolioManager,
                               position_sizer: PositionSizingEngine,
                               strategy_id: str,
                               symbol: str,
                               signal: float,
                               price: float,
                               timestamp: datetime,
                               config: BacktestConfig):
        """Execute a trading signal with position sizing."""
        try:
            # Calculate position size using the position sizing engine
            portfolio_value = portfolio.get_portfolio_value({symbol: price})
            
            position_size = position_sizer.calculate_position_size(
                signal_strength=abs(signal),
                portfolio_value=portfolio_value,
                current_price=price,
                strategy_id=strategy_id,
                symbol=symbol,
                market_data=None  # Would need historical data for advanced sizing
            )
            
            # Apply signal direction
            if signal < 0:
                if not config.enable_short_selling:
                    return  # Skip short signals if disabled
                position_size = -position_size
            
            # Check if we should close existing position first
            position_key = f"{strategy_id}_{symbol}"
            if position_key in portfolio.positions:
                existing_position = portfolio.positions[position_key]
                # Close if signal direction changed
                if (existing_position.size > 0 and signal < 0) or (existing_position.size < 0 and signal > 0):
                    portfolio.close_position(position_key, price, timestamp)
            
            # Open new position if size is significant
            if abs(position_size) > 1e-6:  # Minimum position size threshold
                success = portfolio.open_position(
                    symbol=symbol,
                    strategy_id=strategy_id,
                    size=position_size,
                    price=price,
                    timestamp=timestamp,
                    metadata={'signal_strength': signal, 'position_sizing_method': position_sizer.config.method.value}
                )
                
                if success:
                    # Update position sizing engine performance tracking
                    # (PnL will be updated when position is closed)
                    pass
        
        except Exception as e:
            logger.warning(f"Failed to execute signal for {strategy_id}/{symbol}: {e}")
    
    def _calculate_portfolio_returns(self, portfolio_history: List[Dict]) -> pd.Series:
        """Calculate portfolio returns from history."""
        if not portfolio_history or len(portfolio_history) < 2:
            return pd.Series()
        
        df = pd.DataFrame(portfolio_history)
        returns = df['portfolio_value'].pct_change().dropna()
        
        # Set index to timestamps if available
        if 'timestamp' in df.columns:
            returns.index = pd.to_datetime(df['timestamp'].iloc[1:])
        
        return returns
    
    def _extract_benchmark_data(self, data: pd.DataFrame, config: BacktestConfig) -> Optional[pd.DataFrame]:
        """Extract benchmark data if specified."""
        benchmark_symbol = getattr(config, 'benchmark_symbol', None)
        
        if benchmark_symbol is None:
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            # Look for benchmark in multi-level columns
            benchmark_cols = [col for col in data.columns if len(col) > 1 and col[1] == benchmark_symbol and col[0] == 'close']
            if benchmark_cols:
                return data[benchmark_cols[0]].dropna()
        else:
            # Simple column structure
            if benchmark_symbol in data.columns:
                return data[benchmark_symbol].dropna()
        
        return None


# Backward compatibility alias
BacktestEngine = MultiStrategyOrchestrator