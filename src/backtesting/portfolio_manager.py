"""
Portfolio manager for handling multiple assets and strategies in backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    size: float
    entry_price: float
    entry_time: datetime
    position_type: str  # 'long' or 'short'
    strategy_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Trade:
    """Represents a completed trade."""
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
    metadata: Dict[str, Any] = field(default_factory=dict)

class PortfolioManager:
    """
    Manages portfolio state, positions, and trades across multiple strategies and assets.
    """
    
    def __init__(self, initial_capital: float = 100000, commission_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        
        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}  # key: f"{symbol}_{strategy_id}"
        self.closed_trades: List[Trade] = []
        
        # Portfolio history for analysis
        self.portfolio_history: List[Dict[str, Any]] = []
        
        # Risk management
        self.max_position_size = 0.1  # 10% of portfolio per position
        self.max_total_exposure = 0.8  # 80% max total exposure
        
    def can_open_position(self, symbol: str, strategy_id: str, size: float, price: float) -> bool:
        """Check if position can be opened based on risk limits."""
        position_value = abs(size * price)
        
        # Check individual position size limit
        portfolio_value = self.get_portfolio_value()
        if position_value > portfolio_value * self.max_position_size:
            return False
        
        # Check total exposure limit
        total_exposure = self.get_total_exposure() + position_value
        if total_exposure > portfolio_value * self.max_total_exposure:
            return False
        
        # Check sufficient cash (for long positions)
        if size > 0:  # long position
            required_cash = position_value + (position_value * self.commission_rate)
            if required_cash > self.cash:
                return False
        
        return True
    
    def open_position(self, symbol: str, strategy_id: str, size: float, price: float, 
                     timestamp: datetime, metadata: Dict[str, Any] = None) -> bool:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            strategy_id: Strategy identifier
            size: Position size (positive for long, negative for short)
            price: Entry price
            timestamp: Entry timestamp
            metadata: Additional position metadata
            
        Returns:
            True if position opened successfully
        """
        if not self.can_open_position(symbol, strategy_id, size, price):
            logger.warning(f"Cannot open position: {symbol} {size} @ {price} for {strategy_id}")
            return False
        
        position_key = f"{symbol}_{strategy_id}"
        
        # Close existing position if any
        if position_key in self.positions:
            self.close_position(symbol, strategy_id, price, timestamp, "position_replaced")
        
        # Calculate commission
        commission = abs(size * price) * self.commission_rate
        
        # Create new position
        position = Position(
            symbol=symbol,
            size=size,
            entry_price=price,
            entry_time=timestamp,
            position_type='long' if size > 0 else 'short',
            strategy_id=strategy_id,
            metadata=metadata or {}
        )
        
        self.positions[position_key] = position
        
        # Update cash (for long positions, we spend cash; for short, we receive cash)
        if size > 0:  # long
            self.cash -= (abs(size * price) + commission)
        else:  # short
            self.cash += (abs(size * price) - commission)
        
        logger.info(f"Opened {position.position_type} position: {symbol} {size} @ {price}")
        return True
    
    def close_position(self, symbol: str, strategy_id: str, price: float, 
                      timestamp: datetime, reason: str = "manual") -> Optional[Trade]:
        """
        Close an existing position.
        
        Args:
            symbol: Trading symbol
            strategy_id: Strategy identifier  
            price: Exit price
            timestamp: Exit timestamp
            reason: Reason for closing
            
        Returns:
            Trade object if position was closed
        """
        position_key = f"{symbol}_{strategy_id}"
        
        if position_key not in self.positions:
            logger.warning(f"No position to close: {position_key}")
            return None
        
        position = self.positions[position_key]
        
        # Calculate PnL
        if position.position_type == 'long':
            pnl = position.size * (price - position.entry_price)
        else:  # short
            pnl = abs(position.size) * (position.entry_price - price)
        
        pnl_pct = pnl / (abs(position.size) * position.entry_price) * 100
        
        # Calculate commission
        commission = abs(position.size * price) * self.commission_rate
        net_pnl = pnl - commission
        
        # Update cash
        if position.position_type == 'long':
            self.cash += (position.size * price) - commission
        else:  # short
            self.cash += commission  # We already got cash when opening short
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            strategy_id=strategy_id,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=price,
            size=position.size,
            position_type=position.position_type,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            commission=commission * 2,  # entry + exit commission
            metadata={**position.metadata, 'exit_reason': reason}
        )
        
        self.closed_trades.append(trade)
        del self.positions[position_key]
        
        logger.info(f"Closed {position.position_type} position: {symbol} PnL: ${net_pnl:.2f} ({pnl_pct:.2f}%)")
        return trade
    
    def close_all_positions(self, price_data: Dict[str, float], timestamp: datetime, reason: str = "end_of_backtest"):
        """Close all open positions."""
        positions_to_close = list(self.positions.keys())
        
        for position_key in positions_to_close:
            position = self.positions[position_key]
            symbol = position.symbol
            
            if symbol in price_data:
                self.close_position(symbol, position.strategy_id, price_data[symbol], timestamp, reason)
            else:
                logger.warning(f"No price data for {symbol}, cannot close position")
    
    def get_portfolio_value(self, current_prices: Dict[str, float] = None) -> float:
        """Calculate current portfolio value."""
        total_value = self.cash
        
        if current_prices:
            for position in self.positions.values():
                if position.symbol in current_prices:
                    current_price = current_prices[position.symbol]
                    if position.position_type == 'long':
                        position_value = position.size * current_price
                    else:  # short
                        position_value = abs(position.size) * (2 * position.entry_price - current_price)
                    total_value += position_value
        
        return total_value
    
    def get_total_exposure(self, current_prices: Dict[str, float] = None) -> float:
        """Calculate total position exposure."""
        total_exposure = 0
        
        for position in self.positions.values():
            if current_prices and position.symbol in current_prices:
                price = current_prices[position.symbol]
            else:
                price = position.entry_price
            
            exposure = abs(position.size * price)
            total_exposure += exposure
        
        return total_exposure
    
    def update_portfolio_history(self, timestamp: datetime, current_prices: Dict[str, float]):
        """Record portfolio state for analysis."""
        portfolio_value = self.get_portfolio_value(current_prices)
        total_exposure = self.get_total_exposure(current_prices)
        
        self.portfolio_history.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'total_exposure': total_exposure,
            'num_positions': len(self.positions),
            'returns': (portfolio_value - self.initial_capital) / self.initial_capital * 100
        })
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio performance metrics."""
        if not self.closed_trades:
            return {'total_trades': 0}
        
        trades_df = pd.DataFrame([{
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct,
            'symbol': trade.symbol,
            'strategy_id': trade.strategy_id,
            'duration': (trade.exit_time - trade.entry_time).total_seconds() / 3600  # hours
        } for trade in self.closed_trades])
        
        # Overall metrics
        total_pnl = trades_df['pnl'].sum()
        total_return = total_pnl / self.initial_capital * 100
        
        # Win/loss metrics
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Risk metrics
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else float('inf')
        
        # Portfolio history metrics
        if self.portfolio_history:
            portfolio_df = pd.DataFrame(self.portfolio_history)
            max_portfolio_value = portfolio_df['portfolio_value'].max()
            max_drawdown = ((max_portfolio_value - portfolio_df['portfolio_value'].min()) / max_portfolio_value) * 100
        else:
            max_drawdown = 0
        
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'avg_trade_duration': trades_df['duration'].mean(),
            'final_portfolio_value': self.get_portfolio_value()
        }
