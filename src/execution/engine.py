"""
Execution engine for cryptocurrency trading with realistic latency simulation.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Order representation with all execution details."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    latency_ms: float = 0.0
    exchange: str = "simulated"
    strategy_id: str = ""
    
    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]

@dataclass
class Fill:
    """Trade execution fill details."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    exchange: str = "simulated"
    
    @property
    def notional_value(self) -> float:
        return self.quantity * self.price

@dataclass
class Position:
    """Position tracking with P&L calculation."""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    last_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl
    
    def update_unrealized_pnl(self, current_price: float):
        """Update unrealized P&L with current market price."""
        self.last_price = current_price
        if self.quantity != 0:
            self.unrealized_pnl = (current_price - self.avg_price) * self.quantity

class ExecutionEngine:
    """
    Realistic execution engine with latency simulation and market impact.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize execution engine.
        
        Args:
            config: Execution configuration
        """
        self.config = config
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.positions: Dict[str, Position] = defaultdict(lambda: Position(""))
        self.cash_balance = config.get('initial_capital', 100000.0)
        self.commission_rate = config.get('commission_rate', 0.001)  # 0.1%
        self.slippage_model = config.get('slippage_model', 'linear')
        self.latency_simulator = LatencySimulator(config.get('latency_config', {}))
        
        # Performance tracking
        self.trade_count = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
    def submit_order(self, order: Order) -> str:
        """
        Submit order for execution.
        
        Args:
            order: Order to submit
            
        Returns:
            Order ID
        """
        order.timestamp = datetime.now()
        self.orders[order.id] = order
        
        logger.info(f"Order submitted: {order.side.value} {order.quantity} {order.symbol} @ {order.price}")
        
        return order.id
    
    async def process_orders(self, market_data: Dict[str, float]) -> List[Fill]:
        """
        Process pending orders against market data.
        
        Args:
            market_data: Current market prices {symbol: price}
            
        Returns:
            List of new fills
        """
        new_fills = []
        
        for order in list(self.orders.values()):
            if not order.is_active:
                continue
                
            symbol_price = market_data.get(order.symbol)
            if symbol_price is None:
                continue
                
            # Simulate network latency
            latency_ms = await self.latency_simulator.get_latency(order.exchange)
            order.latency_ms = latency_ms
            
            # Check if order can be filled
            fill_result = self._check_fill_conditions(order, symbol_price)
            
            if fill_result['can_fill']:
                fill = await self._execute_fill(order, fill_result, symbol_price)
                if fill:
                    new_fills.append(fill)
                    self.fills.append(fill)
                    self._update_position(fill)
        
        return new_fills
    
    def _check_fill_conditions(self, order: Order, market_price: float) -> Dict[str, Any]:
        """
        Check if order can be filled based on type and market conditions.
        
        Args:
            order: Order to check
            market_price: Current market price
            
        Returns:
            Dictionary with fill conditions
        """
        result = {
            'can_fill': False,
            'fill_price': market_price,
            'fill_quantity': order.remaining_quantity
        }
        
        if order.order_type == OrderType.MARKET:
            result['can_fill'] = True
            
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and market_price <= order.price:
                result['can_fill'] = True
                result['fill_price'] = min(order.price, market_price)
            elif order.side == OrderSide.SELL and market_price >= order.price:
                result['can_fill'] = True
                result['fill_price'] = max(order.price, market_price)
                
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY and market_price >= order.stop_price:
                result['can_fill'] = True
            elif order.side == OrderSide.SELL and market_price <= order.stop_price:
                result['can_fill'] = True
                
        elif order.order_type == OrderType.STOP_LIMIT:
            # Convert to limit order when stop triggered
            if order.side == OrderSide.BUY and market_price >= order.stop_price:
                if market_price <= order.price:
                    result['can_fill'] = True
                    result['fill_price'] = min(order.price, market_price)
            elif order.side == OrderSide.SELL and market_price <= order.stop_price:
                if market_price >= order.price:
                    result['can_fill'] = True
                    result['fill_price'] = max(order.price, market_price)
        
        return result
    
    async def _execute_fill(self, order: Order, fill_result: Dict[str, Any], market_price: float) -> Optional[Fill]:
        """
        Execute order fill with realistic slippage and commission.
        
        Args:
            order: Order to fill
            fill_result: Fill conditions
            market_price: Current market price
            
        Returns:
            Fill object or None if execution fails
        """
        fill_price = fill_result['fill_price']
        fill_quantity = fill_result['fill_quantity']
        
        # Apply slippage model
        slippage = self._calculate_slippage(order, fill_quantity, market_price)
        if order.side == OrderSide.BUY:
            fill_price += slippage
        else:
            fill_price -= slippage
            
        # Calculate commission
        commission = self._calculate_commission(fill_quantity, fill_price)
        
        # Check if we have sufficient balance/position
        if not self._validate_execution(order, fill_quantity, fill_price, commission):
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order rejected: insufficient balance/position")
            return None
        
        # Create fill
        fill = Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            commission=commission,
            timestamp=datetime.now(),
            exchange=order.exchange
        )
        
        # Update order
        order.filled_quantity += fill_quantity
        order.filled_price = ((order.filled_price * (order.filled_quantity - fill_quantity)) + 
                             (fill_price * fill_quantity)) / order.filled_quantity
        order.commission += commission
        order.slippage += slippage
        
        if order.remaining_quantity <= 1e-8:  # Essentially zero
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIAL
            
        # Update cash balance
        notional = fill_quantity * fill_price
        if order.side == OrderSide.BUY:
            self.cash_balance -= (notional + commission)
        else:
            self.cash_balance += (notional - commission)
            
        self.trade_count += 1
        self.total_commission += commission
        self.total_slippage += abs(slippage)
        
        logger.info(f"Fill executed: {fill.side.value} {fill.quantity} {fill.symbol} @ {fill.price:.4f}")
        
        return fill
    
    def _calculate_slippage(self, order: Order, quantity: float, market_price: float) -> float:
        """
        Calculate realistic slippage based on order size and market conditions.
        
        Args:
            order: Order being executed
            quantity: Fill quantity
            market_price: Current market price
            
        Returns:
            Slippage amount (always positive)
        """
        base_slippage = self.config.get('base_slippage_bps', 2) / 10000  # 2 bps default
        
        if self.slippage_model == 'linear':
            # Linear model: slippage increases with order size
            size_factor = min(quantity / 10.0, 2.0)  # Cap at 2x
            slippage = base_slippage * market_price * (1 + size_factor)
            
        elif self.slippage_model == 'sqrt':
            # Square root model: concave slippage function
            size_factor = np.sqrt(min(quantity / 10.0, 4.0))
            slippage = base_slippage * market_price * (1 + size_factor)
            
        elif self.slippage_model == 'fixed':
            # Fixed slippage regardless of size
            slippage = base_slippage * market_price
            
        else:
            slippage = 0.0
            
        # Add random component for realism
        noise_factor = self.config.get('slippage_noise', 0.5)
        random_component = np.random.normal(0, noise_factor) * base_slippage * market_price
        slippage += abs(random_component)
        
        return slippage
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """
        Calculate trading commission.
        
        Args:
            quantity: Trade quantity
            price: Execution price
            
        Returns:
            Commission amount
        """
        notional = quantity * price
        commission = notional * self.commission_rate
        
        # Minimum commission
        min_commission = self.config.get('min_commission', 0.0)
        return max(commission, min_commission)
    
    def _validate_execution(self, order: Order, quantity: float, price: float, commission: float) -> bool:
        """
        Validate that execution is possible given current balances.
        
        Args:
            order: Order to validate
            quantity: Execution quantity
            price: Execution price
            commission: Commission cost
            
        Returns:
            True if execution is valid
        """
        notional = quantity * price
        
        if order.side == OrderSide.BUY:
            # Check cash balance
            required_cash = notional + commission
            return self.cash_balance >= required_cash
        else:
            # Check position
            current_position = self.positions[order.symbol].quantity
            return current_position >= quantity
    
    def _update_position(self, fill: Fill):
        """
        Update position based on fill.
        
        Args:
            fill: Executed fill
        """
        position = self.positions[fill.symbol]
        
        if position.symbol == "":
            position.symbol = fill.symbol
            
        old_quantity = position.quantity
        old_avg_price = position.avg_price
        
        if fill.side == OrderSide.BUY:
            # Adding to position
            new_quantity = old_quantity + fill.quantity
            if new_quantity != 0:
                position.avg_price = ((old_quantity * old_avg_price) + 
                                    (fill.quantity * fill.price)) / new_quantity
            position.quantity = new_quantity
            
        else:
            # Reducing position
            position.quantity = old_quantity - fill.quantity
            
            # Calculate realized P&L for the sold portion
            if old_quantity != 0:
                realized_pnl = (fill.price - old_avg_price) * fill.quantity
                position.realized_pnl += realized_pnl
        
        position.total_commission += fill.commission
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel pending order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successfully cancelled
        """
        if order_id not in self.orders:
            return False
            
        order = self.orders[order_id]
        if order.is_active:
            order.status = OrderStatus.CANCELLED
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        return False
    
    def get_portfolio_value(self, market_data: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            market_data: Current market prices
            
        Returns:
            Total portfolio value
        """
        total_value = self.cash_balance
        
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                current_price = market_data.get(symbol, position.avg_price)
                position.update_unrealized_pnl(current_price)
                total_value += position.market_value
                
        return total_value
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get execution performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        avg_commission_per_trade = self.total_commission / max(self.trade_count, 1)
        avg_slippage_per_trade = self.total_slippage / max(self.trade_count, 1)
        
        return {
            'total_trades': self.trade_count,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'avg_commission_per_trade': avg_commission_per_trade,
            'avg_slippage_per_trade': avg_slippage_per_trade,
            'total_realized_pnl': total_realized_pnl,
            'cash_balance': self.cash_balance,
            'active_orders': len([o for o in self.orders.values() if o.is_active])
        }
    
    def reset(self):
        """Reset execution engine state."""
        self.orders.clear()
        self.fills.clear()
        self.positions.clear()
        self.cash_balance = self.config.get('initial_capital', 100000.0)
        self.trade_count = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0

class LatencySimulator:
    """
    Realistic network latency simulation for different exchanges.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize latency simulator.
        
        Args:
            config: Latency configuration
        """
        self.config = config
        self.exchange_latencies = config.get('exchange_latencies', {
            'binance': {'mean': 50, 'std': 15, 'min': 20, 'max': 200},
            'coinbase': {'mean': 75, 'std': 20, 'min': 30, 'max': 250},
            'kraken': {'mean': 100, 'std': 25, 'min': 40, 'max': 300},
            'simulated': {'mean': 1, 'std': 0.5, 'min': 0.5, 'max': 2}
        })
        
    async def get_latency(self, exchange: str = 'simulated') -> float:
        """
        Get realistic latency for exchange.
        
        Args:
            exchange: Exchange name
            
        Returns:
            Latency in milliseconds
        """
        params = self.exchange_latencies.get(exchange, self.exchange_latencies['simulated'])
        
        # Generate latency from truncated normal distribution
        latency = np.random.normal(params['mean'], params['std'])
        latency = np.clip(latency, params['min'], params['max'])
        
        # Simulate occasional network spikes
        if np.random.random() < 0.05:  # 5% chance of spike
            latency *= np.random.uniform(2, 5)
            
        # Simulate actual network delay
        await asyncio.sleep(latency / 1000.0)
        
        return latency
