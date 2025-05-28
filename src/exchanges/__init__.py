"""
Exchange Adapters for Multi-Exchange Trading

This module provides unified interfaces for interacting with different
cryptocurrency and stock exchanges, enabling seamless strategy execution
across multiple platforms.

Supported Exchanges:
- Bybit: Cryptocurrency exchange with testnet support
- Alpaca: Stock and crypto trading with paper trading support
"""

from .base_exchange import (
    BaseExchange, Balance, OrderRequest, Order, Trade, Ticker, Candle,
    OrderType, OrderSide, OrderStatus, TradingMode
)
from .exchange_factory import ExchangeFactory

# Import adapters for auto-registration
try:
    from .bybit_adapter import BybitAdapter
except ImportError:
    BybitAdapter = None

try:
    from .alpaca_adapter import AlpacaAdapter
except ImportError:
    AlpacaAdapter = None

__all__ = [
    'BaseExchange', 
    'ExchangeFactory',
    'Balance', 
    'OrderRequest', 
    'Order', 
    'Trade', 
    'Ticker', 
    'Candle',
    'OrderType', 
    'OrderSide', 
    'OrderStatus', 
    'TradingMode',
    'BybitAdapter',
    'AlpacaAdapter'
]
