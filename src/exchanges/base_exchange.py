"""
Base Exchange Interface

This module defines the abstract base class for all exchange adapters,
ensuring consistent interface across different trading platforms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime
from decimal import Decimal
import logging

# Import pydantic for data validation
try:
    from pydantic import BaseModel, validator
except ImportError:
    # Fallback for older versions
    from pydantic import BaseModel
    validator = lambda *args, **kwargs: lambda func: func


class OrderType(Enum):
    """Order types supported across exchanges"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status across exchanges"""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TradingMode(Enum):
    """Trading environment modes"""
    LIVE = "live"
    PAPER = "paper"
    TESTNET = "testnet"


# Data Models
class Balance(BaseModel):
    """Account balance for a specific asset"""
    asset: str
    free: Decimal
    locked: Decimal
    total: Decimal


class OrderRequest(BaseModel):
    """Standardized order request"""
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "GTC"  # Good Till Canceled
    client_order_id: Optional[str] = None


class Order(BaseModel):
    """Standardized order response"""
    id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: Decimal
    price: Optional[Decimal]
    stop_price: Optional[Decimal]
    status: OrderStatus
    filled_quantity: Decimal
    remaining_quantity: Decimal
    created_at: datetime
    updated_at: datetime
    fees: Optional[Decimal] = None


class Trade(BaseModel):
    """Executed trade information"""
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee: Decimal
    fee_asset: str
    timestamp: datetime


class Ticker(BaseModel):
    """Market ticker data"""
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: Decimal
    high_24h: Decimal
    low_24h: Decimal
    change_24h: Decimal
    timestamp: datetime


class Candle(BaseModel):
    """OHLCV candle data"""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class BaseExchange(ABC):
    """
    Abstract base class for all exchange adapters.
    
    This class defines the standard interface that all exchange implementations
    must follow, ensuring consistent behavior across different trading platforms.
    """
    
    def __init__(self, 
                 api_key: str, 
                 api_secret: str, 
                 mode: TradingMode = TradingMode.PAPER,
                 **kwargs):
        """
        Initialize exchange adapter
        
        Args:
            api_key: API key for exchange
            api_secret: API secret for exchange
            mode: Trading mode (live, paper, testnet)
            **kwargs: Additional exchange-specific parameters
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.mode = mode
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Initialize exchange-specific client
        self._client = None
        self._initialize_client(**kwargs)
    
    @abstractmethod
    def _initialize_client(self, **kwargs) -> None:
        """Initialize the exchange-specific client"""
        pass
    
    @property
    @abstractmethod
    def exchange_name(self) -> str:
        """Return the name of the exchange"""
        pass
    
    @property
    @abstractmethod
    def supported_symbols(self) -> List[str]:
        """Return list of supported trading symbols"""
        pass
    
    # Connection & Authentication
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to exchange
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to exchange"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test exchange connectivity and authentication
        
        Returns:
            True if connection and auth successful
        """
        pass
    
    # Account Management
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information
        
        Returns:
            Account information including balances, permissions, etc.
        """
        pass
    
    @abstractmethod
    async def get_balances(self) -> List[Balance]:
        """
        Get account balances for all assets
        
        Returns:
            List of Balance objects
        """
        pass
    
    @abstractmethod
    async def get_balance(self, asset: str) -> Optional[Balance]:
        """
        Get balance for specific asset
        
        Args:
            asset: Asset symbol (e.g., 'BTC', 'USD')
        
        Returns:
            Balance object or None if not found
        """
        pass
    
    # Market Data
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """
        Get ticker data for symbol
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Ticker object or None if not found
        """
        pass
    
    @abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, List[Tuple[Decimal, Decimal]]]:
        """
        Get order book for symbol
        
        Args:
            symbol: Trading pair symbol
            limit: Number of price levels to return
        
        Returns:
            Dict with 'bids' and 'asks' lists of [price, quantity] tuples
        """
        pass
    
    @abstractmethod
    async def get_candles(self, 
                         symbol: str, 
                         interval: str, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         limit: int = 500) -> List[Candle]:
        """
        Get historical candle data
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval (e.g., '1m', '1h', '1d')
            start_time: Start time for data
            end_time: End time for data
            limit: Maximum number of candles
        
        Returns:
            List of Candle objects
        """
        pass
    
    # Order Management
    @abstractmethod
    async def place_order(self, order_request: OrderRequest) -> Optional[Order]:
        """
        Place a new order
        
        Args:
            order_request: OrderRequest object with order details
        
        Returns:
            Order object if successful, None otherwise
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an existing order
        
        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol
        
        Returns:
            True if cancellation successful
        """
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """
        Get order details
        
        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol
        
        Returns:
            Order object or None if not found
        """
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all open orders
        
        Args:
            symbol: Optional symbol filter
        
        Returns:
            List of open Order objects
        """
        pass
    
    @abstractmethod
    async def get_order_history(self, 
                               symbol: Optional[str] = None,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None,
                               limit: int = 500) -> List[Order]:
        """
        Get order history
        
        Args:
            symbol: Optional symbol filter
            start_time: Start time for history
            end_time: End time for history
            limit: Maximum number of orders
        
        Returns:
            List of Order objects
        """
        pass
    
    @abstractmethod
    async def get_trade_history(self,
                               symbol: Optional[str] = None,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None,
                               limit: int = 500) -> List[Trade]:
        """
        Get trade history
        
        Args:
            symbol: Optional symbol filter
            start_time: Start time for history
            end_time: End time for history
            limit: Maximum number of trades
        
        Returns:
            List of Trade objects
        """
        pass
    
    # Utility Methods
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format for this exchange
        
        Args:
            symbol: Symbol to normalize
        
        Returns:
            Normalized symbol
        """
        return symbol.upper()
    
    def get_minimum_order_size(self, symbol: str) -> Decimal:
        """
        Get minimum order size for symbol
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Minimum order quantity
        """
        # Default implementation - should be overridden by specific exchanges
        return Decimal('0.001')
    
    def get_price_precision(self, symbol: str) -> int:
        """
        Get price precision for symbol
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Number of decimal places for price
        """
        # Default implementation - should be overridden by specific exchanges
        return 8
    
    def get_quantity_precision(self, symbol: str) -> int:
        """
        Get quantity precision for symbol
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Number of decimal places for quantity
        """
        # Default implementation - should be overridden by specific exchanges
        return 8
    
    # Context Manager Support
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
