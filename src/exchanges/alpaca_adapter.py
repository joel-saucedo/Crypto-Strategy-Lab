"""
Alpaca Exchange Adapter

This module provides integration with Alpaca stock trading platform.
Supports both paper trading and live trading environments.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import logging

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    tradeapi = None
    APIError = Exception

from .base_exchange import (
    BaseExchange, Balance, OrderRequest, Order, Trade, Ticker, Candle,
    OrderType, OrderSide, OrderStatus, TradingMode
)


class AlpacaAdapter(BaseExchange):
    """
    Alpaca exchange adapter for stock trading
    
    Supports both paper trading and live trading with stocks, ETFs, and cryptocurrencies.
    """
    
    def __init__(self, 
                 api_key: str, 
                 api_secret: str, 
                 mode: TradingMode = TradingMode.PAPER,
                 **kwargs):
        """
        Initialize Alpaca adapter
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            mode: Trading mode (live or paper)
            **kwargs: Additional parameters
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-trade-api library is required for Alpaca integration. Install with: pip install alpaca-trade-api")
        
        self._paper_trading = mode == TradingMode.PAPER
        self._api = None
        
        # Base URLs
        self._base_url = 'https://paper-api.alpaca.markets' if self._paper_trading else 'https://api.alpaca.markets'
        self._data_url = 'https://data.alpaca.markets'
        
        super().__init__(api_key, api_secret, mode, **kwargs)
    
    def _initialize_client(self, **kwargs) -> None:
        """Initialize Alpaca API client"""
        try:
            self._api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self._base_url,
                api_version='v2'
            )
            self.logger.info(f"Initialized Alpaca client ({'paper' if self._paper_trading else 'live'})")
        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca client: {e}")
            raise
    
    @property
    def exchange_name(self) -> str:
        """Return exchange name"""
        return "alpaca"
    
    @property
    def supported_symbols(self) -> List[str]:
        """Get supported trading symbols"""
        # Return some common symbols - full list would be too large
        return [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'SPY', 'QQQ', 'IWM', 'VTI', 'BTC/USD', 'ETH/USD', 'LTC/USD'
        ]
    
    async def connect(self) -> bool:
        """Establish connection and test API"""
        try:
            return await self.test_connection()
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close connection"""
        self._api = None
        self.logger.info("Disconnected from Alpaca")
    
    async def test_connection(self) -> bool:
        """Test connection and authentication"""
        try:
            # Test with account info request
            account = self._api.get_account()
            return account is not None
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def normalize_symbol(self, symbol: str) -> str:
        """Convert symbol format for Alpaca"""
        # Alpaca uses different formats for crypto vs stocks
        if '/' in symbol:
            # Crypto format: BTC/USD -> BTCUSD
            base, quote = symbol.split('/')
            return f"{base}{quote}".upper()
        else:
            # Stock format: keep as is
            return symbol.upper()
    
    def _standardize_symbol(self, alpaca_symbol: str) -> str:
        """Convert Alpaca symbol to standard format"""
        # Common crypto symbols that should have / format
        crypto_symbols = ['BTCUSD', 'ETHUSD', 'LTCUSD', 'ADAUSD', 'DOTUSD']
        
        if alpaca_symbol in crypto_symbols:
            # Convert BTCUSD -> BTC/USD
            if alpaca_symbol.endswith('USD'):
                base = alpaca_symbol[:-3]
                return f"{base}/USD"
        
        return alpaca_symbol
    
    # Account Management
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            account = self._api.get_account()
            return {
                'account_id': account.id,
                'status': account.status,
                'currency': account.currency,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'pattern_day_trader': account.pattern_day_trader,
                'trade_suspended_by_user': account.trade_suspended_by_user,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked,
                'created_at': account.created_at,
                'paper_trading': self._paper_trading
            }
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    async def get_balances(self) -> List[Balance]:
        """Get account balances"""
        try:
            account = self._api.get_account()
            positions = self._api.list_positions()
            
            balances = []
            
            # Add cash balance
            cash_balance = Balance(
                asset='USD',
                free=Decimal(str(account.cash)),
                locked=Decimal('0'),
                total=Decimal(str(account.cash))
            )
            balances.append(cash_balance)
            
            # Add position balances
            for position in positions:
                if float(position.qty) != 0:
                    market_value = Decimal(str(position.market_value))
                    balance = Balance(
                        asset=self._standardize_symbol(position.symbol),
                        free=Decimal(str(position.qty)),
                        locked=Decimal('0'),
                        total=Decimal(str(position.qty))
                    )
                    balances.append(balance)
            
            return balances
        except Exception as e:
            self.logger.error(f"Error getting balances: {e}")
            return []
    
    async def get_balance(self, asset: str) -> Optional[Balance]:
        """Get balance for specific asset"""
        balances = await self.get_balances()
        for balance in balances:
            if balance.asset == asset.upper():
                return balance
        return None
    
    # Market Data
    async def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """Get ticker data"""
        try:
            alpaca_symbol = self.normalize_symbol(symbol)
            
            # Get latest quote
            latest_quote = self._api.get_latest_quote(alpaca_symbol)
            
            # Get latest trade
            latest_trade = self._api.get_latest_trade(alpaca_symbol)
            
            # Get daily bars for 24h stats
            end = datetime.now()
            start = end - timedelta(days=1)
            bars = self._api.get_bars(alpaca_symbol, '1Day', start=start, end=end).df
            
            if not bars.empty:
                latest_bar = bars.iloc[-1]
                high_24h = latest_bar['high']
                low_24h = latest_bar['low']
                volume = latest_bar['volume']
                change_24h = (latest_trade.price - latest_bar['open']) / latest_bar['open'] * 100
            else:
                high_24h = low_24h = volume = change_24h = 0
            
            return Ticker(
                symbol=self._standardize_symbol(alpaca_symbol),
                bid=Decimal(str(latest_quote.bid_price)),
                ask=Decimal(str(latest_quote.ask_price)),
                last=Decimal(str(latest_trade.price)),
                volume=Decimal(str(volume)),
                high_24h=Decimal(str(high_24h)),
                low_24h=Decimal(str(low_24h)),
                change_24h=Decimal(str(change_24h)),
                timestamp=latest_trade.timestamp
            )
        except Exception as e:
            self.logger.error(f"Error getting ticker for {symbol}: {e}")
            return None
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, List[Tuple[Decimal, Decimal]]]:
        """Get order book (limited data available from Alpaca)"""
        try:
            alpaca_symbol = self.normalize_symbol(symbol)
            
            # Alpaca doesn't provide full orderbook, only latest quote
            latest_quote = self._api.get_latest_quote(alpaca_symbol)
            
            # Create minimal orderbook with best bid/ask
            bids = [(Decimal(str(latest_quote.bid_price)), Decimal(str(latest_quote.bid_size)))]
            asks = [(Decimal(str(latest_quote.ask_price)), Decimal(str(latest_quote.ask_size)))]
            
            return {'bids': bids, 'asks': asks}
        except Exception as e:
            self.logger.error(f"Error getting orderbook for {symbol}: {e}")
            return {'bids': [], 'asks': []}
    
    async def get_candles(self, 
                         symbol: str, 
                         interval: str, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         limit: int = 500) -> List[Candle]:
        """Get historical candle data"""
        try:
            alpaca_symbol = self.normalize_symbol(symbol)
            
            # Convert interval to Alpaca format
            interval_map = {
                '1m': '1Min',
                '5m': '5Min',
                '15m': '15Min',
                '30m': '30Min',
                '1h': '1Hour',
                '1d': '1Day'
            }
            alpaca_interval = interval_map.get(interval, '1Day')
            
            # Set default time range if not provided
            if not end_time:
                end_time = datetime.now()
            if not start_time:
                if alpaca_interval in ['1Min', '5Min', '15Min', '30Min']:
                    start_time = end_time - timedelta(days=1)
                else:
                    start_time = end_time - timedelta(days=365)
            
            # Get bars from Alpaca
            bars = self._api.get_bars(
                alpaca_symbol,
                alpaca_interval,
                start=start_time,
                end=end_time,
                limit=limit
            ).df
            
            candles = []
            for index, row in bars.iterrows():
                candle = Candle(
                    timestamp=index.to_pydatetime(),
                    open=Decimal(str(row['open'])),
                    high=Decimal(str(row['high'])),
                    low=Decimal(str(row['low'])),
                    close=Decimal(str(row['close'])),
                    volume=Decimal(str(row['volume']))
                )
                candles.append(candle)
            
            return sorted(candles, key=lambda x: x.timestamp)
        except Exception as e:
            self.logger.error(f"Error getting candles for {symbol}: {e}")
            return []
    
    # Order Management
    def _map_order_type(self, order_type: OrderType) -> str:
        """Map order type to Alpaca format"""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_LOSS: "stop",
            OrderType.TAKE_PROFIT: "limit"
        }
        return mapping.get(order_type, "market")
    
    def _map_order_side(self, side: OrderSide) -> str:
        """Map order side to Alpaca format"""
        return "buy" if side == OrderSide.BUY else "sell"
    
    def _parse_order_status(self, status: str) -> OrderStatus:
        """Parse Alpaca order status to standard format"""
        status_map = {
            'new': OrderStatus.OPEN,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'done_for_day': OrderStatus.CANCELED,
            'canceled': OrderStatus.CANCELED,
            'expired': OrderStatus.EXPIRED,
            'replaced': OrderStatus.CANCELED,
            'pending_cancel': OrderStatus.PENDING,
            'pending_replace': OrderStatus.PENDING,
            'accepted': OrderStatus.OPEN,
            'pending_new': OrderStatus.PENDING,
            'accepted_for_bidding': OrderStatus.OPEN,
            'stopped': OrderStatus.CANCELED,
            'rejected': OrderStatus.REJECTED,
            'suspended': OrderStatus.CANCELED,
            'calculated': OrderStatus.PENDING
        }
        return status_map.get(status, OrderStatus.PENDING)
    
    async def place_order(self, order_request: OrderRequest) -> Optional[Order]:
        """Place a new order"""
        try:
            alpaca_symbol = self.normalize_symbol(order_request.symbol)
            
            # Prepare order parameters
            order_params = {
                'symbol': alpaca_symbol,
                'qty': float(order_request.quantity),
                'side': self._map_order_side(order_request.side),
                'type': self._map_order_type(order_request.type),
                'time_in_force': order_request.time_in_force
            }
            
            if order_request.price and order_request.type == OrderType.LIMIT:
                order_params['limit_price'] = float(order_request.price)
            
            if order_request.stop_price and order_request.type == OrderType.STOP_LOSS:
                order_params['stop_price'] = float(order_request.stop_price)
            
            if order_request.client_order_id:
                order_params['client_order_id'] = order_request.client_order_id
            
            # Submit order
            alpaca_order = self._api.submit_order(**order_params)
            
            return Order(
                id=alpaca_order.id,
                client_order_id=alpaca_order.client_order_id,
                symbol=self._standardize_symbol(alpaca_order.symbol),
                side=order_request.side,
                type=order_request.type,
                quantity=Decimal(str(alpaca_order.qty)),
                price=Decimal(str(alpaca_order.limit_price)) if alpaca_order.limit_price else None,
                stop_price=Decimal(str(alpaca_order.stop_price)) if alpaca_order.stop_price else None,
                status=self._parse_order_status(alpaca_order.status),
                filled_quantity=Decimal(str(alpaca_order.filled_qty or 0)),
                remaining_quantity=Decimal(str(float(alpaca_order.qty) - float(alpaca_order.filled_qty or 0))),
                created_at=alpaca_order.created_at,
                updated_at=alpaca_order.updated_at or alpaca_order.created_at
            )
                
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            self._api.cancel_order(order_id)
            return True
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {e}")
            return False
    
    async def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get order details"""
        try:
            alpaca_order = self._api.get_order(order_id)
            return self._parse_order(alpaca_order)
        except Exception as e:
            self.logger.error(f"Error getting order {order_id}: {e}")
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders"""
        try:
            params = {'status': 'open'}
            if symbol:
                params['symbols'] = self.normalize_symbol(symbol)
            
            alpaca_orders = self._api.list_orders(**params)
            orders = []
            
            for alpaca_order in alpaca_orders:
                order = self._parse_order(alpaca_order)
                if order:
                    orders.append(order)
            
            return orders
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            return []
    
    async def get_order_history(self, 
                               symbol: Optional[str] = None,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None,
                               limit: int = 500) -> List[Order]:
        """Get order history"""
        try:
            params = {
                'status': 'all',
                'limit': limit
            }
            
            if symbol:
                params['symbols'] = self.normalize_symbol(symbol)
            if start_time:
                params['after'] = start_time
            if end_time:
                params['until'] = end_time
            
            alpaca_orders = self._api.list_orders(**params)
            orders = []
            
            for alpaca_order in alpaca_orders:
                order = self._parse_order(alpaca_order)
                if order:
                    orders.append(order)
            
            return orders
        except Exception as e:
            self.logger.error(f"Error getting order history: {e}")
            return []
    
    async def get_trade_history(self,
                               symbol: Optional[str] = None,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None,
                               limit: int = 500) -> List[Trade]:
        """Get trade history"""
        try:
            params = {
                'limit': limit
            }
            
            if start_time:
                params['after'] = start_time
            if end_time:
                params['until'] = end_time
            
            # Get portfolio history to find trades
            portfolio_history = self._api.get_portfolio_history(
                period='1D' if not start_time else None,
                timeframe='1Min',
                **params
            )
            
            # Note: Alpaca doesn't provide direct trade history API
            # This is a simplified implementation
            # In practice, you'd need to track fills from order updates
            
            return []
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []
    
    def _parse_order(self, alpaca_order) -> Optional[Order]:
        """Parse Alpaca order to standard Order object"""
        try:
            # Determine order type
            order_type = OrderType.MARKET
            if alpaca_order.order_type == 'limit':
                order_type = OrderType.LIMIT
            elif alpaca_order.order_type == 'stop':
                order_type = OrderType.STOP_LOSS
            
            return Order(
                id=alpaca_order.id,
                client_order_id=alpaca_order.client_order_id,
                symbol=self._standardize_symbol(alpaca_order.symbol),
                side=OrderSide.BUY if alpaca_order.side == 'buy' else OrderSide.SELL,
                type=order_type,
                quantity=Decimal(str(alpaca_order.qty)),
                price=Decimal(str(alpaca_order.limit_price)) if alpaca_order.limit_price else None,
                stop_price=Decimal(str(alpaca_order.stop_price)) if alpaca_order.stop_price else None,
                status=self._parse_order_status(alpaca_order.status),
                filled_quantity=Decimal(str(alpaca_order.filled_qty or 0)),
                remaining_quantity=Decimal(str(float(alpaca_order.qty) - float(alpaca_order.filled_qty or 0))),
                created_at=alpaca_order.created_at,
                updated_at=alpaca_order.updated_at or alpaca_order.created_at
            )
        except Exception as e:
            self.logger.error(f"Error parsing order data: {e}")
            return None
