"""
Bybit Exchange Adapter

This module provides integration with Bybit cryptocurrency exchange using the pybit library.
Supports both testnet and live trading environments.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import logging

try:
    from pybit.unified_trading import HTTP
    PYBIT_AVAILABLE = True
except ImportError:
    PYBIT_AVAILABLE = False
    HTTP = None

from .base_exchange import (
    BaseExchange, Balance, OrderRequest, Order, Trade, Ticker, Candle,
    OrderType, OrderSide, OrderStatus, TradingMode
)


class BybitAdapter(BaseExchange):
    """
    Bybit exchange adapter using pybit library
    
    Supports unified trading API for spot, linear, and inverse contracts.
    Provides both testnet and live trading capabilities.
    """
    
    def __init__(self, 
                 api_key: str, 
                 api_secret: str, 
                 mode: TradingMode = TradingMode.TESTNET,
                 **kwargs):
        """
        Initialize Bybit adapter
        
        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret
            mode: Trading mode (live, testnet)
            **kwargs: Additional parameters
        """
        if not PYBIT_AVAILABLE:
            raise ImportError("pybit library is required for Bybit integration. Install with: pip install pybit")
        
        self._testnet = mode == TradingMode.TESTNET
        self._session = None
        
        # Symbol mappings for Bybit
        self._symbol_map = {}
        self._reverse_symbol_map = {}
        
        super().__init__(api_key, api_secret, mode, **kwargs)
    
    def _initialize_client(self, **kwargs) -> None:
        """Initialize Bybit HTTP client"""
        try:
            self._session = HTTP(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self._testnet,
                **kwargs
            )
            self.logger.info(f"Initialized Bybit client ({'testnet' if self._testnet else 'live'})")
        except Exception as e:
            self.logger.error(f"Failed to initialize Bybit client: {e}")
            raise
    
    @property
    def exchange_name(self) -> str:
        """Return exchange name"""
        return "bybit"
    
    @property
    def supported_symbols(self) -> List[str]:
        """Get supported trading symbols"""
        # This will be populated dynamically
        return list(self._symbol_map.keys())
    
    async def connect(self) -> bool:
        """Establish connection and load market info"""
        try:
            # Test connection
            if not await self.test_connection():
                return False
            
            # Load symbol information
            await self._load_symbol_info()
            
            self.logger.info("Successfully connected to Bybit")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Bybit: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close connection"""
        self._session = None
        self.logger.info("Disconnected from Bybit")
    
    async def test_connection(self) -> bool:
        """Test connection and authentication"""
        try:
            # Test with account info request
            response = self._session.get_wallet_balance(accountType="UNIFIED")
            return response.get('retCode') == 0
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    async def _load_symbol_info(self) -> None:
        """Load symbol information and create mappings"""
        try:
            # Get spot symbols
            response = self._session.get_instruments_info(category="spot")
            if response.get('retCode') == 0:
                for symbol_info in response.get('result', {}).get('list', []):
                    symbol = symbol_info['symbol']
                    base_coin = symbol_info['baseCoin']
                    quote_coin = symbol_info['quoteCoin']
                    
                    # Map to standard format
                    standard_symbol = f"{base_coin}/{quote_coin}"
                    self._symbol_map[standard_symbol] = symbol
                    self._reverse_symbol_map[symbol] = standard_symbol
            
            self.logger.info(f"Loaded {len(self._symbol_map)} symbols")
        except Exception as e:
            self.logger.error(f"Failed to load symbol info: {e}")
    
    def normalize_symbol(self, symbol: str) -> str:
        """Convert standard symbol format to Bybit format"""
        if symbol in self._symbol_map:
            return self._symbol_map[symbol]
        
        # Try direct conversion
        if '/' in symbol:
            base, quote = symbol.split('/')
            bybit_symbol = f"{base}{quote}"
            if bybit_symbol.upper() in self._reverse_symbol_map:
                return bybit_symbol.upper()
        
        return symbol.upper()
    
    def _standardize_symbol(self, bybit_symbol: str) -> str:
        """Convert Bybit symbol to standard format"""
        if bybit_symbol in self._reverse_symbol_map:
            return self._reverse_symbol_map[bybit_symbol]
        return bybit_symbol
    
    # Account Management
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            response = self._session.get_wallet_balance(accountType="UNIFIED")
            if response.get('retCode') == 0:
                return response.get('result', {})
            else:
                self.logger.error(f"Failed to get account info: {response.get('retMsg')}")
                return {}
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    async def get_balances(self) -> List[Balance]:
        """Get account balances"""
        try:
            response = self._session.get_wallet_balance(accountType="UNIFIED")
            balances = []
            
            if response.get('retCode') == 0:
                coins = response.get('result', {}).get('list', [{}])[0].get('coin', [])
                for coin_info in coins:
                    balance = Balance(
                        asset=coin_info['coin'],
                        free=Decimal(coin_info['availableToWithdraw']),
                        locked=Decimal(coin_info['locked']),
                        total=Decimal(coin_info['walletBalance'])
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
            bybit_symbol = self.normalize_symbol(symbol)
            response = self._session.get_tickers(category="spot", symbol=bybit_symbol)
            
            if response.get('retCode') == 0:
                ticker_data = response.get('result', {}).get('list', [])
                if ticker_data:
                    data = ticker_data[0]
                    return Ticker(
                        symbol=self._standardize_symbol(data['symbol']),
                        bid=Decimal(data['bid1Price']),
                        ask=Decimal(data['ask1Price']),
                        last=Decimal(data['lastPrice']),
                        volume=Decimal(data['volume24h']),
                        high_24h=Decimal(data['highPrice24h']),
                        low_24h=Decimal(data['lowPrice24h']),
                        change_24h=Decimal(data['price24hPcnt']),
                        timestamp=datetime.now()
                    )
            return None
        except Exception as e:
            self.logger.error(f"Error getting ticker for {symbol}: {e}")
            return None
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, List[Tuple[Decimal, Decimal]]]:
        """Get order book"""
        try:
            bybit_symbol = self.normalize_symbol(symbol)
            response = self._session.get_orderbook(category="spot", symbol=bybit_symbol, limit=limit)
            
            if response.get('retCode') == 0:
                data = response.get('result', {})
                bids = [(Decimal(price), Decimal(size)) for price, size in data.get('b', [])]
                asks = [(Decimal(price), Decimal(size)) for price, size in data.get('a', [])]
                return {'bids': bids, 'asks': asks}
            
            return {'bids': [], 'asks': []}
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
            bybit_symbol = self.normalize_symbol(symbol)
            
            # Convert interval to Bybit format
            interval_map = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720,
                '1d': 'D', '1w': 'W', '1M': 'M'
            }
            bybit_interval = interval_map.get(interval, interval)
            
            # Prepare time parameters
            params = {
                'category': 'spot',
                'symbol': bybit_symbol,
                'interval': str(bybit_interval),
                'limit': limit
            }
            
            if start_time:
                params['start'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['end'] = int(end_time.timestamp() * 1000)
            
            response = self._session.get_kline(**params)
            candles = []
            
            if response.get('retCode') == 0:
                klines = response.get('result', {}).get('list', [])
                for kline in klines:
                    candle = Candle(
                        timestamp=datetime.fromtimestamp(int(kline[0]) / 1000),
                        open=Decimal(kline[1]),
                        high=Decimal(kline[2]),
                        low=Decimal(kline[3]),
                        close=Decimal(kline[4]),
                        volume=Decimal(kline[5])
                    )
                    candles.append(candle)
            
            return sorted(candles, key=lambda x: x.timestamp)
        except Exception as e:
            self.logger.error(f"Error getting candles for {symbol}: {e}")
            return []
    
    # Order Management
    def _map_order_type(self, order_type: OrderType) -> str:
        """Map order type to Bybit format"""
        mapping = {
            OrderType.MARKET: "Market",
            OrderType.LIMIT: "Limit",
            OrderType.STOP_LOSS: "StopLoss",
            OrderType.TAKE_PROFIT: "TakeProfit"
        }
        return mapping.get(order_type, "Limit")
    
    def _map_order_side(self, side: OrderSide) -> str:
        """Map order side to Bybit format"""
        return "Buy" if side == OrderSide.BUY else "Sell"
    
    def _parse_order_status(self, status: str) -> OrderStatus:
        """Parse Bybit order status to standard format"""
        status_map = {
            'New': OrderStatus.OPEN,
            'PartiallyFilled': OrderStatus.PARTIALLY_FILLED,
            'Filled': OrderStatus.FILLED,
            'Cancelled': OrderStatus.CANCELED,
            'Rejected': OrderStatus.REJECTED,
            'PartiallyFilledCanceled': OrderStatus.CANCELED
        }
        return status_map.get(status, OrderStatus.PENDING)
    
    async def place_order(self, order_request: OrderRequest) -> Optional[Order]:
        """Place a new order"""
        try:
            bybit_symbol = self.normalize_symbol(order_request.symbol)
            
            params = {
                'category': 'spot',
                'symbol': bybit_symbol,
                'side': self._map_order_side(order_request.side),
                'orderType': self._map_order_type(order_request.type),
                'qty': str(order_request.quantity),
                'timeInForce': order_request.time_in_force
            }
            
            if order_request.price and order_request.type == OrderType.LIMIT:
                params['price'] = str(order_request.price)
            
            if order_request.client_order_id:
                params['orderLinkId'] = order_request.client_order_id
            
            response = self._session.place_order(**params)
            
            if response.get('retCode') == 0:
                result = response.get('result', {})
                return Order(
                    id=result['orderId'],
                    client_order_id=result.get('orderLinkId'),
                    symbol=self._standardize_symbol(bybit_symbol),
                    side=order_request.side,
                    type=order_request.type,
                    quantity=order_request.quantity,
                    price=order_request.price,
                    stop_price=order_request.stop_price,
                    status=OrderStatus.PENDING,
                    filled_quantity=Decimal('0'),
                    remaining_quantity=order_request.quantity,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
            else:
                self.logger.error(f"Failed to place order: {response.get('retMsg')}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            bybit_symbol = self.normalize_symbol(symbol)
            response = self._session.cancel_order(
                category="spot",
                symbol=bybit_symbol,
                orderId=order_id
            )
            return response.get('retCode') == 0
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {e}")
            return False
    
    async def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get order details"""
        try:
            bybit_symbol = self.normalize_symbol(symbol)
            response = self._session.get_open_orders(
                category="spot",
                symbol=bybit_symbol,
                orderId=order_id
            )
            
            if response.get('retCode') == 0:
                orders = response.get('result', {}).get('list', [])
                if orders:
                    return self._parse_order(orders[0])
            return None
        except Exception as e:
            self.logger.error(f"Error getting order {order_id}: {e}")
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders"""
        try:
            params = {'category': 'spot'}
            if symbol:
                params['symbol'] = self.normalize_symbol(symbol)
            
            response = self._session.get_open_orders(**params)
            orders = []
            
            if response.get('retCode') == 0:
                for order_data in response.get('result', {}).get('list', []):
                    order = self._parse_order(order_data)
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
                'category': 'spot',
                'limit': limit
            }
            
            if symbol:
                params['symbol'] = self.normalize_symbol(symbol)
            if start_time:
                params['startTime'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['endTime'] = int(end_time.timestamp() * 1000)
            
            response = self._session.get_order_history(**params)
            orders = []
            
            if response.get('retCode') == 0:
                for order_data in response.get('result', {}).get('list', []):
                    order = self._parse_order(order_data)
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
                'category': 'spot',
                'limit': limit
            }
            
            if symbol:
                params['symbol'] = self.normalize_symbol(symbol)
            if start_time:
                params['startTime'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['endTime'] = int(end_time.timestamp() * 1000)
            
            response = self._session.get_executions(**params)
            trades = []
            
            if response.get('retCode') == 0:
                for trade_data in response.get('result', {}).get('list', []):
                    trade = Trade(
                        id=trade_data['execId'],
                        order_id=trade_data['orderId'],
                        symbol=self._standardize_symbol(trade_data['symbol']),
                        side=OrderSide.BUY if trade_data['side'] == 'Buy' else OrderSide.SELL,
                        quantity=Decimal(trade_data['execQty']),
                        price=Decimal(trade_data['execPrice']),
                        fee=Decimal(trade_data['execFee']),
                        fee_asset=trade_data['feeTokenId'],
                        timestamp=datetime.fromtimestamp(int(trade_data['execTime']) / 1000)
                    )
                    trades.append(trade)
            
            return trades
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []
    
    def _parse_order(self, order_data: Dict[str, Any]) -> Optional[Order]:
        """Parse Bybit order data to standard Order object"""
        try:
            return Order(
                id=order_data['orderId'],
                client_order_id=order_data.get('orderLinkId'),
                symbol=self._standardize_symbol(order_data['symbol']),
                side=OrderSide.BUY if order_data['side'] == 'Buy' else OrderSide.SELL,
                type=OrderType.LIMIT if order_data['orderType'] == 'Limit' else OrderType.MARKET,
                quantity=Decimal(order_data['qty']),
                price=Decimal(order_data['price']) if order_data.get('price') else None,
                stop_price=Decimal(order_data['stopPrice']) if order_data.get('stopPrice') else None,
                status=self._parse_order_status(order_data['orderStatus']),
                filled_quantity=Decimal(order_data['cumExecQty']),
                remaining_quantity=Decimal(order_data['leavesQty']),
                created_at=datetime.fromtimestamp(int(order_data['createdTime']) / 1000),
                updated_at=datetime.fromtimestamp(int(order_data['updatedTime']) / 1000)
            )
        except Exception as e:
            self.logger.error(f"Error parsing order data: {e}")
            return None
