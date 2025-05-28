"""
Multi-Exchange Integration Tests

This module provides comprehensive tests for the multi-exchange integration,
testing both Bybit and Alpaca adapters with mock and real connections.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from decimal import Decimal
import os

from src.exchanges import (
    ExchangeFactory, BaseExchange, TradingMode,
    OrderRequest, OrderType, OrderSide, Balance
)


class TestExchangeFactory:
    """Test the exchange factory functionality"""
    
    def test_get_available_exchanges(self):
        """Test getting available exchanges"""
        exchanges = ExchangeFactory.get_available_exchanges()
        assert isinstance(exchanges, list)
        # Should have at least bybit and alpaca if imports work
    
    def test_register_exchange(self):
        """Test registering a new exchange"""
        class MockExchange(BaseExchange):
            @property
            def exchange_name(self):
                return "mock"
            
            @property
            def supported_symbols(self):
                return ["TEST/USD"]
            
            def _initialize_client(self, **kwargs):
                pass
            
            async def connect(self):
                return True
            
            async def disconnect(self):
                pass
            
            async def test_connection(self):
                return True
            
            async def get_account_info(self):
                return {}
            
            async def get_balances(self):
                return []
            
            async def get_balance(self, asset):
                return None
            
            async def get_ticker(self, symbol):
                return None
            
            async def get_orderbook(self, symbol, limit=100):
                return {'bids': [], 'asks': []}
            
            async def get_candles(self, symbol, interval, start_time=None, end_time=None, limit=500):
                return []
            
            async def place_order(self, order_request):
                return None
            
            async def cancel_order(self, order_id, symbol):
                return False
            
            async def get_order(self, order_id, symbol):
                return None
            
            async def get_open_orders(self, symbol=None):
                return []
            
            async def get_order_history(self, symbol=None, start_time=None, end_time=None, limit=500):
                return []
            
            async def get_trade_history(self, symbol=None, start_time=None, end_time=None, limit=500):
                return []
        
        ExchangeFactory.register_exchange('mock', MockExchange)
        assert 'mock' in ExchangeFactory.get_available_exchanges()
    
    def test_create_exchange_invalid(self):
        """Test creating exchange with invalid name"""
        with pytest.raises(ValueError, match="Exchange 'invalid' not registered"):
            ExchangeFactory.create_exchange('invalid', 'key', 'secret')
    
    def test_create_exchange_missing_credentials(self):
        """Test creating exchange without credentials"""
        # Clear environment variables
        env_backup = {}
        for key in os.environ:
            if 'BYBIT' in key or 'ALPACA' in key:
                env_backup[key] = os.environ[key]
                del os.environ[key]
        
        try:
            with pytest.raises(ValueError, match="Missing API credentials"):
                ExchangeFactory.create_exchange('bybit')
        finally:
            # Restore environment
            for key, value in env_backup.items():
                os.environ[key] = value


class TestBybitAdapter:
    """Test Bybit adapter functionality"""
    
    @pytest.fixture
    def mock_bybit_session(self):
        """Mock Bybit HTTP session"""
        with patch('src.exchanges.bybit_adapter.HTTP') as mock_http:
            mock_session = Mock()
            mock_http.return_value = mock_session
            yield mock_session
    
    @pytest.fixture
    def bybit_adapter(self, mock_bybit_session):
        """Create Bybit adapter with mocked session"""
        from src.exchanges.bybit_adapter import BybitAdapter
        return BybitAdapter('test_key', 'test_secret', TradingMode.TESTNET)
    
    def test_initialize_without_pybit(self):
        """Test initialization without pybit library"""
        with patch('src.exchanges.bybit_adapter.PYBIT_AVAILABLE', False):
            from src.exchanges.bybit_adapter import BybitAdapter
            with pytest.raises(ImportError, match="pybit library is required"):
                BybitAdapter('key', 'secret')
    
    def test_exchange_name(self, bybit_adapter):
        """Test exchange name property"""
        assert bybit_adapter.exchange_name == "bybit"
    
    def test_normalize_symbol(self, bybit_adapter):
        """Test symbol normalization"""
        # Setup symbol mapping
        bybit_adapter._symbol_map = {'BTC/USD': 'BTCUSD'}
        
        assert bybit_adapter.normalize_symbol('BTC/USD') == 'BTCUSD'
        assert bybit_adapter.normalize_symbol('UNKNOWN') == 'UNKNOWN'
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self, bybit_adapter, mock_bybit_session):
        """Test successful connection test"""
        mock_bybit_session.get_wallet_balance.return_value = {'retCode': 0}
        
        result = await bybit_adapter.test_connection()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_test_connection_failure(self, bybit_adapter, mock_bybit_session):
        """Test failed connection test"""
        mock_bybit_session.get_wallet_balance.side_effect = Exception("Connection error")
        
        result = await bybit_adapter.test_connection()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_balances(self, bybit_adapter, mock_bybit_session):
        """Test getting account balances"""
        mock_response = {
            'retCode': 0,
            'result': {
                'list': [{
                    'coin': [{
                        'coin': 'BTC',
                        'availableToWithdraw': '1.5',
                        'locked': '0.1',
                        'walletBalance': '1.6'
                    }]
                }]
            }
        }
        mock_bybit_session.get_wallet_balance.return_value = mock_response
        
        balances = await bybit_adapter.get_balances()
        
        assert len(balances) == 1
        assert balances[0].asset == 'BTC'
        assert balances[0].free == Decimal('1.5')
        assert balances[0].locked == Decimal('0.1')
        assert balances[0].total == Decimal('1.6')


class TestAlpacaAdapter:
    """Test Alpaca adapter functionality"""
    
    @pytest.fixture
    def mock_alpaca_api(self):
        """Mock Alpaca API"""
        with patch('src.exchanges.alpaca_adapter.tradeapi.REST') as mock_rest:
            mock_api = Mock()
            mock_rest.return_value = mock_api
            yield mock_api
    
    @pytest.fixture
    def alpaca_adapter(self, mock_alpaca_api):
        """Create Alpaca adapter with mocked API"""
        from src.exchanges.alpaca_adapter import AlpacaAdapter
        return AlpacaAdapter('test_key', 'test_secret', TradingMode.PAPER)
    
    def test_initialize_without_alpaca(self):
        """Test initialization without alpaca-trade-api library"""
        with patch('src.exchanges.alpaca_adapter.ALPACA_AVAILABLE', False):
            from src.exchanges.alpaca_adapter import AlpacaAdapter
            with pytest.raises(ImportError, match="alpaca-trade-api library is required"):
                AlpacaAdapter('key', 'secret')
    
    def test_exchange_name(self, alpaca_adapter):
        """Test exchange name property"""
        assert alpaca_adapter.exchange_name == "alpaca"
    
    def test_normalize_symbol(self, alpaca_adapter):
        """Test symbol normalization"""
        assert alpaca_adapter.normalize_symbol('BTC/USD') == 'BTCUSD'
        assert alpaca_adapter.normalize_symbol('AAPL') == 'AAPL'
    
    def test_standardize_symbol(self, alpaca_adapter):
        """Test symbol standardization"""
        assert alpaca_adapter._standardize_symbol('BTCUSD') == 'BTC/USD'
        assert alpaca_adapter._standardize_symbol('AAPL') == 'AAPL'
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self, alpaca_adapter, mock_alpaca_api):
        """Test successful connection test"""
        mock_account = Mock()
        mock_alpaca_api.get_account.return_value = mock_account
        
        result = await alpaca_adapter.test_connection()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_test_connection_failure(self, alpaca_adapter, mock_alpaca_api):
        """Test failed connection test"""
        mock_alpaca_api.get_account.side_effect = Exception("Connection error")
        
        result = await alpaca_adapter.test_connection()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_account_info(self, alpaca_adapter, mock_alpaca_api):
        """Test getting account information"""
        mock_account = Mock()
        mock_account.id = 'test-account'
        mock_account.status = 'ACTIVE'
        mock_account.currency = 'USD'
        mock_account.buying_power = '10000.00'
        mock_account.cash = '5000.00'
        mock_account.portfolio_value = '15000.00'
        mock_account.pattern_day_trader = False
        mock_account.trade_suspended_by_user = False
        mock_account.trading_blocked = False
        mock_account.transfers_blocked = False
        mock_account.account_blocked = False
        mock_account.created_at = datetime.now()
        
        mock_alpaca_api.get_account.return_value = mock_account
        
        account_info = await alpaca_adapter.get_account_info()
        
        assert account_info['account_id'] == 'test-account'
        assert account_info['buying_power'] == 10000.0
        assert account_info['paper_trading'] is True


class TestMultiExchangeIntegration:
    """Integration tests for multi-exchange functionality"""
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using exchange as async context manager"""
        # Mock the exchange to avoid real API calls
        mock_exchange = AsyncMock(spec=BaseExchange)
        mock_exchange.connect.return_value = True
        
        async with mock_exchange as exchange:
            assert exchange is mock_exchange
            mock_exchange.connect.assert_called_once()
        
        mock_exchange.disconnect.assert_called_once()
    
    def test_order_request_validation(self):
        """Test order request validation"""
        # Valid order request
        order = OrderRequest(
            symbol='BTC/USD',
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=Decimal('0.1'),
            price=Decimal('50000')
        )
        
        assert order.symbol == 'BTC/USD'
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.LIMIT
        assert order.quantity == Decimal('0.1')
        assert order.price == Decimal('50000')
    
    def test_balance_model(self):
        """Test balance data model"""
        balance = Balance(
            asset='BTC',
            free=Decimal('1.5'),
            locked=Decimal('0.1'),
            total=Decimal('1.6')
        )
        
        assert balance.asset == 'BTC'
        assert balance.free == Decimal('1.5')
        assert balance.locked == Decimal('0.1')
        assert balance.total == Decimal('1.6')


@pytest.mark.integration
class TestRealExchangeConnections:
    """
    Integration tests with real exchange connections
    These tests require valid API credentials and should be run separately
    """
    
    @pytest.mark.skipif(
        not os.getenv('BYBIT_TESTNET_API_KEY'),
        reason="Bybit testnet credentials not available"
    )
    @pytest.mark.asyncio
    async def test_bybit_testnet_connection(self):
        """Test real connection to Bybit testnet"""
        try:
            exchange = ExchangeFactory.create_exchange(
                'bybit',
                mode=TradingMode.TESTNET
            )
            
            async with exchange:
                # Test basic functionality
                account_info = await exchange.get_account_info()
                assert isinstance(account_info, dict)
                
                balances = await exchange.get_balances()
                assert isinstance(balances, list)
                
                # Test market data
                ticker = await exchange.get_ticker('BTC/USDT')
                if ticker:
                    assert ticker.symbol is not None
                    assert ticker.bid > 0
                    assert ticker.ask > 0
        
        except ImportError:
            pytest.skip("Bybit dependencies not available")
    
    @pytest.mark.skipif(
        not os.getenv('ALPACA_PAPER_API_KEY'),
        reason="Alpaca paper trading credentials not available"
    )
    @pytest.mark.asyncio
    async def test_alpaca_paper_connection(self):
        """Test real connection to Alpaca paper trading"""
        try:
            exchange = ExchangeFactory.create_exchange(
                'alpaca',
                mode=TradingMode.PAPER
            )
            
            async with exchange:
                # Test basic functionality
                account_info = await exchange.get_account_info()
                assert isinstance(account_info, dict)
                assert account_info.get('paper_trading') is True
                
                balances = await exchange.get_balances()
                assert isinstance(balances, list)
                
                # Test market data
                ticker = await exchange.get_ticker('AAPL')
                if ticker:
                    assert ticker.symbol is not None
                    assert ticker.bid > 0
                    assert ticker.ask > 0
        
        except ImportError:
            pytest.skip("Alpaca dependencies not available")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
