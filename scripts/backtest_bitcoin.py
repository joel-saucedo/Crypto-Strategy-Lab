"""
Bitcoin backtesting script using pre-fetched data.
Tests multiple strategies on the historical Bitcoin data.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from backtesting.enhanced_backtester import EnhancedBacktester, BacktestConfig
    from backtesting.portfolio_manager import PortfolioManager
    from strategies.base_strategy import BaseStrategy
    from visualization.backtest_analyzer import BacktestAnalyzer
    from data.data_manager import DataManager
except ImportError:
    # Handle direct execution
    from src.backtesting.enhanced_backtester import EnhancedBacktester, BacktestConfig
    from src.backtesting.portfolio_manager import PortfolioManager
    from src.strategies.base_strategy import BaseStrategy
    from src.visualization.backtest_analyzer import BacktestAnalyzer
    from src.data.data_manager import DataManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MAStrategy(BaseStrategy):
    """Moving Average Crossover Strategy."""
    
    def __init__(self, short_period=20, long_period=50):
        super().__init__()
        self.short_period = short_period
        self.long_period = long_period
    
    def generate_signal(self, data: pd.DataFrame) -> float:
        """Generate signal based on MA crossover."""
        if len(data) < self.long_period:
            return 0.0
        
        short_ma = data['close'].rolling(window=self.short_period).mean()
        long_ma = data['close'].rolling(window=self.long_period).mean()
        
        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]
        
        if pd.isna(current_short) or pd.isna(current_long):
            return 0.0
        
        # Signal strength based on MA difference
        if current_short > current_long and prev_short <= prev_long:
            return 0.7  # Strong buy on crossover
        elif current_short < current_long and prev_short >= prev_long:
            return -0.7  # Strong sell on crossover
        elif current_short > current_long:
            return 0.3  # Weak buy when above
        elif current_short < current_long:
            return -0.3  # Weak sell when below
        
        return 0.0

class RSIStrategy(BaseStrategy):
    """RSI Mean Reversion Strategy."""
    
    def __init__(self, period=14, oversold=30, overbought=70):
        super().__init__()
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signal(self, data: pd.DataFrame) -> float:
        """Generate signal based on RSI levels."""
        if len(data) < self.period + 1:
            return 0.0
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        if pd.isna(current_rsi):
            return 0.0
        
        # Signal strength based on RSI levels
        if current_rsi < self.oversold:
            return min(0.8, (self.oversold - current_rsi) / 10)  # Buy signal
        elif current_rsi > self.overbought:
            return max(-0.8, -(current_rsi - self.overbought) / 10)  # Sell signal
        
        return 0.0

class MomentumStrategy(BaseStrategy):
    """Price Momentum Strategy."""
    
    def __init__(self, lookback=10, threshold=0.02):
        super().__init__()
        self.lookback = lookback
        self.threshold = threshold
    
    def generate_signal(self, data: pd.DataFrame) -> float:
        """Generate signal based on price momentum."""
        if len(data) < self.lookback + 1:
            return 0.0
        
        # Calculate momentum
        momentum = data['close'].iloc[-1] / data['close'].iloc[-self.lookback-1] - 1
        
        if pd.isna(momentum):
            return 0.0
        
        # Signal strength based on momentum magnitude
        if momentum > self.threshold:
            return min(0.6, momentum * 10)  # Buy on positive momentum
        elif momentum < -self.threshold:
            return max(-0.6, momentum * 10)  # Sell on negative momentum
        
        return 0.0

def load_bitcoin_data():
    """Load the pre-fetched Bitcoin data."""
    
    base_dir = Path(__file__).parent.parent
    processed_file = base_dir / 'data' / 'processed' / 'BTCUSD_daily_processed.parquet'
    raw_file = base_dir / 'data' / 'raw' / 'BTCUSD_daily_ohlcv.csv'
    
    if processed_file.exists():
        logger.info(f"📁 Loading processed data from: {processed_file}")
        data = pd.read_parquet(processed_file)
        
        # Ensure proper datetime index if not already set
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data.set_index('timestamp', inplace=True)
            elif 'date' in data.columns:
                data.set_index('date', inplace=True)
        
        logger.info(f"✅ Loaded {len(data)} days of processed Bitcoin data")
        logger.info(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
        logger.info(f"💰 Price range: ${data['low'].min():,.0f} - ${data['high'].max():,.0f}")
        
        return data
        
    elif raw_file.exists():
        logger.info(f"📁 Loading raw data from: {raw_file}")
        data = pd.read_csv(raw_file, index_col=0, parse_dates=True)
        logger.info(f"✅ Loaded {len(data)} days of raw Bitcoin data")
        return data
        
    else:
        raise FileNotFoundError(
            "No Bitcoin data found! Please run 'python scripts/fetch_btc_data.py' first"
        )

async def run_bitcoin_backtest():
    """Run backtesting on Bitcoin data with multiple strategies."""
    
    logger.info("🚀 Starting Bitcoin Backtesting")
    logger.info("=" * 50)
    
    try:
        # Load Bitcoin data
        data = load_bitcoin_data()
        
        # Create a simple data manager for backtesting
        data_manager = DataManager()
        
        # Initialize backtester
        backtester = EnhancedBacktester(data_manager)
        
        # Add strategies
        strategies = [
            (MAStrategy(20, 50), "MA_20_50"),
            (RSIStrategy(14, 30, 70), "RSI_14"),
            (MomentumStrategy(10, 0.02), "Momentum_10")
        ]
        
        for strategy, strategy_id in strategies:
            backtester.add_strategy(
                strategy=strategy,
                symbols=['BTCUSD'],
                strategy_id=strategy_id
            )
            logger.info(f"✅ Added strategy: {strategy_id}")
        
        # Set up backtest configuration
        start_date = data.index[0] + timedelta(days=365)  # Skip first year for warmup
        end_date = data.index[-1] - timedelta(days=30)    # Keep some recent data for validation
        
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000,  # $100k starting capital
            commission_rate=0.001,   # 0.1% commission
            interval='1d'
        )
        
        logger.info(f"📅 Backtest period: {start_date.date()} to {end_date.date()}")
        logger.info(f"💰 Initial capital: ${config.initial_capital:,}")
        logger.info(f"💸 Commission rate: {config.commission_rate:.1%}")
        
        # Override the backtester's data fetching with our pre-loaded data
        backtester._data_cache = {'BTCUSD': data}
        
        # Run backtest
        logger.info("\n🔄 Running backtest...")
        result = await backtester.run_backtest(config)
        
        # Display results
        logger.info("\n📊 Backtest Results")
        logger.info("=" * 30)
        
        metrics = result.metrics
        logger.info(f"Initial Capital:     ${config.initial_capital:,}")
        logger.info(f"Final Portfolio:     ${metrics['final_portfolio_value']:,.0f}")
        logger.info(f"Total Return:        {metrics['total_return']:.2f}%")
        logger.info(f"Annualized Return:   {metrics.get('annualized_return', 0):.2f}%")
        logger.info(f"Max Drawdown:        {metrics['max_drawdown']:.2f}%")
        logger.info(f"Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"Total Trades:        {metrics['total_trades']}")
        
        if metrics['total_trades'] > 0:
            logger.info(f"Win Rate:            {metrics['win_rate']:.1f}%")
            logger.info(f"Avg Trade Return:    {metrics.get('avg_trade_return', 0):.2f}%")
        
        # Strategy breakdown
        if hasattr(result, 'strategy_performance'):
            logger.info("\n📈 Strategy Performance")
            logger.info("-" * 25)
            for strategy_id, perf in result.strategy_performance.items():
                logger.info(f"{strategy_id:15} | Return: {perf.get('return', 0):.2f}% | Trades: {perf.get('trades', 0)}")
        
        # Save results
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"bitcoin_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Convert portfolio history to DataFrame and save
        if hasattr(result, 'portfolio_history') and result.portfolio_history:
            portfolio_df = pd.DataFrame(result.portfolio_history)
            portfolio_df.to_csv(results_file, index=False)
            logger.info(f"💾 Results saved to: {results_file}")
        
        logger.info("\n🎉 Bitcoin backtesting completed successfully!")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    logger.info("🚀 Starting Bitcoin Backtesting Script")
    result = asyncio.run(run_bitcoin_backtest())
    
    if result:
        logger.info("✅ Backtesting completed successfully!")
    else:
        logger.error("❌ Backtesting failed!")
