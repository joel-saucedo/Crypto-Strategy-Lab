"""
Comprehensive demo of the enhanced backtesting system with multiple strategies and assets.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_manager import DataManager
from backtesting.enhanced_backtester import EnhancedBacktester, BacktestConfig
from strategies.base_strategy import BaseStrategy
from visualization.backtest_analyzer import BacktestAnalyzer

# Example strategies for demonstration
class MovingAverageCrossover(BaseStrategy):
    """Simple moving average crossover strategy."""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signal(self, data: pd.DataFrame) -> float:
        """Generate trading signal based on MA crossover."""
        if len(data) < self.slow_period:
            return 0.0
        
        # Calculate moving averages
        fast_ma = data['close'].rolling(window=self.fast_period).mean()
        slow_ma = data['close'].rolling(window=self.slow_period).mean()
        
        # Current and previous values
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]
        
        # Crossover detection
        if pd.isna(current_fast) or pd.isna(current_slow):
            return 0.0
        
        # Bullish crossover (fast MA crosses above slow MA)
        if prev_fast <= prev_slow and current_fast > current_slow:
            return 1.0  # Strong buy signal
        
        # Bearish crossover (fast MA crosses below slow MA)
        if prev_fast >= prev_slow and current_fast < current_slow:
            return -1.0  # Strong sell signal
        
        return 0.0  # No signal

class RSIStrategy(BaseStrategy):
    """RSI-based mean reversion strategy."""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__()
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signal(self, data: pd.DataFrame) -> float:
        """Generate trading signal based on RSI levels."""
        if len(data) < self.rsi_period + 1:
            return 0.0
        
        rsi = self.calculate_rsi(data['close'])
        current_rsi = rsi.iloc[-1]
        
        if pd.isna(current_rsi):
            return 0.0
        
        # Generate signals based on RSI levels
        if current_rsi < self.oversold:
            return 0.8  # Buy signal (oversold)
        elif current_rsi > self.overbought:
            return -0.8  # Sell signal (overbought)
        
        return 0.0

class MomentumStrategy(BaseStrategy):
    """Price momentum strategy."""
    
    def __init__(self, lookback_period: int = 20, momentum_threshold: float = 0.02):
        super().__init__()
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
    
    def generate_signal(self, data: pd.DataFrame) -> float:
        """Generate signal based on price momentum."""
        if len(data) < self.lookback_period:
            return 0.0
        
        # Calculate momentum as percentage change
        current_price = data['close'].iloc[-1]
        past_price = data['close'].iloc[-self.lookback_period]
        
        momentum = (current_price - past_price) / past_price
        
        # Generate signals based on momentum strength
        if momentum > self.momentum_threshold:
            return min(momentum * 5, 1.0)  # Scale momentum, cap at 1.0
        elif momentum < -self.momentum_threshold:
            return max(momentum * 5, -1.0)  # Scale momentum, cap at -1.0
        
        return 0.0

async def run_enhanced_backtest_demo():
    """Run a comprehensive demonstration of the enhanced backtesting system."""
    
    print("🚀 Enhanced Backtesting System Demo")
    print("=" * 50)
    
    # Configuration
    config = BacktestConfig(
        start_date='2023-01-01',
        end_date='2023-12-31',
        initial_capital=100000,
        commission_rate=0.001,
        timeframe='1d',  # Daily data for demo
        benchmark_symbol='BTC-USD',
        enable_short_selling=True,
        max_position_size=0.15,  # 15% max per position
        max_total_exposure=0.8   # 80% max total exposure
    )
    
    # Initialize data manager
    data_manager = DataManager({
        'providers': {
            'yahoo': {},  # Use default Yahoo Finance settings
        },
        'provider_priority': ['yahoo'],
        'enable_cache': True
    })
    
    # Initialize backtester
    backtester = EnhancedBacktester(data_manager)
    
    # Add multiple strategies with different symbols
    print("\n📊 Adding strategies...")
    
    # Strategy 1: MA Crossover on major crypto pairs
    ma_strategy = MovingAverageCrossover(fast_period=10, slow_period=30)
    backtester.add_strategy(
        strategy=ma_strategy,
        symbols=['BTC-USD', 'ETH-USD'],
        strategy_id='MA_Crossover_Major'
    )
    
    # Strategy 2: RSI on smaller crypto pairs
    rsi_strategy = RSIStrategy(rsi_period=14, oversold=25, overbought=75)
    backtester.add_strategy(
        strategy=rsi_strategy,
        symbols=['ADA-USD', 'DOT-USD'],
        strategy_id='RSI_MeanReversion'
    )
    
    # Strategy 3: Momentum on mixed assets
    momentum_strategy = MomentumStrategy(lookback_period=20, momentum_threshold=0.03)
    backtester.add_strategy(
        strategy=momentum_strategy,
        symbols=['SOL-USD', 'MATIC-USD'],
        strategy_id='Momentum_Breakout'
    )
    
    print("✅ Added 3 strategies across 6 different assets")
    
    # Run backtest
    print(f"\n🔄 Running backtest from {config.start_date} to {config.end_date}...")
    
    try:
        result = await backtester.run_backtest(config)
        print("✅ Backtest completed successfully!")
        
        # Analyze results
        print(f"\n📈 Portfolio Performance:")
        print(f"Initial Capital: ${config.initial_capital:,.2f}")
        print(f"Final Value: ${result.metrics['final_portfolio_value']:,.2f}")
        print(f"Total Return: {result.metrics['total_return']:.2f}%")
        print(f"Total PnL: ${result.metrics['total_pnl']:,.2f}")
        
        print(f"\n📊 Trading Statistics:")
        print(f"Total Trades: {result.metrics['total_trades']}")
        print(f"Win Rate: {result.metrics['win_rate']:.1f}%")
        print(f"Profit Factor: {result.metrics['profit_factor']:.2f}")
        print(f"Max Drawdown: {result.metrics['max_drawdown']:.2f}%")
        
        # Strategy breakdown
        print(f"\n🎯 Strategy Performance:")
        strategy_breakdown = result.get_strategy_breakdown()
        for strategy_id, stats in strategy_breakdown.items():
            print(f"\n{strategy_id}:")
            print(f"  Trades: {stats['total_trades']}")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  Total PnL: ${stats['total_pnl']:,.2f}")
        
        # Symbol breakdown
        print(f"\n💰 Asset Performance:")
        symbol_breakdown = result.get_symbol_breakdown()
        for symbol, stats in symbol_breakdown.items():
            print(f"\n{symbol}:")
            print(f"  Trades: {stats['total_trades']}")
            print(f"  Total PnL: ${stats['total_pnl']:,.2f}")
            print(f"  Strategies Used: {stats['strategies_used']}")
        
        # Create analyzer for visualizations
        analyzer = BacktestAnalyzer(result)
        
        # Generate comprehensive report
        print(f"\n📋 Generating comprehensive report...")
        report = analyzer.generate_report()
        
        print(f"\nRisk Metrics:")
        risk_metrics = report.get('risk_metrics', {})
        print(f"  Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Volatility: {risk_metrics.get('volatility', 0):.2f}%")
        print(f"  Recovery Factor: {risk_metrics.get('recovery_factor', 0):.2f}")
        
        # Export results
        print(f"\n💾 Exporting results...")
        analyzer.export_results('backtest_results.json')
        print("✅ Results exported to backtest_results.json")
        
        # Generate visualizations (optional - requires display)
        print(f"\n📊 Generating visualizations...")
        try:
            # Portfolio performance plot
            portfolio_fig = analyzer.plot_portfolio_performance()
            portfolio_fig.write_html('portfolio_performance.html')
            
            # Strategy comparison plot
            strategy_fig = analyzer.plot_strategy_comparison()
            strategy_fig.write_html('strategy_comparison.html')
            
            # Trade analysis plot
            trade_fig = analyzer.plot_trade_analysis()
            trade_fig.write_html('trade_analysis.html')
            
            print("✅ Visualizations saved as HTML files")
            
        except Exception as e:
            print(f"⚠️  Visualization generation failed: {e}")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"Final portfolio value: ${result.metrics['final_portfolio_value']:,.2f}")
        print(f"Total return: {result.metrics['total_return']:.2f}%")
        
        return result
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the demo
    result = asyncio.run(run_enhanced_backtest_demo())
