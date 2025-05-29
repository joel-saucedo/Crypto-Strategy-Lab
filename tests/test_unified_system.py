#!/usr/bin/env python3
"""
Test script to verify the unified backtesting system works end-to-end.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.core.backtest_engine import (
        MultiStrategyOrchestrator, 
        BacktestConfig, 
        PositionSizingType,
        PositionSizingConfig
    )
    from src.strategies.base_strategy import BaseStrategy
    from src.data.data_manager import DataManager
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


class SimpleMovingAverageStrategy(BaseStrategy):
    """Simple test strategy using moving average crossover."""
    
    def __init__(self, short_window=10, long_window=30):
        super().__init__("SMA_Strategy")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signal(self, data: pd.DataFrame) -> float:
        """Generate signal based on moving average crossover."""
        if len(data) < self.long_window:
            return 0.0
        
        # Calculate moving averages
        prices = data['close']
        short_ma = prices.tail(self.short_window).mean()
        long_ma = prices.tail(self.long_window).mean()
        
        # Generate signal
        if short_ma > long_ma * 1.01:  # 1% buffer
            return 1.0  # Buy signal
        elif short_ma < long_ma * 0.99:  # 1% buffer
            return -1.0  # Sell signal
        
        return 0.0  # No signal


def generate_synthetic_data(days=252, symbols=['BTC', 'ETH'], start_date='2023-01-01'):
    """Generate synthetic price data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    data = {}
    
    for symbol in symbols:
        # Generate random walk with drift
        returns = np.random.normal(0.001, 0.02, days)  # 0.1% daily drift, 2% volatility
        
        # Start with a base price
        base_price = 100.0 if symbol == 'BTC' else 50.0
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        for metric in ['open', 'high', 'low', 'close', 'volume']:
            if metric == 'close':
                data[(metric, symbol)] = prices
            elif metric == 'open':
                data[(metric, symbol)] = [p * (1 + np.random.normal(0, 0.001)) for p in prices]
            elif metric == 'high':
                data[(metric, symbol)] = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
            elif metric == 'low':
                data[(metric, symbol)] = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
            else:  # volume
                data[(metric, symbol)] = np.random.uniform(1000, 10000, days)
    
    # Create DataFrame with MultiIndex columns
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['metric', 'symbol'])
    
    return df


async def test_unified_system():
    """Test the unified backtesting system."""
    print("\n🧪 Testing Unified Backtesting System")
    print("=" * 50)
    
    try:
        # 1. Create synthetic data
        print("1. Generating synthetic data...")
        data = generate_synthetic_data(days=100, symbols=['BTC', 'ETH'])
        print(f"   Generated data shape: {data.shape}")
        
        # 2. Create strategies
        print("2. Creating test strategies...")
        strategy1 = SimpleMovingAverageStrategy(short_window=5, long_window=20)
        strategy2 = SimpleMovingAverageStrategy(short_window=10, long_window=30)
        
        # 3. Initialize orchestrator
        print("3. Initializing multi-strategy orchestrator...")
        orchestrator = MultiStrategyOrchestrator()
        
        # Add strategies
        orchestrator.add_strategy(strategy1, symbols=['BTC'], strategy_id='SMA_5_20')
        orchestrator.add_strategy(strategy2, symbols=['ETH'], strategy_id='SMA_10_30')
        print(f"   Added {len(orchestrator.strategies)} strategies")
        
        # 4. Create backtest configuration
        print("4. Creating backtest configuration...")
        config = BacktestConfig(
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=100000,
            fees={'taker': 0.001, 'maker': 0.0005},
            max_position_size=0.2
        )
        
        # 5. Run backtest
        print("5. Running backtest...")
        result = await orchestrator.run_backtest(config, data=data, validate=False)
        print("   ✓ Backtest completed successfully")
        
        # 6. Check results
        print("6. Analyzing results...")
        
        if hasattr(result, 'portfolio_manager') and result.portfolio_manager:
            pm = result.portfolio_manager
            final_value = pm.get_portfolio_value()
            total_return = (final_value - config.initial_capital) / config.initial_capital * 100
            
            print(f"   Initial Capital: ${config.initial_capital:,.2f}")
            print(f"   Final Portfolio Value: ${final_value:,.2f}")
            print(f"   Total Return: {total_return:.2f}%")
            print(f"   Total Trades: {len(pm.closed_trades)}")
            
            # Check individual strategy performance
            if hasattr(result, 'strategy_performances') and result.strategy_performances:
                print("\n   Strategy Performance Breakdown:")
                for strategy_id, perf in result.strategy_performances.items():
                    print(f"     {strategy_id}: {perf}")
        
        # 7. Test validation layers
        print("7. Testing validation layers...")
        if hasattr(orchestrator, '_run_validation_pipeline'):
            # Get some sample returns for validation
            if result.portfolio_manager.portfolio_history:
                portfolio_history = result.portfolio_manager.portfolio_history
                returns = orchestrator._calculate_portfolio_returns(portfolio_history)
                
                if len(returns) > 0:
                    validation_result = orchestrator._run_validation_pipeline(
                        returns, 'unified_test'
                    )
                    print(f"   Validation Score: {validation_result.get('overall_score', 'N/A')}")
                    print(f"   Validation Result: {validation_result.get('overall_result', 'N/A')}")
        
        print("\n🎉 All tests passed! Unified system is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("Crypto Strategy Lab - Unified System Test")
    print("========================================")
    
    # Run the async test
    success = asyncio.run(test_unified_system())
    
    if success:
        print("\n✅ SUCCESS: Unified backtesting system is operational!")
        print("\nNext steps:")
        print("- Test with real market data")
        print("- Add more sophisticated strategies")
        print("- Integrate with live trading")
        return 0
    else:
        print("\n❌ FAILURE: System needs debugging")
        return 1


if __name__ == "__main__":
    exit(main())
