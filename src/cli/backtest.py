"""
CLI module for backtesting functionality using the unified engine.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
import json

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backtest_engine import (
    UnifiedBacktestEngine, 
    BacktestConfig, 
    PositionSizingConfig,
    PositionSizingType
)
from data.data_manager import DataManager
from strategies.base_strategy import BaseStrategy

def run_backtest(args):
    """
    Run a comprehensive backtest using the unified engine.
    """
    try:
        print("=" * 60)
        print("🚀 CRYPTO STRATEGY LAB - UNIFIED BACKTESTING ENGINE")
        print("=" * 60)
        
        # Parse dates
        if args.start:
            start_date = datetime.strptime(args.start, '%Y-%m-%d')
        else:
            start_date = datetime.now() - timedelta(days=365)
            
        if args.end:
            end_date = datetime.strptime(args.end, '%Y-%m-%d')
        else:
            end_date = datetime.now()
        
        # Configure backtesting
        config = BacktestConfig(
            initial_capital=getattr(args, 'capital', 100000),
            start_date=start_date,
            end_date=end_date,
            max_position_size=getattr(args, 'position_size', 0.1),
            max_total_exposure=getattr(args, 'exposure', 0.8),
            monte_carlo_trials=getattr(args, 'mc_trials', 10000),
            min_dsr=getattr(args, 'min_dsr', 0.95),
            min_psr=getattr(args, 'min_psr', 0.95)
        )
        
        print(f"📊 Configuration:")
        print(f"   Strategy: {args.strategy}")
        print(f"   Symbol: {args.symbol}")
        print(f"   Period: {start_date.date()} to {end_date.date()}")
        print(f"   Capital: ${config.initial_capital:,.2f}")
        print(f"   Position Size: {config.max_position_size:.1%}")
        print(f"   Monte Carlo Trials: {config.monte_carlo_trials:,}")
        print()
        
        # Initialize unified engine
        engine = UnifiedBacktestEngine(config)
        
        # Load strategy
        print(f"📈 Loading strategy: {args.strategy}")
        strategy_instance = load_strategy(args.strategy)
        
        # Add strategy to engine
        strategy_id = engine.add_strategy(
            strategy=strategy_instance,
            symbols=[args.symbol],
            strategy_id=args.strategy
        )
        
        # Load market data
        print(f"📉 Loading market data for {args.symbol}")
        data = load_market_data(args.symbol, start_date, end_date)
        
        if data.empty:
            print("❌ No market data available for the specified period")
            return 1
        
        print(f"   Loaded {len(data)} data points")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        print()
        
        # Run backtest
        print("🔄 Running unified backtest with comprehensive validation...")
        results = engine.run_backtest(data, validate=True)
        
        # Display results
        display_backtest_results(results, strategy_id)
        
        # Save results
        if getattr(args, 'output', None):
            save_results(results, args.output)
        
        # Validation status
        validation_passed = results.get('validation_passed', False)
        if validation_passed:
            print("✅ STRATEGY VALIDATION: PASSED")
            print("   Strategy meets all validation criteria including DSR ≥ 0.95")
            return_code = 0
        else:
            print("❌ STRATEGY VALIDATION: FAILED")
            print("   Strategy does not meet validation requirements")
            return_code = 1
        
        print("=" * 60)
        return return_code
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1

def load_strategy(strategy_name: str):
    """Load a strategy by name."""
    try:
        # Try to import the strategy
        strategy_module = __import__(f'strategies.{strategy_name}', fromlist=[strategy_name])
        
        # Look for strategy class
        strategy_class = None
        for attr_name in dir(strategy_module):
            attr = getattr(strategy_module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, BaseStrategy) and 
                attr != BaseStrategy):
                strategy_class = attr
                break
        
        if strategy_class is None:
            raise ImportError(f"No strategy class found in {strategy_name}")
        
        return strategy_class()
        
    except ImportError as e:
        print(f"❌ Failed to load strategy '{strategy_name}': {e}")
        print("Available strategies:")
        list_available_strategies()
        raise

def load_market_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Load market data for backtesting."""
    try:
        # Try to load from processed data first
        data_path = Path(f"data/processed/{symbol}_daily_processed.parquet")
        
        if data_path.exists():
            print(f"   Loading from processed data: {data_path}")
            data = pd.read_parquet(data_path)
            data.index = pd.to_datetime(data.index)
            
            # Filter by date range
            mask = (data.index >= start_date) & (data.index <= end_date)
            return data[mask]
        
        # Fallback to raw data
        raw_data_path = Path(f"data/raw/{symbol}_daily_ohlcv.csv")
        if raw_data_path.exists():
            print(f"   Loading from raw data: {raw_data_path}")
            data = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
            
            # Filter by date range
            mask = (data.index >= start_date) & (data.index <= end_date)
            return data[mask]
        
        # Try data manager as last resort
        print("   Attempting to fetch data using DataManager...")
        data_manager = DataManager()
        data = data_manager.get_data(symbol, start_date, end_date)
        
        return data
        
    except Exception as e:
        print(f"   Warning: Error loading data: {e}")
        # Return sample data for testing
        print("   Generating sample data for testing...")
        return generate_sample_data(start_date, end_date)

def generate_sample_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Simple random walk
    np.random.seed(42)
    price_changes = np.random.normal(0.001, 0.02, len(dates))
    prices = 50000 * np.exp(np.cumsum(price_changes))  # Start at $50,000
    
    # Generate OHLCV
    data = pd.DataFrame(index=dates)
    data['open'] = prices * (1 + np.random.normal(0, 0.005, len(dates)))
    data['high'] = np.maximum(data['open'], prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))))
    data['low'] = np.minimum(data['open'], prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))))
    data['close'] = prices
    data['volume'] = np.random.uniform(1000, 10000, len(dates))
    
    return data

def display_backtest_results(results: dict, strategy_id: str):
    """Display comprehensive backtest results."""
    metrics = results.get('portfolio_metrics', {})
    validation = results.get('validation', {})
    
    print("📊 BACKTEST RESULTS")
    print("-" * 40)
    
    # Portfolio performance
    print(f"💰 Portfolio Performance:")
    print(f"   Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"   Annualized Return: {metrics.get('annualized_return', 0):.2%}")
    print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"   Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
    print(f"   Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"   Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
    print()
    
    # Trading statistics
    trades = results.get('trades', [])
    if trades:
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        print(f"📈 Trading Statistics:")
        print(f"   Total Trades: {len(trades)}")
        print(f"   Winning Trades: {len(winning_trades)}")
        print(f"   Losing Trades: {len(losing_trades)}")
        print(f"   Win Rate: {len(winning_trades)/len(trades)*100:.1f}%")
        
        if winning_trades:
            avg_win = np.mean([t['pnl'] for t in winning_trades])
            print(f"   Average Win: ${avg_win:.2f}")
        
        if losing_trades:
            avg_loss = np.mean([t['pnl'] for t in losing_trades])
            print(f"   Average Loss: ${avg_loss:.2f}")
        
        total_pnl = sum(t['pnl'] for t in trades)
        print(f"   Total P&L: ${total_pnl:.2f}")
        print()
    
    # Validation results
    if validation:
        print(f"🔍 VALIDATION RESULTS:")
        
        # Critical metrics
        critical_metrics = validation.get('validation_summary', {}).get('critical_metrics', {})
        print(f"   DSR (Deflated Sharpe): {critical_metrics.get('dsr', 'N/A')}")
        print(f"   PSR (Probabilistic Sharpe): {critical_metrics.get('psr', 'N/A')}")
        
        # Monte Carlo
        mc_results = validation.get('monte_carlo_simulation', {})
        if mc_results:
            print(f"   Monte Carlo Trials: {mc_results.get('n_simulations', 'N/A')}")
            print(f"   Sharpe Percentile: {mc_results.get('percentile_ranks', {}).get('sharpe_percentile', 'N/A'):.1f}%")
            print(f"   Return Percentile: {mc_results.get('percentile_ranks', {}).get('return_percentile', 'N/A'):.1f}%")
        
        # Overall validation
        validation_passed = results.get('validation_passed', False)
        status = "✅ PASSED" if validation_passed else "❌ FAILED"
        print(f"   Overall Validation: {status}")
        print()

def save_results(results: dict, output_path: str):
    """Save backtest results to file."""
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize results for JSON
        serializable_results = serialize_for_json(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"📁 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save results: {e}")

def serialize_for_json(obj):
    """Convert complex objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

def list_available_strategies():
    """List available strategies."""
    strategies_path = Path("src/strategies")
    
    if not strategies_path.exists():
        print("   No strategies directory found")
        return
    
    for strategy_dir in strategies_path.iterdir():
        if (strategy_dir.is_dir() and 
            not strategy_dir.name.startswith('_') and
            strategy_dir.name != '__pycache__'):
            print(f"   - {strategy_dir.name}")
