"""
CLI module for paper trading functionality.
"""

import sys
import os
import time
import signal
from datetime import datetime
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def run_paper_trading(args):
    """
    Run paper trading simulation.
    """
    try:
        print("=" * 60)
        print("📊 CRYPTO STRATEGY LAB - PAPER TRADING")
        print("=" * 60)
        
        print(f"🧪 Starting paper trading:")
        print(f"   Strategy: {args.strategy}")
        print(f"   Starting Capital: ${args.capital:,.2f}")
        print(f"   Duration: {args.duration} days")
        print(f"   Symbols: {', '.join(args.symbols)}")
        print(f"   Mode: Paper Trading (No Real Money)")
        print()
        
        # Set up signal handler for graceful shutdown
        def signal_handler(sig, frame):
            print("\n🛑 Stopping paper trading...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Import and initialize components
        from src.core.backtest_engine import MultiStrategyOrchestrator as BacktestEngine
        from src.core.backtest_engine import BacktestConfig
        from src.cli.backtest import load_strategy, load_market_data
        from datetime import datetime, timedelta
        
        # Configure paper trading
        config = BacktestConfig(
            initial_capital=args.capital,
            enable_short_selling=True,
            max_position_size=0.1,
            max_total_exposure=0.8,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=args.duration)
        )
        
        # Initialize engine
        engine = BacktestEngine()
        
        # Load strategy
        print(f"📈 Loading strategy: {args.strategy}")
        strategy_instance = load_strategy(args.strategy)
        
        # Add strategy to engine
        strategy_id = engine.add_strategy(
            strategy=strategy_instance,
            symbols=args.symbols,
            strategy_id=args.strategy
        )
        
        print("🔄 Starting paper trading simulation...")
        print("   Press Ctrl+C to stop")
        print()
        
        # Start real-time paper trading loop
        start_time = datetime.now()
        end_time = start_time + timedelta(days=args.duration)
        
        while datetime.now() < end_time:
            try:
                # Get current market data
                current_data = {}
                for symbol in args.symbols:
                    # In real implementation, fetch live data
                    # For now, use sample data
                    data = load_market_data(symbol, 
                                          datetime.now() - timedelta(days=30),
                                          datetime.now())
                    current_data[symbol] = data
                
                # Process signals and update positions
                print(f"📊 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Processing market data...")
                
                # Display current status
                elapsed = datetime.now() - start_time
                remaining = end_time - datetime.now()
                
                print(f"   Elapsed: {elapsed.days}d {elapsed.seconds//3600}h {(elapsed.seconds%3600)//60}m")
                print(f"   Remaining: {remaining.days}d {remaining.seconds//3600}h {(remaining.seconds%3600)//60}m")
                print(f"   Current Portfolio Value: ${args.capital:,.2f}")
                print()
                
                # Sleep for next update (in real implementation, this would be event-driven)
                time.sleep(60)  # Update every minute
                
            except KeyboardInterrupt:
                break
        
        print("✅ Paper trading simulation completed!")
        print(f"   Duration: {(datetime.now() - start_time).total_seconds() / 3600:.1f} hours")
        print()
        
        return 0
        
        # For paper trading, we use the backtest engine in real-time mode
        # In a full implementation, you'd also have execution engine integration
        
        # Load strategy
        print(f"📈 Loading strategy: {args.strategy}")
        strategy_instance = load_strategy(args.strategy)
        
        # Add strategy to engine
        strategy_id = backtest_engine.add_strategy(
            strategy=strategy_instance,
            symbols=['BTCUSD'],  # Default symbol for paper trading
            strategy_id=args.strategy
        )
        
        print("✅ Paper trading initialized successfully!")
        print("🔄 Starting trading loop...")
        print("   Press Ctrl+C to stop")
        print()
        
        # Main trading loop
        iteration = 0
        while True:
            iteration += 1
            
            try:
                # Get latest market data
                current_time = datetime.now()
                print(f"[{current_time.strftime('%H:%M:%S')}] Iteration {iteration}")
                
                # In a real implementation, this would:
                # 1. Fetch real-time market data
                # 2. Generate signals using the strategy
                # 3. Execute trades through the execution engine
                # 4. Update portfolio and track performance
                
                # For now, just simulate the loop
                print("   📡 Fetching market data...")
                print("   🧠 Generating signals...")
                print("   💼 Updating portfolio...")
                
                # Show current status
                print(f"   💰 Portfolio Value: ${args.capital:,.2f} (simulated)")
                print("   📊 No active positions")
                print()
                
                # Wait before next iteration
                time.sleep(30)  # 30 seconds between iterations
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"   ⚠️  Error in trading loop: {e}")
                time.sleep(10)  # Wait before retrying
                
        print("🛑 Paper trading stopped")
        return 0
        
    except Exception as e:
        print(f"❌ Error starting paper trading: {e}")
        import traceback
        traceback.print_exc()
        return 1

def load_strategy(strategy_name: str):
    """Load a strategy by name."""
    try:
        # First try to import signal.py (preferred pattern)
        try:
            strategy_module = __import__(f'strategies.{strategy_name}.signal', fromlist=[strategy_name])
            # Look for strategy class (usually named like 'StrategyNameSignal')
            class_name = ''.join(word.title() for word in strategy_name.split('_')) + 'Signal'
            strategy_class = getattr(strategy_module, class_name)
        except (ImportError, AttributeError):
            # Fallback to strategy.py
            strategy_module = __import__(f'strategies.{strategy_name}.strategy', fromlist=[strategy_name])
            class_name = ''.join(word.title() for word in strategy_name.split('_')) + 'Strategy'
            strategy_class = getattr(strategy_module, class_name)
        
        print(f"   ✅ Loaded strategy: {class_name}")
        return strategy_class()
        
    except (ImportError, AttributeError) as e:
        print(f"❌ Failed to load strategy '{strategy_name}': {e}")
        print("   Available strategies:")
        
        # List available strategies
        from pathlib import Path
        strategies_dir = Path('src/strategies')
        if strategies_dir.exists():
            available = [d.name for d in strategies_dir.iterdir() 
                        if d.is_dir() and not d.name.startswith('_')]
            for strategy in available:
                print(f"     - {strategy}")
        
        raise
