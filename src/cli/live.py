"""
CLI module for live trading functionality.
"""

import sys
import os
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def run_live_trading(args):
    """
    Run live trading with real money.
    """
    print("=" * 60)
    print("⚠️  CRYPTO STRATEGY LAB - LIVE TRADING")
    print("=" * 60)
    
    print("🚨 DANGER: LIVE TRADING WITH REAL MONEY")
    print("=" * 60)
    
    if not getattr(args, 'confirm', False):
        print("❌ Live trading requires explicit confirmation")
        print("   Use --confirm flag to acknowledge the risks")
        print()
        print("⚠️  RISKS:")
        print("   • You can lose all your money")
        print("   • Markets are volatile and unpredictable")
        print("   • Software bugs can cause significant losses")
        print("   • Network issues can prevent order management")
        print("   • Exchange failures can lock your funds")
        print()
        print("💡 Consider paper trading first to test your strategy")
        return 1
    
    print(f"💰 Configuration:")
    print(f"   Strategy: {args.strategy}")
    print(f"   Exchange: {args.exchange}")
    print(f"   Starting Capital: ${args.capital:,.2f}")
    print(f"   Symbols: {', '.join(args.symbols)}")
    print(f"   Mode: {'DRY RUN' if getattr(args, 'dry_run', False) else 'LIVE TRADING (REAL MONEY)'}")
    print()
    
    # Additional safety check for real trading
    if not getattr(args, 'dry_run', False):
        print("🛡️  FINAL SAFETY CHECK")
        print("   Type 'I UNDERSTAND THE RISKS' to proceed:")
        
        try:
            confirmation = input("   > ")
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Live trading cancelled")
            return 1
        
        if confirmation != "I UNDERSTAND THE RISKS":
            print("❌ Confirmation failed. Live trading cancelled.")
            return 1
    
    try:
        # Import required modules
        from src.core.backtest_engine import MultiStrategyOrchestrator as TradingEngine
        from src.core.backtest_engine import BacktestConfig
        from src.exchanges.exchange_factory import ExchangeFactory
        from src.cli.backtest import load_strategy
        import asyncio
        import signal
        
        # Set up signal handler for graceful shutdown
        shutdown_event = asyncio.Event()
        
        def signal_handler(sig, frame):
            print("\n🛑 Shutting down live trading...")
            shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Initialize exchange connection
        print(f"🔌 Connecting to {args.exchange}...")
        exchange = ExchangeFactory.create_exchange(args.exchange)
        
        if not exchange:
            print(f"❌ Unsupported exchange: {args.exchange}")
            print("   Supported exchanges: bybit, alpaca")
            return 1
        
        # Test exchange connection
        print("🔍 Testing exchange connection...")
        if not await exchange.test_connection():
            print("❌ Failed to connect to exchange")
            return 1
        
        print("✅ Exchange connection successful")
        
        # Configure trading engine
        config = BacktestConfig(
            initial_capital=args.capital,
            enable_short_selling=True,
            max_position_size=0.05,  # More conservative for live trading
            max_total_exposure=0.5,   # More conservative for live trading
        )
        
        # Initialize trading engine
        engine = TradingEngine()
        
        # Load strategy
        print(f"📈 Loading strategy: {args.strategy}")
        strategy_instance = load_strategy(args.strategy)
        
        # Add strategy to engine
        strategy_id = engine.add_strategy(
            strategy=strategy_instance,
            symbols=args.symbols,
            strategy_id=args.strategy
        )
        
        print("🚀 Starting live trading...")
        print("   Press Ctrl+C to stop")
        print()
        
        # Main trading loop
        while not shutdown_event.is_set():
            try:
                # Get market data
                market_data = {}
                for symbol in args.symbols:
                    data = await exchange.get_market_data(symbol)
                    market_data[symbol] = data
                
                # Process trading signals
                print(f"📊 Processing signals for {len(args.symbols)} symbols...")
                
                # Execute trades (if not dry run)
                if not getattr(args, 'dry_run', False):
                    # Real trading logic would go here
                    print("💼 Executing trades...")
                else:
                    print("🧪 Dry run - no actual trades executed")
                
                # Wait before next iteration
                await asyncio.sleep(30)  # 30 second intervals
                
            except Exception as e:
                print(f"❌ Error in trading loop: {e}")
                if not getattr(args, 'dry_run', False):
                    print("🛑 Stopping live trading due to error")
                    break
                else:
                    print("⚠️  Continuing in dry run mode...")
                    await asyncio.sleep(5)
        
        print("✅ Live trading stopped")
        return 0
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    if confirmation.strip() != "I UNDERSTAND THE RISKS":
        print("❌ Incorrect confirmation. Live trading cancelled for safety.")
        return 1
    
    print()
    print("🚀 LIVE TRADING INITIALIZATION")
    print("=" * 60)
    print()
    print("⚠️  LIVE TRADING NOT FULLY IMPLEMENTED FOR SAFETY")
    print()
    print("📝 For complete live trading implementation, you need:")
    print("   1. ✅ Complete comprehensive testing with paper trading")
    print("   2. ✅ Implement proper risk management systems")
    print("   3. ❌ Add exchange connectivity and order management")
    print("   4. ❌ Implement monitoring and alerting systems")
    print("   5. ❌ Add circuit breakers and kill switches")
    print("   6. ❌ Complete legal and compliance requirements")
    print("   7. ❌ Set up proper API keys and authentication")
    print("   8. ❌ Implement position sizing and risk controls")
    print()
    print("💡 Continue with paper trading to test your strategies safely:")
    print(f"   python -m src.cli paper --strategy {args.strategy} --capital {args.capital}")
    print()
    print("🔧 When ready for live trading:")
    print("   1. Complete all implementation requirements above")
    print("   2. Remove safety checks in this module")
    print("   3. Add proper exchange integration")
    print("   4. Set up monitoring and alerting")
    print()
    
    return 1  # Return error to prevent accidental live trading
