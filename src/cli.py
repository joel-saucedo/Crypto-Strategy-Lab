#!/usr/bin/env python3
"""
Main CLI entry point for Crypto Strategy Lab.
"""

import argparse
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from cli.backtest import run_backtest
from cli.validate import run_validation
from cli.paper import run_paper_trading
from cli.live import run_live_trading
from cli.monitor import run_monitor

def create_parser():
    """Create the main CLI parser."""
    parser = argparse.ArgumentParser(
        description="Crypto Strategy Lab - Professional cryptocurrency trading strategy development",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a backtest
  crypto-lab backtest --strategy lag_autocorr --symbol BTCUSD --start 2023-01-01 --end 2023-12-31

  # Validate a strategy
  crypto-lab validate --strategy hurst_exponent --trials 10000

  # Run paper trading
  crypto-lab paper --strategy variance_ratio --capital 10000 --duration 7

  # Deploy live trading
  crypto-lab live --strategy permutation_entropy --capital 5000 --exchange bybit

  # Monitor performance
  crypto-lab monitor --port 8080
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run strategy backtests')
    backtest_parser.add_argument('--strategy', required=True, help='Strategy to test')
    backtest_parser.add_argument('--symbol', default='BTCUSD', help='Symbol to trade')
    backtest_parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    backtest_parser.add_argument('--position-size', type=float, default=0.1, help='Max position size')
    backtest_parser.add_argument('--exposure', type=float, default=0.8, help='Max total exposure')
    backtest_parser.add_argument('--mc-trials', type=int, default=10000, help='Monte Carlo trials')
    backtest_parser.add_argument('--min-dsr', type=float, default=0.95, help='Minimum DSR threshold')
    backtest_parser.add_argument('--min-psr', type=float, default=0.95, help='Minimum PSR threshold')
    backtest_parser.add_argument('--output', help='Output file for results')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate strategy statistical significance')
    validate_parser.add_argument('--strategy', help='Strategy to validate')
    validate_parser.add_argument('--config', help='Config file to validate')
    validate_parser.add_argument('--trials', type=int, default=10000, help='Monte Carlo trials')
    validate_parser.add_argument('--min-dsr', type=float, default=0.95, help='Minimum DSR threshold')
    
    # Paper trading command
    paper_parser = subparsers.add_parser('paper', help='Run paper trading simulation')
    paper_parser.add_argument('--strategy', required=True, help='Strategy to trade')
    paper_parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    paper_parser.add_argument('--duration', type=int, default=7, help='Duration in days')
    paper_parser.add_argument('--symbols', nargs='+', default=['BTCUSD'], help='Symbols to trade')
    
    # Live trading command
    live_parser = subparsers.add_parser('live', help='Deploy live trading')
    live_parser.add_argument('--strategy', required=True, help='Strategy to deploy')
    live_parser.add_argument('--exchange', required=True, help='Exchange to use')
    live_parser.add_argument('--capital', type=float, required=True, help='Trading capital')
    live_parser.add_argument('--symbols', nargs='+', default=['BTCUSD'], help='Symbols to trade')
    live_parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start monitoring dashboard')
    monitor_parser.add_argument('--port', type=int, default=8080, help='Dashboard port')
    monitor_parser.add_argument('--host', default='localhost', help='Dashboard host')
    
    return parser

async def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'backtest':
            return await run_backtest(args)
        elif args.command == 'validate':
            return await run_validation(args)
        elif args.command == 'paper':
            return await run_paper_trading(args)
        elif args.command == 'live':
            return await run_live_trading(args)
        elif args.command == 'monitor':
            return await run_monitor(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(asyncio.run(main()))