"""
Command Line Interface for Crypto Strategy Lab
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Crypto Strategy Lab - Multi-Exchange Trading Framework"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run strategy backtests')
    backtest_parser.add_argument('--strategy', required=True, help='Strategy name')
    backtest_parser.add_argument('--symbol', default='BTCUSD', help='Trading symbol')
    backtest_parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    
    # Paper trade command
    paper_parser = subparsers.add_parser('paper', help='Start paper trading')
    paper_parser.add_argument('--strategy', required=True, help='Strategy name')
    paper_parser.add_argument('--capital', type=float, default=10000, help='Starting capital')
    
    # Live trade command
    live_parser = subparsers.add_parser('live', help='Start live trading')
    live_parser.add_argument('--strategy', required=True, help='Strategy name')
    live_parser.add_argument('--capital', type=float, required=True, help='Starting capital')
    live_parser.add_argument('--confirm', action='store_true', help='Confirm live trading')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Launch monitoring dashboard')
    monitor_parser.add_argument('--port', type=int, default=8501, help='Dashboard port')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('--config', help='Config file to validate')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'backtest':
            from src.cli.backtest import run_backtest
            return run_backtest(args)
        elif args.command == 'paper':
            from src.cli.paper import run_paper_trading
            return run_paper_trading(args)
        elif args.command == 'live':
            from src.cli.live import run_live_trading
            return run_live_trading(args)
        elif args.command == 'monitor':
            from src.cli.monitor import run_monitor
            return run_monitor(args)
        elif args.command == 'validate':
            from src.cli.validate import run_validation
            return run_validation(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
    except ImportError as e:
        print(f"Error importing command module: {e}")
        print("Some CLI features may not be fully implemented yet.")
        return 1
    except Exception as e:
        print(f"Error executing command: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
