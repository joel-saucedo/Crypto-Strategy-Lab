"""
Command Line Interface for Crypto Strategy Lab
Unified backtesting framework with advanced validation and position sizing.
"""

import argparse
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main CLI entry point for the unified backtesting framework."""
    parser = argparse.ArgumentParser(
        description="Crypto Strategy Lab - Unified Backtesting Framework",
        epilog="For detailed help on each command, use: python -m src.cli <command> --help"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backtest command - Enhanced with validation layers
    backtest_parser = subparsers.add_parser('backtest', help='Run strategy backtests with advanced validation')
    backtest_parser.add_argument('--strategy', required=True, help='Strategy name or path to strategy file')
    backtest_parser.add_argument('--symbol', default='BTCUSD', help='Trading symbol (default: BTCUSD)')
    backtest_parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital (default: 100000)')
    backtest_parser.add_argument('--commission', type=float, default=0.001, help='Commission rate (default: 0.001)')
    backtest_parser.add_argument('--position-sizing', choices=['fixed_fractional', 'kelly', 'volatility_targeting', 'regime_aware', 'bayesian'], 
                                default='fixed_fractional', help='Position sizing method')
    backtest_parser.add_argument('--validation-layers', type=int, choices=[1,2,3,4,5], default=5, help='Number of validation layers (1-5)')
    backtest_parser.add_argument('--output', help='Output file for results (JSON format)')
    backtest_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Paper trade command - Safe simulation mode
    paper_parser = subparsers.add_parser('paper', help='Start paper trading simulation')
    paper_parser.add_argument('--strategy', required=True, help='Strategy name or path')
    paper_parser.add_argument('--capital', type=float, default=10000, help='Starting capital (default: 10000)')
    paper_parser.add_argument('--duration', type=int, help='Duration in days (default: run indefinitely)')
    paper_parser.add_argument('--save-trades', action='store_true', help='Save trade history to file')
    
    # Live trade command - Production trading with safety checks
    live_parser = subparsers.add_parser('live', help='Start live trading (CAUTION: Real money)')
    live_parser.add_argument('--strategy', required=True, help='Strategy name or path')
    live_parser.add_argument('--capital', type=float, required=True, help='Starting capital (required)')
    live_parser.add_argument('--exchange', required=True, choices=['bybit', 'alpaca'], help='Exchange to use')
    live_parser.add_argument('--confirm', action='store_true', help='Confirm live trading')
    live_parser.add_argument('--max-drawdown', type=float, default=0.1, help='Maximum drawdown limit (default: 10%)')
    live_parser.add_argument('--stop-loss', type=float, help='Global stop loss percentage')
    
    # Monitor command - Dashboard and monitoring
    monitor_parser = subparsers.add_parser('monitor', help='Launch monitoring dashboard')
    monitor_parser.add_argument('--port', type=int, default=8501, help='Dashboard port (default: 8501)')
    monitor_parser.add_argument('--mode', choices=['streamlit', 'html'], default='streamlit', help='Dashboard mode')
    monitor_parser.add_argument('--refresh', type=int, default=30, help='Refresh interval in seconds')
    
    # Validate command - System and configuration validation
    validate_parser = subparsers.add_parser('validate', help='Validate system, configs, and data')
    validate_parser.add_argument('--config', help='Specific config file to validate')
    validate_parser.add_argument('--strategy', help='Validate specific strategy')
    validate_parser.add_argument('--data', help='Validate data files')
    validate_parser.add_argument('--dependencies', action='store_true', help='Check system dependencies')
    validate_parser.add_argument('--all', action='store_true', help='Run all validations')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'backtest':
            from src.cli.backtest import main as backtest_main
            return backtest_main(args)
        
        elif args.command == 'paper':
            from src.cli.paper import main as paper_main
            return paper_main(args)
        
        elif args.command == 'live':
            from src.cli.live import main as live_main
            return live_main(args)
        
        elif args.command == 'monitor':
            from src.cli.monitor import main as monitor_main
            return monitor_main(args)
        
        elif args.command == 'validate':
            from src.cli.validate import main as validate_main
            return validate_main(args)
        
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose if hasattr(args, 'verbose') else False:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
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
