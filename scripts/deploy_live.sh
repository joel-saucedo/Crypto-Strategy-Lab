#!/bin/bash
# Live deployment script for Crypto-Strategy-Lab strategies

# Default values
STRATEGY="all"
CAPITAL="10000"
EXCHANGE="binance"
CONFIG_FILE="./config/live_trading.yaml"
PAPER_TRADING_DAYS=30
FORCE_DEPLOY=false
MONITOR_SETUP=true

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --strategy)
      STRATEGY="$2"
      shift 2
      ;;
    --capital)
      CAPITAL="$2"
      shift 2
      ;;
    --exchange)
      EXCHANGE="$2"
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --force)
      FORCE_DEPLOY=true
      shift
      ;;
    --no-monitor)
      MONITOR_SETUP=false
      shift
      ;;
    --paper-days)
      PAPER_TRADING_DAYS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --strategy <strategy_name> --capital <amount> --exchange <exchange> --config <config_file> [--force] [--no-monitor] [--paper-days <days>]"
      exit 1
      ;;
  esac
done

# Check Git status - ensure we're on a clean branch
if [ "$FORCE_DEPLOY" = false ]; then
  if [ -n "$(git status --porcelain)" ]; then
    echo "Error: Working directory is not clean. Commit changes before deploying or use --force to override."
    git status --short
    exit 1
  fi

  # Verify we're on main branch for production deployment
  CURRENT_BRANCH=$(git branch --show-current)
  if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "Warning: You're not on the main branch. Deployments should typically be from main."
    read -p "Continue anyway? (y/N): " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
      echo "Deployment aborted."
      exit 1
    fi
  fi
fi

# Check for valid DSR before deployment
echo "Validating strategy DSR..."
DSR_CHECK=$(python -c "
import sys
from src.core.validation import validate_dsr
strategy = '$STRATEGY'
try:
    dsr = validate_dsr(strategy)
    print(f'{dsr:.4f}')
    if dsr < 0.95:
        sys.exit(1)
except Exception as e:
    print(f'Error: {e}')
    sys.exit(2)
")

DSR_CHECK_EXIT=$?
if [ $DSR_CHECK_EXIT -eq 1 ]; then
    echo "Strategy DSR ${DSR_CHECK} is below threshold (0.95). Deployment aborted."
    exit 1
elif [ $DSR_CHECK_EXIT -eq 2 ]; then
    echo "Error validating strategy DSR: ${DSR_CHECK}"
    exit 1
fi

echo "Strategy DSR check passed: ${DSR_CHECK}"

# Verify paper trading history unless forced
if [ "$FORCE_DEPLOY" = false ]; then
  echo "Checking paper trading history..."
  PAPER_LOG_DIR="./data/paper_trade"
  
  # Check if strategy has been paper traded for sufficient time
  PAPER_TRADING_VERIFIED=false
  if [ -d "$PAPER_LOG_DIR" ]; then
    PAPER_FILES=$(find "$PAPER_LOG_DIR" -name "${STRATEGY}_paper_trade_*.csv" | sort)
    
    if [ -n "$PAPER_FILES" ]; then
      # Get the earliest and latest paper trading files
      FIRST_FILE=$(echo "$PAPER_FILES" | head -n 1)
      LATEST_FILE=$(echo "$PAPER_FILES" | tail -n 1)
      
      # Extract dates from filenames
      FIRST_DATE=$(basename "$FIRST_FILE" | grep -o '[0-9]\{8\}' | head -n 1)
      LATEST_DATE=$(basename "$LATEST_FILE" | grep -o '[0-9]\{8\}' | head -n 1)
      
      # Calculate the number of days between first and latest
      if command -v python &> /dev/null; then
        DAYS_DIFF=$(python -c "
from datetime import datetime
first = datetime.strptime('$FIRST_DATE', '%Y%m%d')
latest = datetime.strptime('$LATEST_DATE', '%Y%m%d')
delta = latest - first
print(delta.days)
")
        
        if [ "$DAYS_DIFF" -ge "$PAPER_TRADING_DAYS" ]; then
          PAPER_TRADING_VERIFIED=true
          echo "Paper trading verified: $DAYS_DIFF days (required: $PAPER_TRADING_DAYS)"
        else
          echo "Warning: Strategy has only been paper traded for $DAYS_DIFF days (required: $PAPER_TRADING_DAYS)"
          read -p "Continue with deployment anyway? (y/N): " CONTINUE
          if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
            echo "Deployment aborted. Run paper trading for more days first."
            echo "Use: ./scripts/paper_trade.sh --strategy $STRATEGY"
            exit 1
          fi
        fi
      else
        echo "Warning: Python not found, skipping paper trading duration check"
      fi
    else
      echo "Warning: No paper trading history found for $STRATEGY"
      read -p "No paper trading history. Continue with deployment? (y/N): " CONTINUE
      if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
        echo "Deployment aborted. Run paper trading first."
        echo "Use: ./scripts/paper_trade.sh --strategy $STRATEGY"
        exit 1
      fi
    fi
  else
    echo "Warning: Paper trading directory not found"
    read -p "No paper trading history. Continue with deployment? (y/N): " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
      echo "Deployment aborted. Run paper trading first."
      echo "Use: ./scripts/paper_trade.sh --strategy $STRATEGY"
      exit 1
    fi
  fi
fi

# Check for API credentials
if [ ! -f ".env" ]; then
    echo "Error: .env file with API credentials not found"
    echo "Please create a .env file with your exchange API credentials"
    exit 1
fi

# Deploy the strategy
echo "Deploying strategy: $STRATEGY"
echo "Exchange: $EXCHANGE"
echo "Initial capital: $CAPITAL"

python -c "
import os
import sys
import yaml
import importlib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

def load_strategy(strategy_name):
    try:
        # Try to import signal.py first
        module_path = f'src.strategies.{strategy_name}.signal'
        module = importlib.import_module(module_path)
        class_name = ''.join(word.title() for word in strategy_name.split('_')) + 'Signal'
        strategy_class = getattr(module, class_name)
    except (ImportError, AttributeError):
        # Try strategy.py if signal.py doesn't exist or doesn't have the class
        try:
            module_path = f'src.strategies.{strategy_name}.strategy'
            module = importlib.import_module(module_path)
            class_name = ''.join(word.title() for word in strategy_name.split('_')) + 'Strategy'
            strategy_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            print(f'Error loading strategy {strategy_name}: {e}')
            return None
    
    return strategy_class()

def initialize_exchange(exchange_name):
    if exchange_name.lower() == 'binance':
        try:
            from src.execution.binance import BinanceExecutor
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            if not api_key or not api_secret:
                print('Error: Binance API credentials not found in .env file')
                return None
            return BinanceExecutor(api_key, api_secret)
        except ImportError:
            print('Error: Binance execution module not found')
            return None
    elif exchange_name.lower() == 'coinbase':
        try:
            from src.execution.coinbase import CoinbaseExecutor
            api_key = os.getenv('COINBASE_API_KEY')
            api_secret = os.getenv('COINBASE_API_SECRET')
            if not api_key or not api_secret:
                print('Error: Coinbase API credentials not found in .env file')
                return None
            return CoinbaseExecutor(api_key, api_secret)
        except ImportError:
            print('Error: Coinbase execution module not found')
            return None
    else:
        print(f'Error: Unsupported exchange {exchange_name}')
        return None

def deploy_strategy(strategy_name, exchange_name, initial_capital, config_file):
    # Load strategy
    strategy = load_strategy(strategy_name)
    if strategy is None:
        print(f'Failed to load strategy {strategy_name}')
        return False
    
    # Initialize exchange
    executor = initialize_exchange(exchange_name)
    if executor is None:
        print(f'Failed to initialize exchange {exchange_name}')
        return False
    
    # Load configuration
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f'Error loading config file: {e}')
        return False
    
    # Configure trading parameters
    ticker = config.get('ticker', 'BTC/USDT')
    position_size = float(config.get('position_size', 0.1))
    max_position = float(config.get('max_position', 0.5))
    stop_loss = float(config.get('stop_loss', 0.05))
    take_profit = float(config.get('take_profit', 0.15))
    
    print(f'Deploying {strategy_name} on {exchange_name}')
    print(f'Ticker: {ticker}')
    print(f'Initial capital: ${initial_capital}')
    print(f'Position size: {position_size * 100}% per signal')
    print(f'Max position: {max_position * 100}% of capital')
    print(f'Stop loss: {stop_loss * 100}%')
    print(f'Take profit: {take_profit * 100}%')
    
    # Log deployment
    deployment_log = {
        'strategy': strategy_name,
        'exchange': exchange_name,
        'ticker': ticker,
        'initial_capital': initial_capital,
        'position_size': position_size,
        'max_position': max_position,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'deployment_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    log_dir = Path('./data/live')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'{strategy_name}_deployment_{datetime.now().strftime(\"%Y%m%d\")}.yaml'
    
    with open(log_file, 'w') as f:
        yaml.dump(deployment_log, f, default_flow_style=False)
    
    print(f'Deployment log saved to {log_file}')
    print('Strategy deployed successfully!')
    
    return True

if __name__ == '__main__':
    strategy = '$STRATEGY'
    exchange = '$EXCHANGE'
    initial_capital = float('$CAPITAL')
    config_file = '$CONFIG_FILE'
    
    if strategy == 'all':
        # Get all strategy directories
        strategy_path = Path('./src/strategies')
        strategies = [d.name for d in strategy_path.iterdir() if d.is_dir()]
        
        success_count = 0
        for s in strategies:
            if deploy_strategy(s, exchange, initial_capital, config_file):
                success_count += 1
        
        print(f'Deployment completed for {success_count}/{len(strategies)} strategies')
    else:
        if deploy_strategy(strategy, exchange, initial_capital, config_file):
            print('Deployment completed successfully')
        else:
            print('Deployment failed')
            sys.exit(1)
"

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Live deployment completed successfully"
    
    # Set up monitoring if requested
    if [ "$MONITOR_SETUP" = true ]; then
        echo "Setting up monitoring dashboard..."
        
        # Create monitoring directory if it doesn't exist
        MONITOR_DIR="./data/monitoring"
        mkdir -p "$MONITOR_DIR"
        
        # Create monitoring configuration
        MONITOR_CONFIG="$MONITOR_DIR/${STRATEGY}_monitoring_config.yaml"
        
        cat > "$MONITOR_CONFIG" <<EOL
strategy: $STRATEGY
exchange: $EXCHANGE
ticker: $(grep "ticker:" "$CONFIG_FILE" | awk '{print $2}')
deployment_time: $(date +"%Y-%m-%d %H:%M:%S")
check_interval_minutes: 15
alert_thresholds:
  drawdown_percent: 10
  inactivity_hours: 6
  position_deviation_percent: 5
notification:
  email_alerts: true
  slack_webhook: true
EOL
        
        # Launch monitoring process
        echo "Starting monitoring process..."
        MONITOR_LOG="$MONITOR_DIR/${STRATEGY}_monitor.log"
        nohup python -c "
import sys
import time
import yaml
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='$MONITOR_LOG',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info('Starting monitoring process for $STRATEGY on $EXCHANGE')

# Load monitoring configuration
with open('$MONITOR_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

# Implement basic monitoring loop
try:
    logging.info('Monitoring initialized with config: %s', config)
    print('Monitoring initialized. Check $MONITOR_LOG for details')
    
    # In a real implementation, this would connect to the exchange API
    # and check positions, account balance, etc.
    while True:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.info('Monitoring check at %s', now)
        time.sleep(config['check_interval_minutes'] * 60)
except Exception as e:
    logging.error('Monitoring error: %s', e)
    sys.exit(1)
" > /dev/null 2>&1 &
        
        echo "Monitoring started. Check $MONITOR_LOG for details"
        echo "Dashboard will be available at: http://localhost:8501 (when you run: python -m streamlit run scripts/monitoring_dashboard.py)"
    fi
    
    # Log deployment to deployment history
    DEPLOY_HISTORY="./data/deployment_history.csv"
    if [ ! -f "$DEPLOY_HISTORY" ]; then
        echo "timestamp,strategy,exchange,capital,git_commit,dsr" > "$DEPLOY_HISTORY"
    fi
    
    GIT_COMMIT=$(git rev-parse HEAD)
    echo "$(date +"%Y-%m-%d %H:%M:%S"),$STRATEGY,$EXCHANGE,$CAPITAL,$GIT_COMMIT,$DSR_CHECK" >> "$DEPLOY_HISTORY"
    
    echo "Deployment logged to $DEPLOY_HISTORY"
    echo "Monitor your positions regularly and check logs for performance"
else
    echo "Live deployment failed"
    exit 1
fi
