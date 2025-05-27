#!/bin/bash
# Paper trading script for Crypto-Strategy-Lab

# Default values
STRATEGY="all"
DAYS=30
INITIAL_CAPITAL=10000
OUTPUT_DIR="./data/paper_trade"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --strategy)
      STRATEGY="$2"
      shift 2
      ;;
    --days)
      DAYS="$2"
      shift 2
      ;;
    --capital)
      INITIAL_CAPITAL="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --strategy <strategy_name> --days <days> --capital <amount> --output <dir>"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting paper trading simulation"
echo "Strategy: $STRATEGY"
echo "Duration: $DAYS days"
echo "Initial capital: $INITIAL_CAPITAL"
echo "Output directory: $OUTPUT_DIR"

# Run the Python script
python -c "
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import importlib
import yaml

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

def run_paper_trading(strategy_name, days, initial_capital, output_dir):
    print(f'Running paper trading for {strategy_name}...')
    
    # Load the strategy
    strategy = load_strategy(strategy_name)
    if strategy is None:
        print(f'Failed to load strategy {strategy_name}')
        return False
    
    # Load historical data
    data_path = Path('./data/processed/prices.csv')
    if not data_path.exists():
        print(f'Data file not found: {data_path}')
        return False
    
    prices = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Get the last N days of data
    end_date = returns.index[-1]
    start_date = end_date - timedelta(days=days)
    paper_returns = returns[returns.index >= start_date]
    
    # Generate signals
    signals = strategy.generate(paper_returns)
    
    # Simple portfolio simulation
    portfolio_value = pd.Series(initial_capital, index=signals.index)
    position = 0
    
    for i in range(1, len(signals)):
        signal = signals.iloc[i-1]  # Signal from yesterday determines today's position
        ret = paper_returns.iloc[i]  # Today's return
        
        # Update position based on signal
        if signal != position:
            position = signal
        
        # Update portfolio value
        portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + position * ret)
    
    # Calculate metrics
    total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
    daily_returns = portfolio_value.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    max_drawdown = (portfolio_value / portfolio_value.cummax() - 1).min()
    
    # Create results dataframe
    results = pd.DataFrame({
        'Date': signals.index,
        'Signal': signals.values,
        'Price': prices.loc[signals.index].values,
        'Return': paper_returns.values,
        'Portfolio': portfolio_value.values
    })
    
    # Save results
    output_file = Path(output_dir) / f'{strategy_name}_paper_trade_{datetime.now().strftime(\"%Y%m%d\")}.csv'
    results.to_csv(output_file)
    
    # Print summary
    print(f'Strategy: {strategy_name}')
    print(f'Period: {start_date.strftime(\"%Y-%m-%d\")} to {end_date.strftime(\"%Y-%m-%d\")}')
    print(f'Total Return: {total_return:.2%}')
    print(f'Sharpe Ratio: {sharpe:.2f}')
    print(f'Max Drawdown: {max_drawdown:.2%}')
    print(f'Results saved to {output_file}')
    
    return True

if __name__ == '__main__':
    strategy = '$STRATEGY'
    days = int('$DAYS')
    initial_capital = float('$INITIAL_CAPITAL')
    output_dir = '$OUTPUT_DIR'
    
    if strategy == 'all':
        # Get all strategy directories
        strategy_path = Path('./src/strategies')
        strategies = [d.name for d in strategy_path.iterdir() if d.is_dir()]
        
        success_count = 0
        for s in strategies:
            if run_paper_trading(s, days, initial_capital, output_dir):
                success_count += 1
        
        print(f'Paper trading completed for {success_count}/{len(strategies)} strategies')
    else:
        if run_paper_trading(strategy, days, initial_capital, output_dir):
            print('Paper trading completed successfully')
        else:
            print('Paper trading failed')
            sys.exit(1)
"

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Paper trading completed successfully"
else
    echo "Paper trading failed"
    exit 1
fi
