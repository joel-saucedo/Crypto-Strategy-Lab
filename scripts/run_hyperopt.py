#!/bin/bash

"""
Hyperparameter optimization runner script.
Executes nested CV optimization for specified strategy.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pipelines.hyperopt import HyperOptimizer

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization')
    parser.add_argument('strategy', help='Strategy name')
    parser.add_argument('--data-path', default='data/processed/btc_1d.csv', 
                       help='Path to price data')
    parser.add_argument('--output-dir', default='reports/hyperopt',
                       help='Output directory for results')
    parser.add_argument('--n-trials', type=int, default=1000,
                       help='Number of trials for DSR calculation')
    
    args = parser.parse_args()
    
    print(f"Running hyperparameter optimization for {args.strategy}")
    print(f"Data source: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 50)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    try:
        if Path(args.data_path).exists():
            prices = pd.read_csv(args.data_path, index_col=0, parse_dates=True)
            returns = prices['close'].pct_change(fill_method=None).dropna()
            print(f"Loaded {len(prices)} price observations")
        else:
            print(f"Data file not found: {args.data_path}")
            print("Generating synthetic data for testing...")
            
            # Generate synthetic data for testing
            dates = pd.date_range('2020-01-01', periods=1000, freq='D')
            
            # Create synthetic price data with some autocorrelation
            returns = np.random.normal(0.0005, 0.02, len(dates))
            for i in range(1, len(returns)):
                returns[i] += 0.05 * returns[i-1]  # Small momentum
                
            prices_synthetic = np.exp(np.cumsum(returns)) * 10000
            
            prices = pd.DataFrame({
                'open': prices_synthetic * 0.999,
                'high': prices_synthetic * 1.002,
                'low': prices_synthetic * 0.998,
                'close': prices_synthetic,
                'volume': np.random.uniform(1000000, 5000000, len(dates))
            }, index=dates)
            
            returns = pd.Series(returns, index=dates)
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Run optimization
    try:
        optimizer = HyperOptimizer(args.strategy)
        results = optimizer.optimize(prices, returns)
        
        print("\\nOptimization Results:")
        print(f"Best DSR: {results['best_score']:.4f}")
        print(f"Validation passed: {results['validation_passed']}")
        print(f"Best parameters: {results['best_params']}")
        
        # Save detailed results
        results_file = output_path / f"{args.strategy}_hyperopt_results.json"
        import json
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj
        
        serializable_results = json.loads(json.dumps(results, default=convert_numpy))
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\\nDetailed results saved to: {results_file}")
        
        # Update strategy configuration if validation passed
        if results['validation_passed']:
            print("\\n" + "="*50)
            print("VALIDATION PASSED - UPDATING CONFIGURATION")
            print("="*50)
            
            import yaml
            config_path = f"config/strategies/{args.strategy}.yaml"
            
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                config['best_params'] = results['best_params']
                config['validation_results'] = {
                    'dsr': float(results['best_score']),
                    'validation_date': pd.Timestamp.now().isoformat(),
                    'validation_passed': True
                }
                
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                print(f"Updated configuration: {config_path}")
                
            except Exception as e:
                print(f"Warning: Could not update configuration: {e}")
        else:
            print("\\n" + "="*50)
            print("VALIDATION FAILED - DSR THRESHOLD NOT MET")
            print("="*50)
            print(f"Required DSR: ≥ 0.95")
            print(f"Achieved DSR: {results['best_score']:.4f}")
            print("\\nStrategy needs improvement before deployment.")
            
            return 1
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\\nHyperparameter optimization completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
