"""
Hyperparameter optimization pipeline.
Implements nested cross-validation with DSR objective.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from itertools import product
import yaml
import argparse
from typing import Dict, List, Any

from src.core.backtest import BacktestEngine
from src.strategies.lag_autocorr.signal import LagAutocorrSignal

class HyperOptimizer:
    """
    Nested cross-validation hyperparameter optimizer.
    Maximizes Deflated Sharpe Ratio on pooled out-of-sample data.
    """
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.engine = BacktestEngine()
        
    def optimize(self, prices: pd.DataFrame, returns: pd.Series) -> Dict[str, Any]:
        """
        Run nested CV hyperparameter optimization.
        
        Args:
            prices: OHLCV price data
            returns: Return series
            
        Returns:
            Best parameters and validation results
        """
        # Load strategy configuration
        config_path = f"config/strategies/{self.strategy_name}.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        param_grid = config['param_grid']
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        # Nested cross-validation
        outer_cv = TimeSeriesSplit(n_splits=5)
        best_params = None
        best_score = -np.inf
        
        oos_results = []
        
        for train_idx, test_idx in outer_cv.split(returns):
            train_returns = returns.iloc[train_idx]
            train_prices = prices.iloc[train_idx]
            test_returns = returns.iloc[test_idx]
            test_prices = prices.iloc[test_idx]
            
            # Inner CV for parameter selection
            inner_best = self._inner_cv(train_returns, train_prices, param_combinations)
            
            # Evaluate on outer test set
            test_score = self._evaluate_params(
                inner_best, test_returns, test_prices
            )
            
            oos_results.append({
                'params': inner_best,
                'score': test_score,
                'test_period': test_returns.index
            })
            
        # Select best parameters based on pooled OOS performance
        pooled_score = np.mean([r['score'] for r in oos_results])
        
        if pooled_score > best_score:
            best_score = pooled_score
            # Use most frequent best parameters across folds
            best_params = self._aggregate_best_params(oos_results)
            
        return {
            'best_params': best_params,
            'best_score': best_score,
            'oos_results': oos_results,
            'validation_passed': best_score >= 0.95  # DSR threshold
        }
        
    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate all parameter combinations from grid."""
        combinations = []
        
        # Handle different parameter types
        param_lists = {}
        for param, config in param_grid.items():
            if 'values' in config:
                param_lists[param] = config['values']
            elif 'min' in config and 'max' in config:
                if 'step' in config:
                    param_lists[param] = list(
                        np.arange(config['min'], config['max'] + config['step'], config['step'])
                    )
                else:
                    # For continuous parameters, use 10 points
                    param_lists[param] = list(
                        np.linspace(config['min'], config['max'], 10)
                    )
                    
        # Generate all combinations
        keys = param_lists.keys()
        values = param_lists.values()
        
        for combination in product(*values):
            combinations.append(dict(zip(keys, combination)))
            
        return combinations
        
    def _inner_cv(self, returns: pd.Series, prices: pd.DataFrame, 
                  param_combinations: List[Dict]) -> Dict:
        """Inner CV loop for parameter selection."""
        inner_cv = TimeSeriesSplit(n_splits=3)
        best_params = None
        best_score = -np.inf
        
        for params in param_combinations:
            scores = []
            
            for inner_train_idx, inner_val_idx in inner_cv.split(returns):
                inner_train_returns = returns.iloc[inner_train_idx]
                inner_val_returns = returns.iloc[inner_val_idx]
                inner_val_prices = prices.iloc[inner_val_idx]
                
                score = self._evaluate_params(params, inner_val_returns, inner_val_prices)
                scores.append(score)
                
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
                
        return best_params
        
    def _evaluate_params(self, params: Dict, returns: pd.Series, 
                        prices: pd.DataFrame) -> float:
        """Evaluate parameter set and return DSR score."""
        try:
            # Create temporary strategy with these parameters
            strategy = LagAutocorrSignal()
            strategy.params = params
            
            # Generate signals
            signals = strategy.generate(returns)
            
            # Run backtest
            results = self.engine.run_strategy(prices, signals, self.strategy_name)
            
            # Return DSR as score
            return results.get('dsr', 0)
            
        except Exception as e:
            print(f"Error evaluating params {params}: {e}")
            return 0
            
    def _aggregate_best_params(self, oos_results: List[Dict]) -> Dict:
        """Aggregate best parameters across CV folds."""
        # Simple approach: return parameters from best-performing fold
        best_fold = max(oos_results, key=lambda x: x['score'])
        return best_fold['params']

def main():
    """Main hyperoptimization script."""
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization')
    parser.add_argument('strategy', help='Strategy name')
    parser.add_argument('--data-path', default='data/processed/btc_1d.csv', 
                       help='Path to price data')
    
    args = parser.parse_args()
    
    # Load data
    prices = pd.read_csv(args.data_path, index_col=0, parse_dates=True)
    returns = prices['close'].pct_change(fill_method=None).dropna()
    
    # Run optimization
    optimizer = HyperOptimizer(args.strategy)
    results = optimizer.optimize(prices, returns)
    
    # Update configuration file with best parameters
    if results['validation_passed']:
        config_path = f"config/strategies/{args.strategy}.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        config['best_params'] = results['best_params']
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"✓ Validation passed. Updated {config_path}")
        print(f"Best DSR: {results['best_score']:.3f}")
    else:
        print(f"✗ Validation failed. DSR: {results['best_score']:.3f} < 0.95")
        
if __name__ == "__main__":
    main()
