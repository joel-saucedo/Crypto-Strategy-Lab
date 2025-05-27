"""
Drift Significance Strategy

Mathematical foundation:
t = μ̂/(σ/√N),  Var(μ̂) ≥ σ²/N (Cramér-Rao lower bound)

Edge: Only drifts with |t| > 2 on a 90-day window beat the noise floor
Trade rule: Position size ∝ |t|; skip trades when |t| ≤ 2
Risk hooks: Dynamic position sizing based on signal strength
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import yaml
from scipy import stats

class DriftsignificanceSignal:
    """
    Drift Significance trading signal generator using Cramér-Rao lower bound.
    
    Detects statistically significant drifts using t-statistic analysis.
    Only signals when drift significance exceeds noise floor threshold.
    """
    
    def __init__(self, config_path: str = "config/strategies/drift_significance.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.params = config['best_params']
        
    def generate(self, returns: pd.Series, **context) -> pd.Series:
        """
        Generate trading signals based on drift significance.
        
        Args:
            returns: Price return series
            **context: Additional market context
            
        Returns:
            Signal series with values {-1, 0, 1}
        """
        signals = pd.Series(0, index=returns.index)
        
        lookback = self.params['lookback_days']
        min_t_stat = self.params['min_t_stat']
        confidence_level = self.params.get('confidence_level', 0.95)
        
        for i in range(lookback, len(returns)):
            window_data = returns.iloc[i-lookback:i]
            signal_value = self._calculate_signal(window_data, min_t_stat, confidence_level)
            signals.iloc[i] = signal_value
            
        return signals
        
    def _calculate_signal(self, data: pd.Series, min_t_stat: float, confidence_level: float) -> int:
        """
        Calculate signal for a given data window using drift significance test.
        
        Args:
            data: Return series window
            min_t_stat: Minimum t-statistic threshold
            confidence_level: Statistical confidence level
            
        Returns:
            Signal value {-1, 0, 1}
        """
        if len(data) < 10:  # Need sufficient data
            return 0
            
        # Calculate drift parameters
        mu_hat, sigma_hat, t_stat, p_value = self._calculate_drift_stats(data)
        
        # Apply Cramér-Rao significance filter
        if abs(t_stat) < min_t_stat:
            return 0  # Insufficient statistical significance
            
        # Additional confidence check
        critical_value = stats.t.ppf((1 + confidence_level) / 2, len(data) - 1)
        if abs(t_stat) < critical_value:
            return 0
            
        # Trading logic based on drift direction and significance
        if t_stat > min_t_stat and p_value < (1 - confidence_level):
            return 1  # Significant positive drift
        elif t_stat < -min_t_stat and p_value < (1 - confidence_level):
            return -1  # Significant negative drift
        else:
            return 0
            
    def _calculate_drift_stats(self, returns: pd.Series) -> tuple:
        """
        Calculate drift statistics using Cramér-Rao framework.
        
        Args:
            returns: Return series
            
        Returns:
            Tuple of (mu_hat, sigma_hat, t_stat, p_value)
        """
        n = len(returns)
        
        # Maximum likelihood estimators
        mu_hat = returns.mean()  # Sample mean
        sigma_hat = returns.std(ddof=1)  # Sample standard deviation
        
        # Standard error using Cramér-Rao lower bound
        # Var(μ̂) ≥ σ²/N (equality holds for normal distribution)
        standard_error = sigma_hat / np.sqrt(n)
        
        # t-statistic for testing H0: μ = 0
        if standard_error > 0:
            t_stat = mu_hat / standard_error
        else:
            t_stat = 0
            
        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
        
        return mu_hat, sigma_hat, t_stat, p_value
        
    def get_signal_strength(self, returns: pd.Series) -> float:
        """
        Calculate signal strength for position sizing.
        
        Args:
            returns: Return series
            
        Returns:
            Signal strength (0 to 1)
        """
        if len(returns) < 10:
            return 0.0
            
        _, _, t_stat, _ = self._calculate_drift_stats(returns)
        
        # Normalize t-statistic to [0, 1] range
        # Stronger signals get higher weights
        max_t = 5.0  # Practical upper bound for t-statistics
        strength = min(abs(t_stat) / max_t, 1.0)
        
        return strength
        
    def get_param_grid(self) -> Dict[str, Any]:
        """Return parameter grid for hyperoptimization."""
        with open(f"config/strategies/drift_significance.yaml", 'r') as f:
            config = yaml.safe_load(f)
        return config['param_grid']
