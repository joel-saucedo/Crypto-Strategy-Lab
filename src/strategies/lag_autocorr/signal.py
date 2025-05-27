"""
Lag-k Autocorrelation Strategy

Mathematical foundation:
ρ(k) = Σ(r_t - r̄)(r_{t-k} - r̄) / Σ(r_t - r̄)²

Edge: Daily crypto returns exhibit statistically significant serial dependence.
Trade rule: Long if ρ(1) > 0 and p-value < 0.05; short if ρ(1) < 0.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any
import yaml

class LagAutocorrSignal:
    """
    Lag-k autocorrelation momentum/reversal filter.
    Implements the mathematical framework from docs/pdf/lag_autocorr.pdf
    """
    
    def __init__(self, config_path: str = "config/strategies/lag_autocorr.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.params = config['best_params']
        
    def generate(self, returns: pd.Series, **context) -> pd.Series:
        """
        Generate trading signals based on lag-k autocorrelation.
        
        Args:
            returns: Price return series
            **context: Additional market context
            
        Returns:
            Signal series with values {-1, 0, 1}
        """
        signals = pd.Series(0, index=returns.index)
        
        lookback = self.params['lookback_days']
        lag_k = self.params['lag_k']
        sig_threshold = self.params['significance_threshold']
        min_corr = self.params['min_correlation']
        
        for i in range(lookback, len(returns)):
            # Get lookback window
            window_returns = returns.iloc[i-lookback:i]
            
            if len(window_returns) < lookback:
                continue
                
            # Calculate lag-k autocorrelation
            autocorr, p_value = self._calculate_autocorr(window_returns, lag_k)
            
            # Generate signal based on statistical significance
            if p_value < sig_threshold and abs(autocorr) > min_corr:
                if autocorr > 0:
                    signals.iloc[i] = 1  # Momentum signal
                else:
                    signals.iloc[i] = -1  # Reversal signal
            else:
                signals.iloc[i] = 0  # No signal
                
        return signals
        
    def _calculate_autocorr(self, returns: pd.Series, lag: int) -> tuple:
        """
        Calculate lag-k autocorrelation with statistical test.
        
        Returns:
            tuple: (autocorrelation, p_value)
        """
        if len(returns) <= lag:
            return 0, 1
            
        # Calculate autocorrelation
        autocorr = returns.autocorr(lag=lag)
        
        if pd.isna(autocorr):
            return 0, 1
            
        # Calculate t-statistic for significance test
        n = len(returns)
        se = 1 / np.sqrt(n)  # Standard error under null hypothesis
        t_stat = autocorr / se
        
        # Two-tailed test
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        return autocorr, p_value
        
    def get_param_grid(self) -> Dict[str, Any]:
        """Return parameter grid for hyperoptimization."""
        with open("config/strategies/lag_autocorr.yaml", 'r') as f:
            config = yaml.safe_load(f)
        return config['param_grid']
