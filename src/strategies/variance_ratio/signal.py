"""
Variance Ratio Strategy

Mathematical foundation:
VR(q) = Var[P_{t+q} - P_t] / (q × Var[P_{t+1} - P_t])

Edge: VR(q) > 1 signals persistent drift; < 1 signals mean-reverting noise
Trade rule: In a 60-day window trade momentum when VR(5) > 1+δ (δ=0.1) and Lo-MacKinlay Z > 2; 
           trade contrarian when VR(5) < 1-δ
Risk hooks: Scale position by Z-statistic confidence
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import yaml
from scipy import stats

class VarianceRatioSignal:
    """
    Variance Ratio trading signal generator.
    
    Implements the Lo-MacKinlay variance ratio test for drift detection.
    """
    
    def __init__(self, config_path: str = "config/strategies/variance_ratio.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.params = config['best_params']
        
    def generate(self, returns: pd.Series, **context) -> pd.Series:
        """
        Generate trading signals based on variance ratio test.
        
        Args:
            returns: Price return series
            **context: Additional market context
            
        Returns:
            Signal series with values {-1, 0, 1}
        """
        signals = pd.Series(0, index=returns.index)
        
        lookback = self.params['lookback_days']
        q = self.params['lag_q']
        threshold_delta = self.params['threshold_delta']
        min_z_stat = self.params['min_z_stat']
        
        for i in range(lookback, len(returns)):
            window_data = returns.iloc[i-lookback:i]
            signal_value = self._calculate_signal(window_data, q, threshold_delta, min_z_stat)
            signals.iloc[i] = signal_value
            
        return signals
        
    def _calculate_signal(self, data: pd.Series, q: int, threshold_delta: float, min_z_stat: float) -> int:
        """Calculate signal for a given data window using variance ratio test."""
        if len(data) < q + 10:  # Need sufficient data
            return 0
            
        # Calculate variance ratio VR(q)
        vr, z_stat = self._calculate_variance_ratio(data, q)
        
        if abs(z_stat) < min_z_stat:
            return 0  # Insufficient statistical significance
            
        # Trading rules based on VR and Z-statistic
        if vr > (1 + threshold_delta) and z_stat > min_z_stat:
            return 1  # Momentum signal (persistent drift)
        elif vr < (1 - threshold_delta) and z_stat > min_z_stat:
            return -1  # Contrarian signal (mean-reverting noise)
        else:
            return 0
    
    def _calculate_variance_ratio(self, returns: pd.Series, q: int) -> tuple:
        """
        Calculate Lo-MacKinlay variance ratio and test statistic.
        
        Args:
            returns: Return series
            q: Lag parameter
            
        Returns:
            Tuple of (variance_ratio, z_statistic)
        """
        n = len(returns)
        
        if n < q + 10:
            return 1.0, 0.0
            
        # Calculate q-period returns
        returns_q = returns.rolling(window=q).sum().dropna()
        
        if len(returns_q) < 10:
            return 1.0, 0.0
            
        # Variance of 1-period returns
        var_1 = returns.var()
        
        # Variance of q-period returns
        var_q = returns_q.var()
        
        if var_1 <= 0:
            return 1.0, 0.0
            
        # Variance ratio
        vr = var_q / (q * var_1)
        
        # Lo-MacKinlay test statistic (assuming homoskedasticity)
        nq = len(returns_q)
        z_stat = (vr - 1) * np.sqrt(nq) / np.sqrt((2 * (q - 1)) / (3 * q))
        
        return vr, z_stat
        
    def get_param_grid(self) -> Dict[str, Any]:
        """Return parameter grid for hyperoptimization."""
        with open(f"config/strategies/variance_ratio.yaml", 'r') as f:
            config = yaml.safe_load(f)
        return config['param_grid']
