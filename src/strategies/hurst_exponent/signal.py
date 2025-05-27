"""
Hurst Exponent Strategy

Mathematical foundation:
(R/S)_n ∝ n^H  ⟹  H = log((R/S)_n) / log(n)

Edge: H > 0.5 indicates trending persistence; H < 0.5 indicates anti-persistence
Trade rule: Compute DFA-based H_200 (≈9 months). If H > 0.55 use trend-following with wider stops; 
           if H < 0.45 use mean-reversion with tighter targets
Risk hooks: Stop-loss width proportional to H estimate confidence
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import yaml
from scipy import stats

class HurstExponentSignal:
    """
    Hurst Exponent trading signal generator.
    
    Implements DFA-based Hurst exponent estimation for regime detection.
    """
    
    def __init__(self, config_path: str = "config/strategies/hurst_exponent.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.params = config['best_params']
        
    def generate(self, returns: pd.Series, **context) -> pd.Series:
        """
        Generate trading signals based on Hurst exponent regime detection.
        
        Args:
            returns: Price return series
            **context: Additional market context
            
        Returns:
            Signal series with values {-1, 0, 1}
        """
        signals = pd.Series(0, index=returns.index)
        
        lookback = self.params['lookback_days']
        h_threshold_high = self.params['h_threshold_high']
        h_threshold_low = self.params['h_threshold_low']
        min_confidence = self.params['min_confidence']
        
        for i in range(lookback, len(returns)):
            window_data = returns.iloc[i-lookback:i]
            signal_value = self._calculate_signal(window_data, h_threshold_high, h_threshold_low, min_confidence)
            signals.iloc[i] = signal_value
            
        return signals
        
    def _calculate_signal(self, data: pd.Series, h_high: float, h_low: float, min_conf: float) -> int:
        """Calculate signal based on Hurst exponent regime."""
        if len(data) < 50:  # Need sufficient data for reliable estimate
            return 0
            
        # Calculate Hurst exponent using DFA
        hurst, confidence = self._calculate_hurst_dfa(data)
        
        if confidence < min_conf:
            return 0  # Insufficient confidence in estimate
            
        # Trading rules based on Hurst regime
        if hurst > h_high:
            return 1  # Persistent/trending regime - momentum
        elif hurst < h_low:
            return -1  # Anti-persistent regime - mean reversion
        else:
            return 0  # Neutral regime
    
    def _calculate_hurst_dfa(self, returns: pd.Series) -> tuple:
        """
        Calculate Hurst exponent using Detrended Fluctuation Analysis (DFA).
        
        Args:
            returns: Return series
            
        Returns:
            Tuple of (hurst_exponent, confidence)
        """
        if len(returns) < 50:
            return 0.5, 0.0
            
        # Convert returns to cumulative sum (integrate)
        y = np.cumsum(returns - returns.mean())
        n = len(y)
        
        # Define scales for DFA
        scales = np.logspace(1, np.log10(n//4), 20).astype(int)
        scales = np.unique(scales)
        
        if len(scales) < 4:
            return 0.5, 0.0
            
        fluctuations = []
        
        for scale in scales:
            # Divide time series into non-overlapping segments
            n_segments = n // scale
            
            if n_segments < 2:
                continue
                
            # Calculate local trend for each segment
            segment_flucts = []
            
            for i in range(n_segments):
                start_idx = i * scale
                end_idx = (i + 1) * scale
                segment = y[start_idx:end_idx]
                
                # Fit linear trend
                x = np.arange(len(segment))
                if len(x) > 1:
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)
                    
                    # Calculate detrended fluctuation
                    detrended = segment - trend
                    fluct = np.sqrt(np.mean(detrended**2))
                    segment_flucts.append(fluct)
            
            if segment_flucts:
                avg_fluct = np.mean(segment_flucts)
                fluctuations.append(avg_fluct)
            else:
                fluctuations.append(np.nan)
        
        # Remove NaN values
        valid_indices = ~np.isnan(fluctuations)
        if np.sum(valid_indices) < 3:
            return 0.5, 0.0
            
        log_scales = np.log10(scales[valid_indices])
        log_fluctuations = np.log10(np.array(fluctuations)[valid_indices])
        
        # Linear regression to find Hurst exponent
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_fluctuations)
            hurst = slope
            confidence = r_value**2  # R-squared as confidence measure
            
            # Bound Hurst exponent to reasonable range
            hurst = np.clip(hurst, 0.1, 0.9)
            
            return hurst, confidence
            
        except:
            return 0.5, 0.0
        
    def get_param_grid(self) -> Dict[str, Any]:
        """Return parameter grid for hyperoptimization."""
        with open(f"config/strategies/hurst_exponent.yaml", 'r') as f:
            config = yaml.safe_load(f)
        return config['param_grid']
