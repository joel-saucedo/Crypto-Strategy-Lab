"""
Hurst Exponent Strategy

Mathematical foundation:
(R/S)_n ∝ n^H  ⟹  H = log((R/S)_n) / log(n)

Edge: H > 0.5 indicates trending persistence; H < 0.5 indicates anti-persistence
Trade rule: Compute DFA-based H_200 (≈9 months). If H > 0.55 use trend-following with wider stops; 
           if H < 0.45 use mean-reversion with tighter targets.
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
        trend_threshold = self.params['trend_threshold']
        mean_revert_threshold = self.params['mean_revert_threshold']
        min_confidence = self.params['min_confidence']
        
        for i in range(lookback, len(returns)):
            window_data = returns.iloc[i-lookback:i]
            signal_value = self._calculate_signal(window_data, trend_threshold, 
                                                mean_revert_threshold, min_confidence)
            signals.iloc[i] = signal_value
            
        return signals
        
    def _calculate_signal(self, data: pd.Series, trend_threshold: float, 
                         mean_revert_threshold: float, min_confidence: float) -> int:
        """Calculate signal based on Hurst exponent estimation."""
        if len(data) < 50:  # Need sufficient data for reliable H estimation
            return 0
            
        # Calculate Hurst exponent using DFA method
        hurst, confidence = self._estimate_hurst_dfa(data)
        
        if confidence < min_confidence:
            return 0  # Insufficient confidence in H estimate
            
        # Determine regime and signal direction
        if hurst > trend_threshold:
            # Trending regime - use momentum
            recent_return = data.iloc[-1]
            return 1 if recent_return > 0 else -1
        elif hurst < mean_revert_threshold:
            # Mean-reverting regime - fade moves
            recent_return = data.iloc[-1]
            return -1 if recent_return > 0 else 1
        else:
            return 0  # Neutral regime
    
    def _estimate_hurst_dfa(self, returns: pd.Series) -> tuple:
        """
        Estimate Hurst exponent using Detrended Fluctuation Analysis (DFA).
        
        Args:
            returns: Return series
            
        Returns:
            Tuple of (hurst_exponent, confidence)
        """
        try:
            # Convert returns to cumulative sum (integrated series)
            y = np.cumsum(returns - returns.mean())
            n = len(y)
            
            if n < 50:
                return 0.5, 0.0
            
            # Define box sizes (scales)
            min_box = 10
            max_box = n // 4
            
            if max_box <= min_box:
                return 0.5, 0.0
                
            box_sizes = np.logspace(np.log10(min_box), np.log10(max_box), 
                                  num=min(20, max_box - min_box + 1), dtype=int)
            box_sizes = np.unique(box_sizes)
            
            fluctuations = []
            
            for box_size in box_sizes:
                if box_size >= n:
                    continue
                    
                # Calculate local trends and fluctuations
                n_boxes = n // box_size
                box_fluctuations = []
                
                for i in range(n_boxes):
                    start_idx = i * box_size
                    end_idx = (i + 1) * box_size
                    
                    if end_idx > n:
                        break
                        
                    box_data = y[start_idx:end_idx]
                    
                    # Fit linear trend
                    x_vals = np.arange(len(box_data))
                    if len(x_vals) > 1:
                        poly_coeffs = np.polyfit(x_vals, box_data, 1)
                        trend = np.polyval(poly_coeffs, x_vals)
                        
                        # Calculate fluctuation (RMS deviation from trend)
                        fluctuation = np.sqrt(np.mean((box_data - trend) ** 2))
                        box_fluctuations.append(fluctuation)
                
                if box_fluctuations:
                    avg_fluctuation = np.mean(box_fluctuations)
                    fluctuations.append(avg_fluctuation)
                else:
                    fluctuations.append(np.nan)
            
            # Remove NaN values
            valid_data = [(box, fluc) for box, fluc in zip(box_sizes, fluctuations) 
                         if not np.isnan(fluc) and fluc > 0]
            
            if len(valid_data) < 5:
                return 0.5, 0.0
                
            valid_boxes, valid_fluctuations = zip(*valid_data)
            
            # Fit power law: F(n) ∝ n^H
            log_boxes = np.log10(valid_boxes)
            log_fluctuations = np.log10(valid_fluctuations)
            
            # Linear regression to get Hurst exponent
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_boxes, log_fluctuations)
            
            hurst = slope
            confidence = r_value ** 2  # R-squared as confidence measure
            
            # Ensure Hurst is in reasonable range
            hurst = np.clip(hurst, 0.0, 1.0)
            
            return hurst, confidence
            
        except Exception:
            return 0.5, 0.0  # Return neutral values on error
        
    def get_param_grid(self) -> Dict[str, Any]:
        """Return parameter grid for hyperoptimization."""
        with open(f"config/strategies/hurst_exponent.yaml", 'r') as f:
            config = yaml.safe_load(f)
        return config['param_grid']
