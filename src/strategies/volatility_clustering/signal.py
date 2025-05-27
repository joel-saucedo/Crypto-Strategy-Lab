"""
Volatility Clustering Strategy

Mathematical foundation:
σ²_{t+1} = ω + α r_t² + β σ_t² (GARCH(1,1))

Edge: Crypto vol clusters; calm regimes yield cleaner drift
Trade rule: Label Calm if 20-day realized vol < 20-percentile; Storm if > 80-percentile.
           Run trend-following only in Calm; reduce size or switch to mean-reversion in Storm
Risk hooks: Regime-dependent position sizing and strategy selection
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import yaml
from scipy import stats
import warnings

class VolatilityclusteringSignal:
    """
    Volatility Clustering trading signal generator using GARCH-based regime detection.
    
    Detects calm vs storm volatility regimes and adjusts trading strategy accordingly.
    Uses trend-following in calm periods and mean-reversion in storm periods.
    """
    
    def __init__(self, config_path: str = "config/strategies/volatility_clustering.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.params = config['best_params']
        
    def generate(self, returns: pd.Series, **context) -> pd.Series:
        """
        Generate trading signals based on volatility clustering regimes.
        
        Args:
            returns: Price return series
            **context: Additional market context
            
        Returns:
            Signal series with values {-1, 0, 1}
        """
        signals = pd.Series(0, index=returns.index)
        
        lookback_days = self.params['lookback_days']
        regime_window = self.params['regime_window']
        calm_percentile = self.params['calm_percentile'] 
        storm_percentile = self.params['storm_percentile']
        trend_threshold = self.params['trend_threshold']
        
        for i in range(max(lookback_days, regime_window), len(returns)):
            # Get data windows
            regime_data = returns.iloc[i-regime_window:i]
            trend_data = returns.iloc[i-lookback_days:i]
            
            # Detect volatility regime
            regime = self._detect_volatility_regime(
                regime_data, calm_percentile, storm_percentile
            )
            
            # Generate signal based on regime
            signal_value = self._calculate_signal(
                trend_data, regime, trend_threshold
            )
            signals.iloc[i] = signal_value
            
        return signals
        
    def _detect_volatility_regime(
        self, 
        data: pd.Series, 
        calm_percentile: float, 
        storm_percentile: float
    ) -> str:
        """
        Detect current volatility regime (Calm, Storm, or Neutral).
        
        Args:
            data: Return series for regime detection
            calm_percentile: Percentile threshold for calm regime
            storm_percentile: Percentile threshold for storm regime
            
        Returns:
            Regime label: 'calm', 'storm', or 'neutral'
        """
        if len(data) < 10:
            return 'neutral'
            
        # Calculate realized volatility
        realized_vol = self._calculate_realized_volatility(data)
        
        # Get historical volatility distribution for percentile calculation
        vol_history = self._get_volatility_history(data)
        
        if len(vol_history) < 20:
            return 'neutral'
            
        # Calculate regime thresholds
        calm_threshold = np.percentile(vol_history, calm_percentile * 100)
        storm_threshold = np.percentile(vol_history, storm_percentile * 100)
        
        # Classify regime
        if realized_vol < calm_threshold:
            return 'calm'
        elif realized_vol > storm_threshold:
            return 'storm'
        else:
            return 'neutral'
            
    def _calculate_realized_volatility(self, returns: pd.Series) -> float:
        """
        Calculate realized volatility using exponentially weighted moving average.
        
        Args:
            returns: Return series
            
        Returns:
            Realized volatility (annualized)
        """
        if len(returns) < 2:
            return 0.0
            
        # Use EWMA for more responsive volatility estimation
        alpha = 0.06  # RiskMetrics lambda = 0.94, so alpha = 1-0.94
        
        squared_returns = returns ** 2
        ewma_var = squared_returns.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
        
        # Annualize (assuming daily data)
        realized_vol = np.sqrt(ewma_var * 252)
        
        return realized_vol
        
    def _get_volatility_history(self, data: pd.Series, history_length: int = 252) -> np.ndarray:
        """
        Get historical volatility for percentile calculation.
        
        Args:
            data: Return series
            history_length: Number of periods for volatility history
            
        Returns:
            Array of historical volatility values
        """
        if len(data) < 30:
            return np.array([])
            
        # Calculate rolling volatility
        window_size = 20  # 20-day volatility
        vol_history = []
        
        for i in range(window_size, min(len(data), history_length)):
            window = data.iloc[i-window_size:i]
            vol = window.std() * np.sqrt(252)  # Annualized
            vol_history.append(vol)
            
        return np.array(vol_history)
        
    def _calculate_signal(
        self, 
        data: pd.Series, 
        regime: str, 
        trend_threshold: float
    ) -> int:
        """
        Calculate trading signal based on regime and trend analysis.
        
        Args:
            data: Return series for trend detection
            regime: Current volatility regime
            trend_threshold: Threshold for trend significance
            
        Returns:
            Signal value {-1, 0, 1}
        """
        if len(data) < 10:
            return 0
            
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(data)
        
        # Apply regime-dependent strategy
        if regime == 'calm':
            # Use trend-following in calm periods
            if trend_strength > trend_threshold:
                return 1  # Follow uptrend
            elif trend_strength < -trend_threshold:
                return -1  # Follow downtrend
            else:
                return 0
                
        elif regime == 'storm':
            # Use mean-reversion in storm periods  
            if trend_strength > trend_threshold:
                return -1  # Fade uptrend (expect reversion)
            elif trend_strength < -trend_threshold:
                return 1  # Fade downtrend (expect reversion)
            else:
                return 0
                
        else:  # neutral regime
            # Reduced activity in neutral periods
            if abs(trend_strength) > trend_threshold * 1.5:  # Higher threshold
                return 1 if trend_strength > 0 else -1
            else:
                return 0
                
    def _calculate_trend_strength(self, returns: pd.Series) -> float:
        """
        Calculate trend strength using multiple indicators.
        
        Args:
            returns: Return series
            
        Returns:
            Trend strength (-1 to 1)
        """
        if len(returns) < 10:
            return 0.0
            
        # Method 1: Linear regression slope
        x = np.arange(len(returns))
        cumulative_returns = returns.cumsum()
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, cumulative_returns)
            
            # Normalize slope by volatility
            vol = returns.std() if returns.std() > 0 else 1
            normalized_slope = slope / vol
            
            # Weight by R-squared (trend quality)
            trend_strength = normalized_slope * (r_value ** 2)
            
        except (ValueError, FloatingPointError):
            trend_strength = 0.0
            
        # Method 2: Simple momentum (for robustness)
        if len(returns) >= 5:
            recent_mean = returns.tail(5).mean()
            overall_mean = returns.mean()
            momentum = (recent_mean - overall_mean) / (returns.std() + 1e-8)
            
            # Combine with regression-based measure
            trend_strength = 0.7 * trend_strength + 0.3 * momentum
            
        # Clip to reasonable range
        trend_strength = np.clip(trend_strength, -2.0, 2.0)
        
        return trend_strength
        
    def get_regime_info(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Get detailed regime information for analysis.
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary with regime analysis
        """
        regime_window = self.params['regime_window']
        
        if len(returns) < regime_window:
            return {'regime': 'insufficient_data', 'volatility': 0, 'trend_strength': 0}
            
        recent_data = returns.tail(regime_window)
        
        # Detect regime
        regime = self._detect_volatility_regime(
            recent_data, 
            self.params['calm_percentile'], 
            self.params['storm_percentile']
        )
        
        # Calculate metrics
        current_vol = self._calculate_realized_volatility(recent_data)
        trend_strength = self._calculate_trend_strength(recent_data)
        
        return {
            'regime': regime,
            'realized_volatility': current_vol,
            'trend_strength': trend_strength,
            'recommendation': self._get_regime_recommendation(regime)
        }
        
    def _get_regime_recommendation(self, regime: str) -> str:
        """Get trading recommendation based on regime."""
        if regime == 'calm':
            return 'trend_following'
        elif regime == 'storm':
            return 'mean_reversion'
        else:
            return 'neutral'
        
    def get_param_grid(self) -> Dict[str, Any]:
        """Return parameter grid for hyperoptimization."""
        with open(f"config/strategies/volatility_clustering.yaml", 'r') as f:
            config = yaml.safe_load(f)
        return config['param_grid']
