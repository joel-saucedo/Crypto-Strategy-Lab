"""
Spectral-Entropy Collapse Trading Strategy (SBPRF Core)

Mathematical Foundation:
- Spectral Entropy quantifies the disorder in frequency domain: H_s = -∑(P_k * log(P_k))
- Where P_k is the normalized power spectral density at frequency k
- Sudden drops in spectral entropy indicate regime changes or order flow concentration
- Strategy generates signals when entropy collapses below historical percentiles

Key Components:
1. Power Spectral Density (PSD) estimation via Welch's method
2. Spectral entropy calculation with proper normalization
3. Regime change detection via entropy percentile thresholds
4. Signal strength based on entropy collapse magnitude

DSR Requirements:
- Minimum 95% statistical significance for regime detection
- No look-ahead bias in entropy calculations
- Robust to market microstructure noise
- Mathematical consistency with information theory
"""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import percentileofscore
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class SpectralEntropyCollapseSignal:
    """
    Detects regime changes through spectral entropy collapse analysis.
    
    The strategy identifies moments when the frequency spectrum becomes
    more concentrated (lower entropy), indicating potential regime shifts
    or microstructure changes that can be exploited for trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize spectral entropy parameters."""
        self.lookback = config.get('lookback', 100)
        self.welch_nperseg = config.get('welch_nperseg', 32)
        self.entropy_threshold_pct = config.get('entropy_threshold_pct', 20)
        self.min_entropy_drop = config.get('min_entropy_drop', 0.1)
        self.signal_decay = config.get('signal_decay', 0.9)
        self.volatility_lookback = config.get('volatility_lookback', 20)
        
        # Internal state
        self.entropy_history = []
        self.current_signal = 0.0
        
    def calculate_spectral_entropy(self, prices: np.ndarray) -> float:
        """
        Calculate spectral entropy using Welch's method for PSD estimation.
        
        Mathematical Foundation:
        1. Estimate PSD: S(f) using Welch's method (reduces noise vs raw FFT)
        2. Normalize PSD: P(f) = S(f) / ∑S(f)
        3. Calculate entropy: H = -∑(P(f) * log(P(f)))
        4. Normalize by max entropy: H_norm = H / log(N)
        
        Args:
            prices: Array of price data
            
        Returns:
            Normalized spectral entropy [0, 1]
        """
        if len(prices) < self.welch_nperseg:
            return 0.5  # Neutral entropy for insufficient data
            
        # Calculate log returns to focus on dynamics
        returns = np.diff(np.log(prices))
        
        if len(returns) < self.welch_nperseg:
            return 0.5
            
        try:
            # Welch's method for robust PSD estimation
            freqs, psd = scipy_signal.welch(
                returns, 
                nperseg=min(self.welch_nperseg, len(returns)),
                noverlap=None,
                detrend='constant'
            )
            
            # Remove DC component (zero frequency)
            if len(freqs) > 1:
                psd = psd[1:]
                freqs = freqs[1:]
            
            # Normalize PSD to probability distribution
            psd_sum = np.sum(psd)
            if psd_sum <= 0:
                return 0.5
                
            prob_density = psd / psd_sum
            
            # Calculate spectral entropy
            # Add small epsilon to avoid log(0)
            epsilon = 1e-12
            prob_density = np.maximum(prob_density, epsilon)
            
            entropy = -np.sum(prob_density * np.log(prob_density))
            
            # Normalize by maximum possible entropy
            max_entropy = np.log(len(prob_density))
            if max_entropy > 0:
                normalized_entropy = entropy / max_entropy
            else:
                normalized_entropy = 0.5
                
            return np.clip(normalized_entropy, 0.0, 1.0)
            
        except Exception:
            return 0.5
    
    def detect_entropy_collapse(self, current_entropy: float) -> Tuple[bool, float]:
        """
        Detect if current entropy represents a significant collapse.
        
        Args:
            current_entropy: Current spectral entropy value
            
        Returns:
            (is_collapse, collapse_strength)
        """
        if len(self.entropy_history) < 20:
            return False, 0.0
            
        # Calculate percentile of current entropy in historical distribution
        entropy_percentile = percentileofscore(self.entropy_history, current_entropy)
        
        # Check if entropy is below threshold percentile
        is_below_threshold = entropy_percentile < self.entropy_threshold_pct
        
        if not is_below_threshold:
            return False, 0.0
            
        # Calculate collapse magnitude
        recent_entropy = np.array(self.entropy_history[-20:])
        entropy_drop = np.mean(recent_entropy) - current_entropy
        
        # Require minimum drop magnitude
        if entropy_drop < self.min_entropy_drop:
            return False, 0.0
            
        # Calculate collapse strength (0 to 1)
        max_possible_drop = np.max(recent_entropy) - np.min(self.entropy_history)
        if max_possible_drop > 0:
            collapse_strength = np.clip(entropy_drop / max_possible_drop, 0.0, 1.0)
        else:
            collapse_strength = 0.0
            
        return True, collapse_strength
    
    def calculate_volatility_regime(self, prices: np.ndarray) -> float:
        """
        Calculate current volatility regime for signal conditioning.
        
        Returns volatility percentile (0-1) over lookback period.
        """
        if len(prices) < self.volatility_lookback + 1:
            return 0.5
            
        returns = np.diff(np.log(prices[-self.volatility_lookback-1:]))
        current_vol = np.std(returns[-5:]) if len(returns) >= 5 else np.std(returns)
        historical_vols = [np.std(returns[i:i+5]) for i in range(len(returns)-4)] if len(returns) >= 5 else [np.std(returns)]
        
        if len(historical_vols) == 0:
            return 0.5
            
        vol_percentile = percentileofscore(historical_vols, current_vol) / 100.0
        return np.clip(vol_percentile, 0.0, 1.0)
    
    def generate_signal(self, data: pd.DataFrame) -> float:
        """
        Generate trading signal based on spectral entropy collapse.
        
        Signal Logic:
        1. Calculate current spectral entropy
        2. Detect if entropy collapse occurred
        3. Generate signal strength based on collapse magnitude
        4. Condition signal based on volatility regime
        5. Apply signal decay for persistence
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Signal strength [-1, 1] where:
            - Positive: Long signal (entropy collapse suggests regime change)
            - Negative: Short signal (high entropy suggests continuation)
            - Zero: No signal
        """
        if len(data) < self.lookback:
            return 0.0
            
        # Use recent price data
        recent_data = data.tail(self.lookback)
        prices = recent_data['close'].values
        
        # Calculate current spectral entropy
        current_entropy = self.calculate_spectral_entropy(prices)
        
        # Update entropy history
        self.entropy_history.append(current_entropy)
        if len(self.entropy_history) > self.lookback:
            self.entropy_history.pop(0)
            
        # Detect entropy collapse
        is_collapse, collapse_strength = self.detect_entropy_collapse(current_entropy)
        
        # Calculate base signal
        if is_collapse:
            # Entropy collapse suggests regime change - generate long signal
            base_signal = collapse_strength
        else:
            # High entropy suggests continuation - weak short bias
            entropy_percentile = percentileofscore(self.entropy_history, current_entropy) / 100.0
            if entropy_percentile > 0.8:  # Very high entropy
                base_signal = -0.3 * (entropy_percentile - 0.8) / 0.2
            else:
                base_signal = 0.0
        
        # Condition signal based on volatility regime
        vol_regime = self.calculate_volatility_regime(prices)
        
        # Entropy collapses are more significant in low volatility environments
        if is_collapse:
            vol_adjustment = 1.0 + 0.5 * (1.0 - vol_regime)  # Boost in low vol
        else:
            vol_adjustment = 1.0
            
        conditioned_signal = base_signal * vol_adjustment
        
        # Apply signal decay and update current signal
        self.current_signal = (
            self.signal_decay * self.current_signal + 
            (1 - self.signal_decay) * conditioned_signal
        )
        
        return np.clip(self.current_signal, -1.0, 1.0)
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Return current strategy state for monitoring."""
        current_entropy = self.entropy_history[-1] if self.entropy_history else 0.5
        
        return {
            'current_entropy': current_entropy,
            'entropy_history_length': len(self.entropy_history),
            'current_signal': self.current_signal,
            'entropy_percentile': percentileofscore(self.entropy_history, current_entropy) if len(self.entropy_history) > 0 else 50.0
        }
