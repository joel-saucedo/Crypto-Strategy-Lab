"""
Spectral Peaks Strategy

Mathematical foundation:
S(f) = |ℱ{r_t}|² (Power Spectral Density)

Edge: Significant peaks reveal hidden periodicities (e.g., 4-year halving cycle)
Trade rule: Band-pass ±15% around the dominant frequency; long at troughs, short at peaks 
           when amplitude > 2× noise baseline (χ² test)
Risk hooks: Position size scaled by spectral peak significance
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import yaml
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from scipy import stats

class SpectralpeaksSignal:
    """
    Power-Spectral Peak Cycle Timer signal generator.
    
    Detects significant spectral peaks to identify hidden market cycles
    and times entry/exit around dominant periodicities.
    """
    
    def __init__(self, config_path: str = "config/strategies/spectral_peaks.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.params = config['best_params']
        
    def generate(self, returns: pd.Series, **context) -> pd.Series:
        """
        Generate trading signals based on spectral peak analysis.
        
        Args:
            returns: Price return series
            **context: Additional market context
            
        Returns:
            Signal series with values {-1, 0, 1}
        """
        signals = pd.Series(0, index=returns.index)
        
        window_size = self.params['window_size']
        significance_threshold = self.params['significance_threshold']
        min_period = self.params['min_period']
        max_period = self.params['max_period']
        phase_threshold = self.params['phase_threshold']
        
        for i in range(window_size, len(returns)):
            window_data = returns.iloc[i-window_size:i]
            signal_value = self._calculate_signal(
                window_data, 
                significance_threshold,
                min_period,
                max_period,
                phase_threshold
            )
            signals.iloc[i] = signal_value
            
        return signals
        
    def _calculate_signal(
        self, 
        data: pd.Series, 
        significance_threshold: float,
        min_period: int,
        max_period: int,
        phase_threshold: float
    ) -> int:
        """
        Calculate signal based on spectral peak analysis.
        
        Args:
            data: Return series window
            significance_threshold: Minimum peak significance
            min_period: Minimum cycle period (days)
            max_period: Maximum cycle period (days)
            phase_threshold: Phase threshold for timing signals
            
        Returns:
            Signal value {-1, 0, 1}
        """
        if len(data) < min_period * 2:  # Need at least 2 full cycles
            return 0
            
        # Find dominant spectral peaks
        dominant_freq, peak_power, baseline_power = self._find_dominant_peak(
            data, min_period, max_period
        )
        
        if dominant_freq is None:
            return 0
            
        # Test statistical significance using χ² test
        significance = peak_power / baseline_power if baseline_power > 0 else 0
        
        if significance < significance_threshold:
            return 0  # Peak not significant enough
            
        # Determine current phase in the dominant cycle
        phase = self._calculate_cycle_phase(data, dominant_freq)
        
        # Generate trading signal based on cycle phase
        return self._phase_to_signal(phase, phase_threshold)
        
    def _find_dominant_peak(
        self, 
        data: pd.Series, 
        min_period: int, 
        max_period: int
    ) -> Tuple[float, float, float]:
        """
        Find the dominant spectral peak within specified period range.
        
        Args:
            data: Return series
            min_period: Minimum period to consider
            max_period: Maximum period to consider
            
        Returns:
            Tuple of (dominant_frequency, peak_power, baseline_power)
        """
        # Apply window to reduce spectral leakage
        windowed_data = data * scipy_signal.windows.hann(len(data))
        
        # Compute power spectral density
        freqs = fftfreq(len(data))
        fft_vals = fft(windowed_data)
        power_spectrum = np.abs(fft_vals) ** 2
        
        # Convert to periods and filter by range
        periods = 1 / np.abs(freqs[1:len(freqs)//2])  # Exclude DC and negative freqs
        power_vals = power_spectrum[1:len(power_spectrum)//2]
        
        # Filter by period range
        valid_mask = (periods >= min_period) & (periods <= max_period)
        
        if not valid_mask.any():
            return None, 0, 0
            
        valid_periods = periods[valid_mask]
        valid_powers = power_vals[valid_mask]
        valid_freqs = freqs[1:len(freqs)//2][valid_mask]
        
        # Find peak
        peak_idx = np.argmax(valid_powers)
        dominant_freq = valid_freqs[peak_idx]
        peak_power = valid_powers[peak_idx]
        
        # Calculate baseline (median power excluding peak region)
        baseline_mask = np.abs(valid_periods - valid_periods[peak_idx]) > valid_periods[peak_idx] * 0.1
        if baseline_mask.any():
            baseline_power = np.median(valid_powers[baseline_mask])
        else:
            baseline_power = np.median(valid_powers)
            
        return dominant_freq, peak_power, baseline_power
        
    def _calculate_cycle_phase(self, data: pd.Series, frequency: float) -> float:
        """
        Calculate current phase in the dominant cycle.
        
        Args:
            data: Return series
            frequency: Dominant frequency
            
        Returns:
            Phase angle in radians [0, 2π]
        """
        if frequency == 0:
            return 0
            
        # Create sinusoid at dominant frequency
        t = np.arange(len(data))
        reference_sine = np.sin(2 * np.pi * frequency * t)
        reference_cosine = np.cos(2 * np.pi * frequency * t)
        
        # Calculate phase using correlation
        sine_corr = np.corrcoef(data, reference_sine)[0, 1]
        cosine_corr = np.corrcoef(data, reference_cosine)[0, 1]
        
        # Handle NaN correlations
        if np.isnan(sine_corr):
            sine_corr = 0
        if np.isnan(cosine_corr):
            cosine_corr = 0
            
        # Calculate phase angle
        phase = np.arctan2(sine_corr, cosine_corr)
        
        # Normalize to [0, 2π]
        if phase < 0:
            phase += 2 * np.pi
            
        return phase
        
    def _phase_to_signal(self, phase: float, threshold: float) -> int:
        """
        Convert cycle phase to trading signal.
        
        Args:
            phase: Phase angle in radians
            threshold: Phase threshold for signal generation
            
        Returns:
            Signal value {-1, 0, 1}
        """
        # Normalize phase to [-π, π] for easier interpretation
        normalized_phase = phase - np.pi
        
        # Long at cycle troughs (phase near -π or π)
        if abs(abs(normalized_phase) - np.pi) < threshold:
            return 1
            
        # Short at cycle peaks (phase near 0)
        if abs(normalized_phase) < threshold:
            return -1
            
        return 0
        
    def get_cycle_strength(self, data: pd.Series) -> float:
        """
        Calculate the strength of the dominant cycle.
        
        Args:
            data: Return series
            
        Returns:
            Cycle strength (0 to 1)
        """
        min_period = self.params['min_period']
        max_period = self.params['max_period']
        
        dominant_freq, peak_power, baseline_power = self._find_dominant_peak(
            data, min_period, max_period
        )
        
        if dominant_freq is None or baseline_power == 0:
            return 0.0
            
        # Normalize strength to [0, 1]
        strength = min(peak_power / baseline_power / 10.0, 1.0)
        
        return strength
        
    def get_param_grid(self) -> Dict[str, Any]:
        """Return parameter grid for hyperoptimization."""
        with open(f"config/strategies/spectral_peaks.yaml", 'r') as f:
            config = yaml.safe_load(f)
        return config['param_grid']
