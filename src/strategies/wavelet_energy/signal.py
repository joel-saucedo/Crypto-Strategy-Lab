"""
Wavelet Energy-Ratio Breakout Trading Strategy

Mathematical Foundation:
- Wavelet decomposition separates price signals into different frequency components
- Energy ratios between scales indicate regime changes and breakout patterns
- High-frequency energy spikes suggest breakouts; low-frequency dominance suggests trends
- Strategy detects energy redistribution across wavelet scales

Key Components:
1. Discrete Wavelet Transform (DWT) with multiple decomposition levels
2. Energy calculation per wavelet scale: E_j = Σ|W_j(t)|²
3. Energy ratio analysis: R_j = E_j / Σ(E_k) for breakout detection
4. Multi-scale signal generation based on energy concentration

DSR Requirements:
- Minimum 95% statistical significance for breakout detection
- No look-ahead bias in wavelet calculations
- Robust to market microstructure noise
- Mathematical consistency with wavelet theory
"""

import numpy as np
import pandas as pd
import pywt
from scipy.stats import percentileofscore
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class WaveletEnergyBreakoutSignal:
    """
    Detects breakouts through wavelet energy redistribution analysis.
    
    The strategy identifies moments when energy concentrates in specific
    frequency bands, indicating potential breakouts or regime changes
    that can be exploited for trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize wavelet energy parameters."""
        self.wavelet_type = config.get('wavelet_type', 'db4')
        self.decomposition_levels = config.get('decomposition_levels', 4)
        self.lookback = config.get('lookback', 100)
        self.energy_threshold_pct = config.get('energy_threshold_pct', 80)
        self.breakout_threshold_pct = config.get('breakout_threshold_pct', 90)
        self.signal_decay = config.get('signal_decay', 0.85)
        self.trend_lookback = config.get('trend_lookback', 20)
        
        # Internal state
        self.energy_ratios_history = []
        self.current_signal = 0.0
        self.energy_history = {i: [] for i in range(self.decomposition_levels + 1)}
        
    def calculate_wavelet_decomposition(self, prices: np.ndarray) -> List[np.ndarray]:
        """
        Perform discrete wavelet transform decomposition.
        
        Mathematical Foundation:
        1. DWT: W_j,k = Σ x(t) * ψ_j,k(t) where ψ is wavelet function
        2. Multi-resolution: x(t) = Σ_j Σ_k W_j,k * ψ_j,k(t)
        3. Energy at scale j: E_j = Σ_k |W_j,k|²
        
        Args:
            prices: Array of price data
            
        Returns:
            List of wavelet coefficients [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        """
        if len(prices) < 2**self.decomposition_levels:
            # Insufficient data - return neutral decomposition
            return [np.array([0.0]) for _ in range(self.decomposition_levels + 1)]
            
        # Calculate log returns for stationarity
        returns = np.diff(np.log(prices))
        
        if len(returns) < 8:  # Minimum for meaningful wavelet analysis
            return [np.array([0.0]) for _ in range(self.decomposition_levels + 1)]
            
        try:
            # Perform multi-level discrete wavelet transform
            coeffs = pywt.wavedec(returns, self.wavelet_type, level=self.decomposition_levels)
            
            # Ensure we have the expected number of coefficient arrays
            while len(coeffs) < self.decomposition_levels + 1:
                coeffs.append(np.array([0.0]))
                
            return coeffs
            
        except Exception:
            # Fallback for any wavelet computation errors
            return [np.array([0.0]) for _ in range(self.decomposition_levels + 1)]
    
    def calculate_energy_ratios(self, coeffs: List[np.ndarray]) -> np.ndarray:
        """
        Calculate energy ratios for each wavelet scale.
        
        Energy at scale j: E_j = Σ|W_j,k|²
        Energy ratio: R_j = E_j / Σ(E_k)
        
        Args:
            coeffs: Wavelet coefficients from decomposition
            
        Returns:
            Array of energy ratios for each scale
        """
        energies = []
        
        for coeff in coeffs:
            if len(coeff) > 0:
                energy = np.sum(coeff**2)
            else:
                energy = 0.0
            energies.append(energy)
        
        energies = np.array(energies)
        total_energy = np.sum(energies)
        
        if total_energy > 0:
            energy_ratios = energies / total_energy
        else:
            # Uniform distribution if no energy
            energy_ratios = np.ones(len(energies)) / len(energies)
            
        return energy_ratios
    
    def detect_energy_breakout(self, energy_ratios: np.ndarray) -> Tuple[bool, float, int]:
        """
        Detect if current energy distribution indicates a breakout.
        
        Breakout signals:
        1. High-frequency energy concentration (short-term breakout)
        2. Sudden energy redistribution from historical patterns
        3. Energy ratio exceeding historical percentiles
        
        Args:
            energy_ratios: Current energy ratios
            
        Returns:
            (is_breakout, breakout_strength, dominant_scale)
        """
        if len(self.energy_ratios_history) < 20:
            return False, 0.0, 0
            
        # Find dominant scale (highest energy ratio)
        dominant_scale = np.argmax(energy_ratios)
        dominant_energy = energy_ratios[dominant_scale]
        
        # Calculate historical energy distribution for this scale
        historical_energies = [hist[dominant_scale] for hist in self.energy_ratios_history]
        
        # Check if current energy ratio is extreme
        energy_percentile = percentileofscore(historical_energies, dominant_energy)
        
        # Breakout condition: energy concentration above threshold
        is_concentration_breakout = energy_percentile > self.breakout_threshold_pct
        
        # Additional condition: dominant energy above absolute threshold
        is_energy_breakout = dominant_energy > (self.energy_threshold_pct / 100.0)
        
        is_breakout = bool(is_concentration_breakout and is_energy_breakout)
        
        if is_breakout:
            # Calculate breakout strength (0 to 1)
            percentile_strength = (energy_percentile - self.breakout_threshold_pct) / (100 - self.breakout_threshold_pct)
            energy_strength = (dominant_energy - 0.5) / 0.5  # Scale from 0.5 to 1.0
            
            breakout_strength = np.clip(np.mean([percentile_strength, energy_strength]), 0.0, 1.0)
        else:
            breakout_strength = 0.0
            
        return is_breakout, float(breakout_strength), int(dominant_scale)
    
    def interpret_breakout_scale(self, dominant_scale: int, energy_ratios: np.ndarray) -> str:
        """
        Interpret the meaning of the dominant wavelet scale.
        
        Scale interpretation:
        - Scale 0 (approximation): Long-term trend
        - Scale 1 (detail 1): Medium-term cycles  
        - Scale 2+ (detail 2+): Short-term breakouts
        
        Args:
            dominant_scale: Index of dominant scale
            energy_ratios: Current energy ratios
            
        Returns:
            Breakout type: 'trend', 'cycle', 'breakout'
        """
        if dominant_scale == 0:
            return 'trend'  # Approximation coefficients - trend
        elif dominant_scale == 1:
            return 'cycle'  # First detail level - medium-term
        else:
            return 'breakout'  # Higher detail levels - short-term breakouts
    
    def calculate_trend_direction(self, prices: np.ndarray) -> float:
        """
        Calculate trend direction for signal conditioning.
        
        Returns:
            Trend direction [-1, 1] where 1 is uptrend, -1 is downtrend
        """
        if len(prices) < self.trend_lookback:
            return 0.0
            
        recent_prices = prices[-self.trend_lookback:]
        
        # Linear regression slope
        x = np.arange(len(recent_prices))
        slope = np.polyfit(x, recent_prices, 1)[0]
        
        # Normalize slope by price level
        price_level = np.mean(recent_prices)
        if price_level > 0:
            normalized_slope = slope / price_level
        else:
            normalized_slope = 0.0
            
        # Convert to trend direction [-1, 1]
        trend_direction = np.tanh(normalized_slope * 100)  # Scale and bound
        
        return np.clip(trend_direction, -1.0, 1.0)
    
    def generate_signal(self, data: pd.DataFrame) -> float:
        """
        Generate trading signal based on wavelet energy breakout analysis.
        
        Signal Logic:
        1. Decompose price series into wavelet scales
        2. Calculate energy ratios for each scale
        3. Detect energy concentration breakouts
        4. Generate signal based on breakout type and trend direction
        5. Apply signal decay for persistence
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Signal strength [-1, 1] where:
            - Positive: Long signal (breakout in trend direction)
            - Negative: Short signal (breakout against trend or reversal)
            - Zero: No signal
        """
        if len(data) < self.lookback:
            return 0.0
            
        # Use recent price data
        recent_data = data.tail(self.lookback)
        prices = recent_data['close'].values
        
        # Perform wavelet decomposition
        coeffs = self.calculate_wavelet_decomposition(prices)
        
        # Calculate energy ratios
        energy_ratios = self.calculate_energy_ratios(coeffs)
        
        # Update energy history
        self.energy_ratios_history.append(energy_ratios)
        if len(self.energy_ratios_history) > self.lookback:
            self.energy_ratios_history.pop(0)
            
        # Update individual scale histories
        for i, energy in enumerate(energy_ratios):
            if i < len(self.energy_history):
                self.energy_history[i].append(energy)
                if len(self.energy_history[i]) > self.lookback:
                    self.energy_history[i].pop(0)
        
        # Detect energy breakout
        is_breakout, breakout_strength, dominant_scale = self.detect_energy_breakout(energy_ratios)
        
        # Calculate base signal
        if is_breakout:
            # Interpret breakout type
            breakout_type = self.interpret_breakout_scale(dominant_scale, energy_ratios)
            
            # Calculate trend direction
            trend_direction = self.calculate_trend_direction(prices)
            
            # Generate signal based on breakout type and trend
            if breakout_type == 'breakout':
                # Short-term breakout - trade in trend direction
                base_signal = breakout_strength * np.sign(trend_direction) if trend_direction != 0 else breakout_strength * 0.5
            elif breakout_type == 'cycle':
                # Medium-term cycle - moderate signal
                base_signal = breakout_strength * 0.7 * np.sign(trend_direction) if trend_direction != 0 else 0.0
            else:  # trend
                # Long-term trend - conservative signal
                base_signal = breakout_strength * 0.5 * np.sign(trend_direction) if trend_direction != 0 else 0.0
                
        else:
            # No breakout detected
            base_signal = 0.0
        
        # Apply signal decay and update current signal
        self.current_signal = (
            self.signal_decay * self.current_signal + 
            (1 - self.signal_decay) * base_signal
        )
        
        return np.clip(self.current_signal, -1.0, 1.0)
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Return current strategy state for monitoring."""
        if len(self.energy_ratios_history) > 0:
            current_energy_ratios = self.energy_ratios_history[-1]
            dominant_scale = np.argmax(current_energy_ratios)
            dominant_energy = current_energy_ratios[dominant_scale]
            
            # Calculate energy concentration metric
            energy_concentration = np.max(current_energy_ratios) - np.mean(current_energy_ratios)
        else:
            current_energy_ratios = np.array([])
            dominant_scale = 0
            dominant_energy = 0.0
            energy_concentration = 0.0
        
        return {
            'current_energy_ratios': current_energy_ratios.tolist() if len(current_energy_ratios) > 0 else [],
            'dominant_scale': int(dominant_scale),
            'dominant_energy': float(dominant_energy),
            'energy_concentration': float(energy_concentration),
            'energy_history_length': len(self.energy_ratios_history),
            'current_signal': float(self.current_signal),
            'decomposition_levels': self.decomposition_levels
        }
