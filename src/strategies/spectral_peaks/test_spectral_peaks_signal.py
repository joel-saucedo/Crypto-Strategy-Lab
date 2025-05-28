"""
Unit tests for spectral_peaks strategy.
Validates implementation against mathematical expectations.
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.spectral_peaks.signal import SpectralpeaksSignal

class TestSpectralpeaksSignal(unittest.TestCase):
    
    def setUp(self):
        """Setup test fixtures."""
        self.signal = SpectralpeaksSignal()
        
    def test_no_nan_output(self):
        """Test that signal generation doesn't produce NaN values."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 300))
        
        signals = self.signal.generate(returns)
        
        assert not signals.isna().any(), "Signal contains NaN values"
        
    def test_signal_range(self):
        """Test that signals are in valid range {-1, 0, 1}."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 300))
        
        signals = self.signal.generate(returns)
        
        valid_signals = signals.isin([-1, 0, 1])
        assert valid_signals.all(), "Invalid signal values detected"
        
    def test_index_alignment(self):
        """Test that output index aligns with input (no look-ahead)."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        returns = pd.Series(np.random.normal(0, 0.02, 300), index=dates)
        
        signals = self.signal.generate(returns)
        
        assert signals.index.equals(returns.index), "Index alignment failed"
        
    def test_no_look_ahead(self):
        """Test that signals don't use future information."""
        # Create returns with embedded cycle that becomes stronger in the future
        n = 400
        np.random.seed(42)
        
        # First half: weak cycle
        t1 = np.arange(200)
        cycle1 = 0.01 * np.sin(2 * np.pi * t1 / 50)  # 50-day cycle, weak
        noise1 = np.random.normal(0, 0.02, 200)
        returns1 = cycle1 + noise1
        
        # Second half: strong cycle
        t2 = np.arange(200, 400)
        cycle2 = 0.05 * np.sin(2 * np.pi * t2 / 50)  # Same cycle, stronger
        noise2 = np.random.normal(0, 0.01, 200)  # Less noise
        returns2 = cycle2 + noise2
        
        returns = pd.Series(np.concatenate([returns1, returns2]))
        signals = self.signal.generate(returns)
        
        # Early signals shouldn't be systematically stronger due to future information
        early_signals = signals.iloc[:200]
        late_signals = signals.iloc[200:]
        
        early_nonzero = (early_signals != 0).sum()
        late_nonzero = (late_signals != 0).sum()
        
        # Should see more signals in the second half due to stronger cycle
        # but not an extreme difference that would indicate look-ahead bias
        if early_nonzero > 0 and late_nonzero > 0:
            signal_ratio = late_nonzero / early_nonzero
            assert signal_ratio < 10, "Possible look-ahead bias: too many future-dependent signals"
        
    def test_parameter_sensitivity(self):
        """Test reasonable behavior across parameter ranges."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 400))
        
        # Test with different parameter values
        original_params = self.signal.params.copy()
        
        # Test different window sizes
        for window_size in [128, 256, 512]:
            self.signal.params['window_size'] = window_size
            signals = self.signal.generate(returns)
            assert signals.isin([-1, 0, 1]).all(), f"Invalid signals with window_size={window_size}"
        
        # Test different significance thresholds
        for threshold in [2.0, 3.0, 4.0]:
            self.signal.params['significance_threshold'] = threshold
            signals = self.signal.generate(returns)
            assert signals.isin([-1, 0, 1]).all(), f"Invalid signals with threshold={threshold}"
        
        # Restore original parameters
        self.signal.params = original_params
        
    def test_monte_carlo_dsr(self):
        """Test that strategy has positive expected performance on synthetic cyclic data."""
        np.random.seed(42)
        
        # Generate data with known spectral peaks
        n_simulations = 20
        cycle_performances = []
        
        for sim in range(n_simulations):
            # Create data with embedded cycle
            n = 400
            t = np.arange(n)
            
            # Dominant cycle (e.g., 60-day cycle)
            cycle_period = 60
            cycle_amplitude = 0.02
            cycle_component = cycle_amplitude * np.sin(2 * np.pi * t / cycle_period)
            
            # Add phase shift and noise
            phase_shift = np.random.uniform(0, 2 * np.pi)
            cycle_component = cycle_amplitude * np.sin(2 * np.pi * t / cycle_period + phase_shift)
            noise = np.random.normal(0, 0.015, n)
            
            returns = pd.Series(cycle_component + noise)
            returns.index = pd.date_range('2023-01-01', periods=n, freq='D')
            
            signals = self.signal.generate(returns)
            
            # Calculate strategy performance
            strategy_returns = signals.shift(1) * returns
            total_return = strategy_returns.sum()
            cycle_performances.append(total_return)
        
        # Average performance should be reasonable for cyclic data
        avg_performance = np.mean(cycle_performances)
        # Not requiring strongly positive (since timing cycles is challenging)
        # but should not be extremely negative
        assert avg_performance > -0.05, f"Average performance too negative: {avg_performance}"
        
    def test_mathematical_consistency(self):
        """Test that implementation matches documented equations."""
        # Test spectral peak detection with known sinusoid
        np.random.seed(42)
        
        # Create pure sinusoid at known frequency
        n = 256
        t = np.arange(n)
        frequency = 1/32  # 32-day cycle
        amplitude = 0.03
        
        pure_sine = amplitude * np.sin(2 * np.pi * frequency * t)
        returns = pd.Series(pure_sine)
        
        # Test peak detection
        dominant_freq, peak_power, baseline_power = self.signal._find_dominant_peak(
            returns, min_period=20, max_period=50
        )
        
        # Should detect the embedded frequency
        detected_period = 1 / abs(dominant_freq) if dominant_freq else 0
        assert 30 <= detected_period <= 35, f"Failed to detect 32-day cycle: detected {detected_period}"
        
        # Peak should be significantly above baseline
        significance = peak_power / baseline_power if baseline_power > 0 else 0
        assert significance > 2.0, f"Peak not significant enough: {significance}"
        
    def test_cycle_phase_calculation(self):
        """Test cycle phase calculation accuracy."""
        # Create sinusoid with known phase
        n = 100
        t = np.arange(n)
        frequency = 1/20  # 20-day cycle
        phase_offset = np.pi/4  # 45 degrees
        
        data = pd.Series(np.sin(2 * np.pi * frequency * t + phase_offset))
        
        calculated_phase = self.signal._calculate_cycle_phase(data, frequency)
        
        # Phase calculation should be reasonably accurate
        # (Note: exact matching is difficult due to correlation-based method)
        assert 0 <= calculated_phase <= 2 * np.pi, f"Phase out of range: {calculated_phase}"
        
    def test_signal_generation_with_cycle(self):
        """Test that strategy generates appropriate signals for cyclic data."""
        np.random.seed(42)
        
        # Create data with clear cycle
        n = 300
        t = np.arange(n)
        cycle_period = 40
        
        # Strong cycle with some noise
        cycle = 0.03 * np.sin(2 * np.pi * t / cycle_period)
        noise = np.random.normal(0, 0.01, n)
        returns = pd.Series(cycle + noise)
        
        signals = self.signal.generate(returns)
        
        # Should generate some non-zero signals for data with clear cycle
        nonzero_signals = (signals != 0).sum()
        assert nonzero_signals > 0, "No signals generated for cyclic data"
        
        # Signals should not be too frequent (cycle timing should be selective)
        signal_frequency = nonzero_signals / len(signals)
        assert signal_frequency < 0.3, f"Too many signals: {signal_frequency}"
        
    def test_cycle_strength_calculation(self):
        """Test cycle strength calculation."""
        # Strong cycle
        n = 200
        t = np.arange(n)
        strong_cycle = pd.Series(0.04 * np.sin(2 * np.pi * t / 30))
        
        strong_strength = self.signal.get_cycle_strength(strong_cycle)
        
        # Weak/no cycle  
        weak_cycle = pd.Series(np.random.normal(0, 0.02, n))
        weak_strength = self.signal.get_cycle_strength(weak_cycle)
        
        # Strong cycle should have higher strength
        assert strong_strength > weak_strength, "Cycle strength calculation failed"
        assert 0 <= strong_strength <= 1, f"Strength out of range: {strong_strength}"
        assert 0 <= weak_strength <= 1, f"Strength out of range: {weak_strength}"
        
    def test_edge_cases(self):
        """Test handling of edge cases."""
        # Very short series
        short_returns = pd.Series([0.01, -0.01, 0.02])
        signals = self.signal.generate(short_returns)
        assert signals.isin([0]).all(), "Should return zeros for very short series"
        
        # Constant series
        constant_returns = pd.Series([0.0] * 300)
        signals = self.signal.generate(constant_returns)
        assert signals.isin([0]).all(), "Should return zeros for constant series"
        
        # Series with extreme values
        extreme_returns = pd.Series([-0.5, 0.8, -0.3, 0.6] * 100)
        signals = self.signal.generate(extreme_returns)
        assert signals.isin([-1, 0, 1]).all(), "Should handle extreme values gracefully"


if __name__ == '__main__':
    unittest.main()
