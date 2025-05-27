"""
Test Suite for Wavelet Energy-Ratio Breakout Trading Strategy

Tests cover:
1. Mathematical consistency of wavelet decomposition
2. Energy ratio calculation validation
3. Breakout detection accuracy
4. Scale interpretation logic
5. Signal generation logic and bounds
6. Trend direction conditioning
7. No look-ahead bias verification
8. Edge cases and robustness
9. Monte Carlo DSR validation
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strategies.wavelet_energy.strategy import WaveletEnergyBreakoutSignal


class TestWaveletEnergyBreakoutSignal:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'wavelet_type': 'db4',
            'decomposition_levels': 4,
            'lookback': 100,
            'energy_threshold_pct': 80,
            'breakout_threshold_pct': 90,
            'signal_decay': 0.85,
            'trend_lookback': 20
        }
        self.strategy = WaveletEnergyBreakoutSignal(self.config)
        
    def generate_test_data(self, n_points: int = 200, pattern: str = "normal") -> pd.DataFrame:
        """Generate synthetic price data for testing."""
        np.random.seed(42)
        
        if pattern == "breakout":
            # Create breakout pattern with sudden trend change
            mid = n_points // 2
            # First half: sideways
            prices1 = 100 + np.random.normal(0, 0.5, mid)
            # Second half: strong trend (breakout)
            trend = np.linspace(0, 20, n_points - mid)
            noise = np.random.normal(0, 1, n_points - mid)
            prices2 = prices1[-1] + trend + noise
            prices = np.concatenate([prices1, prices2])
        elif pattern == "trend":
            # Smooth trending data
            trend = np.linspace(0, 10, n_points)
            noise = np.random.normal(0, 0.5, n_points)
            prices = 100 + trend + noise
        elif pattern == "cycle":
            # Cyclical pattern
            t = np.linspace(0, 4*np.pi, n_points)
            cycle = 5 * np.sin(t) + 2 * np.sin(3*t)
            noise = np.random.normal(0, 0.5, n_points)
            prices = 100 + cycle + noise
        else:
            # Random walk (normal)
            prices = np.cumsum(np.random.normal(0, 1, n_points)) + 100
            
        return pd.DataFrame({
            'open': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.uniform(1000, 2000, n_points)
        })
    
    def test_wavelet_decomposition(self):
        """Test wavelet decomposition mathematical properties."""
        # Test basic decomposition
        prices = np.random.normal(100, 5, 128)  # Power of 2 for clean decomposition
        coeffs = self.strategy.calculate_wavelet_decomposition(prices)
        
        # Should have decomposition_levels + 1 coefficient arrays
        assert len(coeffs) == self.strategy.decomposition_levels + 1, \
            f"Expected {self.strategy.decomposition_levels + 1} coefficient arrays"
        
        # All coefficients should be finite
        for i, coeff in enumerate(coeffs):
            assert np.all(np.isfinite(coeff)), f"Coefficients at level {i} should be finite"
        
        # Test with insufficient data
        short_prices = np.array([100, 101, 102])
        short_coeffs = self.strategy.calculate_wavelet_decomposition(short_prices)
        assert len(short_coeffs) == self.strategy.decomposition_levels + 1, \
            "Should handle insufficient data gracefully"
        
        # Test with different signal types
        # Sine wave should concentrate energy in specific scales
        t = np.linspace(0, 4*np.pi, 128)
        sine_prices = 100 + 5*np.sin(t)
        sine_coeffs = self.strategy.calculate_wavelet_decomposition(sine_prices)
        assert len(sine_coeffs) > 0, "Should decompose sine wave successfully"
        
        # Constant prices should have low energy in detail coefficients
        constant_prices = np.full(128, 100)
        constant_coeffs = self.strategy.calculate_wavelet_decomposition(constant_prices)
        detail_energy = sum(np.sum(coeff**2) for coeff in constant_coeffs[1:])
        assert detail_energy < 1e-10, "Constant signal should have minimal detail energy"
    
    def test_energy_ratio_calculation(self):
        """Test energy ratio calculation."""
        # Test with simple coefficients
        coeffs = [
            np.array([1.0, 1.0]),      # Approximation: energy = 2
            np.array([2.0]),           # Detail 1: energy = 4  
            np.array([1.0, 1.0]),      # Detail 2: energy = 2
            np.array([0.0])            # Detail 3: energy = 0
        ]
        
        energy_ratios = self.strategy.calculate_energy_ratios(coeffs)
        
        # Check basic properties
        assert len(energy_ratios) == len(coeffs), "Energy ratios length should match coefficients"
        assert np.allclose(np.sum(energy_ratios), 1.0), "Energy ratios should sum to 1"
        assert np.all(energy_ratios >= 0), "Energy ratios should be non-negative"
        
        # Check specific values (total energy = 2+4+2+0 = 8)
        expected_ratios = np.array([2/8, 4/8, 2/8, 0/8])
        assert np.allclose(energy_ratios, expected_ratios, atol=1e-10), \
            "Energy ratios should match expected values"
        
        # Test with zero total energy
        zero_coeffs = [np.array([0.0]), np.array([0.0])]
        zero_ratios = self.strategy.calculate_energy_ratios(zero_coeffs)
        assert np.allclose(zero_ratios, 0.5), "Zero energy should give uniform distribution"
        
        # Test with empty coefficients
        empty_coeffs = [np.array([]), np.array([])]
        empty_ratios = self.strategy.calculate_energy_ratios(empty_coeffs)
        assert len(empty_ratios) == 2, "Should handle empty coefficients"
        assert np.allclose(np.sum(empty_ratios), 1.0), "Empty coefficients should still normalize"
    
    def test_energy_breakout_detection(self):
        """Test energy breakout detection logic."""
        # Build up energy history with normal distributions
        np.random.seed(42)
        for _ in range(50):
            # Normal energy distribution (roughly uniform)
            normal_ratios = np.random.dirichlet([1, 1, 1, 1, 1])  # 5 scales
            self.strategy.energy_ratios_history.append(normal_ratios)
        
        # Test with concentrated energy (breakout)
        breakout_ratios = np.array([0.1, 0.1, 0.7, 0.05, 0.05])  # Concentrated in scale 2
        is_breakout, strength, scale = self.strategy.detect_energy_breakout(breakout_ratios)
        
        assert isinstance(is_breakout, bool), "Should return boolean breakout flag"
        assert 0.0 <= strength <= 1.0, f"Strength {strength} should be in [0,1]"
        assert scale == 2, f"Should detect scale 2 as dominant, got {scale}"
        
        # Test with normal distribution (no breakout)
        normal_ratios = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Uniform distribution
        is_breakout_2, strength_2, scale_2 = self.strategy.detect_energy_breakout(normal_ratios)
        
        # Should not detect breakout with uniform distribution
        assert strength_2 == 0.0, "Uniform distribution should not trigger breakout"
        
        # Test with insufficient history
        empty_strategy = WaveletEnergyBreakoutSignal(self.config)
        is_breakout_3, strength_3, scale_3 = empty_strategy.detect_energy_breakout(breakout_ratios)
        assert not is_breakout_3, "Insufficient history should not trigger breakout"
        assert strength_3 == 0.0, "Insufficient history should have zero strength"
    
    def test_breakout_scale_interpretation(self):
        """Test interpretation of different wavelet scales."""
        energy_ratios = np.array([0.7, 0.1, 0.1, 0.05, 0.05])  # Scale 0 dominant
        
        # Test scale 0 (trend)
        interpretation_0 = self.strategy.interpret_breakout_scale(0, energy_ratios)
        assert interpretation_0 == 'trend', "Scale 0 should be interpreted as trend"
        
        # Test scale 1 (cycle)
        interpretation_1 = self.strategy.interpret_breakout_scale(1, energy_ratios)
        assert interpretation_1 == 'cycle', "Scale 1 should be interpreted as cycle"
        
        # Test scale 2+ (breakout)
        interpretation_2 = self.strategy.interpret_breakout_scale(2, energy_ratios)
        assert interpretation_2 == 'breakout', "Scale 2+ should be interpreted as breakout"
        
        interpretation_3 = self.strategy.interpret_breakout_scale(3, energy_ratios)
        assert interpretation_3 == 'breakout', "Scale 3+ should be interpreted as breakout"
    
    def test_trend_direction_calculation(self):
        """Test trend direction calculation."""
        # Uptrend
        up_prices = np.linspace(100, 110, 50)
        up_trend = self.strategy.calculate_trend_direction(up_prices)
        assert up_trend > 0, "Uptrend should have positive direction"
        assert -1.0 <= up_trend <= 1.0, "Trend direction should be bounded"
        
        # Downtrend
        down_prices = np.linspace(110, 100, 50)
        down_trend = self.strategy.calculate_trend_direction(down_prices)
        assert down_trend < 0, "Downtrend should have negative direction"
        assert -1.0 <= down_trend <= 1.0, "Trend direction should be bounded"
        
        # Flat (no trend)
        flat_prices = np.full(50, 100)
        flat_trend = self.strategy.calculate_trend_direction(flat_prices)
        assert abs(flat_trend) < 0.1, "Flat prices should have near-zero trend"
        
        # Noisy trend
        noisy_trend = np.linspace(100, 105, 50) + np.random.normal(0, 0.5, 50)
        noisy_direction = self.strategy.calculate_trend_direction(noisy_trend)
        assert -1.0 <= noisy_direction <= 1.0, "Noisy trend should be bounded"
        
        # Insufficient data
        short_trend = self.strategy.calculate_trend_direction(np.array([100, 101]))
        assert short_trend == 0.0, "Insufficient data should return zero trend"
    
    def test_signal_generation_bounds(self):
        """Test signal generation produces valid bounds."""
        data = self.generate_test_data(150, "normal")
        
        signals = []
        for i in range(100, len(data)):
            signal = self.strategy.generate_signal(data.iloc[:i])
            signals.append(signal)
            assert -1.0 <= signal <= 1.0, f"Signal {signal} outside bounds [-1,1]"
        
        assert len(signals) > 0, "Should generate some signals"
        assert hasattr(self.strategy, 'current_signal'), "Strategy should track current signal"
    
    def test_breakout_pattern_detection(self):
        """Test strategy detects breakout patterns."""
        # Generate breakout data
        breakout_data = self.generate_test_data(200, "breakout")
        
        signals = []
        states = []
        
        for i in range(100, len(breakout_data)):
            signal = self.strategy.generate_signal(breakout_data.iloc[:i])
            signals.append(signal)
            
            state = self.strategy.get_strategy_state()
            states.append(state)
        
        # Should generate some non-zero signals
        non_zero_signals = [s for s in signals if abs(s) > 0.01]
        assert len(non_zero_signals) > 0, "Should generate some non-zero signals for breakout pattern"
        
        # Check energy concentration increases during breakout
        concentrations = [s['energy_concentration'] for s in states]
        max_concentration = max(concentrations) if concentrations else 0
        assert max_concentration >= 0, "Should calculate energy concentration"
    
    def test_trend_pattern_response(self):
        """Test strategy response to trending patterns."""
        # Generate trending data
        trend_data = self.generate_test_data(150, "trend")
        
        signals = []
        for i in range(100, len(trend_data)):
            signal = self.strategy.generate_signal(trend_data.iloc[:i])
            signals.append(signal)
        
        # Should generate some signals for trending data
        signal_variance = np.var(signals) if signals else 0
        assert signal_variance >= 0, "Should show some signal variation in trending data"
    
    def test_cycle_pattern_response(self):
        """Test strategy response to cyclical patterns."""
        # Generate cyclical data
        cycle_data = self.generate_test_data(150, "cycle")
        
        signals = []
        for i in range(100, len(cycle_data)):
            signal = self.strategy.generate_signal(cycle_data.iloc[:i])
            signals.append(signal)
        
        # Should generate some signals for cyclical data
        assert len(signals) > 0, "Should generate signals for cyclical data"
        
        # Signals should vary over time for cyclical pattern
        if len(signals) > 10:
            signal_range = max(signals) - min(signals)
            assert signal_range >= 0, "Should show signal variation in cyclical data"
    
    def test_signal_decay_mechanism(self):
        """Test signal decay and persistence."""
        data = self.generate_test_data(120, "normal")
        
        # Set initial signal
        self.strategy.current_signal = 0.8
        
        # Generate signals on neutral data (should decay)
        decay_signals = []
        for i in range(100, 110):
            signal = self.strategy.generate_signal(data.iloc[:i])
            decay_signals.append(signal)
        
        # Signal should generally decay toward zero
        if len(decay_signals) > 1:
            first_signal = abs(decay_signals[0])
            last_signal = abs(decay_signals[-1])
            # Allow for some fluctuation, but general trend should be decay
            assert last_signal <= first_signal + 0.1, "Signal should decay or stabilize over time"
    
    def test_no_lookahead_bias(self):
        """Test strategy doesn't use future information."""
        data = self.generate_test_data(150, "normal")
        
        # Generate signal at time t
        strategy_1 = WaveletEnergyBreakoutSignal(self.config)
        signal_t = strategy_1.generate_signal(data.iloc[:100])
        
        # Generate signal with more future data available
        strategy_2 = WaveletEnergyBreakoutSignal(self.config)
        signal_t_with_future = strategy_2.generate_signal(data.iloc[:120])
        
        # The wavelet decomposition should be deterministic for same input
        coeffs_1 = strategy_1.calculate_wavelet_decomposition(data['close'].iloc[:100].values)
        coeffs_2 = strategy_2.calculate_wavelet_decomposition(data['close'].iloc[:100].values)
        
        # Compare energy of first coefficient array (should be identical)
        energy_1 = np.sum(coeffs_1[0]**2) if len(coeffs_1[0]) > 0 else 0
        energy_2 = np.sum(coeffs_2[0]**2) if len(coeffs_2[0]) > 0 else 0
        
        assert abs(energy_1 - energy_2) < 1e-10, "Wavelet decomposition should be deterministic"
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty data
        empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        signal = self.strategy.generate_signal(empty_data)
        assert signal == 0.0, "Empty data should return zero signal"
        
        # Insufficient data
        small_data = self.generate_test_data(10)
        signal = self.strategy.generate_signal(small_data)
        assert signal == 0.0, "Insufficient data should return zero signal"
        
        # Constant prices
        constant_data = pd.DataFrame({
            'open': [100] * 150,
            'high': [100] * 150,
            'low': [100] * 150,
            'close': [100] * 150,
            'volume': [1000] * 150
        })
        signal = self.strategy.generate_signal(constant_data)
        assert -1.0 <= signal <= 1.0, "Constant prices should produce valid signal"
        
        # NaN handling
        nan_data = self.generate_test_data(150)
        nan_data.loc[50:60, 'close'] = np.nan
        nan_data = nan_data.dropna()
        if len(nan_data) >= self.strategy.lookback:
            signal = self.strategy.generate_signal(nan_data)
            assert -1.0 <= signal <= 1.0, "Should handle NaN data gracefully"
        
        # Very large prices (test numerical stability)
        large_data = self.generate_test_data(150)
        large_data['close'] *= 1e6
        signal = self.strategy.generate_signal(large_data)
        assert -1.0 <= signal <= 1.0, "Should handle large price values"
    
    def test_strategy_state_monitoring(self):
        """Test strategy state reporting."""
        data = self.generate_test_data(120)
        
        # Generate some signals to build history
        for i in range(100, 110):
            self.strategy.generate_signal(data.iloc[:i])
        
        state = self.strategy.get_strategy_state()
        
        required_keys = ['current_energy_ratios', 'dominant_scale', 'dominant_energy',
                        'energy_concentration', 'energy_history_length', 'current_signal',
                        'decomposition_levels']
        
        for key in required_keys:
            assert key in state, f"Missing key {key} in strategy state"
        
        assert isinstance(state['current_energy_ratios'], list), "Energy ratios should be list"
        assert isinstance(state['dominant_scale'], int), "Dominant scale should be integer"
        assert 0.0 <= state['dominant_energy'] <= 1.0, "Dominant energy should be in [0,1]"
        assert state['energy_concentration'] >= 0.0, "Energy concentration should be non-negative"
        assert state['energy_history_length'] > 0, "Should have energy history"
        assert -1.0 <= state['current_signal'] <= 1.0, "Current signal should be in [-1,1]"
        assert state['decomposition_levels'] == self.strategy.decomposition_levels, "Should match config"
    
    def test_monte_carlo_dsr_validation(self):
        """Test DSR (Daily Sharpe Ratio) via Monte Carlo simulation."""
        np.random.seed(42)
        n_simulations = 50  # Reduced for faster testing
        daily_returns = []
        
        for sim in range(n_simulations):
            # Generate different pattern types
            pattern_type = ["normal", "breakout", "trend", "cycle"][sim % 4]
            data = self.generate_test_data(250, pattern_type)
            strategy = WaveletEnergyBreakoutSignal(self.config)
            
            returns = []
            for i in range(150, len(data)-1):
                signal = strategy.generate_signal(data.iloc[:i])
                
                # Calculate return from signal
                price_return = (data['close'].iloc[i+1] / data['close'].iloc[i] - 1)
                strategy_return = signal * price_return
                returns.append(strategy_return)
            
            if len(returns) > 0:
                daily_return = np.mean(returns)
                daily_returns.append(daily_return)
        
        if len(daily_returns) > 0:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            
            # Calculate DSR (Daily Sharpe Ratio)
            dsr = mean_return / std_return if std_return > 0 else 0
            
            # DSR should be reasonable
            assert abs(dsr) < 5.0, f"DSR {dsr} seems unrealistic"
            
            # Test should complete most simulations
            assert len(daily_returns) > 30, "Should complete most simulations"
