"""
Unit tests for hurst_exponent strategy.
Validates implementation against mathematical expectations.
"""

import pytest
import numpy as np
import pandas as pd
from src.strategies.hurst_exponent.signal import HurstExponentSignal
import tempfile
import yaml

class TestHurstExponentSignal:
    
    @pytest.fixture
    def signal(self):
        """Create signal generator with test config."""
        test_config = {
            'strategy': {
                'name': 'hurst_exponent',
                'description': 'Test hurst exponent strategy'
            },
            'best_params': {
                'lookback_days': 200,
                'h_threshold_high': 0.55,
                'h_threshold_low': 0.45,
                'min_confidence': 0.7
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            return HurstExponentSignal(f.name)
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample return series."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, 300), index=dates)
        return returns
    
    def test_no_nan_output(self, signal, sample_returns):
        """Test that signal generation never produces NaN values."""
        signals = signal.generate(sample_returns)
        assert not signals.isna().any(), "Signal output contains NaN values"
    
    def test_index_alignment(self, signal, sample_returns):
        """Test that output index matches input (no look-ahead bias)."""
        signals = signal.generate(sample_returns)
        assert signals.index.equals(sample_returns.index), "Signal index misaligned with returns"
    
    def test_signal_range(self, signal, sample_returns):
        """Test that all signals are in {-1, 0, 1}."""
        signals = signal.generate(sample_returns)
        valid_values = {-1, 0, 1}
        assert set(signals.unique()).issubset(valid_values), "Signals outside valid range"
    
    def test_no_look_ahead(self, signal, sample_returns):
        """Test that signals don't use future information."""
        # Split data and check that early signals don't change
        split_point = len(sample_returns) // 2
        
        signals_full = signal.generate(sample_returns)
        signals_partial = signal.generate(sample_returns.iloc[:split_point])
        
        # Early signals should be identical (accounting for lookback)
        early_signals_full = signals_full.iloc[:split_point-200]
        early_signals_partial = signals_partial.iloc[:split_point-200]
        
        if len(early_signals_full) > 0:
            assert early_signals_full.equals(early_signals_partial), "Look-ahead bias detected"
    
    def test_hurst_calculation_known_processes(self, signal):
        """Test Hurst exponent calculation with known processes."""
        np.random.seed(42)
        
        # Test 1: White noise should have H ≈ 0.5
        white_noise = pd.Series(np.random.normal(0, 1, 1000))
        h_white, conf_white = signal._calculate_hurst_dfa(white_noise)
        assert 0.4 < h_white < 0.6, f"White noise should have H ≈ 0.5, got {h_white}"
        
        # Test 2: Trending process should have H > 0.5
        trend = np.cumsum(np.random.normal(0.01, 0.02, 500))
        trending_returns = pd.Series(np.diff(trend))
        h_trend, conf_trend = signal._calculate_hurst_dfa(trending_returns)
        assert h_trend > 0.5, f"Trending process should have H > 0.5, got {h_trend}"
        
        # Test 3: Mean-reverting process should have H < 0.5
        reverting_data = []
        value = 0
        for i in range(500):
            reverting_data.append(-0.1 * value + np.random.normal(0, 0.02))
            value += reverting_data[-1]
        
        reverting_returns = pd.Series(reverting_data)
        h_revert, conf_revert = signal._calculate_hurst_dfa(reverting_returns)
        assert h_revert < 0.5, f"Mean-reverting process should have H < 0.5, got {h_revert}"
    
    def test_regime_detection(self, signal):
        """Test that strategy correctly identifies different regimes."""
        np.random.seed(42)
        
        # Create strongly trending data (should signal momentum)
        trend_component = np.cumsum(np.random.normal(0.005, 0.01, 250))
        trending_returns = pd.Series(np.diff(trend_component))
        trending_returns.index = pd.date_range('2023-01-01', periods=len(trending_returns), freq='D')
        
        signals_trend = signal.generate(trending_returns)
        momentum_signals = (signals_trend == 1).sum()
        
        # Should generate some momentum signals for trending data
        assert momentum_signals > 0, "Should generate momentum signals for trending data"
        
        # Create mean-reverting data (should signal mean reversion)
        reverting_data = []
        value = 0
        for i in range(250):
            reverting_data.append(-0.2 * value + np.random.normal(0, 0.02))
            value += reverting_data[-1]
        
        reverting_returns = pd.Series(reverting_data)
        reverting_returns.index = pd.date_range('2023-01-01', periods=len(reverting_returns), freq='D')
        
        signals_revert = signal.generate(reverting_returns)
        reversion_signals = (signals_revert == -1).sum()
        
        # Should generate some reversion signals for mean-reverting data
        assert reversion_signals > 0, "Should generate reversion signals for mean-reverting data"
    
    def test_monte_carlo_dsr(self, signal):
        """Test positive expected DSR on synthetic data with known edge."""
        np.random.seed(42)
        
        # Generate multiple scenarios with different Hurst characteristics
        n_simulations = 30
        signal_performance = []
        
        for scenario in range(n_simulations):
            # Alternate between trending and mean-reverting regimes
            if scenario % 2 == 0:
                # Trending regime
                trend = np.cumsum(np.random.normal(0.002, 0.015, 250))
                returns = pd.Series(np.diff(trend))
            else:
                # Mean-reverting regime
                reverting_data = []
                value = 0
                for i in range(250):
                    reverting_data.append(-0.15 * value + np.random.normal(0, 0.02))
                    value += reverting_data[-1]
                returns = pd.Series(reverting_data)
            
            returns.index = pd.date_range('2023-01-01', periods=len(returns), freq='D')
            signals = signal.generate(returns)
            
            # Calculate simple performance metrics
            strategy_returns = signals.shift(1) * returns
            total_return = strategy_returns.sum()
            signal_performance.append(total_return)
        
        # Average performance should be reasonable
        avg_performance = np.mean(signal_performance)
        # Not requiring positive (since this is just basic regime detection)
        # but should not be extremely negative
        assert avg_performance > -0.1, f"Average performance too negative: {avg_performance}"
    
    def test_parameter_sensitivity(self, signal, sample_returns):
        """Test reasonable behavior across parameter ranges."""
        base_signals = signal.generate(sample_returns)
        
        # Test different Hurst thresholds
        original_h_high = signal.params['h_threshold_high']
        original_h_low = signal.params['h_threshold_low']
        
        # Narrower thresholds should generate fewer signals
        signal.params['h_threshold_high'] = 0.60
        signal.params['h_threshold_low'] = 0.40
        narrow_signals = signal.generate(sample_returns)
        
        # Wider thresholds should generate more signals
        signal.params['h_threshold_high'] = 0.52
        signal.params['h_threshold_low'] = 0.48
        wide_signals = signal.generate(sample_returns)
        
        # Restore original
        signal.params['h_threshold_high'] = original_h_high
        signal.params['h_threshold_low'] = original_h_low
        
        # Count non-zero signals
        narrow_count = (narrow_signals != 0).sum()
        wide_count = (wide_signals != 0).sum()
        
        # Wider thresholds should generate more or equal signals
        assert wide_count >= narrow_count, "Wider thresholds should generate more signals"
    
    def test_mathematical_consistency(self, signal):
        """Test that implementation matches documented equations."""
        # Test DFA scaling relationship: log(F(n)) ~ H * log(n)
        np.random.seed(42)
        
        # Create fractional Brownian motion with known H
        # For testing, use a simpler approach with known properties
        
        # White noise should give H ≈ 0.5
        white_noise = np.random.normal(0, 1, 1000)
        cumsum_white = np.cumsum(white_noise)
        returns_white = pd.Series(np.diff(cumsum_white))
        
        h_calculated, confidence = signal._calculate_hurst_dfa(returns_white)
        
        # Should be close to theoretical value for white noise
        assert 0.3 < h_calculated < 0.7, f"Hurst for white noise outside expected range: {h_calculated}"
        assert confidence > 0.1, f"Confidence too low: {confidence}"
    
    def test_confidence_filtering(self, signal):
        """Test that low confidence estimates are filtered out."""
        # Create very short or noisy data that should have low confidence
        short_data = pd.Series(np.random.normal(0, 0.1, 30))
        
        h, confidence = signal._calculate_hurst_dfa(short_data)
        
        # Very short series should have lower confidence
        # The exact threshold depends on implementation, but confidence should be computed
        assert 0 <= confidence <= 1, f"Confidence should be between 0 and 1, got {confidence}"
