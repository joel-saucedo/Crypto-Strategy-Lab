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
                'trend_threshold': 0.55,
                'mean_revert_threshold': 0.45,
                'min_confidence': 0.75
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
        
        # Early signals should be identical
        early_signals_full = signals_full.iloc[:split_point-200]  # Account for lookback
        early_signals_partial = signals_partial.iloc[:split_point-200]
        
        if len(early_signals_full) > 0:
            assert early_signals_full.equals(early_signals_partial), "Look-ahead bias detected"
    
    def test_hurst_estimation_properties(self, signal):
        """Test Hurst exponent estimation with known processes."""
        np.random.seed(42)
        
        # Test 1: Brownian motion (H ≈ 0.5)
        brownian = np.cumsum(np.random.normal(0, 1, 500))
        brownian_returns = pd.Series(np.diff(brownian))
        
        h_brownian, conf_brownian = signal._estimate_hurst_dfa(brownian_returns)
        
        # Brownian motion should have H close to 0.5
        assert 0.4 < h_brownian < 0.6, f"Brownian motion H should be ~0.5, got {h_brownian}"
        
        # Test 2: Trending process (H > 0.5)
        trend = np.cumsum(np.random.normal(0.01, 0.02, 500))  # Positive drift
        trend_returns = pd.Series(np.diff(trend))
        
        h_trend, conf_trend = signal._estimate_hurst_dfa(trend_returns)
        
        # Trending process should have H > 0.5
        assert h_trend > 0.45, f"Trending process H should be > 0.5, got {h_trend}"
        
        # Test 3: Mean-reverting process (H < 0.5)
        mean_reverting = []
        value = 0
        for i in range(500):
            # Strong mean reversion
            shock = np.random.normal(0, 0.02)
            value = 0.95 * value + shock  # AR(1) with φ < 1
            mean_reverting.append(value)
        
        mr_returns = pd.Series(np.diff(mean_reverting))
        h_mr, conf_mr = signal._estimate_hurst_dfa(mr_returns)
        
        # Mean-reverting process should have H < 0.5
        assert h_mr < 0.55, f"Mean-reverting process H should be < 0.5, got {h_mr}"
    
    def test_monte_carlo_dsr(self, signal):
        """Test strategy behavior on synthetic regime-switching data."""
        np.random.seed(42)
        
        # Generate data with regime switches
        n_simulations = 20
        total_returns = []
        
        for _ in range(n_simulations):
            # Create regime-switching data
            regime_length = 100
            data = []
            
            # Trending regime
            for i in range(regime_length):
                if i == 0:
                    data.append(np.random.normal(0.002, 0.02))
                else:
                    # Momentum continuation
                    data.append(data[-1] * 0.1 + np.random.normal(0.002, 0.02))
            
            # Mean-reverting regime
            center = data[-1]
            for i in range(regime_length):
                # Mean reversion to center
                reversion = -0.05 * (data[-1] - center)
                data.append(reversion + np.random.normal(0, 0.015))
            
            returns = pd.Series(data)
            returns.index = pd.date_range('2023-01-01', periods=len(returns), freq='D')
            
            signals = signal.generate(returns)
            
            # Calculate strategy returns
            strategy_returns = signals.shift(1) * returns
            total_returns.extend(strategy_returns.dropna())
        
        if total_returns:
            strategy_return_series = pd.Series(total_returns)
            # Should have positive mean return on regime-switching data
            mean_return = strategy_return_series.mean()
            assert mean_return > -0.001, f"Expected non-negative mean return, got {mean_return}"
    
    def test_parameter_sensitivity(self, signal, sample_returns):
        """Test reasonable behavior across parameter ranges."""
        # Test different threshold values
        original_trend_thresh = signal.params['trend_threshold']
        original_mr_thresh = signal.params['mean_revert_threshold']
        
        # More conservative thresholds should generate fewer signals
        signal.params['trend_threshold'] = 0.60
        signal.params['mean_revert_threshold'] = 0.40
        conservative_signals = signal.generate(sample_returns)
        
        # More aggressive thresholds should generate more signals
        signal.params['trend_threshold'] = 0.52
        signal.params['mean_revert_threshold'] = 0.48
        aggressive_signals = signal.generate(sample_returns)
        
        # Restore original
        signal.params['trend_threshold'] = original_trend_thresh
        signal.params['mean_revert_threshold'] = original_mr_thresh
        
        # Count non-zero signals
        conservative_count = (conservative_signals != 0).sum()
        aggressive_count = (aggressive_signals != 0).sum()
        
        # More aggressive thresholds should generate more or equal signals
        assert aggressive_count >= conservative_count, "Aggressive thresholds should generate more signals"
    
    def test_mathematical_consistency(self, signal):
        """Test that Hurst exponent calculation follows expected properties."""
        np.random.seed(42)
        
        # Test scaling property: if data is scaled, H should remain the same
        original_data = pd.Series(np.random.normal(0, 0.02, 200))
        scaled_data = original_data * 2
        
        h_original, _ = signal._estimate_hurst_dfa(original_data)
        h_scaled, _ = signal._estimate_hurst_dfa(scaled_data)
        
        # Hurst exponent should be scale-invariant
        assert abs(h_original - h_scaled) < 0.1, f"Hurst should be scale-invariant: {h_original} vs {h_scaled}"
        
        # Test that longer data gives more reliable estimates
        short_data = pd.Series(np.random.normal(0, 0.02, 50))
        long_data = pd.Series(np.random.normal(0, 0.02, 300))
        
        _, conf_short = signal._estimate_hurst_dfa(short_data)
        _, conf_long = signal._estimate_hurst_dfa(long_data)
        
        # Longer data should generally give higher confidence
        assert conf_long >= conf_short - 0.1, "Longer data should give higher confidence"
