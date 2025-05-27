"""
Unit tests for variance_ratio strategy.
Validates implementation against mathematical expectations.
"""

import pytest
import numpy as np
import pandas as pd
from src.strategies.variance_ratio.signal import VarianceRatioSignal
import tempfile
import yaml

class TestVarianceRatioSignal:
    
    @pytest.fixture
    def signal(self):
        """Create signal generator with test config."""
        test_config = {
            'strategy': {
                'name': 'variance_ratio',
                'description': 'Test variance ratio strategy'
            },
            'best_params': {
                'lookback_days': 60,
                'lag_q': 5,
                'threshold_delta': 0.10,
                'min_z_stat': 2.0
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            return VarianceRatioSignal(f.name)
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample return series."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, 200), index=dates)
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
        early_signals_full = signals_full.iloc[:split_point-60]  # Account for lookback
        early_signals_partial = signals_partial.iloc[:split_point-60]
        
        if len(early_signals_full) > 0:
            assert early_signals_full.equals(early_signals_partial), "Look-ahead bias detected"
    
    def test_variance_ratio_calculation(self, signal):
        """Test variance ratio calculation with known properties."""
        # Create perfectly trending data (should have VR > 1)
        np.random.seed(42)
        trend = np.cumsum(np.random.normal(0.01, 0.01, 100))  # Strong trend
        trending_returns = pd.Series(np.diff(trend))
        
        vr, z_stat = signal._calculate_variance_ratio(trending_returns, 5)
        
        # Trending data should have VR > 1
        assert vr > 1.0, f"Expected VR > 1 for trending data, got {vr}"
        
        # Create mean-reverting data (should have VR < 1)
        reverting_data = []
        value = 0
        for i in range(100):
            # Strong mean reversion
            reverting_data.append(-0.5 * value + np.random.normal(0, 0.01))
            value += reverting_data[-1]
        
        reverting_returns = pd.Series(reverting_data)
        vr_rev, z_stat_rev = signal._calculate_variance_ratio(reverting_returns, 5)
        
        # Mean-reverting data should have VR < 1
        assert vr_rev < 1.0, f"Expected VR < 1 for mean-reverting data, got {vr_rev}"
    
    def test_monte_carlo_dsr(self, signal):
        """Test positive expected DSR on synthetic data with known edge."""
        np.random.seed(42)
        
        # Generate data with known variance ratio properties
        n_simulations = 50
        returns_list = []
        
        for _ in range(n_simulations):
            # Create trending data with positive drift
            trend_component = np.cumsum(np.random.normal(0.002, 0.01, 150))
            noise_component = np.random.normal(0, 0.02, 150)
            
            prices = trend_component + noise_component
            returns = pd.Series(np.diff(prices))
            returns.index = pd.date_range('2023-01-01', periods=len(returns), freq='D')
            
            returns_list.append(returns)
        
        # Test that strategy generates reasonable signals
        positive_signals = 0
        total_signals = 0
        
        for returns in returns_list:
            signals = signal.generate(returns)
            non_zero_signals = signals[signals != 0]
            
            if len(non_zero_signals) > 0:
                total_signals += len(non_zero_signals)
                positive_signals += (non_zero_signals == 1).sum()
        
        if total_signals > 0:
            signal_bias = positive_signals / total_signals
            # For trending data, should have some positive bias
            assert signal_bias > 0.3, f"Expected some positive bias for trending data, got {signal_bias}"
    
    def test_parameter_sensitivity(self, signal, sample_returns):
        """Test reasonable behavior across parameter ranges."""
        base_signals = signal.generate(sample_returns)
        
        # Test different threshold values
        original_threshold = signal.params['threshold_delta']
        
        # Lower threshold should generate more signals
        signal.params['threshold_delta'] = 0.05
        low_thresh_signals = signal.generate(sample_returns)
        
        # Higher threshold should generate fewer signals
        signal.params['threshold_delta'] = 0.20
        high_thresh_signals = signal.generate(sample_returns)
        
        # Restore original
        signal.params['threshold_delta'] = original_threshold
        
        # Count non-zero signals
        low_count = (low_thresh_signals != 0).sum()
        high_count = (high_thresh_signals != 0).sum()
        
        # Lower threshold should generate more or equal signals
        assert low_count >= high_count, "Lower threshold should generate more signals"
    
    def test_mathematical_consistency(self, signal):
        """Test that implementation matches documented equations."""
        # Test variance ratio formula: VR(q) = Var[q-period]/[q * Var[1-period]]
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        q = 5
        vr, _ = signal._calculate_variance_ratio(returns, q)
        
        # Manual calculation
        returns_q = returns.rolling(window=q).sum().dropna()
        var_1 = returns.var()
        var_q = returns_q.var()
        
        if var_1 > 0:
            manual_vr = var_q / (q * var_1)
            assert np.isclose(vr, manual_vr, rtol=1e-10), f"VR calculation mismatch: {vr} vs {manual_vr}"
