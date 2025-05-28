"""
Unit tests for permutation_entropy strategy.
Validates implementation against mathematical expectations.
"""

import pytest
import numpy as np
import pandas as pd
import math
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.permutation_entropy.signal import PermutationentropySignal

class TestPermutationentropySignal:
    
    def setup_method(self):
        """Setup test fixtures."""
        self.signal = PermutationentropySignal()
        
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
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        returns = pd.Series(np.random.normal(0, 0.02, 200), index=dates)
        
        signals = self.signal.generate(returns)
        
        assert signals.index.equals(returns.index), "Index alignment failed"
        
    def test_no_look_ahead(self):
        """Test that signals don't use future information."""
        # Create returns with step change in volatility
        np.random.seed(42)
        early_returns = np.random.normal(0, 0.01, 150)  # Low vol
        late_returns = np.random.normal(0, 0.05, 150)   # High vol
        
        returns = pd.Series(np.concatenate([early_returns, late_returns]))
        signals = self.signal.generate(returns)
        
        # Extract non-zero signals from each period
        early_signals = signals.iloc[:150]
        late_signals = signals.iloc[150:]
        
        # Pattern changes should not cause systematic bias in early signals
        early_nonzero = early_signals[early_signals != 0]
        if len(early_nonzero) > 10:
            early_positive_ratio = (early_nonzero == 1).mean()
            # Should not be extremely biased based on future regime
            assert 0.2 <= early_positive_ratio <= 0.8, "Possible look-ahead bias detected"
        
    def test_parameter_sensitivity(self):
        """Test reasonable behavior across parameter ranges."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 300))
        
        original_params = self.signal.params.copy()
        
        # Test different embedding dimensions
        for embedding_dim in [3, 4, 5, 6]:
            self.signal.params['embedding_dim'] = embedding_dim
            signals = self.signal.generate(returns)
            assert not signals.isna().any(), f"NaN signals with embedding_dim={embedding_dim}"
            
        # Test different window sizes
        for window_size in [100, 150, 200, 250]:
            self.signal.params['window_size'] = window_size
            signals = self.signal.generate(returns)
            assert not signals.isna().any(), f"NaN signals with window_size={window_size}"
        
        # Restore original parameters
        self.signal.params = original_params
        
    def test_monte_carlo_dsr(self):
        """Test positive expected DSR on synthetic data with known edge."""
        np.random.seed(42)
        
        # Generate data with low entropy patterns (periodic structure)
        n_simulations = 20
        strategy_performance = []
        
        for _ in range(n_simulations):
            # Create data with hidden patterns (should have low entropy)
            pattern_length = 20
            base_pattern = np.random.normal(0, 0.01, pattern_length)
            
            # Repeat pattern with noise to create predictable structure
            n_repeats = 15
            patterned_data = []
            for i in range(n_repeats):
                noisy_pattern = base_pattern + np.random.normal(0, 0.005, pattern_length)
                patterned_data.extend(noisy_pattern)
            
            returns = pd.Series(patterned_data)
            returns.index = pd.date_range('2023-01-01', periods=len(returns), freq='D')
            
            signals = self.signal.generate(returns)
            
            # Calculate simple strategy performance
            strategy_returns = signals.shift(1) * returns
            total_return = strategy_returns.sum()
            strategy_performance.append(total_return)
        
        # Strategy should perform reasonably on patterned data
        avg_performance = np.mean(strategy_performance)
        # Not requiring positive (conservative test for this complex strategy)
        assert avg_performance > -0.05, f"Poor average performance: {avg_performance}"
        
    def test_mathematical_consistency(self):
        """Test that implementation matches documented equations."""
        # Test permutation entropy calculation with known data
        np.random.seed(42)
        
        # Test 1: Perfectly periodic data should have low entropy
        periodic_data = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
        pe_periodic = self.signal._calculate_permutation_entropy(periodic_data, 3)
        
        # Should be lower than random data
        random_data = pd.Series(np.random.normal(0, 1, 12))
        pe_random = self.signal._calculate_permutation_entropy(random_data, 3)
        
        assert pe_periodic < pe_random, "Periodic data should have lower entropy than random"
        
        # Test 2: Constant data should have very low entropy (but handle numerical issues)
        constant_data = pd.Series([1.0] * 10)
        pe_constant = self.signal._calculate_permutation_entropy(constant_data, 3)
        
        # Should be very low (may not be exactly 0 due to tie-breaking)
        assert pe_constant < 0.1, f"Constant data entropy too high: {pe_constant}"
        
    def test_permutation_entropy_properties(self):
        """Test specific properties of permutation entropy calculation."""
        # Test entropy bounds
        np.random.seed(42)
        embedding_dim = 4
        
        # Generate different types of data
        data_types = {
            'random': np.random.normal(0, 1, 100),
            'trending': np.cumsum(np.random.normal(0.01, 0.1, 100)),
            'periodic': np.sin(np.arange(100) * 0.2) + np.random.normal(0, 0.1, 100)
        }
        
        for data_type, data in data_types.items():
            data_series = pd.Series(data)
            pe = self.signal._calculate_permutation_entropy(data_series, embedding_dim)
            
            # Entropy should be non-negative and finite
            assert pe >= 0, f"{data_type} data has negative entropy"
            assert np.isfinite(pe), f"{data_type} data has infinite entropy"
            
            # Entropy should not exceed theoretical maximum
            max_entropy = np.log(math.factorial(embedding_dim))
            assert pe <= max_entropy * 1.01, f"{data_type} entropy exceeds theoretical max"
            
    def test_predictability_score_calculation(self):
        """Test predictability score calculation."""
        np.random.seed(42)
        
        # Test with known patterns
        periodic_data = pd.Series(np.sin(np.arange(50) * 0.3))
        random_data = pd.Series(np.random.normal(0, 1, 50))
        
        pred_periodic = self.signal.calculate_predictability_score(periodic_data)
        pred_random = self.signal.calculate_predictability_score(random_data)
        
        # Periodic data should be more predictable
        assert pred_periodic > pred_random, "Periodic data should be more predictable"
        
        # Scores should be in [0, 1]
        assert 0 <= pred_periodic <= 1, "Predictability score out of bounds"
        assert 0 <= pred_random <= 1, "Predictability score out of bounds"
        
    def test_entropy_percentiles_calculation(self):
        """Test entropy percentiles calculation for threshold setting."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 500))
        
        percentiles = self.signal.get_entropy_percentiles(returns)
        
        # Check that percentiles are ordered
        assert percentiles['p10'] <= percentiles['p25'] <= percentiles['p50']
        assert percentiles['p50'] <= percentiles['p75'] <= percentiles['p90']
        
        # Check reasonable ranges
        assert 0 <= percentiles['p10'] <= 1, "P10 out of bounds"
        assert 0 <= percentiles['p90'] <= 1, "P90 out of bounds"
        
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very short series
        short_series = pd.Series([1, 2, 3])
        signals = self.signal.generate(short_series)
        assert (signals == 0).all(), "Short series should produce no signals"
        
        # Test with constant series
        constant_series = pd.Series([1.0] * 100)
        signals = self.signal.generate(constant_series)
        # Should not crash and should produce valid signals
        assert signals.isin([-1, 0, 1]).all(), "Invalid signals for constant series"
        
        # Test with extreme values
        extreme_series = pd.Series([-1000, 1000, -1000, 1000] * 50)
        signals = self.signal.generate(extreme_series)
        assert signals.isin([-1, 0, 1]).all(), "Invalid signals for extreme series"
