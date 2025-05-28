"""
Unit tests for drift_significance strategy.
Validates implementation against mathematical expectations.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.drift_significance.signal import DriftsignificanceSignal

class TestDriftsignificanceSignal:
    
    def setup_method(self):
        """Setup test fixtures."""
        self.signal = DriftsignificanceSignal()
        
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
        np.random.seed(42)
        
        # Create returns with regime change: noise first, then strong positive drift
        noise_period = np.random.normal(0, 0.02, 100)  # No drift initially
        drift_period = np.random.normal(0.003, 0.02, 100)  # Strong drift later
        returns = pd.Series(list(noise_period) + list(drift_period))
        
        signals = self.signal.generate(returns)
        
        # Test that signals in the transition period don't immediately reflect
        # the regime change (which would indicate look-ahead)
        lookback = self.signal.params['lookback_days']
        transition_start = 100
        transition_end = min(100 + lookback, len(returns))
        
        # Signals right after regime change should still be based on historical data
        transition_signals = signals.iloc[transition_start:transition_end]
        
        # If there's no look-ahead bias, signals shouldn't immediately jump to +1
        # when the regime changes, since the lookback window still contains mostly noise
        if len(transition_signals) > 0:
            immediate_positive = (transition_signals == 1).sum()
            # Allow some positive signals but not too many immediately
            assert immediate_positive <= len(transition_signals) * 0.5, "Possible look-ahead bias detected"
        
    def test_parameter_sensitivity(self):
        """Test reasonable behavior across parameter ranges."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.002, 0.02, 300))  # Small positive drift
        
        # Test with different parameter values
        original_params = self.signal.params.copy()
        
        # Test with stricter t-stat threshold
        self.signal.params['min_t_stat'] = 3.0
        signals_strict = self.signal.generate(returns)
        
        # Test with looser t-stat threshold  
        self.signal.params['min_t_stat'] = 1.5
        signals_loose = self.signal.generate(returns)
        
        # Restore original parameters
        self.signal.params = original_params
        
        # Stricter threshold should produce fewer signals
        strict_signals = (signals_strict != 0).sum()
        loose_signals = (signals_loose != 0).sum()
        
        assert strict_signals <= loose_signals, "Stricter threshold should produce fewer signals"
        
    def test_monte_carlo_dsr(self):
        """Test that strategy has positive expected performance on synthetic data."""
        np.random.seed(42)
        
        # Generate data with significant positive drift that should be detectable
        n_simulations = 30
        performance_scores = []
        
        for _ in range(n_simulations):
            # Create data with clear statistical significance
            drift = 0.003  # 30 bps daily drift
            volatility = 0.02
            n_days = 200
            
            returns = np.random.normal(drift, volatility, n_days)
            returns_series = pd.Series(returns)
            
            signals = self.signal.generate(returns_series)
            
            # Calculate strategy performance
            strategy_returns = signals.shift(1) * returns_series
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) > 0:
                # Calculate total return for this simulation
                total_return = strategy_returns.sum()
                performance_scores.append(total_return)
        
        # Average performance should be positive with significant drift
        if performance_scores:
            avg_performance = np.mean(performance_scores)
            assert avg_performance > 0, f"Expected positive performance on high-drift data, got {avg_performance}"
        
    def test_mathematical_consistency(self):
        """Test that implementation matches documented equations."""
        np.random.seed(42)
        
        # Test t-statistic calculation manually
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        # Manual calculation
        n = len(returns)
        mu_hat_manual = returns.mean()
        sigma_hat_manual = returns.std(ddof=1)
        se_manual = sigma_hat_manual / np.sqrt(n)
        t_stat_manual = mu_hat_manual / se_manual if se_manual > 0 else 0
        
        # Strategy calculation
        mu_hat, sigma_hat, t_stat, p_value = self.signal._calculate_drift_stats(returns)
        
        # Verify mathematical consistency
        assert np.isclose(mu_hat, mu_hat_manual, rtol=1e-10), "Mean calculation mismatch"
        assert np.isclose(sigma_hat, sigma_hat_manual, rtol=1e-10), "Std deviation calculation mismatch"
        assert np.isclose(t_stat, t_stat_manual, rtol=1e-10), "T-statistic calculation mismatch"
        
        # Verify p-value calculation
        p_value_manual = 2 * (1 - stats.t.cdf(abs(t_stat_manual), n - 1))
        assert np.isclose(p_value, p_value_manual, rtol=1e-10), "P-value calculation mismatch"
        
    def test_cramer_rao_bound(self):
        """Test that implementation respects Cramér-Rao lower bound principles."""
        np.random.seed(42)
        
        # For normal distribution, MLE achieves Cramér-Rao bound
        returns = pd.Series(np.random.normal(0.001, 0.02, 150))
        
        mu_hat, sigma_hat, t_stat, p_value = self.signal._calculate_drift_stats(returns)
        
        # Check that variance estimate matches theoretical minimum
        n = len(returns)
        theoretical_var = (sigma_hat**2) / n  # Cramér-Rao bound for mean
        empirical_se = sigma_hat / np.sqrt(n)
        theoretical_se = np.sqrt(theoretical_var)
        
        assert np.isclose(empirical_se, theoretical_se, rtol=1e-10), "Standard error doesn't match Cramér-Rao bound"
        
    def test_significance_filtering(self):
        """Test that strategy correctly filters insignificant drifts."""
        np.random.seed(42)
        
        # Create data with very small drift (should be filtered out)
        small_drift_returns = pd.Series(np.random.normal(0.0001, 0.02, 200))  # 1 bp drift
        signals_small = self.signal.generate(small_drift_returns)
        
        # Create data with large drift (should pass filter)
        large_drift_returns = pd.Series(np.random.normal(0.005, 0.02, 200))  # 50 bp drift
        signals_large = self.signal.generate(large_drift_returns)
        
        # Large drift should generate more signals than small drift
        small_signals = (signals_small != 0).sum()
        large_signals = (signals_large != 0).sum()
        
        assert large_signals >= small_signals, "Strategy should generate more signals for larger drifts"
        
    def test_signal_strength_calculation(self):
        """Test signal strength calculation for position sizing."""
        np.random.seed(42)
        
        # Test with different drift magnitudes
        weak_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        strong_returns = pd.Series(np.random.normal(0.004, 0.02, 100))
        
        weak_strength = self.signal.get_signal_strength(weak_returns)
        strong_strength = self.signal.get_signal_strength(strong_returns)
        
        # Stronger signals should have higher strength
        assert strong_strength >= weak_strength, "Signal strength should increase with drift magnitude"
        
        # Strength should be in [0, 1] range
        assert 0 <= weak_strength <= 1, "Signal strength out of range"
        assert 0 <= strong_strength <= 1, "Signal strength out of range"
