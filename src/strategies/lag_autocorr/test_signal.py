"""
Unit tests for lag autocorrelation strategy.
Validates the implementation against mathematical expectations.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.lag_autocorr.signal import LagAutocorrSignal

class TestLagAutocorrSignal:
    
    def setup_method(self):
        """Setup test fixtures."""
        self.signal = LagAutocorrSignal()
        
    def test_no_nan_output(self):
        """Test that signal generation doesn't produce NaN values."""
        # Generate dummy returns
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 300))
        
        signals = self.signal.generate(returns)
        
        # Check no NaN values in signals
        assert not signals.isna().any(), "Signal contains NaN values"
        
    def test_signal_range(self):
        """Test that signals are in valid range {-1, 0, 1}."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 300))
        
        signals = self.signal.generate(returns)
        
        # Check signal values are valid
        valid_signals = signals.isin([-1, 0, 1])
        assert valid_signals.all(), "Invalid signal values detected"
        
    def test_index_alignment(self):
        """Test that output index aligns with input (no look-ahead)."""
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        returns = pd.Series(np.random.normal(0, 0.02, 200), index=dates)
        
        signals = self.signal.generate(returns)
        
        # Check index alignment
        assert signals.index.equals(returns.index), "Index alignment failed"
        
    def test_autocorr_calculation(self):
        """Test autocorrelation calculation with known data."""
        # Create data with known autocorrelation
        n = 100
        ar_coeff = 0.3
        noise = np.random.normal(0, 1, n)
        ar_series = np.zeros(n)
        ar_series[0] = noise[0]
        
        for i in range(1, n):
            ar_series[i] = ar_coeff * ar_series[i-1] + noise[i]
            
        returns = pd.Series(ar_series)
        
        # Calculate autocorrelation manually
        autocorr, p_value = self.signal._calculate_autocorr(returns, lag=1)
        
        # Should detect positive autocorrelation
        assert autocorr > 0, "Failed to detect positive autocorrelation"
        assert p_value < 0.05, "Failed to detect statistical significance"
        
    def test_monte_carlo_dsr(self):
        """Test that strategy has positive expected DSR on synthetic data."""
        np.random.seed(42)
        
        # Generate data with slight momentum
        n = 1000
        returns = np.random.normal(0.0005, 0.02, n)  # Small positive drift
        
        # Add some autocorrelation
        for i in range(1, n):
            returns[i] += 0.1 * returns[i-1]
            
        returns_series = pd.Series(returns)
        signals = self.signal.generate(returns_series)
        
        # Calculate strategy returns
        strategy_returns = signals.shift(1) * returns_series
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            
            # Should have positive Sharpe on momentum data
            assert sharpe > 0, f"Negative Sharpe ratio: {sharpe}"
            
    def test_parameter_grid_access(self):
        """Test that parameter grid is accessible."""
        param_grid = self.signal.get_param_grid()
        
        assert isinstance(param_grid, dict), "Parameter grid must be dictionary"
        assert 'lookback_days' in param_grid, "Missing lookback_days parameter"
        assert 'lag_k' in param_grid, "Missing lag_k parameter"
