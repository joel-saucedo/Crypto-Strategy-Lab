"""
Unit tests for Variance Ratio Signal Strategy

Tests the mathematical correctness of Lo-MacKinlay variance ratio test
and signal generation logic.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest
import numpy as np
import pandas as pd
import tempfile
import yaml
from unittest.mock import patch, mock_open

from src.strategies.variance_ratio.signal import VarianceRatioSignal


class TestVarianceRatioSignal(unittest.TestCase):
    """Test suite for VarianceRatioSignal class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock config
        self.mock_config = {
            'best_params': {
                'lookback_days': 60,
                'lag_q': 5,
                'threshold_delta': 0.1,
                'min_z_stat': 2.0
            },
            'param_grid': {
                'lookback_days': [30, 60, 120],
                'lag_q': [3, 5, 10],
                'threshold_delta': [0.05, 0.1, 0.15],
                'min_z_stat': [1.5, 2.0, 2.5]
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.mock_config, f)
            self.config_path = f.name
            
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'config_path'):
            os.unlink(self.config_path)
    
    def test_initialization(self):
        """Test signal class initialization."""
        signal = VarianceRatioSignal(self.config_path)
        
        self.assertEqual(signal.params['lookback_days'], 60)
        self.assertEqual(signal.params['lag_q'], 5)
        self.assertEqual(signal.params['threshold_delta'], 0.1)
        self.assertEqual(signal.params['min_z_stat'], 2.0)
    
    def test_generate_insufficient_data(self):
        """Test signal generation with insufficient data."""
        signal = VarianceRatioSignal(self.config_path)
        
        # Create small dataset
        returns = pd.Series(np.random.randn(30), 
                          index=pd.date_range('2023-01-01', periods=30))
        
        signals = signal.generate(returns)
        
        # All signals should be 0 due to insufficient lookback
        self.assertTrue((signals == 0).all())
        self.assertEqual(len(signals), len(returns))
        
    
    def test_generate_signals_momentum_data(self):
        """Test signal generation with momentum (persistent drift) data."""
        signal = VarianceRatioSignal(self.config_path)
        
        # Create momentum data with high variance ratio
        np.random.seed(42)
        n = 150
        momentum_returns = []
        drift = 0.002  # Persistent positive drift
        
        for i in range(n):
            # Add momentum with some noise
            if i == 0:
                momentum_returns.append(np.random.normal(drift, 0.01))
            else:
                # Momentum continuation
                momentum = 0.3 * momentum_returns[-1] + np.random.normal(drift, 0.008)
                momentum_returns.append(momentum)
        
        returns = pd.Series(momentum_returns, 
                          index=pd.date_range('2023-01-01', periods=n))
        
        signals = signal.generate(returns)
        
        # Should generate some signals for momentum data
        self.assertTrue(signals.max() <= 1)
        self.assertTrue(signals.min() >= -1)
        self.assertTrue(len(signals) == len(returns))
    
    def test_generate_signals_mean_reverting_data(self):
        """Test signal generation with mean-reverting data."""
        signal = VarianceRatioSignal(self.config_path)
        
        # Create mean-reverting data with low variance ratio
        np.random.seed(123)
        n = 150
        ar_returns = []
        mean_level = 0.0
        
        for i in range(n):
            if i == 0:
                ar_returns.append(np.random.normal(0, 0.02))
            else:
                # Mean reversion: return towards mean level
                reversion = -0.4 * (ar_returns[-1] - mean_level) + np.random.normal(0, 0.015)
                ar_returns.append(reversion)
        
        returns = pd.Series(ar_returns, 
                          index=pd.date_range('2023-01-01', periods=n))
        
        signals = signal.generate(returns)
        
        # Should have valid signal range
        self.assertTrue(signals.max() <= 1)
        self.assertTrue(signals.min() >= -1)
    
    def test_calculate_signal_logic(self):
        """Test signal calculation logic."""
        signal = VarianceRatioSignal(self.config_path)
        
        # Test with short data (should return 0)
        short_data = pd.Series(np.random.randn(10))
        result = signal._calculate_signal(short_data, 5, 0.1, 2.0)
        self.assertEqual(result, 0)
        
        # Test with sufficient data
        sufficient_data = pd.Series(np.random.randn(80))
        result = signal._calculate_signal(sufficient_data, 5, 0.1, 2.0)
        self.assertIn(result, [-1, 0, 1])
    
    def test_calculate_variance_ratio_edge_cases(self):
        """Test variance ratio calculation edge cases."""
        signal = VarianceRatioSignal(self.config_path)
        
        # Test with insufficient data
        short_series = pd.Series(np.random.randn(10))
        vr, z_stat = signal._calculate_variance_ratio(short_series, 5)
        self.assertEqual(vr, 1.0)
        self.assertEqual(z_stat, 0.0)
        
        # Test with constant data (zero variance)
        constant_series = pd.Series([0.01] * 100)
        vr, z_stat = signal._calculate_variance_ratio(constant_series, 5)
        # For constant data, variance should be ~0 and handled properly
        self.assertIsInstance(vr, float)
        self.assertIsInstance(z_stat, float)
        self.assertGreater(vr, 0)  # Variance ratio should be positive
        
        # Test with normal random data
        np.random.seed(42)
        random_series = pd.Series(np.random.randn(100))
        vr, z_stat = signal._calculate_variance_ratio(random_series, 5)
        self.assertIsInstance(vr, float)
        self.assertIsInstance(z_stat, float)
        self.assertGreater(vr, 0)  # Variance ratio should be positive
    
    def test_calculate_variance_ratio_mathematical_properties(self):
        """Test mathematical properties of variance ratio calculation."""
        signal = VarianceRatioSignal(self.config_path)
        
        # Test with white noise (VR should be around 1.0)
        np.random.seed(42)
        white_noise = pd.Series(np.random.randn(200))
        vr, z_stat = signal._calculate_variance_ratio(white_noise, 5)
        
        # For white noise, VR should be close to 1
        self.assertGreater(vr, 0.5)
        self.assertLess(vr, 2.0)
        
        # Test with trending data (VR should be > 1)
        trend_returns = pd.Series(np.random.randn(200) + 0.001)  # Small positive drift
        vr_trend, z_trend = signal._calculate_variance_ratio(trend_returns, 5)
        self.assertIsInstance(vr_trend, float)
        self.assertIsInstance(z_trend, float)
    
    def test_lo_mackinlay_test_statistic(self):
        """Test the Lo-MacKinlay test statistic calculation."""
        signal = VarianceRatioSignal(self.config_path)
        
        # Test multiple lag values
        np.random.seed(42)
        returns = pd.Series(np.random.randn(150))
        
        for q in [2, 5, 10]:
            vr, z_stat = signal._calculate_variance_ratio(returns, q)
            
            # Z-statistic should be finite
            self.assertFalse(np.isnan(z_stat))
            self.assertFalse(np.isinf(z_stat))
            
            # VR should be positive
            self.assertGreater(vr, 0)
    
    def test_signal_values_validity(self):
        """Test that generated signals are always valid."""
        signal = VarianceRatioSignal(self.config_path)
        
        # Test multiple random series
        for seed in [1, 42, 123, 456, 789]:
            np.random.seed(seed)
            returns = pd.Series(np.random.randn(120), 
                              index=pd.date_range('2023-01-01', periods=120))
            
            signals = signal.generate(returns)
            
            # All signals should be in valid range
            self.assertTrue(signals.isin([-1, 0, 1]).all())
            self.assertEqual(len(signals), len(returns))
    
    def test_param_grid(self):
        """Test parameter grid retrieval."""
        signal = VarianceRatioSignal(self.config_path)
        
        # Mock the file reading since get_param_grid hardcodes the config path
        mock_param_grid = {
            'lookback_days': {'min': 30, 'max': 120, 'step': 10},
            'lag_q': {'min': 2, 'max': 10, 'step': 1},
            'threshold_delta': {'min': 0.05, 'max': 0.25, 'step': 0.05},
            'min_z_stat': {'min': 1.5, 'max': 3.0, 'step': 0.25}
        }
        
        with patch('builtins.open', mock_open(read_data=yaml.dump({'param_grid': mock_param_grid}))):
            param_grid = signal.get_param_grid()
        
        self.assertIn('lookback_days', param_grid)
        self.assertIn('lag_q', param_grid)
        self.assertIn('threshold_delta', param_grid)
        self.assertIn('min_z_stat', param_grid)
        
        # Check the structure matches what we expect
        self.assertIn('min', param_grid['lookback_days'])
        self.assertIn('max', param_grid['lookback_days'])
        self.assertIn('step', param_grid['lookback_days'])
    
    def test_reproducibility(self):
        """Test that signals are reproducible with same input."""
        signal = VarianceRatioSignal(self.config_path)
        
        # Same random seed should produce same results
        np.random.seed(42)
        returns1 = pd.Series(np.random.randn(100), 
                           index=pd.date_range('2023-01-01', periods=100))
        
        np.random.seed(42)
        returns2 = pd.Series(np.random.randn(100), 
                           index=pd.date_range('2023-01-01', periods=100))
        
        signals1 = signal.generate(returns1)
        signals2 = signal.generate(returns2)
        
        pd.testing.assert_series_equal(signals1, signals2)


if __name__ == '__main__':
    unittest.main()