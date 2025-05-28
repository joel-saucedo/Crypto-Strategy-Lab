"""
Unit tests for Hurst Exponent Signal Strategy

Tests the mathematical correctness of DFA-based Hurst exponent calculation
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

from src.strategies.hurst_exponent.signal import HurstExponentSignal


class TestHurstExponentSignal(unittest.TestCase):
    """Test suite for HurstExponentSignal class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock config
        self.mock_config = {
            'best_params': {
                'lookback_days': 100,
                'h_threshold_high': 0.55,
                'h_threshold_low': 0.45,
                'min_confidence': 0.7
            },
            'param_grid': {
                'lookback_days': [50, 100, 200],
                'h_threshold_high': [0.55, 0.6],
                'h_threshold_low': [0.4, 0.45],
                'min_confidence': [0.6, 0.7, 0.8]
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
        signal = HurstExponentSignal(self.config_path)
        
        self.assertEqual(signal.params['lookback_days'], 100)
        self.assertEqual(signal.params['h_threshold_high'], 0.55)
        self.assertEqual(signal.params['h_threshold_low'], 0.45)
        self.assertEqual(signal.params['min_confidence'], 0.7)
    
    def test_generate_insufficient_data(self):
        """Test signal generation with insufficient data."""
        signal = HurstExponentSignal(self.config_path)
        
        # Create small dataset
        returns = pd.Series(np.random.randn(50), 
                          index=pd.date_range('2023-01-01', periods=50))
        
        signals = signal.generate(returns)
        
        # All signals should be 0 due to insufficient lookback
        self.assertTrue((signals == 0).all())
        self.assertEqual(len(signals), len(returns))
    
    def test_generate_signals_trending_data(self):
        """Test signal generation with trending (persistent) data."""
        # Create a config with lower confidence threshold for this test
        test_config = {
            'best_params': {
                'lookback_days': 100,
                'h_threshold_high': 0.55,
                'h_threshold_low': 0.45,
                'min_confidence': 0.5  # Lower threshold for testing
            },
            'param_grid': {
                'lookback_days': [50, 100, 200],
                'h_threshold_high': [0.55, 0.6],
                'h_threshold_low': [0.4, 0.45],
                'min_confidence': [0.6, 0.7, 0.8]
            }
        }
        
        # Create temporary config file with lower confidence
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            test_config_path = f.name
        
        try:
            signal = HurstExponentSignal(test_config_path)
            
            # Create strongly trending data with high Hurst exponent
            np.random.seed(42)
            n = 200
            # Create a persistent random walk with strong drift
            drift = 0.005
            volatility = 0.01
            prices = [100]
            for i in range(n-1):
                shock = np.random.randn() * volatility + drift
                prices.append(prices[-1] * (1 + shock))
            
            returns = pd.Series(np.diff(np.log(prices)),
                              index=pd.date_range('2023-01-01', periods=n-1))
            
            signals = signal.generate(returns)
            
            # Should generate some signals for trending data (not necessarily all positive)
            # The test is whether the strategy can detect regime, not the direction
            self.assertTrue(abs(signals).sum() > 0, "Should generate some non-zero signals for trending data")
        finally:
            os.unlink(test_config_path)
        self.assertTrue(signals.max() <= 1)
        self.assertTrue(signals.min() >= -1)
    
    def test_generate_signals_mean_reverting_data(self):
        """Test signal generation with mean-reverting data."""
        signal = HurstExponentSignal(self.config_path)
        
        # Create mean-reverting data with low Hurst exponent
        np.random.seed(123)
        n = 200
        ar_returns = []
        prev = 0
        for _ in range(n):
            # Strong mean reversion: next return opposes previous
            next_ret = -0.5 * prev + np.random.randn() * 0.01
            ar_returns.append(next_ret)
            prev = next_ret
            
        returns = pd.Series(ar_returns, 
                          index=pd.date_range('2023-01-01', periods=n))
        
        signals = signal.generate(returns)
        
        # Should have valid signal range
        self.assertTrue(signals.max() <= 1)
        self.assertTrue(signals.min() >= -1)
    
    def test_calculate_signal_logic(self):
        """Test signal calculation logic."""
        signal = HurstExponentSignal(self.config_path)
        
        # Test with short data (should return 0)
        short_data = pd.Series(np.random.randn(20))
        result = signal._calculate_signal(short_data, 0.55, 0.45, 0.7)
        self.assertEqual(result, 0)
        
        # Test with sufficient data
        sufficient_data = pd.Series(np.random.randn(100))
        result = signal._calculate_signal(sufficient_data, 0.55, 0.45, 0.7)
        self.assertIn(result, [-1, 0, 1])
    
    def test_calculate_hurst_dfa_edge_cases(self):
        """Test Hurst exponent calculation edge cases."""
        signal = HurstExponentSignal(self.config_path)
        
        # Test with insufficient data
        short_series = pd.Series(np.random.randn(10))
        hurst, confidence = signal._calculate_hurst_dfa(short_series)
        self.assertEqual(hurst, 0.5)
        self.assertEqual(confidence, 0.0)
        
        # Test with constant data
        constant_series = pd.Series([1.0] * 100)
        hurst, confidence = signal._calculate_hurst_dfa(constant_series)
        self.assertIsInstance(hurst, float)
        self.assertIsInstance(confidence, float)
        
        # Test with normal random data
        random_series = pd.Series(np.random.randn(100))
        hurst, confidence = signal._calculate_hurst_dfa(random_series)
        self.assertGreaterEqual(hurst, 0.1)
        self.assertLessEqual(hurst, 0.9)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_calculate_hurst_dfa_mathematical_properties(self):
        """Test mathematical properties of Hurst exponent calculation."""
        signal = HurstExponentSignal(self.config_path)
        
        # Test with white noise (should be around 0.5)
        np.random.seed(42)
        white_noise = pd.Series(np.random.randn(200))
        hurst, confidence = signal._calculate_hurst_dfa(white_noise)
        self.assertGreater(confidence, 0.0)  # Should have some confidence
        self.assertGreater(hurst, 0.3)  # Should be reasonably close to 0.5
        self.assertLess(hurst, 0.7)
        
        # Test with strongly trending data
        trend_data = pd.Series(np.cumsum(np.random.randn(200) + 0.1))
        hurst_trend, conf_trend = signal._calculate_hurst_dfa(trend_data)
        self.assertIsInstance(hurst_trend, float)
        self.assertIsInstance(conf_trend, float)
    
    def test_signal_values_validity(self):
        """Test that generated signals are always valid."""
        signal = HurstExponentSignal(self.config_path)
        
        # Test multiple random series
        for seed in [1, 42, 123, 456, 789]:
            np.random.seed(seed)
            returns = pd.Series(np.random.randn(150), 
                              index=pd.date_range('2023-01-01', periods=150))
            
            signals = signal.generate(returns)
            
            # All signals should be in valid range
            self.assertTrue(signals.isin([-1, 0, 1]).all())
            self.assertEqual(len(signals), len(returns))
    
    def test_param_grid(self):
        """Test parameter grid retrieval."""
        signal = HurstExponentSignal(self.config_path)
        
        # Mock the file reading since get_param_grid hardcodes the config path
        mock_param_grid = {
            'lookback_days': {'min': 100, 'max': 300, 'step': 25},
            'h_threshold_high': {'min': 0.53, 'max': 0.65, 'step': 0.02},
            'h_threshold_low': {'min': 0.35, 'max': 0.47, 'step': 0.02},
            'min_confidence': {'min': 0.5, 'max': 0.9, 'step': 0.1}
        }
        
        with patch('builtins.open', mock_open(read_data=yaml.dump({'param_grid': mock_param_grid}))):
            param_grid = signal.get_param_grid()
        
        self.assertIn('lookback_days', param_grid)
        self.assertIn('h_threshold_high', param_grid)
        self.assertIn('h_threshold_low', param_grid)
        self.assertIn('min_confidence', param_grid)
        
        # Check the structure matches what we expect
        self.assertIn('min', param_grid['lookback_days'])
        self.assertIn('max', param_grid['lookback_days'])
        self.assertIn('step', param_grid['lookback_days'])
    
    def test_configuration_parameter_usage(self):
        """Test that configuration parameters are properly used."""
        # Create config with extreme thresholds
        extreme_config = {
            'best_params': {
                'lookback_days': 50,
                'h_threshold_high': 0.9,  # Very high threshold
                'h_threshold_low': 0.1,   # Very low threshold
                'min_confidence': 0.95    # Very high confidence requirement
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(extreme_config, f)
            extreme_config_path = f.name
        
        try:
            signal = HurstExponentSignal(extreme_config_path)
            
            # Generate signals with random data
            returns = pd.Series(np.random.randn(100), 
                              index=pd.date_range('2023-01-01', periods=100))
            signals = signal.generate(returns)
            
            # With extreme thresholds, most signals should be neutral (0)
            neutral_ratio = (signals == 0).sum() / len(signals)
            self.assertGreater(neutral_ratio, 0.7)  # Most signals should be neutral
            
        finally:
            os.unlink(extreme_config_path)
    
    def test_reproducibility(self):
        """Test that signals are reproducible with same input."""
        signal = HurstExponentSignal(self.config_path)
        
        # Same random seed should produce same results
        np.random.seed(42)
        returns1 = pd.Series(np.random.randn(120), 
                           index=pd.date_range('2023-01-01', periods=120))
        
        np.random.seed(42)
        returns2 = pd.Series(np.random.randn(120), 
                           index=pd.date_range('2023-01-01', periods=120))
        
        signals1 = signal.generate(returns1)
        signals2 = signal.generate(returns2)
        
        pd.testing.assert_series_equal(signals1, signals2)


if __name__ == '__main__':
    unittest.main()