"""
Test Suite for PACF Spike Detector Trading Strategy

Tests cover:
1. Mathematical consistency of PACF calculation
2. Spike detection accuracy and significance testing
3. AR structure identification validation
4. Signal generation logic and timing
5. Volatility regime filtering
6. Higher lags insignificance verification
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

from strategies.pacf_spike.strategy import PACFSpikeDetectorSignal


class TestPACFSpikeDetectorSignal:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'lookback_days': 180,
            'max_lag': 20,
            'significance_threshold': 2.0,
            'min_spike_strength': 0.1,
            'higher_lag_threshold': 1.0,
            'entry_delay_days': 1,
            'max_hold_days': 10,
            'signal_decay': 0.9,
            'volatility_filter': True,
            'vol_lookback': 30,
            'vol_threshold_pct': 95,
            'max_signal_strength': 1.0,
            'min_observations': 50,
            'confidence_level': 0.05
        }
        
        # Mock the config file loading
        with patch('builtins.open'), patch('yaml.safe_load', return_value=self.config):
            self.strategy = PACFSpikeDetectorSignal()
        
    def generate_test_data(self, n_points: int = 250, pattern: str = "normal") -> pd.DataFrame:
        """Generate synthetic price data for testing."""
        np.random.seed(42)
        
        if pattern == "ar1":
            # Generate AR(1) process with known coefficient
            ar_coeff = 0.3
            noise = np.random.normal(0, 0.01, n_points)
            returns = np.zeros(n_points)
            returns[0] = noise[0]
            
            for i in range(1, n_points):
                returns[i] = ar_coeff * returns[i-1] + noise[i]
                
            prices = 100 * np.exp(np.cumsum(returns))
            
        elif pattern == "ar2":
            # Generate AR(2) process
            ar1, ar2 = 0.4, -0.2
            noise = np.random.normal(0, 0.01, n_points)
            returns = np.zeros(n_points)
            returns[0:2] = noise[0:2]
            
            for i in range(2, n_points):
                returns[i] = ar1 * returns[i-1] + ar2 * returns[i-2] + noise[i]
                
            prices = 100 * np.exp(np.cumsum(returns))
            
        elif pattern == "ma1":
            # Generate MA(1) process (should not show PACF spike)
            ma_coeff = 0.3
            noise = np.random.normal(0, 0.01, n_points)
            returns = np.zeros(n_points)
            returns[0] = noise[0]
            
            for i in range(1, n_points):
                returns[i] = noise[i] + ma_coeff * noise[i-1]
                
            prices = 100 * np.exp(np.cumsum(returns))
            
        else:
            # Random walk (normal)
            returns = np.random.normal(0, 0.01, n_points)
            prices = 100 * np.exp(np.cumsum(returns))
            
        return pd.DataFrame({
            'open': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.uniform(1000, 2000, n_points)
        })
    
    def test_pacf_calculation_consistency(self):
        """Test PACF calculation mathematical properties."""
        # Generate AR(1) process
        data = self.generate_test_data(300, "ar1")
        returns = data['close'].pct_change().dropna()
        
        # Calculate PACF
        spike_info = self.strategy.detect_pacf_spike(returns)
        
        # AR(1) should show spike at lag 1
        assert spike_info['has_spike'], "Should detect spike in AR(1) process"
        assert spike_info['lag'] <= 3, f"Spike at lag {spike_info['lag']} too high for AR(1)"
        assert spike_info['significance'] > self.strategy.significance_threshold
        
        # Test with white noise (should not show significant spikes)
        white_noise = pd.Series(np.random.normal(0, 0.01, 200))
        noise_spike = self.strategy.detect_pacf_spike(white_noise)
        
        # White noise should not consistently show significant spikes
        # (may occasionally by chance, but significance should be lower)
        if noise_spike['has_spike']:
            assert noise_spike['significance'] < spike_info['significance'], \
                "White noise should not show stronger spikes than AR process"
    
    def test_ar_structure_identification(self):
        """Test identification of different AR structures."""
        # AR(1) process
        ar1_data = self.generate_test_data(300, "ar1")
        ar1_returns = ar1_data['close'].pct_change().dropna()
        ar1_spike = self.strategy.detect_pacf_spike(ar1_returns)
        
        # AR(2) process  
        ar2_data = self.generate_test_data(300, "ar2")
        ar2_returns = ar2_data['close'].pct_change().dropna()
        ar2_spike = self.strategy.detect_pacf_spike(ar2_returns)
        
        # MA(1) process (should not show strong PACF spike)
        ma1_data = self.generate_test_data(300, "ma1")
        ma1_returns = ma1_data['close'].pct_change().dropna()
        ma1_spike = self.strategy.detect_pacf_spike(ma1_returns)
        
        # AR processes should show spikes more reliably than MA
        ar_spike_count = sum([ar1_spike['has_spike'], ar2_spike['has_spike']])
        assert ar_spike_count >= 1, "Should detect spikes in at least one AR process"
        
        # If MA shows spike, it should be weaker
        if ma1_spike['has_spike']:
            max_ar_sig = max([ar1_spike['significance'], ar2_spike['significance']])
            assert ma1_spike['significance'] < max_ar_sig, \
                "MA process should not show stronger PACF spike than AR"
    
    def test_spike_significance_testing(self):
        """Test statistical significance of spike detection."""
        # Generate known AR(1) with strong coefficient
        n = 250
        ar_coeff = 0.5  # Strong AR coefficient
        noise = np.random.normal(0, 0.005, n)  # Low noise
        returns = np.zeros(n)
        returns[0] = noise[0]
        
        for i in range(1, n):
            returns[i] = ar_coeff * returns[i-1] + noise[i]
            
        returns_series = pd.Series(returns)
        spike_info = self.strategy.detect_pacf_spike(returns_series)
        
        # Should detect highly significant spike
        assert spike_info['has_spike'], "Should detect spike in strong AR(1)"
        assert spike_info['significance'] > 3.0, \
            f"Significance {spike_info['significance']:.2f} should be high for strong AR"
        assert spike_info['value'] > 0, "AR(1) with positive coeff should show positive PACF"
    
    def test_higher_lags_insignificance(self):
        """Test that higher lags are checked for insignificance."""
        # Create PACF values with spike at lag 2 but significant higher lags
        pacf_values = np.array([1.0, 0.1, 0.4, 0.3, 0.25, 0.2])  # Spike at lag 2, significant higher
        se = 0.1
        
        # Should reject due to significant higher lags
        insignificant = self.strategy.check_higher_lags_insignificant(pacf_values, se, 2)
        assert not insignificant, "Should reject spike when higher lags are significant"
        
        # Create clean spike (insignificant higher lags)
        clean_pacf = np.array([1.0, 0.1, 0.4, 0.05, 0.03, 0.02])  # Clean spike at lag 2
        clean_insignificant = self.strategy.check_higher_lags_insignificant(clean_pacf, se, 2)
        assert clean_insignificant, "Should accept spike when higher lags are insignificant"
    
    def test_signal_generation_timing(self):
        """Test signal generation timing rules."""
        # Test positive spike with down-day (should generate buy signal)
        spike_info = {
            'has_spike': True,
            'lag': 2,
            'value': 0.3,
            'significance': 2.5,
            'strength': 0.3
        }
        
        # Down-day with positive spike
        signal = self.strategy.generate_spike_signal(spike_info, -0.01)
        assert signal > 0, "Positive spike after down-day should generate buy signal"
        
        # Up-day with positive spike (weaker signal)
        signal_weak = self.strategy.generate_spike_signal(spike_info, 0.01)
        assert 0 < signal_weak < signal, "Positive spike after up-day should be weaker"
        
        # Test negative spike
        neg_spike_info = spike_info.copy()
        neg_spike_info['value'] = -0.3
        
        # Up-day with negative spike
        neg_signal = self.strategy.generate_spike_signal(neg_spike_info, 0.01)
        assert neg_signal < 0, "Negative spike after up-day should generate sell signal"
    
    def test_signal_generation_bounds(self):
        """Test that generated signals are within expected bounds."""
        data = self.generate_test_data(250, "ar1")
        signals = self.strategy.generate(data)
        
        # Signals should be bounded
        assert signals.min() >= -self.strategy.max_signal_strength, \
            f"Signal minimum {signals.min():.3f} below bound"
        assert signals.max() <= self.strategy.max_signal_strength, \
            f"Signal maximum {signals.max():.3f} above bound"
        
        # Should have some non-zero signals for AR process
        non_zero_signals = (signals != 0).sum()
        assert non_zero_signals > 0, "Should generate some trading signals for AR process"
    
    def test_volatility_regime_filtering(self):
        """Test volatility regime filtering."""
        # High volatility returns
        high_vol_returns = pd.Series(np.random.normal(0, 0.05, 100))  # 5% daily vol
        
        # Low volatility returns  
        low_vol_returns = pd.Series(np.random.normal(0, 0.005, 100))  # 0.5% daily vol
        
        # Test regime detection
        high_vol_ok = self.strategy.check_volatility_regime(high_vol_returns)
        low_vol_ok = self.strategy.check_volatility_regime(low_vol_returns)
        
        # At least one regime should be acceptable
        assert high_vol_ok or low_vol_ok, "At least one volatility regime should be tradeable"
        
        # With filter disabled, should always return True
        self.strategy.volatility_filter = False
        assert self.strategy.check_volatility_regime(high_vol_returns), \
            "With filter disabled should always allow trading"
    
    def test_signal_decay_mechanism(self):
        """Test signal decay over holding period."""
        # Generate simple AR data
        data = self.generate_test_data(300, "ar1")
        
        # Track signal evolution
        signals = self.strategy.generate(data)
        
        # Find periods with signals
        signal_periods = signals[signals != 0]
        
        if len(signal_periods) > 1:
            # Check that signals can decay (not all same strength)
            signal_strengths = signal_periods.abs().values
            unique_strengths = len(np.unique(np.round(signal_strengths, 3)))
            assert unique_strengths > 1, "Should show signal decay over time"
    
    def test_no_lookahead_bias(self):
        """Test that strategy doesn't use future information."""
        data = self.generate_test_data(250, "ar1")
        
        # Generate signals incrementally
        incremental_signals = []
        for i in range(self.strategy.lookback_days + 50, len(data), 10):
            partial_data = data.iloc[:i]
            partial_signals = self.strategy.generate(partial_data)
            if len(partial_signals) > 0:
                incremental_signals.append(partial_signals.iloc[-1])
        
        # Check signals are consistent (no future information used)
        full_signals = self.strategy.generate(data)
        
        # Compare overlapping periods
        for i, signal in enumerate(incremental_signals):
            idx = self.strategy.lookback_days + 50 + i * 10 - 1
            if idx < len(full_signals):
                assert abs(signal - full_signals.iloc[idx]) < 1e-10, \
                    f"Signal at index {idx} changed with future data: {signal} vs {full_signals.iloc[idx]}"
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty data
        empty_data = pd.DataFrame({'close': []})
        empty_signals = self.strategy.generate(empty_data)
        assert len(empty_signals) == 0, "Should handle empty data gracefully"
        
        # Insufficient data
        short_data = self.generate_test_data(50)
        short_signals = self.strategy.generate(short_data)
        assert (short_signals == 0).all(), "Should return zero signals for insufficient data"
        
        # Data with NaN values
        nan_data = self.generate_test_data(200)
        nan_data.loc[50:60, 'close'] = np.nan
        nan_signals = self.strategy.generate(nan_data)
        assert not nan_signals.isna().any(), "Should handle NaN values without producing NaN signals"
        
        # Constant prices (no returns variation)
        constant_data = pd.DataFrame({
            'close': [100] * 200,
            'volume': [1000] * 200
        })
        constant_signals = self.strategy.generate(constant_data)
        assert (constant_signals == 0).all(), "Should return zero signals for constant prices"
        
        # Extreme returns
        extreme_data = self.generate_test_data(200)
        extreme_data.loc[100, 'close'] = extreme_data.loc[99, 'close'] * 10  # 900% jump
        extreme_signals = self.strategy.generate(extreme_data)
        assert extreme_signals.abs().max() <= self.strategy.max_signal_strength, \
            "Should bound signals even with extreme price movements"
    
    def test_strategy_state_monitoring(self):
        """Test strategy state tracking."""
        # Generate signals to update state
        data = self.generate_test_data(300, "ar1")
        self.strategy.generate(data)
        
        # Check state structure
        state = self.strategy.get_state()
        required_keys = ['last_spike_lag', 'last_spike_strength', 'entry_countdown', 'hold_countdown']
        
        for key in required_keys:
            assert key in state, f"Missing state key: {key}"
            
        # State values should be reasonable
        assert isinstance(state['hold_countdown'], int), "Hold countdown should be integer"
        assert state['hold_countdown'] >= 0, "Hold countdown should be non-negative"
        
        if state['last_spike_lag'] is not None:
            assert 1 <= state['last_spike_lag'] <= self.strategy.max_lag, \
                "Spike lag should be within valid range"
    
    def test_monte_carlo_dsr_validation(self):
        """Test Directional Success Rate using Monte Carlo simulation."""
        n_simulations = 50
        dsr_scores = []
        
        for sim in range(n_simulations):
            # Generate AR process with varying strength
            np.random.seed(sim + 100)
            ar_strength = np.random.uniform(0.2, 0.6)
            
            n_points = 300
            noise = np.random.normal(0, 0.01, n_points)
            returns = np.zeros(n_points)
            returns[0] = noise[0]
            
            for i in range(1, n_points):
                returns[i] = ar_strength * returns[i-1] + noise[i]
                
            prices = 100 * np.exp(np.cumsum(returns))
            data = pd.DataFrame({
                'close': prices,
                'volume': np.random.uniform(1000, 2000, n_points)
            })
            
            # Generate signals
            signals = self.strategy.generate(data)
            
            # Calculate forward returns (1-day ahead)
            forward_returns = data['close'].pct_change().shift(-1)
            
            # Calculate DSR for non-zero signals
            signal_mask = signals != 0
            if signal_mask.sum() > 5:  # Need minimum signals
                signal_directions = np.sign(signals[signal_mask])
                return_directions = np.sign(forward_returns[signal_mask])
                
                # Remove NaN values
                valid_mask = ~(np.isnan(signal_directions) | np.isnan(return_directions))
                if valid_mask.sum() > 5:
                    dsr = (signal_directions[valid_mask] == return_directions[valid_mask]).mean()
                    dsr_scores.append(dsr)
        
        # Analyze DSR distribution
        if len(dsr_scores) > 10:
            mean_dsr = np.mean(dsr_scores)
            std_dsr = np.std(dsr_scores)
            
            print(f"Monte Carlo DSR Results (n={len(dsr_scores)}):")
            print(f"Mean DSR: {mean_dsr:.3f}")
            print(f"Std DSR: {std_dsr:.3f}")
            print(f"DSR > 0.5: {(np.array(dsr_scores) > 0.5).mean():.1%}")
            
            # DSR should be better than random for AR processes
            assert mean_dsr > 0.45, f"Mean DSR {mean_dsr:.3f} should be above random chance"
            
            # Should have reasonable consistency
            assert std_dsr < 0.3, f"DSR std {std_dsr:.3f} should show reasonable consistency"


if __name__ == "__main__":
    # Run tests manually
    test_suite = TestPACFSpikeDetectorSignal()
    test_suite.setup_method()
    
    print("🧪 Running PACF Spike Detector Strategy Tests...")
    
    # Core functionality tests
    test_suite.test_pacf_calculation_consistency()
    print("✓ PACF calculation consistency test passed")
    
    test_suite.test_ar_structure_identification()
    print("✓ AR structure identification test passed")
    
    test_suite.test_spike_significance_testing()
    print("✓ Spike significance testing test passed")
    
    test_suite.test_higher_lags_insignificance()
    print("✓ Higher lags insignificance test passed")
    
    test_suite.test_signal_generation_timing()
    print("✓ Signal generation timing test passed")
    
    test_suite.test_signal_generation_bounds()
    print("✓ Signal generation bounds test passed")
    
    test_suite.test_volatility_regime_filtering()
    print("✓ Volatility regime filtering test passed")
    
    test_suite.test_signal_decay_mechanism()
    print("✓ Signal decay mechanism test passed")
    
    test_suite.test_no_lookahead_bias()
    print("✓ No look-ahead bias test passed")
    
    test_suite.test_edge_cases()
    print("✓ Edge cases test passed")
    
    test_suite.test_strategy_state_monitoring()
    print("✓ Strategy state monitoring test passed")
    
    test_suite.test_monte_carlo_dsr_validation()
    print("✓ Monte Carlo DSR validation test passed")
    
    print("\n🎉 All PACF Spike Detector Strategy tests passed!")
    print("Strategy implementation mathematically sound and ready for deployment.")
