"""
True-Range Divergence Mean-Reversion Trading Strategy

Mathematical Foundation:
This strategy exploits divergences between True Range volatility and price momentum
to identify mean-reversion opportunities. True Range captures the largest price movement
within a single period, incorporating gaps and intraday volatility.

The strategy identifies:
1. High True Range periods (volatility spikes)
2. Divergences between TR trends and price trends
3. Mean-reversion signals when volatility-momentum relationships break down

Key Mathematical Components:
- True Range: TR = max(H-L, |H-C_prev|, |L-C_prev|)
- TR Relative Strength: TR_t / MA(TR, n)  
- Price Momentum: Close_t / MA(Close, n)
- Divergence Signal: when TR trend ≠ Price trend direction
- Mean-reversion timing based on volatility regime normalization

DSR Requirements: ≥ 0.95 for production deployment
"""

import numpy as np
import pandas as pd
import yaml
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class TrueRangeDivergenceSignal:
    """
    True-Range Divergence Mean-Reversion Strategy
    
    Generates mean-reversion signals when volatility (True Range) and price momentum
    show divergent patterns, indicating potential reversal opportunities.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize strategy with configuration parameters."""
        if config_path is None:
            config_path = "config/strategies/true_range_divergence.yaml"
        
        self.config_path = config_path
        self.load_config()
        self.reset_state()
        
    def load_config(self) -> None:
        """Load strategy configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Core parameters
            self.tr_lookback = config.get('tr_lookback', 20)
            self.momentum_lookback = config.get('momentum_lookback', 14)
            self.divergence_threshold = config.get('divergence_threshold', 0.3)
            self.min_tr_strength = config.get('min_tr_strength', 1.2)
            self.max_tr_strength = config.get('max_tr_strength', 3.0)
            
            # Signal generation
            self.entry_delay_days = config.get('entry_delay_days', 1)
            self.max_hold_days = config.get('max_hold_days', 8)
            self.signal_decay = config.get('signal_decay', 0.85)
            self.max_signal_strength = config.get('max_signal_strength', 1.0)
            
            # Volatility filtering
            self.volatility_filter = config.get('volatility_filter', True)
            self.vol_lookback = config.get('vol_lookback', 30)
            self.vol_percentile_low = config.get('vol_percentile_low', 25)
            self.vol_percentile_high = config.get('vol_percentile_high', 75)
            
            # Risk management
            self.min_observations = config.get('min_observations', 50)
            self.confidence_level = config.get('confidence_level', 0.05)
            
        except FileNotFoundError:
            # Use default parameters if config file not found
            self.tr_lookback = 20
            self.momentum_lookback = 14
            self.divergence_threshold = 0.3
            self.min_tr_strength = 1.2
            self.max_tr_strength = 3.0
            self.entry_delay_days = 1
            self.max_hold_days = 8
            self.signal_decay = 0.85
            self.max_signal_strength = 1.0
            self.volatility_filter = True
            self.vol_lookback = 30
            self.vol_percentile_low = 25
            self.vol_percentile_high = 75
            self.min_observations = 50
            self.confidence_level = 0.05
    
    def reset_state(self) -> None:
        """Reset strategy state for new run."""
        self.state = {
            'last_divergence_strength': None,
            'last_tr_direction': None,
            'last_momentum_direction': None,
            'entry_countdown': 0,
            'hold_countdown': 0,
            'current_signal': 0.0
        }
    
    def calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate True Range for each period.
        
        TR = max(H-L, |H-C_prev|, |L-C_prev|)
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            True Range series
        """
        high = data['high']
        low = data['low']
        close = data['close']
        prev_close = close.shift(1)
        
        # Calculate the three components
        hl = high - low  # High - Low
        hc = (high - prev_close).abs()  # |High - Previous Close|
        lc = (low - prev_close).abs()   # |Low - Previous Close|
        
        # True Range is the maximum of these three
        true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        
        return true_range
    
    def calculate_tr_relative_strength(self, true_range: pd.Series) -> pd.Series:
        """
        Calculate True Range relative strength vs moving average.
        
        Args:
            true_range: True Range series
            
        Returns:
            TR relative strength (TR / MA(TR))
        """
        tr_ma = true_range.rolling(window=self.tr_lookback, min_periods=self.tr_lookback//2).mean()
        tr_strength = true_range / tr_ma
        
        return tr_strength
    
    def calculate_price_momentum(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate price momentum relative to moving average.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Price momentum (Close / MA(Close))
        """
        close = data['close']
        price_ma = close.rolling(window=self.momentum_lookback, min_periods=self.momentum_lookback//2).mean()
        momentum = close / price_ma
        
        return momentum
    
    def detect_divergence(self, tr_strength: pd.Series, momentum: pd.Series) -> Dict[str, Any]:
        """
        Simplified divergence detection - compares longer-term trends.
        
        Args:
            tr_strength: TR relative strength series
            momentum: Price momentum series
            
        Returns:
            Dictionary with divergence information
        """
        min_length = 15  # Need sufficient data for reliable trends
        if len(tr_strength) < min_length or len(momentum) < min_length:
            return {
                'has_divergence': False,
                'strength': 0.0,
                'tr_direction': 0,
                'momentum_direction': 0,
                'signal_direction': 0
            }
        
        # Use longer lookback for more stable trend detection
        lookback = min(15, len(tr_strength) // 3)  # Use 1/3 of available data, max 15
        
        tr_recent = tr_strength.tail(lookback)
        momentum_recent = momentum.tail(lookback)
        
        # Simple trend detection: compare first half vs second half
        tr_first_half = tr_recent.iloc[:len(tr_recent)//2].mean()
        tr_second_half = tr_recent.iloc[len(tr_recent)//2:].mean()
        
        momentum_first_half = momentum_recent.iloc[:len(momentum_recent)//2].mean()
        momentum_second_half = momentum_recent.iloc[len(momentum_recent)//2:].mean()
        
        # Calculate relative changes (more robust than slopes)
        tr_change = (tr_second_half - tr_first_half) / tr_first_half
        momentum_change = (momentum_second_half - momentum_first_half) / momentum_first_half
        
        # Determine directions with more lenient threshold
        change_threshold = 0.005  # 0.5% change required
        
        tr_direction = 1 if tr_change > change_threshold else (-1 if tr_change < -change_threshold else 0)
        momentum_direction = 1 if momentum_change > change_threshold else (-1 if momentum_change < -change_threshold else 0)
        
        # Check for divergence (opposite non-zero directions) 
        has_divergence = (tr_direction != 0 and momentum_direction != 0 and tr_direction != momentum_direction)
        
        if not has_divergence:
            return {
                'has_divergence': False,
                'strength': 0.0,
                'tr_direction': tr_direction,
                'momentum_direction': momentum_direction,
                'signal_direction': 0
            }
        
        # Calculate divergence strength based on magnitude of opposing changes
        divergence_magnitude = abs(tr_change) + abs(momentum_change)
        
        # Scale by current levels
        tr_current = tr_strength.iloc[-1]
        momentum_current = momentum.iloc[-1]
        
        # Basic strength calculation
        divergence_strength = divergence_magnitude * max(0.5, tr_current - 0.5)
        
        # Signal direction: fade the momentum (mean reversion)
        signal_direction = -momentum_direction
        
        return {
            'has_divergence': divergence_strength >= 0.05,  # Much lower threshold
            'strength': divergence_strength,
            'tr_direction': tr_direction,
            'momentum_direction': momentum_direction,
            'signal_direction': signal_direction
        }
    
    def check_volatility_regime(self, tr_series: pd.Series) -> bool:
        """
        Check if current volatility regime is suitable for trading.
        
        Args:
            tr_series: True Range series
            
        Returns:
            True if regime is suitable for trading
        """
        if not self.volatility_filter:
            return True
        
        if len(tr_series) < self.vol_lookback:
            return False
        
        # Get recent volatility level
        recent_tr = tr_series.tail(self.vol_lookback)
        current_tr = tr_series.iloc[-1]
        
        # Calculate percentile position
        percentile = (recent_tr < current_tr).mean() * 100
        
        # Trade in moderate to high volatility regimes
        return self.vol_percentile_low <= percentile <= self.vol_percentile_high
    
    def generate_divergence_signal(self, divergence_info: Dict[str, Any], 
                                 price_change: float) -> float:
        """
        Generate trading signal based on divergence information.
        
        Args:
            divergence_info: Divergence analysis results
            price_change: Recent price change
            
        Returns:
            Signal strength [-1, 1]
        """
        if not divergence_info['has_divergence']:
            return 0.0
        
        base_signal = divergence_info['signal_direction']
        strength = divergence_info['strength']
        
        # Scale signal by divergence strength
        signal = base_signal * min(strength / 2.0, 1.0)
        
        # Timing adjustment: stronger signal when price moves against the reversal
        if base_signal > 0 and price_change < 0:  # Buy signal after decline
            signal *= 1.2
        elif base_signal < 0 and price_change > 0:  # Sell signal after advance
            signal *= 1.2
        else:
            signal *= 0.8  # Weaker signal if price already moving in reversal direction
        
        # Bound signal
        signal = np.clip(signal, -self.max_signal_strength, self.max_signal_strength)
        
        return signal
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals for the entire dataset.
        
        Args:
            data: OHLCV DataFrame with columns [open, high, low, close, volume]
            
        Returns:
            Series of trading signals [-1, 1] aligned with input data
        """
        self.reset_state()
        
        if len(data) < max(self.tr_lookback, self.momentum_lookback) + self.min_observations:
            return pd.Series(0.0, index=data.index)
        
        # Calculate True Range
        true_range = self.calculate_true_range(data)
        
        # Calculate relative strength and momentum
        tr_strength = self.calculate_tr_relative_strength(true_range)
        price_momentum = self.calculate_price_momentum(data)
        
        signals = []
        
        for i in range(len(data)):
            if i < max(self.tr_lookback, self.momentum_lookback):
                signals.append(0.0)
                continue
            
            # Extract data up to current point
            tr_window = tr_strength.iloc[:i+1]
            momentum_window = price_momentum.iloc[:i+1]
            tr_regime_window = true_range.iloc[:i+1]
            
            if len(tr_window) < self.tr_lookback or len(momentum_window) < self.momentum_lookback:
                signals.append(0.0)
                continue
            
            # Check volatility regime
            if not self.check_volatility_regime(tr_regime_window):
                signals.append(0.0)
                continue
            
            # Detect divergence
            divergence_info = self.detect_divergence(tr_window, momentum_window)
            
            # Update state
            self.state['last_divergence_strength'] = divergence_info['strength']
            self.state['last_tr_direction'] = divergence_info['tr_direction']
            self.state['last_momentum_direction'] = divergence_info['momentum_direction']
            
            # Handle existing signal decay and holding period
            if self.state['hold_countdown'] > 0:
                # Decay existing signal
                self.state['current_signal'] *= self.signal_decay
                self.state['hold_countdown'] -= 1
                signals.append(self.state['current_signal'])
                continue
            
            # Check for new signal entry
            if self.state['entry_countdown'] > 0:
                self.state['entry_countdown'] -= 1
                signals.append(0.0)
                continue
            
            # Generate new signal if divergence detected
            if divergence_info['has_divergence']:
                # Calculate recent price change for timing
                price_change = data['close'].iloc[i] / data['close'].iloc[i-1] - 1 if i > 0 else 0
                
                # Generate signal
                signal = self.generate_divergence_signal(divergence_info, price_change)
                
                if abs(signal) > 0.1:  # Minimum signal threshold
                    self.state['current_signal'] = signal
                    self.state['entry_countdown'] = self.entry_delay_days
                    self.state['hold_countdown'] = self.max_hold_days
                    signals.append(signal)
                else:
                    signals.append(0.0)
            else:
                signals.append(0.0)
        
        return pd.Series(signals, index=data.index)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current strategy state for monitoring."""
        return self.state.copy()
