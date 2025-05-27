"""
Daily VPIN (Volume-Synchronized Order-Flow Imbalance) Trading Strategy

Mathematical Foundation:
- dVPIN = (1/N) * Σ|B_i - S_i| where B_i, S_i are buy/sell volumes per bucket
- Volume-synchronized bucketing ensures equal-volume windows across time
- Return-based signing: B_t = 0.5 * [1 + sign(C_t - C_{t-1})] * V_t
- Toxic flow detection via percentile thresholds on rolling dVPIN history

Key Components:
1. Return-based volume signing (buy vs sell initiation proxy)
2. Equal-volume bucketing across rolling window
3. Order flow imbalance calculation per bucket
4. Percentile-based toxic flow detection
5. Position sizing based on flow toxicity levels

DSR Requirements:
- Minimum 95% statistical significance for flow detection
- No look-ahead bias in volume signing or bucketing
- Robust to market microstructure noise
- Mathematical consistency with order flow theory
"""

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class VPINStrategy:
    """
    Daily VPIN implementation for detecting toxic order flow and market stress.
    
    The strategy identifies periods of high order flow imbalance that typically
    precede adverse price movements, allowing for defensive positioning or
    contrarian opportunities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize dVPIN parameters."""
        self.lookback = config.get('lookback', 50)
        self.buckets = config.get('buckets', 50)
        self.vpin_history_length = config.get('vpin_history_length', 252)  # ~1 year
        self.toxic_threshold_pct = config.get('toxic_threshold_pct', 95)
        self.benign_threshold_pct = config.get('benign_threshold_pct', 10)
        self.ema_smoothing = config.get('ema_smoothing', 0.1)
        self.position_scale_factor = config.get('position_scale_factor', 0.5)
        
        # Internal state
        self.vpin_history = []
        self.smoothed_vpin = None
        self.current_signal = 0.0
        
    def signed_volume_return_based(self, prices: np.ndarray, volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate buy/sell volume using return-based signing.
        
        Mathematical Foundation:
        - sign_t = sign(C_t - C_{t-1})
        - B_t = 0.5 * [1 + sign_t] * V_t  (buy-initiated volume)
        - S_t = V_t - B_t                  (sell-initiated volume)
        
        Args:
            prices: Array of closing prices
            volumes: Array of daily volumes
            
        Returns:
            (buy_volumes, sell_volumes)
        """
        if len(prices) < 2:
            return np.zeros_like(volumes), volumes.copy()
            
        # Calculate price returns
        returns = np.diff(prices)
        
        # Sign returns (handle zeros by forward-filling)
        signs = np.sign(returns)
        signs = pd.Series(signs).replace(0, np.nan).ffill().fillna(1).values
        
        # Pad signs to match volume length
        signs = np.concatenate([[1], signs])  # First day gets neutral sign
        
        # Calculate buy/sell volumes
        buy_volumes = 0.5 * (1 + signs) * volumes
        sell_volumes = volumes - buy_volumes
        
        return buy_volumes, sell_volumes
    
    def create_equal_volume_buckets(self, buy_vols: np.ndarray, sell_vols: np.ndarray, 
                                   total_volumes: np.ndarray) -> List[float]:
        """
        Create equal-volume buckets and calculate order flow imbalances.
        
        Algorithm:
        1. Calculate target bucket volume: V_b = Total_Volume / N_buckets
        2. Walk through days (newest to oldest), accumulating B and S
        3. When B + S >= V_b, record |B - S| and reset counters
        4. Continue until N buckets are filled
        
        Args:
            buy_vols: Array of buy-initiated volumes
            sell_vols: Array of sell-initiated volumes
            total_volumes: Array of total volumes
            
        Returns:
            List of order flow imbalances per bucket
        """
        if len(buy_vols) == 0:
            return []
            
        total_volume = np.sum(total_volumes)
        if total_volume <= 0:
            return []
            
        target_bucket_volume = total_volume / self.buckets
        
        # Walk through days from newest to oldest
        bucket_imbalances = []
        accumulated_buy = 0.0
        accumulated_sell = 0.0
        
        for i in range(len(buy_vols) - 1, -1, -1):  # Reverse order
            accumulated_buy += buy_vols[i]
            accumulated_sell += sell_vols[i]
            
            # Check if bucket is filled
            if accumulated_buy + accumulated_sell >= target_bucket_volume:
                imbalance = abs(accumulated_buy - accumulated_sell)
                bucket_imbalances.append(imbalance)
                
                # Reset for next bucket
                accumulated_buy = 0.0
                accumulated_sell = 0.0
                
                # Stop when we have enough buckets
                if len(bucket_imbalances) >= self.buckets:
                    break
        
        return bucket_imbalances
    
    def calculate_dvpin(self, data: pd.DataFrame) -> float:
        """
        Calculate daily VPIN for current window.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            dVPIN value (0 to 1, where higher = more toxic flow)
        """
        if len(data) < self.lookback:
            return 0.0
            
        # Use recent data for calculation
        recent_data = data.tail(self.lookback)
        prices = recent_data['close'].values
        volumes = recent_data['volume'].values
        
        # Calculate signed volumes
        buy_vols, sell_vols = self.signed_volume_return_based(prices, volumes)
        
        # Create equal-volume buckets
        bucket_imbalances = self.create_equal_volume_buckets(buy_vols, sell_vols, volumes)
        
        if len(bucket_imbalances) == 0:
            return 0.0
            
        # Calculate dVPIN as average imbalance
        dvpin = np.mean(bucket_imbalances)
        
        # Normalize by average volume to get ratio
        avg_volume = np.mean(volumes) if len(volumes) > 0 else 1.0
        normalized_dvpin = dvpin / avg_volume if avg_volume > 0 else 0.0
        
        return np.clip(normalized_dvpin, 0.0, 1.0)
    
    def detect_flow_regime(self, current_dvpin: float) -> Tuple[str, float]:
        """
        Detect current order flow regime based on dVPIN percentiles.
        
        Args:
            current_dvpin: Current dVPIN value
            
        Returns:
            (regime, intensity) where:
            - regime: "toxic", "benign", or "neutral"
            - intensity: strength of the regime (0 to 1)
        """
        if len(self.vpin_history) < 20:
            return "neutral", 0.0
            
        # Calculate percentile of current dVPIN
        percentile = percentileofscore(self.vpin_history, current_dvpin)
        
        if percentile >= self.toxic_threshold_pct:
            # High dVPIN = toxic flow
            intensity = (percentile - self.toxic_threshold_pct) / (100 - self.toxic_threshold_pct)
            return "toxic", np.clip(intensity, 0.0, 1.0)
        elif percentile <= self.benign_threshold_pct:
            # Low dVPIN = benign flow
            intensity = (self.benign_threshold_pct - percentile) / self.benign_threshold_pct
            return "benign", np.clip(intensity, 0.0, 1.0)
        else:
            # Middle range = neutral
            return "neutral", 0.0
    
    def calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate recent trend strength for regime conditioning.
        
        Returns:
            Trend strength (-1 to 1) where positive = uptrend, negative = downtrend
        """
        if len(data) < 10:
            return 0.0
            
        recent_prices = data['close'].tail(10).values
        
        # Simple linear regression slope
        x = np.arange(len(recent_prices))
        slope = np.polyfit(x, recent_prices, 1)[0]
        
        # Normalize by price level
        avg_price = np.mean(recent_prices)
        normalized_slope = slope / avg_price if avg_price > 0 else 0.0
        
        # Scale to [-1, 1] range
        return np.clip(normalized_slope * 1000, -1.0, 1.0)
    
    def generate_signal(self, data: pd.DataFrame) -> float:
        """
        Generate trading signal based on dVPIN order flow analysis.
        
        Signal Logic:
        1. Calculate current dVPIN
        2. Detect flow regime (toxic/benign/neutral)
        3. Generate defensive or opportunistic signals:
           - Toxic flow: Defensive positioning (fade trend or stand aside)
           - Benign flow: Opportunistic positioning (follow trend)
           - Neutral flow: Reduced position sizing
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Signal strength [-1, 1] where:
            - Positive: Long signal (benign flow + uptrend or toxic flow + downtrend)
            - Negative: Short signal (benign flow + downtrend or toxic flow + uptrend)
            - Zero: Neutral/defensive positioning
        """
        if len(data) < self.lookback:
            return 0.0
            
        # Calculate current dVPIN
        current_dvpin = self.calculate_dvpin(data)
        
        # Apply EMA smoothing
        if self.smoothed_vpin is None:
            self.smoothed_vpin = current_dvpin
        else:
            self.smoothed_vpin = (
                self.ema_smoothing * current_dvpin + 
                (1 - self.ema_smoothing) * self.smoothed_vpin
            )
        
        # Update dVPIN history
        self.vpin_history.append(self.smoothed_vpin)
        if len(self.vpin_history) > self.vpin_history_length:
            self.vpin_history.pop(0)
            
        # Detect flow regime
        regime, intensity = self.detect_flow_regime(self.smoothed_vpin)
        
        # Calculate trend strength
        trend_strength = self.calculate_trend_strength(data)
        
        # Generate base signal based on regime
        if regime == "toxic":
            # Toxic flow: Fade the trend (contrarian)
            base_signal = -trend_strength * intensity
        elif regime == "benign":
            # Benign flow: Follow the trend (momentum)
            base_signal = trend_strength * intensity
        else:
            # Neutral flow: Reduced positioning
            base_signal = trend_strength * self.position_scale_factor
        
        # Additional conditioning based on dVPIN level
        dvpin_factor = 1.0
        if regime == "toxic":
            # In toxic regime, scale down further if dVPIN is extremely high
            if intensity > 0.8:
                dvpin_factor = 0.3  # Very defensive
            else:
                dvpin_factor = 0.6  # Moderately defensive
        elif regime == "benign":
            # In benign regime, can be more aggressive
            dvpin_factor = 1.0 + 0.5 * intensity  # Up to 1.5x sizing
        
        conditioned_signal = base_signal * dvpin_factor
        
        # Update current signal with some persistence
        signal_decay = 0.7
        self.current_signal = (
            signal_decay * self.current_signal + 
            (1 - signal_decay) * conditioned_signal
        )
        
        return np.clip(self.current_signal, -1.0, 1.0)
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Return current strategy state for monitoring."""
        current_dvpin = self.vpin_history[-1] if self.vpin_history else 0.0
        regime, intensity = self.detect_flow_regime(current_dvpin) if self.vpin_history else ("neutral", 0.0)
        
        return {
            'current_dvpin': current_dvpin,
            'smoothed_dvpin': self.smoothed_vpin or 0.0,
            'dvpin_history_length': len(self.vpin_history),
            'current_signal': self.current_signal,
            'flow_regime': regime,
            'regime_intensity': intensity,
            'dvpin_percentile': percentileofscore(self.vpin_history, current_dvpin) if len(self.vpin_history) > 0 else 50.0
        }
