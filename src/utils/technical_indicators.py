"""
Technical indicators implemented from scratch.
No external dependencies like talib required.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from numba import jit


@jit(nopython=True)
def _rsi_numba(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate RSI using numba for performance.
    
    Args:
        prices: Array of closing prices
        period: RSI period (default 14)
        
    Returns:
        Array of RSI values
    """
    n = len(prices)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # Initial average gain and loss (SMA for first period)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # Calculate first RSI value
    if avg_loss > 0:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    else:
        rsi[period] = 100.0 if avg_gain > 0 else 50.0
    
    # Calculate subsequent RSI values using exponential smoothing
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0 if avg_gain > 0 else 50.0
    
    return rsi


@jit(nopython=True)
def _ema_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using numba.
    
    Args:
        prices: Array of prices
        period: EMA period
        
    Returns:
        Array of EMA values
    """
    n = len(prices)
    ema = np.full(n, np.nan)
    
    if n < period:
        return ema
    
    # Calculate smoothing factor
    alpha = 2.0 / (period + 1.0)
    
    # Initialize with SMA
    ema[period - 1] = np.mean(prices[:period])
    
    # Calculate EMA
    for i in range(period, n):
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1]
    
    return ema


@jit(nopython=True)
def _macd_numba(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate MACD using numba for performance.
    
    Args:
        prices: Array of closing prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    n = len(prices)
    
    # Calculate EMAs
    ema_fast = _ema_numba(prices, fast)
    ema_slow = _ema_numba(prices, slow)
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD line)
    # Filter out NaN values for signal calculation
    macd_valid_start = slow - 1  # First valid MACD value
    signal_line = np.full(n, np.nan)
    
    if n > macd_valid_start:
        # Calculate signal line starting from first valid MACD
        macd_for_signal = macd_line[macd_valid_start:]
        signal_ema = _ema_numba(macd_for_signal, signal)
        signal_line[macd_valid_start:] = signal_ema
    
    # Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


@jit(nopython=True)
def _sma_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average using numba.
    
    Args:
        prices: Array of prices
        period: SMA period
        
    Returns:
        Array of SMA values
    """
    n = len(prices)
    sma = np.full(n, np.nan)
    
    if n < period:
        return sma
    
    # Calculate rolling sum efficiently
    window_sum = np.sum(prices[:period])
    sma[period - 1] = window_sum / period
    
    for i in range(period, n):
        window_sum = window_sum - prices[i - period] + prices[i]
        sma[i] = window_sum / period
    
    return sma


@jit(nopython=True)
def _bollinger_bands_numba(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands using numba.
    
    Args:
        prices: Array of closing prices
        period: Moving average period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)
        
    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    n = len(prices)
    middle_band = np.full(n, np.nan)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    if n < period:
        return middle_band, upper_band, lower_band
    
    # Calculate rolling mean and std
    for i in range(period - 1, n):
        window = prices[i - period + 1:i + 1]
        mean_val = np.mean(window)
        std_val = np.std(window)
        
        middle_band[i] = mean_val
        upper_band[i] = mean_val + std_dev * std_val
        lower_band[i] = mean_val - std_dev * std_val
    
    return middle_band, upper_band, lower_band


class TechnicalIndicators:
    """
    Collection of technical indicators implemented from scratch.
    """
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Series of closing prices
            period: RSI period (default 14)
            
        Returns:
            Series of RSI values (0-100)
        """
        if len(prices) < period + 1:
            return pd.Series(index=prices.index, dtype=float)
        
        rsi_values = _rsi_numba(prices.values, period)
        return pd.Series(rsi_values, index=prices.index)
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Series of closing prices
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        if len(prices) < slow + signal:
            empty_series = pd.Series(index=prices.index, dtype=float)
            return empty_series, empty_series, empty_series
        
        macd_line, signal_line, histogram = _macd_numba(prices.values, fast, slow, signal)
        
        return (
            pd.Series(macd_line, index=prices.index),
            pd.Series(signal_line, index=prices.index),
            pd.Series(histogram, index=prices.index)
        )
    
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: Series of prices
            period: EMA period
            
        Returns:
            Series of EMA values
        """
        if len(prices) < period:
            return pd.Series(index=prices.index, dtype=float)
        
        ema_values = _ema_numba(prices.values, period)
        return pd.Series(ema_values, index=prices.index)
    
    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: Series of prices
            period: SMA period
            
        Returns:
            Series of SMA values
        """
        if len(prices) < period:
            return pd.Series(index=prices.index, dtype=float)
        
        sma_values = _sma_numba(prices.values, period)
        return pd.Series(sma_values, index=prices.index)
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Series of closing prices
            period: Moving average period (default 20)
            std_dev: Standard deviation multiplier (default 2.0)
            
        Returns:
            Tuple of (middle_band, upper_band, lower_band)
        """
        if len(prices) < period:
            empty_series = pd.Series(index=prices.index, dtype=float)
            return empty_series, empty_series, empty_series
        
        middle, upper, lower = _bollinger_bands_numba(prices.values, period, std_dev)
        
        return (
            pd.Series(middle, index=prices.index),
            pd.Series(upper, index=prices.index),
            pd.Series(lower, index=prices.index)
        )
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            k_period: %K period (default 14)
            d_period: %D period (default 3)
            
        Returns:
            Tuple of (%K, %D)
        """
        if len(close) < k_period:
            empty_series = pd.Series(index=close.index, dtype=float)
            return empty_series, empty_series
        
        # Calculate %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        
        # Calculate %D (SMA of %K)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Williams %R.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: Lookback period (default 14)
            
        Returns:
            Series of Williams %R values (-100 to 0)
        """
        if len(close) < period:
            return pd.Series(index=close.index, dtype=float)
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return williams_r
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: ATR period (default 14)
            
        Returns:
            Series of ATR values
        """
        if len(close) < 2:
            return pd.Series(index=close.index, dtype=float)
        
        # Calculate True Range components
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the moving average of True Range
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: ADX period (default 14)
            
        Returns:
            Series of ADX values (0-100)
        """
        if len(close) < period + 1:
            return pd.Series(index=close.index, dtype=float)
        
        # Calculate True Range
        atr_values = TechnicalIndicators.atr(high, low, close, period)
        
        # Calculate Directional Movement
        dm_plus = (high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0)
        dm_minus = (low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0)
        
        dm_plus = dm_plus.where(dm_plus > 0, 0)
        dm_minus = dm_minus.where(dm_minus > 0, 0)
        
        # Calculate smoothed DM and ATR
        dm_plus_smooth = dm_plus.rolling(window=period).mean()
        dm_minus_smooth = dm_minus.rolling(window=period).mean()
        
        # Calculate DI+ and DI-
        di_plus = 100 * dm_plus_smooth / atr_values
        di_minus = 100 * dm_minus_smooth / atr_values
        
        # Calculate DX
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
        
        # Calculate ADX (smoothed DX)
        adx = dx.rolling(window=period).mean()
        
        return adx
