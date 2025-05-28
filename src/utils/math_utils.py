"""
Mathematical and statistical utility functions for backtesting and analysis.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
import scipy.stats as stats
from numba import jit


@jit(nopython=True)
def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, 
                          periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    excess_return = mean_return - (risk_free_rate / periods_per_year)
    return (excess_return / std_return) * np.sqrt(periods_per_year)


@jit(nopython=True)
def calculate_sortino_ratio(returns: np.ndarray, target_return: float = 0.0,
                           periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sortino ratio.
    
    Args:
        returns: Array of returns
        target_return: Target return (annualized)
        periods_per_year: Number of periods per year
        
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    target_daily = target_return / periods_per_year
    
    # Calculate downside deviation
    downside_returns = returns[returns < target_daily]
    if len(downside_returns) == 0:
        return np.inf if mean_return > target_daily else 0.0
    
    downside_deviation = np.sqrt(np.mean((downside_returns - target_daily) ** 2))
    
    if downside_deviation == 0:
        return np.inf if mean_return > target_daily else 0.0
    
    excess_return = mean_return - target_daily
    return (excess_return / downside_deviation) * np.sqrt(periods_per_year)


@jit(nopython=True)
def calculate_max_drawdown(cumulative_returns: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown and its duration.
    
    Args:
        cumulative_returns: Array of cumulative returns
        
    Returns:
        Tuple of (max_drawdown, start_idx, end_idx)
    """
    if len(cumulative_returns) == 0:
        return 0.0, 0, 0
    
    peak = cumulative_returns[0]
    max_dd = 0.0
    start_idx = 0
    end_idx = 0
    temp_start = 0
    
    for i in range(len(cumulative_returns)):
        if cumulative_returns[i] > peak:
            peak = cumulative_returns[i]
            temp_start = i
        
        drawdown = (peak - cumulative_returns[i]) / peak if peak != 0 else 0
        
        if drawdown > max_dd:
            max_dd = drawdown
            start_idx = temp_start
            end_idx = i
    
    return max_dd, start_idx, end_idx


def rolling_correlation(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """
    Calculate rolling correlation between two series.
    
    Args:
        x: First time series
        y: Second time series
        window: Rolling window size
        
    Returns:
        Rolling correlation series
    """
    return x.rolling(window).corr(y)


@jit(nopython=True)
def exponential_moving_average(prices: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calculate exponential moving average.
    
    Args:
        prices: Array of prices
        alpha: Smoothing factor (0 < alpha <= 1)
        
    Returns:
        EMA array
    """
    if len(prices) == 0:
        return np.array([])
    
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema


def bollinger_bands(prices: pd.Series, window: int = 20, 
                   num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Price series
        window: Moving average window
        num_std: Number of standard deviations for bands
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle_band = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band


@jit(nopython=True)
def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).
    
    Args:
        total_return: Total annualized return
        max_drawdown: Maximum drawdown
        
    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return np.inf if total_return > 0 else 0.0
    return abs(total_return / max_drawdown)


def calculate_information_ratio(portfolio_returns: pd.Series, 
                              benchmark_returns: pd.Series) -> float:
    """
    Calculate Information Ratio.
    
    Args:
        portfolio_returns: Portfolio return series
        benchmark_returns: Benchmark return series
        
    Returns:
        Information ratio
    """
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = excess_returns.std()
    
    if tracking_error == 0:
        return 0.0
    
    return excess_returns.mean() / tracking_error


@jit(nopython=True)
def calculate_omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Calculate Omega ratio.
    
    Args:
        returns: Array of returns
        threshold: Threshold return
        
    Returns:
        Omega ratio
    """
    gains = np.sum(np.maximum(returns - threshold, 0))
    losses = np.sum(np.maximum(threshold - returns, 0))
    
    if losses == 0:
        return np.inf if gains > 0 else 1.0
    
    return gains / losses


def hurst_exponent(prices: np.ndarray) -> float:
    """
    Calculate Hurst exponent using R/S analysis.
    
    Args:
        prices: Array of prices
        
    Returns:
        Hurst exponent
    """
    if len(prices) < 20:
        return 0.5  # Random walk default
    
    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    
    # Different time lags
    lags = range(2, min(len(returns) // 4, 100))
    rs_values = []
    
    for lag in lags:
        # Divide series into chunks
        chunks = [returns[i:i+lag] for i in range(0, len(returns), lag) if len(returns[i:i+lag]) == lag]
        
        if len(chunks) == 0:
            continue
            
        rs_chunk = []
        for chunk in chunks:
            if len(chunk) > 1:
                mean_chunk = np.mean(chunk)
                cumdev = np.cumsum(chunk - mean_chunk)
                R = np.max(cumdev) - np.min(cumdev)
                S = np.std(chunk)
                if S > 0:
                    rs_chunk.append(R / S)
        
        if rs_chunk:
            rs_values.append(np.mean(rs_chunk))
    
    if len(rs_values) < 3:
        return 0.5
    
    # Linear regression on log(R/S) vs log(lag)
    log_lags = np.log(list(lags)[:len(rs_values)])
    log_rs = np.log(rs_values)
    
    # Remove any infinite or NaN values
    valid_mask = np.isfinite(log_lags) & np.isfinite(log_rs)
    if np.sum(valid_mask) < 3:
        return 0.5
    
    coeffs = np.polyfit(log_lags[valid_mask], log_rs[valid_mask], 1)
    return coeffs[0]  # Slope is the Hurst exponent
