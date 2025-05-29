"""
Consolidated utility functions to eliminate duplicates across the codebase.
This module serves as the single source of truth for common calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from numba import jit
import warnings


# ============================================================================
# PERFORMANCE METRICS - SINGLE SOURCE OF TRUTH
# ============================================================================

@jit(nopython=True)
def calculate_sharpe_ratio_optimized(returns: np.ndarray, risk_free_rate: float = 0.0, 
                                   periods_per_year: int = 252) -> float:
    """
    Optimized Sharpe ratio calculation - single source of truth.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Annualized Sharpe ratio
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
def calculate_sortino_ratio_optimized(returns: np.ndarray, target_return: float = 0.0,
                                    periods_per_year: int = 252) -> float:
    """
    Optimized Sortino ratio calculation - single source of truth.
    
    Args:
        returns: Array of returns
        target_return: Target return threshold
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    target_daily = target_return / periods_per_year
    
    # Calculate downside deviation
    downside_returns = returns[returns < target_daily]
    
    if len(downside_returns) == 0:
        return float('inf') if mean_return > target_daily else 0.0
    
    downside_deviation = np.std(downside_returns)
    
    if downside_deviation == 0:
        return 0.0
    
    excess_return = mean_return - target_daily
    return (excess_return / downside_deviation) * np.sqrt(periods_per_year)


@jit(nopython=True)
def calculate_max_drawdown_optimized(cumulative_returns: np.ndarray) -> Tuple[float, int, int]:
    """
    Optimized maximum drawdown calculation - single source of truth.
    
    Args:
        cumulative_returns: Array of cumulative returns (equity curve)
        
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


@jit(nopython=True)
def calculate_calmar_ratio_optimized(annualized_return: float, max_drawdown: float) -> float:
    """
    Optimized Calmar ratio calculation - single source of truth.
    
    Args:
        annualized_return: Annualized return
        max_drawdown: Maximum drawdown
        
    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return np.inf if annualized_return > 0 else 0.0
    return abs(annualized_return / max_drawdown)


@jit(nopython=True)
def calculate_information_ratio_optimized(excess_returns: np.ndarray, 
                                        periods_per_year: int = 252) -> float:
    """
    Optimized Information ratio calculation - single source of truth.
    
    Args:
        excess_returns: Array of excess returns vs benchmark
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Information ratio
    """
    if len(excess_returns) == 0:
        return 0.0
    
    mean_excess = np.mean(excess_returns)
    tracking_error = np.std(excess_returns)
    
    if tracking_error == 0:
        return 0.0
    
    return (mean_excess / tracking_error) * np.sqrt(periods_per_year)


# ============================================================================
# COMPREHENSIVE METRICS CALCULATION
# ============================================================================

def calculate_comprehensive_metrics(returns: pd.Series, 
                                  benchmark_returns: Optional[pd.Series] = None,
                                  risk_free_rate: float = 0.02,
                                  periods_per_year: int = 252) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics using optimized functions.
    
    Args:
        returns: Return series
        benchmark_returns: Optional benchmark returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Periods per year for annualization
        
    Returns:
        Dictionary of comprehensive metrics
    """
    if len(returns) == 0:
        return {}
    
    # Convert to numpy array if needed
    returns_array = returns.values if hasattr(returns, 'values') else returns
    
    # Basic metrics
    total_return = np.prod(1 + returns_array) - 1
    annualized_return = np.mean(returns_array) * periods_per_year
    volatility = np.std(returns_array) * np.sqrt(periods_per_year)
    
    # Risk-adjusted metrics
    sharpe_ratio = calculate_sharpe_ratio_optimized(returns_array, risk_free_rate, periods_per_year)
    sortino_ratio = calculate_sortino_ratio_optimized(returns_array, 0.0, periods_per_year)
    
    # Drawdown metrics
    if isinstance(returns, pd.Series):
        cumulative = (1 + returns).cumprod().values
    else:
        cumulative = np.cumprod(1 + returns_array)
    max_dd, dd_start, dd_end = calculate_max_drawdown_optimized(cumulative)
    calmar_ratio = calculate_calmar_ratio_optimized(annualized_return, max_dd)
    
    # Distribution metrics
    if isinstance(returns, pd.Series):
        skewness = float(returns.skew())
        kurtosis = float(returns.kurtosis())
    else:
        from scipy import stats
        skewness = float(stats.skew(returns_array))
        kurtosis = float(stats.kurtosis(returns_array))
    
    # Trading metrics
    win_rate = float(np.mean(returns_array > 0))
    
    # VaR and CVaR
    var_95 = float(np.percentile(returns_array, 5))
    cvar_95 = float(np.mean(returns_array[returns_array <= var_95]))
    
    metrics = {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'volatility': float(volatility),
        'sharpe_ratio': float(sharpe_ratio),
        'sortino_ratio': float(sortino_ratio),
        'calmar_ratio': float(calmar_ratio),
        'max_drawdown': float(max_dd),
        'skewness': skewness,
        'kurtosis': kurtosis,
        'win_rate': win_rate,
        'var_95': var_95,
        'cvar_95': cvar_95
    }
    
    # Benchmark comparison if provided
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        benchmark_aligned = benchmark_returns.reindex(returns.index, method='ffill').dropna()
        returns_aligned = returns.reindex(benchmark_aligned.index)
        
        if len(returns_aligned) > 0:
            excess_returns = (returns_aligned - benchmark_aligned).values
            information_ratio = calculate_information_ratio_optimized(excess_returns, periods_per_year)
            
            # Beta calculation
            if np.std(benchmark_aligned) > 0:
                beta = np.cov(returns_aligned, benchmark_aligned)[0, 1] / np.var(benchmark_aligned)
            else:
                beta = 0.0
            
            # Alpha calculation
            alpha = annualized_return - (risk_free_rate + beta * (np.mean(benchmark_aligned) * periods_per_year - risk_free_rate))
            
            metrics.update({
                'information_ratio': float(information_ratio),
                'beta': float(beta),
                'alpha': float(alpha),
                'tracking_error': float(np.std(excess_returns) * np.sqrt(periods_per_year))
            })
    
    return metrics


# ============================================================================
# DRAWDOWN ANALYSIS
# ============================================================================

def analyze_drawdowns(returns: pd.Series) -> Dict[str, Any]:
    """
    Comprehensive drawdown analysis.
    
    Args:
        returns: Return series
        
    Returns:
        Dictionary with detailed drawdown statistics
    """
    if len(returns) == 0:
        return {}
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    # Find all drawdown periods
    in_drawdown = drawdown < -0.001  # Small threshold to avoid noise
    drawdown_periods = []
    
    current_start = None
    for i, is_dd in enumerate(in_drawdown):
        if is_dd and current_start is None:
            current_start = i
        elif not is_dd and current_start is not None:
            # End of drawdown period
            period_drawdown = drawdown.iloc[current_start:i]
            max_dd_in_period = abs(period_drawdown.min())
            duration = i - current_start
            
            drawdown_periods.append({
                'start_date': returns.index[current_start],
                'end_date': returns.index[i-1],
                'duration': duration,
                'max_drawdown': max_dd_in_period,
                'recovery_date': returns.index[i]
            })
            current_start = None
    
    # Handle case where series ends in drawdown
    if current_start is not None:
        period_drawdown = drawdown.iloc[current_start:]
        max_dd_in_period = abs(period_drawdown.min())
        duration = len(returns) - current_start
        
        drawdown_periods.append({
            'start_date': returns.index[current_start],
            'end_date': returns.index[-1],
            'duration': duration,
            'max_drawdown': max_dd_in_period,
            'recovery_date': None  # Still in drawdown
        })
    
    # Summary statistics
    if drawdown_periods:
        max_drawdown = max(dd['max_drawdown'] for dd in drawdown_periods)
        avg_drawdown = np.mean([dd['max_drawdown'] for dd in drawdown_periods])
        max_duration = max(dd['duration'] for dd in drawdown_periods)
        avg_duration = np.mean([dd['duration'] for dd in drawdown_periods])
    else:
        max_drawdown = 0
        avg_drawdown = 0
        max_duration = 0
        avg_duration = 0
    
    return {
        'max_drawdown': float(max_drawdown),
        'average_drawdown': float(avg_drawdown),
        'max_duration': int(max_duration),
        'average_duration': float(avg_duration),
        'number_of_drawdowns': len(drawdown_periods),
        'drawdown_periods': drawdown_periods,
        'time_in_drawdown': float(in_drawdown.mean())
    }


# ============================================================================
# RISK METRICS
# ============================================================================

@jit(nopython=True)
def calculate_var_optimized(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """
    Optimized Value at Risk calculation.
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
        
    Returns:
        Value at Risk
    """
    if len(returns) == 0:
        return 0.0
    
    return np.percentile(returns, confidence_level * 100)


@jit(nopython=True)
def calculate_cvar_optimized(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """
    Optimized Conditional Value at Risk calculation.
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.05 for 95% CVaR)
        
    Returns:
        Conditional Value at Risk
    """
    if len(returns) == 0:
        return 0.0
    
    var_threshold = np.percentile(returns, confidence_level * 100)
    tail_returns = returns[returns <= var_threshold]
    
    if len(tail_returns) == 0:
        return var_threshold
    
    return np.mean(tail_returns)


# ============================================================================
# EXPORT OPTIMIZED FUNCTIONS
# ============================================================================

# Export the optimized functions to replace duplicates
calculate_sharpe_ratio = calculate_sharpe_ratio_optimized
calculate_sortino_ratio = calculate_sortino_ratio_optimized
calculate_max_drawdown = calculate_max_drawdown_optimized
calculate_calmar_ratio = calculate_calmar_ratio_optimized
calculate_information_ratio = calculate_information_ratio_optimized
calculate_var = calculate_var_optimized
calculate_cvar = calculate_cvar_optimized

__all__ = [
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio', 
    'calculate_max_drawdown',
    'calculate_calmar_ratio',
    'calculate_information_ratio',
    'calculate_var',
    'calculate_cvar',
    'calculate_comprehensive_metrics',
    'analyze_drawdowns'
]
