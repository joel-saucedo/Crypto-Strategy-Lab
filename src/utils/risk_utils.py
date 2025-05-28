"""
Risk management and position sizing utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import warnings


def calculate_var(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
        
    Returns:
        VaR value
    """
    if len(returns) == 0:
        return 0.0
    
    return np.percentile(returns, confidence_level * 100)


def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.05 for 95% CVaR)
        
    Returns:
        CVaR value
    """
    if len(returns) == 0:
        return 0.0
    
    var = calculate_var(returns, confidence_level)
    return np.mean(returns[returns <= var])


def kelly_criterion(returns: np.ndarray, leverage: float = 1.0) -> float:
    """
    Calculate optimal position size using Kelly Criterion.
    
    Args:
        returns: Array of historical returns
        leverage: Maximum leverage allowed
        
    Returns:
        Optimal fraction of capital to risk
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate win rate and average win/loss
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    
    win_rate = len(wins) / len(returns)
    avg_win = np.mean(wins)
    avg_loss = abs(np.mean(losses))
    
    if avg_loss == 0:
        return 0.0
    
    # Kelly formula: f = (bp - q) / b
    # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - win_rate
    
    kelly_fraction = (b * p - q) / b
    
    # Apply leverage constraint and prudent scaling
    kelly_fraction = max(0, min(kelly_fraction, leverage))
    
    # Conservative scaling (use 25% of Kelly)
    return kelly_fraction * 0.25


def position_sizing(method: str, capital: float, risk_per_trade: float,
                   price: float, stop_loss_price: Optional[float] = None,
                   **kwargs) -> int:
    """
    Calculate position size based on different methods.
    
    Args:
        method: Position sizing method
        capital: Total capital available
        risk_per_trade: Risk per trade as fraction of capital
        price: Current price
        stop_loss_price: Stop loss price (required for some methods)
        **kwargs: Additional method-specific parameters
        
    Returns:
        Position size in shares/units
    """
    risk_amount = capital * risk_per_trade
    
    if method == 'fixed_fractional':
        position_value = capital * risk_per_trade
        return int(position_value / price)
    
    elif method == 'fixed_dollar':
        fixed_amount = kwargs.get('fixed_amount', 1000)
        return int(fixed_amount / price)
    
    elif method == 'volatility_adjusted':
        volatility = kwargs.get('volatility', 0.02)
        target_volatility = kwargs.get('target_volatility', 0.01)
        
        if volatility == 0:
            return 0
        
        vol_adjustment = target_volatility / volatility
        position_value = capital * risk_per_trade * vol_adjustment
        return int(position_value / price)
    
    elif method == 'stop_loss_based':
        if stop_loss_price is None:
            raise ValueError("Stop loss price required for stop_loss_based method")
        
        risk_per_share = abs(price - stop_loss_price)
        if risk_per_share == 0:
            return 0
        
        return int(risk_amount / risk_per_share)
    
    elif method == 'kelly':
        returns = kwargs.get('returns', np.array([]))
        kelly_fraction = kelly_criterion(returns)
        position_value = capital * kelly_fraction
        return int(position_value / price)
    
    else:
        raise ValueError(f"Unknown position sizing method: {method}")


def risk_parity_weights(returns_matrix: np.ndarray, target_vol: float = 0.1) -> np.ndarray:
    """
    Calculate risk parity weights for a portfolio.
    
    Args:
        returns_matrix: Matrix of asset returns (assets in columns)
        target_vol: Target portfolio volatility
        
    Returns:
        Array of weights
    """
    if returns_matrix.shape[1] == 0:
        return np.array([])
    
    # Calculate covariance matrix
    cov_matrix = np.cov(returns_matrix.T)
    
    # Equal risk contribution weights (simplified)
    n_assets = returns_matrix.shape[1]
    weights = np.ones(n_assets) / n_assets
    
    # Iterative risk parity optimization (simplified)
    for _ in range(10):  # 10 iterations
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        if portfolio_vol == 0:
            break
        
        # Risk contributions
        marginal_risk = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contributions = weights * marginal_risk
        
        # Update weights to equalize risk contributions
        target_risk = portfolio_vol / n_assets
        weights = weights * target_risk / (risk_contributions + 1e-8)
        weights = weights / np.sum(weights)  # Normalize
    
    # Scale to target volatility
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    if portfolio_vol > 0:
        weights = weights * target_vol / portfolio_vol
    
    return weights


def calculate_portfolio_risk(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Calculate portfolio risk (volatility).
    
    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix of returns
        
    Returns:
        Portfolio volatility
    """
    return np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))


def calculate_maximum_drawdown_duration(returns: pd.Series) -> int:
    """
    Calculate maximum drawdown duration in periods.
    
    Args:
        returns: Return series
        
    Returns:
        Maximum drawdown duration
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    # Find drawdown periods
    in_drawdown = drawdown < 0
    
    if not in_drawdown.any():
        return 0
    
    # Calculate consecutive drawdown periods
    drawdown_periods = []
    current_period = 0
    
    for is_dd in in_drawdown:
        if is_dd:
            current_period += 1
        else:
            if current_period > 0:
                drawdown_periods.append(current_period)
                current_period = 0
    
    # Don't forget the last period if it ends in drawdown
    if current_period > 0:
        drawdown_periods.append(current_period)
    
    return max(drawdown_periods) if drawdown_periods else 0


def calculate_tail_ratio(returns: pd.Series, percentile: float = 0.05) -> float:
    """
    Calculate tail ratio (upside vs downside extremes).
    
    Args:
        returns: Return series
        percentile: Percentile for tail calculation
        
    Returns:
        Tail ratio
    """
    upper_tail = returns.quantile(1 - percentile)
    lower_tail = returns.quantile(percentile)
    
    if lower_tail == 0:
        return np.inf if upper_tail > 0 else 1.0
    
    return abs(upper_tail / lower_tail)


def calculate_skewness_risk(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate skewness-based risk metrics.
    
    Args:
        returns: Return series
        
    Returns:
        Dictionary of skewness risk metrics
    """
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns, fisher=True)
    
    # Jarque-Bera test for normality
    jb_stat, jb_pvalue = stats.jarque_bera(returns)
    
    return {
        'skewness': float(skew),
        'kurtosis': float(kurt),
        'jarque_bera_stat': float(jb_stat),
        'jarque_bera_pvalue': float(jb_pvalue),
        'is_normal': jb_pvalue > 0.05
    }


def dynamic_position_sizing(returns: pd.Series, lookback: int = 252,
                           base_risk: float = 0.02, vol_target: float = 0.15) -> pd.Series:
    """
    Calculate dynamic position sizing based on realized volatility.
    
    Args:
        returns: Return series
        lookback: Lookback period for volatility calculation
        base_risk: Base risk per trade
        vol_target: Target volatility
        
    Returns:
        Series of position size multipliers
    """
    # Calculate rolling volatility
    rolling_vol = returns.rolling(window=lookback).std() * np.sqrt(252)
    
    # Calculate volatility-adjusted position sizes
    vol_adjustment = vol_target / rolling_vol
    position_multiplier = vol_adjustment.clip(0.1, 3.0)  # Limit adjustment range
    
    return position_multiplier * base_risk


def calculate_risk_adjusted_returns(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate comprehensive risk-adjusted return metrics.
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Dictionary of risk-adjusted metrics
    """
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
    
    # Calmar ratio
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
    
    # Information ratio (assuming benchmark return = 0)
    info_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    
    return {
        'annual_return': float(annual_return),
        'annual_volatility': float(annual_vol),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'calmar_ratio': float(calmar),
        'information_ratio': float(info_ratio),
        'max_drawdown': float(max_drawdown)
    }


def stress_test_portfolio(returns: pd.Series, scenarios: Dict[str, Dict]) -> Dict[str, float]:
    """
    Perform stress testing on portfolio returns.
    
    Args:
        returns: Portfolio return series
        scenarios: Dictionary of stress scenarios
        
    Returns:
        Stress test results
    """
    results = {}
    
    for scenario_name, scenario_params in scenarios.items():
        if scenario_name == 'market_crash':
            # Simulate market crash (consecutive negative returns)
            crash_duration = scenario_params.get('duration', 20)
            crash_magnitude = scenario_params.get('magnitude', -0.05)
            
            crash_returns = pd.Series([crash_magnitude] * crash_duration)
            combined_returns = pd.concat([returns, crash_returns])
            
            # Calculate impact
            original_value = (1 + returns).prod()
            stressed_value = (1 + combined_returns).prod()
            impact = (stressed_value - original_value) / original_value
            
            results[f'{scenario_name}_impact'] = float(impact)
        
        elif scenario_name == 'volatility_spike':
            # Simulate volatility spike
            vol_multiplier = scenario_params.get('multiplier', 3.0)
            duration = scenario_params.get('duration', 10)
            
            # Increase volatility for specified duration
            spike_returns = returns.tail(duration) * vol_multiplier
            combined_returns = pd.concat([returns.head(-duration), spike_returns])
            
            # Calculate new risk metrics
            new_vol = combined_returns.std() * np.sqrt(252)
            original_vol = returns.std() * np.sqrt(252)
            vol_impact = (new_vol - original_vol) / original_vol
            
            results[f'{scenario_name}_vol_impact'] = float(vol_impact)
    
    return results
