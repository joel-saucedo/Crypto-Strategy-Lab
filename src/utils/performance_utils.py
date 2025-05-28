"""
Performance analysis and reporting utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings


def performance_summary(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                       risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Generate comprehensive performance summary.
    
    Args:
        returns: Strategy return series
        benchmark_returns: Benchmark return series (optional)
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Performance summary dictionary
    """
    if len(returns) == 0:
        return {'error': 'No returns data provided'}
    
    # Basic statistics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + returns).mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    
    # Risk-adjusted metrics
    sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    
    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
    
    # Calmar ratio
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
    
    # Win rate and profit factor
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
    
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
    profit_factor = (len(winning_trades) * avg_win) / (len(losing_trades) * avg_loss) if avg_loss > 0 else 0
    
    # Tail ratio
    upper_tail = returns.quantile(0.95)
    lower_tail = returns.quantile(0.05)
    tail_ratio = abs(upper_tail / lower_tail) if lower_tail != 0 else 0
    
    # Maximum drawdown duration
    max_dd_duration = calculate_max_drawdown_duration(returns)
    
    summary = {
        'total_return': float(total_return),
        'annual_return': float(annual_return),
        'annual_volatility': float(annual_vol),
        'sharpe_ratio': float(sharpe_ratio),
        'sortino_ratio': float(sortino_ratio),
        'calmar_ratio': float(calmar_ratio),
        'max_drawdown': float(max_drawdown),
        'max_drawdown_duration': int(max_dd_duration),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'tail_ratio': float(tail_ratio),
        'total_trades': len(returns),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss)
    }
    
    # Benchmark comparison if provided
    if benchmark_returns is not None:
        benchmark_comparison = compare_to_benchmark(returns, benchmark_returns, risk_free_rate)
        summary['benchmark_comparison'] = benchmark_comparison
    
    return summary


def calculate_max_drawdown_duration(returns: pd.Series) -> int:
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
    in_drawdown = drawdown < -0.001  # Small threshold to avoid noise
    
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


def compare_to_benchmark(strategy_returns: pd.Series, benchmark_returns: pd.Series,
                        risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Compare strategy performance to benchmark.
    
    Args:
        strategy_returns: Strategy return series
        benchmark_returns: Benchmark return series
        risk_free_rate: Risk-free rate
        
    Returns:
        Comparison metrics
    """
    # Align series by index
    aligned_data = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if len(aligned_data) == 0:
        return {'error': 'No overlapping data between strategy and benchmark'}
    
    strategy_aligned = aligned_data['strategy']
    benchmark_aligned = aligned_data['benchmark']
    
    # Calculate metrics for both
    strategy_annual = strategy_aligned.mean() * 252
    benchmark_annual = benchmark_aligned.mean() * 252
    
    strategy_vol = strategy_aligned.std() * np.sqrt(252)
    benchmark_vol = benchmark_aligned.std() * np.sqrt(252)
    
    # Excess returns
    excess_returns = strategy_aligned - benchmark_aligned
    tracking_error = excess_returns.std() * np.sqrt(252)
    
    # Information ratio
    information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
    
    # Alpha and Beta (simplified CAPM)
    correlation = strategy_aligned.corr(benchmark_aligned)
    beta = correlation * (strategy_vol / benchmark_vol) if benchmark_vol > 0 else 0
    alpha = strategy_annual - (risk_free_rate + beta * (benchmark_annual - risk_free_rate))
    
    # Sharpe ratios
    strategy_sharpe = (strategy_annual - risk_free_rate) / strategy_vol if strategy_vol > 0 else 0
    benchmark_sharpe = (benchmark_annual - risk_free_rate) / benchmark_vol if benchmark_vol > 0 else 0
    
    return {
        'excess_return': float(strategy_annual - benchmark_annual),
        'tracking_error': float(tracking_error),
        'information_ratio': float(information_ratio),
        'alpha': float(alpha),
        'beta': float(beta),
        'correlation': float(correlation),
        'strategy_sharpe': float(strategy_sharpe),
        'benchmark_sharpe': float(benchmark_sharpe),
        'sharpe_difference': float(strategy_sharpe - benchmark_sharpe)
    }


def rolling_performance(returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.
    
    Args:
        returns: Return series
        window: Rolling window size
        
    Returns:
        DataFrame with rolling metrics
    """
    rolling_metrics = pd.DataFrame(index=returns.index)
    
    # Rolling returns and volatility
    rolling_metrics['rolling_return'] = returns.rolling(window).mean() * 252
    rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
    
    # Rolling Sharpe ratio
    rolling_metrics['rolling_sharpe'] = (
        rolling_metrics['rolling_return'] / rolling_metrics['rolling_volatility']
    )
    
    # Rolling maximum drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.rolling(window).max()
    rolling_metrics['rolling_max_drawdown'] = (
        (rolling_max - cumulative) / rolling_max
    ).rolling(window).max()
    
    # Rolling win rate
    rolling_metrics['rolling_win_rate'] = (
        (returns > 0).rolling(window).mean()
    )
    
    return rolling_metrics.dropna()


def drawdown_analysis(returns: pd.Series) -> Dict[str, Any]:
    """
    Detailed drawdown analysis.
    
    Args:
        returns: Return series
        
    Returns:
        Drawdown analysis results
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    # Find all drawdown periods
    in_drawdown = drawdown < -0.001  # Small threshold
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
                'recovery_date': returns.index[i] if i < len(returns) else None
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


def monthly_performance_table(returns: pd.Series) -> pd.DataFrame:
    """
    Create monthly performance table.
    
    Args:
        returns: Daily return series
        
    Returns:
        Monthly performance DataFrame
    """
    # Resample to monthly returns
    monthly_returns = (1 + returns).resample('M').prod() - 1
    
    # Create year-month table
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    monthly_df = monthly_returns.to_frame('returns')
    monthly_df['year'] = monthly_df.index.year
    monthly_df['month'] = monthly_df.index.strftime('%b')
    
    # Pivot to create year vs month table
    performance_table = monthly_df.pivot(index='year', columns='month', values='returns')
    
    # Reorder columns to calendar order
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    performance_table = performance_table.reindex(columns=month_order)
    
    # Add annual returns
    annual_returns = (1 + monthly_returns).groupby(monthly_returns.index.year).prod() - 1
    performance_table['Annual'] = annual_returns
    
    return performance_table


def performance_attribution(returns: pd.Series, factor_returns: pd.DataFrame) -> Dict[str, Any]:
    """
    Simple performance attribution analysis.
    
    Args:
        returns: Strategy return series
        factor_returns: DataFrame with factor return series
        
    Returns:
        Attribution analysis results
    """
    try:
        from sklearn.linear_model import LinearRegression
        
        # Align data
        aligned_data = pd.concat([returns, factor_returns], axis=1).dropna()
        
        if len(aligned_data) < 20:
            return {'error': 'Insufficient data for attribution analysis'}
        
        y = aligned_data.iloc[:, 0].values.reshape(-1, 1)  # Strategy returns
        X = aligned_data.iloc[:, 1:].values  # Factor returns
        
        # Fit regression
        model = LinearRegression()
        model.fit(X, y.ravel())
        
        # Calculate attribution
        factor_contributions = {}
        for i, factor in enumerate(factor_returns.columns):
            contribution = model.coef_[i] * factor_returns[factor].mean() * 252
            factor_contributions[factor] = float(contribution)
        
        # Calculate alpha
        predicted_return = model.predict(X).mean() * 252
        actual_return = returns.mean() * 252
        alpha = actual_return - predicted_return
        
        return {
            'alpha': float(alpha),
            'r_squared': float(model.score(X, y.ravel())),
            'factor_contributions': factor_contributions,
            'total_explained': float(sum(factor_contributions.values())),
            'unexplained': float(alpha)
        }
        
    except ImportError:
        warnings.warn("scikit-learn not available, skipping attribution analysis")
        return {'error': 'scikit-learn not available for attribution analysis'}


def generate_performance_report(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                              strategy_name: str = "Strategy") -> str:
    """
    Generate a text-based performance report.
    
    Args:
        returns: Strategy return series
        benchmark_returns: Benchmark return series (optional)
        strategy_name: Name of the strategy
        
    Returns:
        Formatted performance report string
    """
    summary = performance_summary(returns, benchmark_returns)
    drawdown_stats = drawdown_analysis(returns)
    
    report = f"""
Performance Report: {strategy_name}
{'='*50}

RETURN METRICS
Total Return: {summary['total_return']:.2%}
Annual Return: {summary['annual_return']:.2%}
Annual Volatility: {summary['annual_volatility']:.2%}

RISK-ADJUSTED METRICS
Sharpe Ratio: {summary['sharpe_ratio']:.3f}
Sortino Ratio: {summary['sortino_ratio']:.3f}
Calmar Ratio: {summary['calmar_ratio']:.3f}

DRAWDOWN ANALYSIS
Maximum Drawdown: {summary['max_drawdown']:.2%}
Max Drawdown Duration: {summary['max_drawdown_duration']} periods
Time in Drawdown: {drawdown_stats['time_in_drawdown']:.1%}
Number of Drawdowns: {drawdown_stats['number_of_drawdowns']}

TRADING STATISTICS
Total Trades: {summary['total_trades']}
Win Rate: {summary['win_rate']:.1%}
Profit Factor: {summary['profit_factor']:.2f}
Average Win: {summary['avg_win']:.3%}
Average Loss: {summary['avg_loss']:.3%}
Tail Ratio: {summary['tail_ratio']:.2f}
"""
    
    if 'benchmark_comparison' in summary:
        bc = summary['benchmark_comparison']
        report += f"""
BENCHMARK COMPARISON
Excess Return: {bc['excess_return']:.2%}
Information Ratio: {bc['information_ratio']:.3f}
Alpha: {bc['alpha']:.2%}
Beta: {bc['beta']:.3f}
Correlation: {bc['correlation']:.3f}
"""
    
    return report
