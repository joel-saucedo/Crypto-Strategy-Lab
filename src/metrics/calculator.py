"""
Comprehensive metrics calculator for cryptocurrency trading strategies.
Implements all performance metrics with statistical rigor.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Calculate comprehensive trading strategy performance metrics.
    """
    
    def __init__(self, benchmark_rate: float = 0.0, risk_free_rate: float = 0.02):
        """
        Initialize metrics calculator.
        
        Args:
            benchmark_rate: Benchmark return rate (annualized)
            risk_free_rate: Risk-free rate (annualized)
        """
        self.benchmark_rate = benchmark_rate
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 365 * 24  # For crypto (24/7 trading)
        
    def calculate_all_metrics(
        self,
        returns: pd.Series,
        positions: Optional[pd.Series] = None,
        prices: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Strategy returns series
            positions: Position sizes (optional)
            prices: Price series (optional)
            benchmark_returns: Benchmark returns (optional)
            
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}
        
        # Basic return metrics
        metrics.update(self.calculate_return_metrics(returns))
        
        # Risk metrics
        metrics.update(self.calculate_risk_metrics(returns))
        
        # Risk-adjusted performance
        metrics.update(self.calculate_risk_adjusted_metrics(returns, benchmark_returns))
        
        # Drawdown metrics
        metrics.update(self.calculate_drawdown_metrics(returns))
        
        # Distribution metrics
        metrics.update(self.calculate_distribution_metrics(returns))
        
        # Tail risk metrics
        metrics.update(self.calculate_tail_risk_metrics(returns))
        
        # Trading-specific metrics
        if positions is not None:
            metrics.update(self.calculate_trading_metrics(returns, positions))
            
        # Market timing metrics
        if benchmark_returns is not None:
            metrics.update(self.calculate_market_timing_metrics(returns, benchmark_returns))
            
        # Statistical significance tests
        metrics.update(self.calculate_statistical_tests(returns))
        
        return metrics
    
    def calculate_return_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic return metrics."""
        if len(returns) == 0:
            return {}
            
        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        
        # Annualized return
        periods_per_year = self.trading_days_per_year / len(returns) * len(returns)
        annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
        
        # Compound Annual Growth Rate (CAGR)
        if len(returns) > 1:
            years = len(returns) / self.trading_days_per_year
            cagr = (cumulative_returns.iloc[-1]) ** (1 / years) - 1
        else:
            cagr = 0.0
            
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cagr': cagr,
            'mean_return': returns.mean(),
            'median_return': returns.median(),
            'geometric_mean_return': stats.gmean(1 + returns) - 1 if (returns > -1).all() else np.nan
        }
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics."""
        if len(returns) == 0:
            return {}
            
        # Volatility
        volatility = returns.std()
        annualized_volatility = volatility * np.sqrt(self.trading_days_per_year)
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0.0
        annualized_downside_deviation = downside_deviation * np.sqrt(self.trading_days_per_year)
        
        # Semi-deviation (below mean)
        below_mean = returns[returns < returns.mean()]
        semi_deviation = below_mean.std() if len(below_mean) > 0 else 0.0
        
        return {
            'volatility': volatility,
            'annualized_volatility': annualized_volatility,
            'downside_deviation': downside_deviation,
            'annualized_downside_deviation': annualized_downside_deviation,
            'semi_deviation': semi_deviation,
            'tracking_error': volatility  # Will be updated if benchmark provided
        }
    
    def calculate_risk_adjusted_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        if len(returns) == 0:
            return {}
            
        metrics = {}
        
        # Basic Sharpe ratio
        excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
        if returns.std() > 0:
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(self.trading_days_per_year)
        else:
            sharpe_ratio = 0.0
            
        metrics['sharpe_ratio'] = sharpe_ratio
        
        # Sortino ratio
        downside_returns = returns[returns < self.risk_free_rate / self.trading_days_per_year]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(self.trading_days_per_year)
        else:
            sortino_ratio = 0.0
            
        metrics['sortino_ratio'] = sortino_ratio
        
        # Calmar ratio
        max_drawdown = self.calculate_max_drawdown(returns)
        if max_drawdown > 0:
            calmar_ratio = (returns.mean() * self.trading_days_per_year) / max_drawdown
        else:
            calmar_ratio = 0.0
            
        metrics['calmar_ratio'] = calmar_ratio
        
        # Information ratio (if benchmark provided)
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            active_returns = returns - benchmark_returns
            tracking_error = active_returns.std()
            if tracking_error > 0:
                information_ratio = active_returns.mean() / tracking_error * np.sqrt(self.trading_days_per_year)
            else:
                information_ratio = 0.0
                
            metrics['information_ratio'] = information_ratio
            metrics['tracking_error'] = tracking_error * np.sqrt(self.trading_days_per_year)
            
            # Treynor ratio (requires beta calculation)
            if benchmark_returns.std() > 0:
                beta = np.cov(returns, benchmark_returns)[0, 1] / benchmark_returns.var()
                if beta > 0:
                    treynor_ratio = excess_returns.mean() / beta * self.trading_days_per_year
                else:
                    treynor_ratio = 0.0
                    
                metrics['beta'] = beta
                metrics['treynor_ratio'] = treynor_ratio
        
        return metrics
    
    def calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate drawdown-related metrics."""
        if len(returns) == 0:
            return {}
            
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        max_drawdown = drawdowns.min()
        
        # Average drawdown
        in_drawdown = drawdowns < 0
        drawdown_periods = drawdowns[in_drawdown]
        avg_drawdown = drawdown_periods.mean() if len(drawdown_periods) > 0 else 0.0
        
        # Drawdown duration
        drawdown_duration = 0
        max_drawdown_duration = 0
        current_duration = 0
        
        for dd in drawdowns:
            if dd < 0:
                current_duration += 1
                max_drawdown_duration = max(max_drawdown_duration, current_duration)
            else:
                current_duration = 0
                
        # Recovery time (time to new high after max drawdown)
        max_dd_idx = drawdowns.idxmin()
        recovery_time = 0
        
        if max_dd_idx < len(drawdowns) - 1:
            post_max_dd = drawdowns.loc[max_dd_idx:]
            recovery_idx = (post_max_dd >= 0).idxmax()
            if recovery_idx:
                recovery_time = drawdowns.index.get_loc(recovery_idx) - drawdowns.index.get_loc(max_dd_idx)
        
        return {
            'max_drawdown': abs(max_drawdown),
            'avg_drawdown': abs(avg_drawdown),
            'max_drawdown_duration': max_drawdown_duration,
            'recovery_time': recovery_time,
            'drawdown_frequency': len(drawdown_periods) / len(returns) if len(returns) > 0 else 0
        }
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
            
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        return abs(drawdowns.min())
    
    def calculate_distribution_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate return distribution metrics."""
        if len(returns) == 0:
            return {}
            
        return {
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'jarque_bera_statistic': stats.jarque_bera(returns)[0] if len(returns) > 7 else np.nan,
            'jarque_bera_pvalue': stats.jarque_bera(returns)[1] if len(returns) > 7 else np.nan,
            'positive_periods': (returns > 0).sum() / len(returns),
            'negative_periods': (returns < 0).sum() / len(returns),
            'zero_periods': (returns == 0).sum() / len(returns)
        }
    
    def calculate_tail_risk_metrics(self, returns: pd.Series, confidence_levels: List[float] = [0.05, 0.01]) -> Dict[str, float]:
        """Calculate tail risk metrics (VaR, CVaR, etc.)."""
        if len(returns) == 0:
            return {}
            
        metrics = {}
        
        for alpha in confidence_levels:
            # Value at Risk (VaR)
            var = np.percentile(returns, alpha * 100)
            metrics[f'var_{int(alpha*100)}'] = var
            
            # Conditional Value at Risk (CVaR/Expected Shortfall)
            cvar = returns[returns <= var].mean()
            metrics[f'cvar_{int(alpha*100)}'] = cvar
            
            # Modified VaR (Cornish-Fisher)
            z_score = stats.norm.ppf(alpha)
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)
            
            modified_z = (z_score + 
                         (z_score**2 - 1) * skew / 6 + 
                         (z_score**3 - 3*z_score) * kurt / 24 - 
                         (2*z_score**3 - 5*z_score) * skew**2 / 36)
            
            modified_var = returns.mean() + modified_z * returns.std()
            metrics[f'modified_var_{int(alpha*100)}'] = modified_var
        
        # Maximum loss
        metrics['maximum_loss'] = returns.min()
        
        # Gain-to-pain ratio
        total_gain = returns[returns > 0].sum()
        total_pain = abs(returns[returns < 0].sum())
        
        if total_pain > 0:
            gain_to_pain_ratio = total_gain / total_pain
        else:
            gain_to_pain_ratio = np.inf if total_gain > 0 else 0
            
        metrics['gain_to_pain_ratio'] = gain_to_pain_ratio
        
        return metrics
    
    def calculate_trading_metrics(self, returns: pd.Series, positions: pd.Series) -> Dict[str, float]:
        """Calculate trading-specific metrics."""
        if len(returns) == 0 or len(positions) == 0:
            return {}
            
        # Position changes (trades)
        position_changes = positions.diff().fillna(0)
        trades = position_changes[position_changes != 0]
        
        # Win/loss analysis
        winning_periods = returns > 0
        losing_periods = returns < 0
        
        win_rate = winning_periods.sum() / len(returns) if len(returns) > 0 else 0
        loss_rate = losing_periods.sum() / len(returns) if len(returns) > 0 else 0
        
        # Average win/loss
        avg_win = returns[winning_periods].mean() if winning_periods.any() else 0
        avg_loss = returns[losing_periods].mean() if losing_periods.any() else 0
        
        # Profit factor
        total_wins = returns[winning_periods].sum() if winning_periods.any() else 0
        total_losses = abs(returns[losing_periods].sum()) if losing_periods.any() else 0
        
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        
        # Expectancy
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
        
        # Kelly criterion
        if avg_loss < 0:  # Ensure avg_loss is negative
            kelly_fraction = (win_rate * avg_win + loss_rate * avg_loss) / abs(avg_loss)
        else:
            kelly_fraction = 0
            
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'kelly_fraction': kelly_fraction,
            'largest_win': returns.max(),
            'largest_loss': returns.min(),
            'consecutive_wins': self._max_consecutive(winning_periods),
            'consecutive_losses': self._max_consecutive(losing_periods)
        }
    
    def calculate_market_timing_metrics(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate market timing ability metrics."""
        if len(returns) == 0 or len(benchmark_returns) == 0 or len(returns) != len(benchmark_returns):
            return {}
            
        # Up/down capture ratios
        up_market = benchmark_returns > 0
        down_market = benchmark_returns < 0
        
        if up_market.any() and benchmark_returns[up_market].std() > 0:
            up_capture = returns[up_market].mean() / benchmark_returns[up_market].mean()
        else:
            up_capture = 0
            
        if down_market.any() and benchmark_returns[down_market].std() > 0:
            down_capture = returns[down_market].mean() / benchmark_returns[down_market].mean()
        else:
            down_capture = 0
            
        # Market timing metrics (Treynor-Mazuy)
        excess_benchmark = benchmark_returns - self.risk_free_rate / self.trading_days_per_year
        excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
        
        # Regression: R_p - R_f = alpha + beta*(R_m - R_f) + gamma*(R_m - R_f)^2
        X = np.column_stack([excess_benchmark, excess_benchmark**2])
        if len(X) > 2:
            try:
                coeffs = np.linalg.lstsq(X, excess_returns, rcond=None)[0]
                timing_coefficient = coeffs[1] if len(coeffs) > 1 else 0
            except:
                timing_coefficient = 0
        else:
            timing_coefficient = 0
            
        return {
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture,
            'timing_coefficient': timing_coefficient,
            'up_market_periods': up_market.sum(),
            'down_market_periods': down_market.sum()
        }
    
    def calculate_statistical_tests(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate statistical significance tests."""
        if len(returns) < 10:
            return {}
            
        metrics = {}
        
        # T-test for mean return significance
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        metrics['t_statistic'] = t_stat
        metrics['t_test_pvalue'] = p_value
        
        # Sharpe ratio significance (using Jobson-Korkie test)
        n = len(returns)
        sr = self.calculate_risk_adjusted_metrics(returns)['sharpe_ratio']
        
        if n > 1:
            # Standard error of Sharpe ratio
            sr_se = np.sqrt((1 + 0.5 * sr**2) / n)
            sr_t_stat = sr / sr_se
            sr_p_value = 2 * (1 - stats.norm.cdf(abs(sr_t_stat)))
            
            metrics['sharpe_t_statistic'] = sr_t_stat
            metrics['sharpe_pvalue'] = sr_p_value
        
        # Autocorrelation test
        if len(returns) > 1:
            autocorr_1 = returns.autocorr(lag=1)
            metrics['autocorrelation_lag1'] = autocorr_1 if not np.isnan(autocorr_1) else 0
            
        # Ljung-Box test for serial correlation
        if len(returns) > 10:
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_stat = acorr_ljungbox(returns, lags=5, return_df=True)
                metrics['ljung_box_statistic'] = lb_stat['lb_stat'].iloc[-1]
                metrics['ljung_box_pvalue'] = lb_stat['lb_pvalue'].iloc[-1]
            except:
                pass
                
        return metrics
    
    def calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.
        
        Args:
            returns: Return series
            threshold: Threshold return level
            
        Returns:
            Omega ratio
        """
        if len(returns) == 0:
            return 0.0
            
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if losses.sum() > 0:
            omega = gains.sum() / losses.sum()
        else:
            omega = np.inf if gains.sum() > 0 else 1.0
            
        return omega
    
    def calculate_ulcer_index(self, returns: pd.Series) -> float:
        """
        Calculate Ulcer Index (measure of downside risk).
        
        Args:
            returns: Return series
            
        Returns:
            Ulcer Index
        """
        if len(returns) == 0:
            return 0.0
            
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        ulcer_index = np.sqrt((drawdowns**2).mean())
        
        return ulcer_index
    
    def _max_consecutive(self, condition_series: pd.Series) -> int:
        """Calculate maximum consecutive True values."""
        if len(condition_series) == 0:
            return 0
            
        max_consecutive = 0
        current_consecutive = 0
        
        for value in condition_series:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
    
    def calculate_performance_attribution(
        self,
        returns: pd.Series,
        factor_returns: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """
        Calculate performance attribution to various factors.
        
        Args:
            returns: Strategy returns
            factor_returns: Dictionary of factor name to return series
            
        Returns:
            Dictionary with factor attributions
        """
        if len(returns) == 0 or not factor_returns:
            return {}
            
        attribution = {}
        
        # Prepare factor matrix
        factor_names = list(factor_returns.keys())
        factor_matrix = pd.DataFrame(factor_returns)
        
        # Align dates
        common_dates = returns.index.intersection(factor_matrix.index)
        if len(common_dates) == 0:
            return {}
            
        aligned_returns = returns.loc[common_dates]
        aligned_factors = factor_matrix.loc[common_dates]
        
        # Multiple regression
        try:
            from sklearn.linear_model import LinearRegression
            
            model = LinearRegression()
            model.fit(aligned_factors, aligned_returns)
            
            # Factor loadings (betas)
            for i, factor_name in enumerate(factor_names):
                attribution[f'{factor_name}_beta'] = model.coef_[i]
                
            # Factor contributions to return
            factor_contributions = model.coef_ * aligned_factors.mean()
            for i, factor_name in enumerate(factor_names):
                attribution[f'{factor_name}_contribution'] = factor_contributions[i]
                
            # Alpha (unexplained return)
            attribution['alpha'] = model.intercept_
            attribution['r_squared'] = model.score(aligned_factors, aligned_returns)
            
        except ImportError:
            logger.warning("scikit-learn not available for performance attribution")
            
        return attribution
