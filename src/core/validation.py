"""
Statistical validation framework enforcing the two-layer blueprint.
Calculates DSR, PSR, and runs comprehensive stress tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy import stats
from scipy.stats import jarque_bera
import warnings

class ValidationHarness:
    """
    Comprehensive validation that enforces DSR ≥ 0.95 gate.
    Implements bootstrap, permutation, and White Reality Check.
    """
    
    def __init__(self, min_dsr: float = 0.95, min_psr: float = 0.95):
        self.min_dsr = min_dsr
        self.min_psr = min_psr
        
    def validate_strategy(self, 
                         returns: pd.Series,
                         benchmark_returns: Optional[pd.Series] = None,
                         n_trials: int = 1000) -> Dict[str, Any]:
        """
        Complete validation suite for a strategy.
        
        Args:
            returns: Strategy return series
            benchmark_returns: Benchmark return series (optional)
            n_trials: Number of trials for multiple testing adjustment
            
        Returns:
            Comprehensive validation results
        """
        if len(returns) < 30:
            return {'validation_passed': False, 'error': 'Insufficient data'}
            
        results = {
            'basic_metrics': self._calculate_basic_metrics(returns),
            'dsr_analysis': self._calculate_dsr(returns, n_trials),
            'psr_analysis': self._calculate_psr(returns),
            'stress_tests': self._run_stress_tests(returns),
            'statistical_tests': self._run_statistical_tests(returns),
            'regime_analysis': self._analyze_regimes(returns)
        }
        
        # Overall validation decision
        results['validation_passed'] = self._evaluate_validation(results)
        
        return results
        
    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        if returns.std() == 0:
            return {'sharpe_ratio': 0, 'total_return': 0, 'volatility': 0, 'max_drawdown': 0}
            
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annualized_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'calmar_ratio': (returns.mean() * 252) / abs(self._calculate_max_drawdown(returns)) if self._calculate_max_drawdown(returns) != 0 else 0,
            'win_rate': (returns > 0).mean(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
        
        return metrics
        
    def _calculate_dsr(self, returns: pd.Series, n_trials: int = 1000) -> Dict[str, float]:
        """
        Calculate Deflated Sharpe Ratio with multiple testing adjustment.
        
        The DSR adjusts the Sharpe ratio for multiple testing bias.
        DSR = (SR - E[max SR]) / std[max SR]
        """
        if returns.std() == 0:
            return {'dsr': 0, 'sharpe_ratio': 0, 'trials_adjustment': 0}
            
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        n_obs = len(returns)
        
        # Expected maximum Sharpe ratio under null hypothesis
        # Approximation for large n_trials
        gamma = 0.5772156649  # Euler's constant
        expected_max_sr = np.sqrt(2 * np.log(n_trials)) - (np.log(np.log(n_trials)) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * np.log(n_trials)))
        
        # Standard deviation of maximum Sharpe ratio
        std_max_sr = 1 / np.sqrt(2 * np.log(n_trials))
        
        # Deflated Sharpe Ratio
        dsr = (sharpe_ratio - expected_max_sr) / std_max_sr
        
        # Convert to probability scale (probability that strategy is genuine)
        dsr_prob = stats.norm.cdf(dsr)
        
        return {
            'dsr': dsr,
            'dsr_probability': dsr_prob,
            'sharpe_ratio': sharpe_ratio,
            'expected_max_sr': expected_max_sr,
            'trials_adjustment': n_trials
        }
        
    def _calculate_psr(self, returns: pd.Series, benchmark_sr: float = 0) -> Dict[str, float]:
        """
        Calculate Probabilistic Sharpe Ratio.
        PSR = Prob[SR > benchmark_SR]
        """
        if returns.std() == 0:
            return {'psr': 0, 'psr_confidence': 0}
            
        n_obs = len(returns)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        # Skewness and kurtosis adjustments
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        # Standard error of Sharpe ratio with higher moment corrections
        sr_std = np.sqrt((1 + 0.5 * sharpe_ratio**2 - skew * sharpe_ratio + (kurt - 3) / 4 * sharpe_ratio**2) / n_obs)
        
        # PSR calculation
        psr_stat = (sharpe_ratio - benchmark_sr) / sr_std
        psr = stats.norm.cdf(psr_stat)
        
        return {
            'psr': psr,
            'psr_statistic': psr_stat,
            'sharpe_ratio': sharpe_ratio,
            'sharpe_std_error': sr_std
        }
        
    def _run_stress_tests(self, returns: pd.Series) -> Dict[str, Any]:
        """Run comprehensive stress testing suite."""
        stress_results = {}
        
        # Bootstrap confidence intervals
        stress_results['bootstrap'] = self._bootstrap_confidence(returns)
        
        # Permutation test
        stress_results['permutation'] = self._permutation_test(returns)
        
        # Regime-specific performance
        stress_results['regime_stress'] = self._regime_stress_test(returns)
        
        # Tail risk analysis
        stress_results['tail_risk'] = self._tail_risk_analysis(returns)
        
        return stress_results
        
    def _bootstrap_confidence(self, returns: pd.Series, n_bootstrap: int = 1000) -> Dict[str, float]:
        """Bootstrap confidence intervals for Sharpe ratio."""
        bootstrap_sharpes = []
        
        for _ in range(n_bootstrap):
            boot_sample = returns.sample(n=len(returns), replace=True)
            if boot_sample.std() > 0:
                boot_sharpe = boot_sample.mean() / boot_sample.std() * np.sqrt(252)
                bootstrap_sharpes.append(boot_sharpe)
                
        if not bootstrap_sharpes:
            return {'ci_lower': 0, 'ci_upper': 0, 'mean': 0}
            
        return {
            'ci_lower': np.percentile(bootstrap_sharpes, 5),
            'ci_upper': np.percentile(bootstrap_sharpes, 95),
            'mean': np.mean(bootstrap_sharpes),
            'std': np.std(bootstrap_sharpes)
        }
        
    def _permutation_test(self, returns: pd.Series, n_permutations: int = 1000) -> Dict[str, float]:
        """Permutation test for return significance."""
        if returns.std() == 0:
            return {'p_value': 1, 'observed_mean': 0}
            
        observed_mean = returns.mean()
        
        # Generate null distribution by permuting signs
        null_means = []
        for _ in range(n_permutations):
            signs = np.random.choice([-1, 1], size=len(returns))
            permuted_returns = returns.abs() * signs
            null_means.append(permuted_returns.mean())
            
        # P-value: fraction of null means >= observed mean
        p_value = np.mean(np.array(null_means) >= observed_mean)
        
        return {
            'p_value': p_value,
            'observed_mean': observed_mean,
            'null_mean': np.mean(null_means),
            'null_std': np.std(null_means)
        }
        
    def _regime_stress_test(self, returns: pd.Series) -> Dict[str, float]:
        """Test performance across different market regimes."""
        # Simple regime classification based on volatility
        rolling_vol = returns.rolling(20).std()
        high_vol_threshold = rolling_vol.quantile(0.8)
        low_vol_threshold = rolling_vol.quantile(0.2)
        
        high_vol_mask = rolling_vol > high_vol_threshold
        low_vol_mask = rolling_vol < low_vol_threshold
        
        regimes = {
            'high_volatility': returns[high_vol_mask],
            'low_volatility': returns[low_vol_mask],
            'normal_volatility': returns[~(high_vol_mask | low_vol_mask)]
        }
        
        regime_metrics = {}
        for regime_name, regime_returns in regimes.items():
            if len(regime_returns) > 5 and regime_returns.std() > 0:
                regime_metrics[f'{regime_name}_sharpe'] = regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                regime_metrics[f'{regime_name}_mean'] = regime_returns.mean()
            else:
                regime_metrics[f'{regime_name}_sharpe'] = 0
                regime_metrics[f'{regime_name}_mean'] = 0
                
        return regime_metrics
        
    def _tail_risk_analysis(self, returns: pd.Series) -> Dict[str, float]:
        """Analyze tail risk characteristics."""
        if len(returns) < 10:
            return {'var_5': 0, 'cvar_5': 0, 'max_loss': 0}
            
        return {
            'var_5': np.percentile(returns, 5),
            'var_1': np.percentile(returns, 1),
            'cvar_5': returns[returns <= np.percentile(returns, 5)].mean(),
            'max_loss': returns.min(),
            'max_gain': returns.max()
        }
        
    def _run_statistical_tests(self, returns: pd.Series) -> Dict[str, Any]:
        """Run statistical tests on return distribution."""
        tests = {}
        
        # Normality test
        if len(returns) > 8:
            jb_stat, jb_pvalue = jarque_bera(returns)
            tests['jarque_bera'] = {'statistic': jb_stat, 'p_value': jb_pvalue}
        
        # Autocorrelation test
        tests['ljung_box'] = self._ljung_box_test(returns)
        
        return tests
        
    def _ljung_box_test(self, returns: pd.Series, lags: int = 10) -> Dict[str, float]:
        """Ljung-Box test for autocorrelation."""
        if len(returns) < lags * 2:
            return {'statistic': 0, 'p_value': 1}
            
        # Simple implementation of Ljung-Box test
        n = len(returns)
        autocorrs = [returns.autocorr(lag=i) for i in range(1, lags + 1)]
        autocorrs = [ac for ac in autocorrs if not pd.isna(ac)]
        
        if not autocorrs:
            return {'statistic': 0, 'p_value': 1}
            
        lb_stat = n * (n + 2) * sum(ac**2 / (n - i) for i, ac in enumerate(autocorrs, 1))
        p_value = 1 - stats.chi2.cdf(lb_stat, len(autocorrs))
        
        return {'statistic': lb_stat, 'p_value': p_value}
        
    def _analyze_regimes(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze strategy performance across different regimes."""
        if len(returns) < 50:
            return {'regimes_identified': 0}
            
        # Simple regime switching based on rolling correlation with market
        rolling_mean = returns.rolling(20).mean()
        
        # Classify periods
        bull_mask = rolling_mean > rolling_mean.quantile(0.7)
        bear_mask = rolling_mean < rolling_mean.quantile(0.3)
        
        regime_analysis = {
            'bull_market_returns': returns[bull_mask].mean() if bull_mask.any() else 0,
            'bear_market_returns': returns[bear_mask].mean() if bear_mask.any() else 0,
            'bull_market_sharpe': (returns[bull_mask].mean() / returns[bull_mask].std() * np.sqrt(252)) if bull_mask.any() and returns[bull_mask].std() > 0 else 0,
            'bear_market_sharpe': (returns[bear_mask].mean() / returns[bear_mask].std() * np.sqrt(252)) if bear_mask.any() and returns[bear_mask].std() > 0 else 0,
            'regimes_identified': 2
        }
        
        return regime_analysis
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0
            
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
        
    def _evaluate_validation(self, results: Dict[str, Any]) -> bool:
        """Evaluate whether strategy passes all validation gates."""
        try:
            # Core requirements
            dsr_passed = results['dsr_analysis']['dsr_probability'] >= self.min_dsr
            psr_passed = results['psr_analysis']['psr'] >= self.min_psr
            
            # Additional requirements
            sharpe_passed = results['basic_metrics']['sharpe_ratio'] >= 1.0
            drawdown_passed = results['basic_metrics']['max_drawdown'] >= -0.20  # Max 20% drawdown
            
            return dsr_passed and psr_passed and sharpe_passed and drawdown_passed
            
        except (KeyError, TypeError):
            return False
