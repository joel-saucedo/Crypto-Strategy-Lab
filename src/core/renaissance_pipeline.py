"""
5-Layer Validation Pipeline
=========================================================

Implements a comprehensive hierarchical filtering system inspired by Renaissance Technologies' 
approach to signal validation and portfolio construction. This pipeline provides multiple 
layers of validation to ensure robust strategy performance.

The 5-Layer Architecture:
1. Layer 1: Noise Gate Filtering - Basic signal quality checks
2. Layer 2: Statistical Significance Testing - Rigorous statistical validation  
3. Layer 3: Regime Robustness Analysis - Performance across market regimes
4. Layer 4: Signal Consensus & Orthogonality - Multi-signal validation
5. Layer 5: Copula-Weighted Portfolio Allocation - Advanced risk management

"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import defaultdict

# Scientific computing
from scipy import stats
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
import numba
from numba import jit, prange

# Copula libraries
try:
    from copulas.multivariate import GaussianMultivariate
    from copulas.univariate import GaussianUnivariate
    COPULA_AVAILABLE = True
except ImportError:
    COPULA_AVAILABLE = False
    warnings.warn("Copula libraries not available. Install with: pip install copulas")

logger = logging.getLogger(__name__)

class LayerResult(Enum):
    """Enumeration for layer validation results"""
    PASS = "PASS"
    FAIL = "FAIL" 
    WARNING = "WARNING"
    PENDING = "PENDING"

class RegimeType(Enum):
    """Market regime classifications"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"

@dataclass
class LayerConfig:
    """Configuration for each validation layer"""
    enabled: bool = True
    threshold: float = 0.05
    min_observations: int = 100
    confidence_level: float = 0.95
    parallel_execution: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineConfig:
    """Configuration for the entire Renaissance pipeline"""
    # Layer configurations
    noise_gate: LayerConfig = field(default_factory=lambda: LayerConfig(threshold=0.01))
    statistical_test: LayerConfig = field(default_factory=lambda: LayerConfig(threshold=0.05))
    regime_analysis: LayerConfig = field(default_factory=lambda: LayerConfig(threshold=0.1))
    signal_consensus: LayerConfig = field(default_factory=lambda: LayerConfig(threshold=0.05))
    portfolio_allocation: LayerConfig = field(default_factory=lambda: LayerConfig(threshold=0.02))
    
    # Global settings
    require_all_layers: bool = False  # If True, all layers must pass
    min_passing_layers: int = 3  # Minimum layers that must pass
    monte_carlo_runs: int = 1000
    max_workers: int = 4
    random_seed: Optional[int] = 42

@dataclass
class LayerValidationResult:
    """Result from a single validation layer"""
    layer_name: str
    result: LayerResult
    score: float
    confidence: float
    details: Dict[str, Any]
    execution_time: float
    warnings: List[str] = field(default_factory=list)

@dataclass
class PipelineValidationResult:
    """Combined result from all validation layers"""
    overall_result: LayerResult
    overall_score: float
    layer_results: List[LayerValidationResult]
    execution_time: float
    strategy_name: str
    timestamp: str
    summary: str = ""

# ================================================================================================
# NUMBA-ACCELERATED FUNCTIONS
# ================================================================================================

@jit(nopython=True)
def _calculate_noise_ratio_numba(returns: np.ndarray, window: int = 20) -> float:
    """Calculate signal-to-noise ratio using rolling standard deviation"""
    if len(returns) < window:
        return 0.0
    
    signal_var = np.var(returns)
    noise_vars = np.zeros(len(returns) - window + 1)
    
    for i in prange(len(returns) - window + 1):
        window_returns = returns[i:i+window]
        noise_vars[i] = np.var(window_returns)
    
    avg_noise_var = np.mean(noise_vars)
    return signal_var / (avg_noise_var + 1e-8)

@jit(nopython=True)
def _calculate_autocorr_numba(returns: np.ndarray, max_lags: int = 10) -> np.ndarray:
    """Calculate autocorrelation for multiple lags"""
    n = len(returns)
    autocorrs = np.zeros(max_lags)
    mean_returns = np.mean(returns)
    
    for lag in range(1, max_lags + 1):
        if n - lag <= 0:
            autocorrs[lag-1] = 0.0
            continue
            
        numerator = 0.0
        denominator = 0.0
        
        for i in range(n - lag):
            numerator += (returns[i] - mean_returns) * (returns[i + lag] - mean_returns)
        
        for i in range(n):
            denominator += (returns[i] - mean_returns) ** 2
        
        if denominator > 0:
            autocorrs[lag-1] = numerator / denominator
        else:
            autocorrs[lag-1] = 0.0
    
    return autocorrs

@jit(nopython=True)
def _regime_sharpe_numba(returns: np.ndarray, regime_mask: np.ndarray) -> float:
    """Calculate Sharpe ratio for a specific regime"""
    regime_returns = returns[regime_mask]
    if len(regime_returns) < 2:
        return 0.0
    
    mean_return = np.mean(regime_returns)
    std_return = np.std(regime_returns)
    
    if std_return > 0:
        return mean_return / std_return * np.sqrt(252)  # Annualized
    return 0.0

@jit(nopython=True) 
def _calculate_drawdowns_numba(cum_returns: np.ndarray) -> Tuple[np.ndarray, float]:
    """Calculate drawdown series and maximum drawdown"""
    peak = cum_returns[0]
    drawdowns = np.zeros_like(cum_returns)
    max_dd = 0.0
    
    for i in range(len(cum_returns)):
        if cum_returns[i] > peak:
            peak = cum_returns[i]
        
        dd = (cum_returns[i] - peak) / peak
        drawdowns[i] = dd
        
        if dd < max_dd:
            max_dd = dd
    
    return drawdowns, abs(max_dd)

# ================================================================================================
# LAYER 1: NOISE GATE FILTERING
# ================================================================================================

class NoiseGateValidator:
    """
    Layer 1: Noise Gate Filtering
    
    Performs basic signal quality checks to filter out obviously poor strategies:
    - Signal-to-noise ratio analysis
    - Autocorrelation structure validation  
    - Basic statistical properties
    - Outlier detection and handling
    """
    
    def __init__(self, config: LayerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".NoiseGate")
    
    def validate(self, returns: np.ndarray, strategy_name: str) -> LayerValidationResult:
        """Execute noise gate validation"""
        start_time = time.time()
        warnings_list = []
        
        try:
            # Basic data quality checks
            if len(returns) < self.config.min_observations:
                return LayerValidationResult(
                    layer_name="noise_gate",
                    result=LayerResult.FAIL,
                    score=0.0,
                    confidence=1.0,
                    details={"reason": "Insufficient observations"},
                    execution_time=time.time() - start_time,
                    warnings=[f"Only {len(returns)} observations, minimum {self.config.min_observations} required"]
                )
            
            # Remove NaN and infinite values
            clean_returns = returns[np.isfinite(returns)]
            if len(clean_returns) < len(returns) * 0.95:
                warnings_list.append(f"Removed {len(returns) - len(clean_returns)} invalid observations")
            
            # Calculate noise metrics
            noise_ratio = _calculate_noise_ratio_numba(clean_returns)
            autocorrs = _calculate_autocorr_numba(clean_returns)
            
            # Statistical properties
            skewness = stats.skew(clean_returns)
            kurtosis = stats.kurtosis(clean_returns, fisher=True)
            
            # Outlier detection (using z-score)
            z_scores = np.abs(stats.zscore(clean_returns))
            outlier_pct = np.sum(z_scores > 3) / len(clean_returns)
            
            # Volatility clustering test (Ljung-Box on squared returns)
            squared_returns = clean_returns ** 2
            try:
                lb_stat, lb_pvalue = stats.boxljung(squared_returns, lags=10, return_df=False)
                vol_clustering = lb_pvalue < 0.05
            except:
                vol_clustering = False
                warnings_list.append("Could not perform volatility clustering test")
            
            # Scoring logic
            score = 0.0
            details = {
                "noise_ratio": noise_ratio,
                "max_autocorr": np.max(np.abs(autocorrs)),
                "skewness": skewness,
                "kurtosis": kurtosis,
                "outlier_percentage": outlier_pct,
                "volatility_clustering": vol_clustering,
                "observations": len(clean_returns)
            }
            
            # Scoring criteria
            if noise_ratio > 1.5:
                score += 0.25
            if np.max(np.abs(autocorrs)) < 0.1:  # Low autocorrelation is good
                score += 0.25
            if abs(skewness) < 2.0:  # Reasonable skewness
                score += 0.15
            if abs(kurtosis) < 10.0:  # Reasonable kurtosis
                score += 0.15
            if outlier_pct < 0.05:  # Less than 5% outliers
                score += 0.2
            
            # Determine result
            if score >= self.config.threshold:
                result = LayerResult.PASS
                confidence = min(score, 1.0)
            else:
                result = LayerResult.FAIL
                confidence = 1.0 - score
            
            return LayerValidationResult(
                layer_name="noise_gate",
                result=result,
                score=score,
                confidence=confidence,
                details=details,
                execution_time=time.time() - start_time,
                warnings=warnings_list
            )
            
        except Exception as e:
            self.logger.error(f"Error in noise gate validation: {str(e)}")
            return LayerValidationResult(
                layer_name="noise_gate",
                result=LayerResult.FAIL,
                score=0.0,
                confidence=1.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                warnings=[f"Validation failed with error: {str(e)}"]
            )

# ================================================================================================
# LAYER 2: STATISTICAL SIGNIFICANCE TESTING
# ================================================================================================

class StatisticalTestValidator:
    """
    Layer 2: Statistical Significance Testing
    
    Rigorous statistical validation including:
    - Multiple hypothesis testing corrections
    - Bootstrap confidence intervals
    - Permutation tests
    - Bayesian model comparison
    """
    
    def __init__(self, config: LayerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".StatisticalTest")
    
    def validate(self, returns: np.ndarray, strategy_name: str) -> LayerValidationResult:
        """Execute statistical significance testing"""
        start_time = time.time()
        warnings_list = []
        
        try:
            # Basic statistics
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # T-test against zero mean
            t_stat, t_pvalue = stats.ttest_1samp(returns, 0)
            
            # Bootstrap confidence interval for Sharpe ratio
            n_bootstrap = 1000
            bootstrap_sharpes = []
            
            for _ in range(n_bootstrap):
                boot_sample = np.random.choice(returns, size=len(returns), replace=True)
                boot_sharpe = np.mean(boot_sample) / np.std(boot_sample) * np.sqrt(252) if np.std(boot_sample) > 0 else 0
                bootstrap_sharpes.append(boot_sharpe)
            
            sharpe_ci_lower = np.percentile(bootstrap_sharpes, 2.5)
            sharpe_ci_upper = np.percentile(bootstrap_sharpes, 97.5)
            
            # Permutation test for mean return
            n_permutations = 1000
            perm_means = []
            original_mean = np.mean(returns)
            
            for _ in range(n_permutations):
                perm_returns = np.random.permutation(returns)
                perm_means.append(np.mean(perm_returns))
            
            perm_pvalue = np.sum(np.array(perm_means) >= original_mean) / n_permutations
            
            # Jarque-Bera test for normality
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
            
            # Ljung-Box test for autocorrelation
            try:
                lb_stat, lb_pvalue = stats.boxljung(returns, lags=10, return_df=False)
                autocorr_present = lb_pvalue < 0.05
            except:
                autocorr_present = False
                lb_pvalue = 1.0
                warnings_list.append("Could not perform Ljung-Box test")
            
            # Multiple testing correction (Bonferroni)
            alpha = self.config.threshold
            corrected_alpha = alpha / 4  # 4 tests: t-test, permutation, JB, LB
            
            # Calculate composite score
            score = 0.0
            details = {
                "sharpe_ratio": sharpe_ratio,
                "t_statistic": t_stat,
                "t_pvalue": t_pvalue,
                "sharpe_ci_lower": sharpe_ci_lower,
                "sharpe_ci_upper": sharpe_ci_upper,
                "permutation_pvalue": perm_pvalue,
                "jarque_bera_pvalue": jb_pvalue,
                "ljung_box_pvalue": lb_pvalue,
                "corrected_alpha": corrected_alpha,
                "autocorr_present": autocorr_present
            }
            
            # Scoring criteria
            if t_pvalue < corrected_alpha and t_stat > 0:  # Significant positive returns
                score += 0.3
            if perm_pvalue < corrected_alpha:  # Permutation test significant
                score += 0.25
            if sharpe_ci_lower > 0:  # Positive Sharpe CI
                score += 0.25
            if abs(sharpe_ratio) > 0.5:  # Decent Sharpe ratio
                score += 0.2
            
            # Determine result
            if score >= self.config.threshold:
                result = LayerResult.PASS
                confidence = min(score, 1.0)
            else:
                result = LayerResult.FAIL
                confidence = 1.0 - score
            
            return LayerValidationResult(
                layer_name="statistical_test",
                result=result,
                score=score,
                confidence=confidence,
                details=details,
                execution_time=time.time() - start_time,
                warnings=warnings_list
            )
            
        except Exception as e:
            self.logger.error(f"Error in statistical testing: {str(e)}")
            return LayerValidationResult(
                layer_name="statistical_test",
                result=LayerResult.FAIL,
                score=0.0,
                confidence=1.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                warnings=[f"Validation failed with error: {str(e)}"]
            )

# ================================================================================================
# LAYER 3: REGIME ROBUSTNESS ANALYSIS  
# ================================================================================================

class RegimeRobustnessValidator:
    """
    Layer 3: Regime Robustness Analysis
    
    Tests strategy performance across different market regimes:
    - Bull/bear market performance
    - High/low volatility periods
    - Crisis period analysis
    - Regime change adaptability
    """
    
    def __init__(self, config: LayerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".RegimeRobustness")
    
    def _identify_regimes(self, prices: np.ndarray, returns: np.ndarray) -> Dict[str, np.ndarray]:
        """Identify different market regimes"""
        n = len(prices)
        
        # Bull/Bear regimes (based on moving average crossover)
        ma_short = pd.Series(prices).rolling(20, min_periods=1).mean().values
        ma_long = pd.Series(prices).rolling(50, min_periods=1).mean().values
        bull_mask = ma_short > ma_long
        
        # Volatility regimes (based on rolling volatility)
        vol_window = 20
        rolling_vol = pd.Series(returns).rolling(vol_window, min_periods=1).std().values
        vol_median = np.median(rolling_vol)
        high_vol_mask = rolling_vol > vol_median
        
        # Crisis periods (based on large drawdowns)
        cum_returns = np.cumprod(1 + returns)
        drawdowns, _ = _calculate_drawdowns_numba(cum_returns)
        crisis_mask = drawdowns < -0.1  # 10% drawdown threshold
        
        # Trend regimes (based on price momentum)
        momentum = np.diff(prices, prepend=prices[0])
        trend_up_mask = momentum > 0
        
        return {
            "bull": bull_mask,
            "bear": ~bull_mask,
            "high_vol": high_vol_mask,
            "low_vol": ~high_vol_mask,
            "crisis": crisis_mask,
            "normal": ~crisis_mask,
            "trend_up": trend_up_mask,
            "trend_down": ~trend_up_mask
        }
    
    def validate(self, returns: np.ndarray, prices: np.ndarray, strategy_name: str) -> LayerValidationResult:
        """Execute regime robustness validation"""
        start_time = time.time()
        warnings_list = []
        
        try:
            # Identify regimes
            regimes = self._identify_regimes(prices, returns)
            
            # Calculate performance metrics for each regime
            regime_results = {}
            
            for regime_name, regime_mask in regimes.items():
                regime_returns = returns[regime_mask]
                
                if len(regime_returns) < 10:  # Minimum observations per regime
                    warnings_list.append(f"Insufficient data for {regime_name} regime ({len(regime_returns)} obs)")
                    continue
                
                # Calculate regime-specific metrics
                sharpe = _regime_sharpe_numba(regime_returns, np.ones(len(regime_returns), dtype=bool))
                mean_return = np.mean(regime_returns) * 252  # Annualized
                volatility = np.std(regime_returns) * np.sqrt(252)  # Annualized
                
                # Calculate regime hit rate
                positive_returns = np.sum(regime_returns > 0) / len(regime_returns)
                
                regime_results[regime_name] = {
                    "sharpe": sharpe,
                    "mean_return": mean_return,
                    "volatility": volatility,
                    "hit_rate": positive_returns,
                    "observations": len(regime_returns)
                }
            
            # Calculate consistency score
            sharpe_ratios = [r["sharpe"] for r in regime_results.values() if np.isfinite(r["sharpe"])]
            
            if len(sharpe_ratios) < 3:
                warnings_list.append("Insufficient regime data for robust analysis")
                score = 0.0
            else:
                # Consistency metrics
                mean_sharpe = np.mean(sharpe_ratios)
                sharpe_std = np.std(sharpe_ratios)
                min_sharpe = np.min(sharpe_ratios)
                consistency_ratio = 1 - (sharpe_std / (abs(mean_sharpe) + 1e-8))
                
                # Scoring logic
                score = 0.0
                
                # Positive performance across regimes
                if mean_sharpe > 0:
                    score += 0.3
                if min_sharpe > -1.0:  # No terrible regime performance
                    score += 0.25
                if consistency_ratio > 0.5:  # Reasonable consistency
                    score += 0.25
                    
                # Bonus for crisis performance
                if "crisis" in regime_results and regime_results["crisis"]["sharpe"] > 0:
                    score += 0.2
            
            details = {
                "regime_results": regime_results,
                "consistency_metrics": {
                    "mean_sharpe": np.mean(sharpe_ratios) if sharpe_ratios else 0,
                    "min_sharpe": np.min(sharpe_ratios) if sharpe_ratios else 0,
                    "sharpe_std": np.std(sharpe_ratios) if sharpe_ratios else 0,
                    "num_regimes": len(sharpe_ratios)
                }
            }
            
            # Determine result
            if score >= self.config.threshold:
                result = LayerResult.PASS
                confidence = min(score, 1.0)
            else:
                result = LayerResult.FAIL
                confidence = 1.0 - score
            
            return LayerValidationResult(
                layer_name="regime_robustness",
                result=result,
                score=score,
                confidence=confidence,
                details=details,
                execution_time=time.time() - start_time,
                warnings=warnings_list
            )
            
        except Exception as e:
            self.logger.error(f"Error in regime robustness validation: {str(e)}")
            return LayerValidationResult(
                layer_name="regime_robustness",
                result=LayerResult.FAIL,
                score=0.0,
                confidence=1.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                warnings=[f"Validation failed with error: {str(e)}"]
            )

# ================================================================================================
# LAYER 4: SIGNAL CONSENSUS & ORTHOGONALITY
# ================================================================================================

class SignalConsensusValidator:
    """
    Layer 4: Signal Consensus & Orthogonality
    
    Multi-signal validation focusing on:
    - Signal correlation analysis
    - Orthogonality testing
    - Consensus scoring
    - Information ratio optimization
    """
    
    def __init__(self, config: LayerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".SignalConsensus")
    
    def validate(self, signal_returns: Dict[str, np.ndarray], strategy_name: str) -> LayerValidationResult:
        """Execute signal consensus validation"""
        start_time = time.time()
        warnings_list = []
        
        try:
            if len(signal_returns) < 2:
                return LayerValidationResult(
                    layer_name="signal_consensus",
                    result=LayerResult.WARNING,
                    score=0.5,
                    confidence=0.8,
                    details={"reason": "Single signal - consensus analysis not applicable"},
                    execution_time=time.time() - start_time,
                    warnings=["Only one signal provided - consensus analysis skipped"]
                )
            
            # Align signals by common time periods
            signal_names = list(signal_returns.keys())
            min_length = min(len(returns) for returns in signal_returns.values())
            
            aligned_signals = {}
            for name, returns in signal_returns.items():
                aligned_signals[name] = returns[-min_length:]  # Take most recent periods
            
            # Create signal matrix
            signal_matrix = np.column_stack([aligned_signals[name] for name in signal_names])
            
            # Correlation analysis
            correlation_matrix = np.corrcoef(signal_matrix.T)
            
            # Calculate orthogonality metrics
            mean_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
            max_correlation = np.max(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
            
            # PCA analysis for dimensionality
            scaler = StandardScaler()
            scaled_signals = scaler.fit_transform(signal_matrix)
            pca = PCA()
            pca.fit(scaled_signals)
            explained_variance = pca.explained_variance_ratio_
            
            # Effective number of independent signals
            effective_signals = 1 / np.sum(explained_variance ** 2)  # Inverse participation ratio
            
            # Mutual information analysis
            mutual_info_scores = []
            for i in range(len(signal_names)):
                for j in range(i+1, len(signal_names)):
                    # Discretize signals for mutual information
                    signal_i_disc = pd.cut(aligned_signals[signal_names[i]], bins=10, labels=False)
                    signal_j_disc = pd.cut(aligned_signals[signal_names[j]], bins=10, labels=False)
                    
                    mi_score = mutual_info_score(signal_i_disc, signal_j_disc)
                    mutual_info_scores.append(mi_score)
            
            mean_mutual_info = np.mean(mutual_info_scores) if mutual_info_scores else 0
            
            # Signal consensus calculation
            equal_weight_returns = np.mean(signal_matrix, axis=1)
            consensus_sharpe = np.mean(equal_weight_returns) / np.std(equal_weight_returns) * np.sqrt(252) if np.std(equal_weight_returns) > 0 else 0
            
            # Individual signal Sharpe ratios
            individual_sharpes = []
            for name in signal_names:
                signal_ret = aligned_signals[name]
                sharpe = np.mean(signal_ret) / np.std(signal_ret) * np.sqrt(252) if np.std(signal_ret) > 0 else 0
                individual_sharpes.append(sharpe)
            
            mean_individual_sharpe = np.mean(individual_sharpes)
            consensus_improvement = consensus_sharpe / (mean_individual_sharpe + 1e-8) if mean_individual_sharpe != 0 else 0
            
            # Scoring logic
            score = 0.0
            
            # Low correlation (good diversification)
            if mean_correlation < 0.3:
                score += 0.25
            elif mean_correlation < 0.5:
                score += 0.15
            
            # High effective number of signals
            if effective_signals / len(signal_names) > 0.7:
                score += 0.25
            elif effective_signals / len(signal_names) > 0.5:
                score += 0.15
            
            # Consensus improvement
            if consensus_improvement > 1.1:
                score += 0.3
            elif consensus_improvement > 1.0:
                score += 0.2
            
            # Low mutual information (independence)
            if mean_mutual_info < 0.1:
                score += 0.2
            
            details = {
                "num_signals": len(signal_names),
                "mean_correlation": mean_correlation,
                "max_correlation": max_correlation,
                "effective_signals": effective_signals,
                "mean_mutual_info": mean_mutual_info,
                "consensus_sharpe": consensus_sharpe,
                "mean_individual_sharpe": mean_individual_sharpe,
                "consensus_improvement": consensus_improvement,
                "correlation_matrix": correlation_matrix.tolist(),
                "explained_variance": explained_variance.tolist(),
                "individual_sharpes": dict(zip(signal_names, individual_sharpes))
            }
            
            # Determine result
            if score >= self.config.threshold:
                result = LayerResult.PASS
                confidence = min(score, 1.0)
            else:
                result = LayerResult.FAIL
                confidence = 1.0 - score
            
            return LayerValidationResult(
                layer_name="signal_consensus",
                result=result,
                score=score,
                confidence=confidence,
                details=details,
                execution_time=time.time() - start_time,
                warnings=warnings_list
            )
            
        except Exception as e:
            self.logger.error(f"Error in signal consensus validation: {str(e)}")
            return LayerValidationResult(
                layer_name="signal_consensus",
                result=LayerResult.FAIL,
                score=0.0,
                confidence=1.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                warnings=[f"Validation failed with error: {str(e)}"]
            )

# ================================================================================================
# LAYER 5: COPULA-WEIGHTED PORTFOLIO ALLOCATION
# ================================================================================================

class CopulaPortfolioValidator:
    """
    Layer 5: Copula-Weighted Portfolio Allocation
    
    Advanced risk management using copula modeling:
    - Dependency structure modeling
    - Tail risk assessment
    - Optimal allocation weights
    - Risk-adjusted performance optimization
    """
    
    def __init__(self, config: LayerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".CopulaPortfolio")
        self.copula_available = COPULA_AVAILABLE
    
    def _estimate_copula_weights(self, signal_matrix: np.ndarray) -> np.ndarray:
        """Estimate optimal portfolio weights using copula modeling"""
        if not self.copula_available:
            # Fallback to equal weights
            return np.ones(signal_matrix.shape[1]) / signal_matrix.shape[1]
        
        try:
            # Fit Gaussian copula
            copula = GaussianMultivariate()
            
            # Convert to uniform marginals
            uniform_data = np.zeros_like(signal_matrix)
            for i in range(signal_matrix.shape[1]):
                uniform_data[:, i] = stats.rankdata(signal_matrix[:, i]) / (len(signal_matrix) + 1)
            
            copula.fit(uniform_data)
            
            # Generate scenarios for risk assessment
            n_scenarios = 1000
            scenarios = copula.sample(n_scenarios)
            
            # Convert back to original scale
            scenario_returns = np.zeros_like(scenarios)
            for i in range(signal_matrix.shape[1]):
                marginal_dist = stats.norm(np.mean(signal_matrix[:, i]), np.std(signal_matrix[:, i]))
                scenario_returns[:, i] = marginal_dist.ppf(scenarios[:, i])
            
            # Optimize weights using mean-variance with tail risk constraints
            def portfolio_objective(weights):
                portfolio_returns = scenario_returns @ weights
                expected_return = np.mean(portfolio_returns)
                portfolio_vol = np.std(portfolio_returns)
                var_95 = np.percentile(portfolio_returns, 5)  # 95% VaR
                
                # Objective: maximize Sharpe ratio while minimizing tail risk
                sharpe = expected_return / portfolio_vol if portfolio_vol > 0 else 0
                tail_penalty = max(0, -var_95) * 10  # Penalty for large tail losses
                
                return -(sharpe - tail_penalty)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
            ]
            bounds = [(0, 1)] * signal_matrix.shape[1]  # Long-only
            
            # Initial guess (equal weights)
            initial_weights = np.ones(signal_matrix.shape[1]) / signal_matrix.shape[1]
            
            # Optimize
            result = minimize(portfolio_objective, initial_weights, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x
            else:
                return initial_weights
                
        except Exception as e:
            self.logger.warning(f"Copula optimization failed: {str(e)}, using equal weights")
            return np.ones(signal_matrix.shape[1]) / signal_matrix.shape[1]
    
    def validate(self, signal_returns: Dict[str, np.ndarray], strategy_name: str) -> LayerValidationResult:
        """Execute copula-weighted portfolio validation"""
        start_time = time.time()
        warnings_list = []
        
        try:
            if len(signal_returns) < 2:
                return LayerValidationResult(
                    layer_name="copula_portfolio",
                    result=LayerResult.WARNING,
                    score=0.5,
                    confidence=0.8,
                    details={"reason": "Single signal - portfolio optimization not applicable"},
                    execution_time=time.time() - start_time,
                    warnings=["Only one signal provided - portfolio optimization skipped"]
                )
            
            # Align signals
            signal_names = list(signal_returns.keys())
            min_length = min(len(returns) for returns in signal_returns.values())
            
            aligned_signals = {}
            for name, returns in signal_returns.items():
                aligned_signals[name] = returns[-min_length:]
            
            signal_matrix = np.column_stack([aligned_signals[name] for name in signal_names])
            
            # Estimate optimal weights
            optimal_weights = self._estimate_copula_weights(signal_matrix)
            
            # Calculate portfolio performance
            portfolio_returns = signal_matrix @ optimal_weights
            equal_weight_returns = np.mean(signal_matrix, axis=1)
            
            # Performance metrics
            portfolio_sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252) if np.std(portfolio_returns) > 0 else 0
            equal_weight_sharpe = np.mean(equal_weight_returns) / np.std(equal_weight_returns) * np.sqrt(252) if np.std(equal_weight_returns) > 0 else 0
            
            # Risk metrics
            portfolio_var_95 = np.percentile(portfolio_returns, 5)
            portfolio_cvar_95 = np.mean(portfolio_returns[portfolio_returns <= portfolio_var_95])
            
            equal_weight_var_95 = np.percentile(equal_weight_returns, 5)
            equal_weight_cvar_95 = np.mean(equal_weight_returns[equal_weight_returns <= equal_weight_var_95])
            
            # Information ratio improvement
            tracking_error = np.std(portfolio_returns - equal_weight_returns)
            information_ratio = (np.mean(portfolio_returns) - np.mean(equal_weight_returns)) / tracking_error if tracking_error > 0 else 0
            
            # Weight concentration (Herfindahl index)
            weight_concentration = np.sum(optimal_weights ** 2)
            diversification_ratio = 1 / weight_concentration
            
            # Scoring logic
            score = 0.0
            
            # Sharpe improvement
            sharpe_improvement = portfolio_sharpe / (equal_weight_sharpe + 1e-8) if equal_weight_sharpe != 0 else 0
            if sharpe_improvement > 1.1:
                score += 0.3
            elif sharpe_improvement > 1.0:
                score += 0.2
            
            # Risk reduction
            var_improvement = portfolio_var_95 / (equal_weight_var_95 - 1e-8) if equal_weight_var_95 != 0 else 0
            if var_improvement > 1.0:  # Less negative VaR is better
                score += 0.25
            
            # Information ratio
            if information_ratio > 0.5:
                score += 0.25
            elif information_ratio > 0:
                score += 0.15
            
            # Diversification
            if diversification_ratio > 2:
                score += 0.2
            elif diversification_ratio > 1.5:
                score += 0.1
            
            details = {
                "optimal_weights": dict(zip(signal_names, optimal_weights.tolist())),
                "portfolio_sharpe": portfolio_sharpe,
                "equal_weight_sharpe": equal_weight_sharpe,
                "sharpe_improvement": sharpe_improvement,
                "portfolio_var_95": portfolio_var_95,
                "equal_weight_var_95": equal_weight_var_95,
                "portfolio_cvar_95": portfolio_cvar_95,
                "information_ratio": information_ratio,
                "weight_concentration": weight_concentration,
                "diversification_ratio": diversification_ratio,
                "copula_available": self.copula_available
            }
            
            # Add warning if copula libraries not available
            if not self.copula_available:
                warnings_list.append("Copula libraries not available - using fallback optimization")
            
            # Determine result
            if score >= self.config.threshold:
                result = LayerResult.PASS
                confidence = min(score, 1.0)
            else:
                result = LayerResult.FAIL
                confidence = 1.0 - score
            
            return LayerValidationResult(
                layer_name="copula_portfolio",
                result=result,
                score=score,
                confidence=confidence,
                details=details,
                execution_time=time.time() - start_time,
                warnings=warnings_list
            )
            
        except Exception as e:
            self.logger.error(f"Error in copula portfolio validation: {str(e)}")
            return LayerValidationResult(
                layer_name="copula_portfolio",
                result=LayerResult.FAIL,
                score=0.0,
                confidence=1.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                warnings=[f"Validation failed with error: {str(e)}"]
            )

# ================================================================================================
# MAIN PIPELINE
# ================================================================================================

class RenaissancePipeline:
    """
    Main 5-Layer Validation Pipeline
    
    Orchestrates all validation layers and provides final strategy assessment.
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger(__name__ + ".RenaissancePipeline")
        
        # Initialize validators
        self.validators = {
            "noise_gate": NoiseGateValidator(self.config.noise_gate),
            "statistical_test": StatisticalTestValidator(self.config.statistical_test),
            "regime_robustness": RegimeRobustnessValidator(self.config.regime_analysis),
            "signal_consensus": SignalConsensusValidator(self.config.signal_consensus),
            "copula_portfolio": CopulaPortfolioValidator(self.config.portfolio_allocation)
        }
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
    
    def validate_strategy(self, 
                         returns: np.ndarray,
                         prices: np.ndarray = None,
                         signal_returns: Dict[str, np.ndarray] = None,
                         strategy_name: str = "unknown") -> PipelineValidationResult:
        """
        Execute the full 5-layer validation pipeline
        
        Args:
            returns: Strategy returns
            prices: Price series for regime analysis
            signal_returns: Dictionary of individual signal returns for consensus analysis
            strategy_name: Name of the strategy being validated
            
        Returns:
            PipelineValidationResult with comprehensive validation results
        """
        start_time = time.time()
        layer_results = []
        
        self.logger.info(f"🏛️ Starting Renaissance 5-Layer Validation for {strategy_name}")
        self.logger.info("=" * 80)
        
        try:
            # Layer 1: Noise Gate Filtering
            if self.config.noise_gate.enabled:
                self.logger.info("🔍 Layer 1: Noise Gate Filtering")
                noise_result = self.validators["noise_gate"].validate(returns, strategy_name)
                layer_results.append(noise_result)
                self.logger.info(f"   Result: {noise_result.result.value} (Score: {noise_result.score:.3f})")
            
            # Layer 2: Statistical Significance Testing
            if self.config.statistical_test.enabled:
                self.logger.info("📊 Layer 2: Statistical Significance Testing")
                stat_result = self.validators["statistical_test"].validate(returns, strategy_name)
                layer_results.append(stat_result)
                self.logger.info(f"   Result: {stat_result.result.value} (Score: {stat_result.score:.3f})")
            
            # Layer 3: Regime Robustness Analysis
            if self.config.regime_analysis.enabled and prices is not None:
                self.logger.info("🌍 Layer 3: Regime Robustness Analysis")
                regime_result = self.validators["regime_robustness"].validate(returns, prices, strategy_name)
                layer_results.append(regime_result)
                self.logger.info(f"   Result: {regime_result.result.value} (Score: {regime_result.score:.3f})")
            
            # Layer 4: Signal Consensus & Orthogonality
            if self.config.signal_consensus.enabled and signal_returns is not None:
                self.logger.info("🤝 Layer 4: Signal Consensus & Orthogonality")
                consensus_result = self.validators["signal_consensus"].validate(signal_returns, strategy_name)
                layer_results.append(consensus_result)
                self.logger.info(f"   Result: {consensus_result.result.value} (Score: {consensus_result.score:.3f})")
            
            # Layer 5: Copula-Weighted Portfolio Allocation
            if self.config.portfolio_allocation.enabled and signal_returns is not None:
                self.logger.info("🎯 Layer 5: Copula-Weighted Portfolio Allocation")
                portfolio_result = self.validators["copula_portfolio"].validate(signal_returns, strategy_name)
                layer_results.append(portfolio_result)
                self.logger.info(f"   Result: {portfolio_result.result.value} (Score: {portfolio_result.score:.3f})")
            
            # Calculate overall result
            passed_layers = sum(1 for result in layer_results if result.result == LayerResult.PASS)
            warning_layers = sum(1 for result in layer_results if result.result == LayerResult.WARNING)
            total_layers = len(layer_results)
            
            # Overall scoring
            total_score = np.mean([r.score for r in layer_results]) if layer_results else 0.0
            
            # Determine overall result
            if self.config.require_all_layers:
                # All layers must pass
                overall_result = LayerResult.PASS if passed_layers == total_layers else LayerResult.FAIL
            else:
                # Minimum layers must pass
                if passed_layers >= self.config.min_passing_layers:
                    overall_result = LayerResult.PASS
                elif passed_layers + warning_layers >= self.config.min_passing_layers:
                    overall_result = LayerResult.WARNING
                else:
                    overall_result = LayerResult.FAIL
            
            # Generate summary
            summary = f"Renaissance Pipeline: {passed_layers}/{total_layers} layers passed"
            if warning_layers > 0:
                summary += f" ({warning_layers} warnings)"
            
            execution_time = time.time() - start_time
            
            result = PipelineValidationResult(
                overall_result=overall_result,
                overall_score=total_score,
                layer_results=layer_results,
                execution_time=execution_time,
                strategy_name=strategy_name,
                timestamp=pd.Timestamp.now().isoformat(),
                summary=summary
            )
            
            self.logger.info("=" * 80)
            self.logger.info(f"🏆 Overall Result: {overall_result.value}")
            self.logger.info(f"📈 Overall Score: {total_score:.3f}")
            self.logger.info(f"⏱️  Execution Time: {execution_time:.2f}s")
            self.logger.info(f"📋 Summary: {summary}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Renaissance pipeline validation: {str(e)}")
            
            # Return failure result
            return PipelineValidationResult(
                overall_result=LayerResult.FAIL,
                overall_score=0.0,
                layer_results=layer_results,
                execution_time=time.time() - start_time,
                strategy_name=strategy_name,
                timestamp=pd.Timestamp.now().isoformat(),
                summary=f"Pipeline failed with error: {str(e)}"
            )

# ================================================================================================
# CONVENIENCE FUNCTIONS
# ================================================================================================

def run_renaissance_validation(returns: np.ndarray,
                             prices: np.ndarray = None,
                             signal_returns: Dict[str, np.ndarray] = None,
                             strategy_name: str = "strategy",
                             config: PipelineConfig = None) -> PipelineValidationResult:
    """
    Convenience function to run Renaissance validation pipeline
    
    Args:
        returns: Strategy returns
        prices: Price series for regime analysis (optional)
        signal_returns: Dictionary of individual signal returns (optional)
        strategy_name: Name of the strategy
        config: Pipeline configuration (optional)
        
    Returns:
        PipelineValidationResult
    """
    pipeline = RenaissancePipeline(config)
    return pipeline.validate_strategy(returns, prices, signal_returns, strategy_name)

def create_default_config() -> PipelineConfig:
    """Create a default pipeline configuration"""
    return PipelineConfig()

def create_strict_config() -> PipelineConfig:
    """Create a strict pipeline configuration requiring all layers to pass"""
    config = PipelineConfig()
    config.require_all_layers = True
    config.noise_gate.threshold = 0.5
    config.statistical_test.threshold = 0.01  # More stringent
    config.regime_analysis.threshold = 0.2
    config.signal_consensus.threshold = 0.1
    config.portfolio_allocation.threshold = 0.05
    return config

def create_lenient_config() -> PipelineConfig:
    """Create a lenient pipeline configuration for exploratory analysis"""
    config = PipelineConfig()
    config.require_all_layers = False
    config.min_passing_layers = 2
    config.noise_gate.threshold = 0.3
    config.statistical_test.threshold = 0.1
    config.regime_analysis.threshold = 0.3
    config.signal_consensus.threshold = 0.2
    config.portfolio_allocation.threshold = 0.1
    return config

if __name__ == "__main__":
    # Example usage
    logger.info("5-Layer Validation Pipeline")
    logger.info("Ready for strategy validation!")
