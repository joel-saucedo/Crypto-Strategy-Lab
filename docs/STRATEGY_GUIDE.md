# Strategy Guide

This guide provides the mathematical foundation and implementation details for each of the 12 orthogonal trading strategies in the Crypto Strategy Lab framework.

## 1. Lag-k Autocorrelation Strategy

**Mathematical Foundation:**
```
ρ(k) = Σ(r_t - r̄)(r_{t-k} - r̄) / Σ(r_t - r̄)²
```

**Edge:** Daily cryptocurrency returns exhibit statistically significant serial dependence.

**Trade Rule:** Long if ρ(1) > 0 and p-value < 0.05; short if ρ(1) < 0. Require |ρ| > 1.96/√N over 250-day lookback.

**Implementation:** `src/strategies/lag_autocorr/`

## 2. Variance-Ratio Strategy

**Mathematical Foundation:** Lo-MacKinlay variance ratio test for random walk hypothesis.

**Edge:** Detect persistence or mean reversion in price series.

**Implementation:** `src/strategies/variance_ratio/`

## 3. Hurst Exponent Strategy

**Mathematical Foundation:** Fractional Brownian motion analysis using R/S statistics.

**Edge:** Identify long-term memory in price movements.

**Implementation:** `src/strategies/hurst_exponent/`

## 4. Drift Significance Strategy

**Mathematical Foundation:** Cramér-Rao lower bound for drift parameter estimation.

**Edge:** Statistical significance testing of drift in price series.

**Implementation:** `src/strategies/drift_significance/`

## 5. Spectral Peaks Strategy

**Mathematical Foundation:** Power spectral density analysis for dominant frequencies.

**Edge:** Identify cyclical patterns in price movements.

**Implementation:** `src/strategies/spectral_peaks/`

## 6. Permutation Entropy Strategy

**Mathematical Foundation:** Ordinal pattern analysis for predictability measurement.

**Edge:** Quantify complexity and predictability in time series.

**Implementation:** `src/strategies/permutation_entropy/`

## 7. Volatility Clustering Strategy

**Mathematical Foundation:** GARCH modeling for volatility persistence.

**Edge:** Exploit volatility clustering effects in cryptocurrency markets.

**Implementation:** `src/strategies/volatility_clustering/`

## 8. Spectral Entropy Strategy

**Mathematical Foundation:** Frequency domain entropy measurement.

**Edge:** Detect regime changes through spectral complexity.

**Implementation:** `src/strategies/spectral_entropy/`

## 9. VPIN Strategy

**Mathematical Foundation:** Volume-synchronized Probability of Informed Trading.

**Edge:** Order flow imbalance detection for directional prediction.

**Implementation:** `src/strategies/vpin/`

## 10. Wavelet Energy Strategy

**Mathematical Foundation:** Multi-resolution wavelet analysis.

**Edge:** Breakout detection across multiple time scales.

**Implementation:** `src/strategies/wavelet_energy/`

## 11. PACF Spike Strategy

**Mathematical Foundation:** Partial autocorrelation function analysis.

**Edge:** Autoregressive structure identification.

**Implementation:** `src/strategies/pacf_spike/`

## 12. True Range Divergence Strategy

**Mathematical Foundation:** Price-volatility divergence analysis.

**Edge:** Mean reversion signals from volatility-price disconnects.

**Implementation:** `src/strategies/true_range_divergence/`


---



---

## Unit Test Expectations

Each strategy implementation must pass:

1. **No NaN Output:** Signal generation never produces NaN values
2. **Index Alignment:** Output index matches input (no look-ahead bias)
3. **Signal Range:** All signals in {-1, 0, 1}
4. **Monte Carlo DSR:** Positive expected DSR on synthetic data with known edge
5. **Parameter Sensitivity:** Reasonable behavior across parameter ranges
6. **Mathematical Consistency:** Implementation matches documented equations

## Risk Management Integration

All strategies integrate with the core risk framework:

- **Position Sizing:** Fractional Kelly with VaR caps
- **Regime Awareness:** Volatility and trend regime adjustments
- **Correlation Management:** Portfolio-level diversification
- **Drawdown Control:** Dynamic sizing during adverse periods

## Implementation Notes

- Each strategy lives in `src/strategies/<name>/signal.py`
- Configuration in `config/strategies/<name>.yaml`
- Mathematical proofs in `docs/pdf/<name>.pdf`
- Unit tests in `src/strategies/<name>/test_signal.py`
- Hyperparameter optimization via `src/pipelines/hyperopt.py`

## Implementation Framework

### Strategy Structure
Each strategy follows a consistent implementation pattern:
- Main logic in `src/strategies/<strategy_name>/signal.py`
- Configuration in `config/strategies/<strategy_name>.yaml`
- Unit tests in `src/strategies/<strategy_name>/test_signal.py`
- Mathematical proofs in `docs/pdf/<strategy_name>.pdf`

### Quality Standards
All strategies must meet these requirements:
1. **Statistical Significance:** DSR ≥ 0.95 across multiple validation tests
2. **Orthogonality:** Low correlation with existing strategies
3. **Implementation Quality:** Clean code with comprehensive test coverage
4. **Documentation:** Complete mathematical foundations and trade rules
5. **Parameter Robustness:** Stable performance across parameter ranges

### Framework Extension
The framework supports adding new strategies through:
- Strategy registry with automatic detection
- Common interface for signal generation
- Statistical validation gates
- Hyperparameter optimization support
- Ensemble composition capabilities

### Adding New Strategies
1. Use `scripts/new_strategy.py` to generate boilerplate code
2. Implement signal generation logic following the standard interface
3. Create comprehensive tests validating statistical properties
4. Document mathematical foundation and trade rules
5. Submit pull request with complete implementation and documentation

### Development Workflow
```bash
# Generate new strategy scaffold
python scripts/new_strategy.py my_strategy_name

# Implement strategy logic
# Edit src/strategies/my_strategy_name/signal.py

# Test implementation
pytest src/strategies/my_strategy_name/test_signal.py -v

# Validate statistical significance
python scripts/validate_strategy.py my_strategy_name

# Run hyperparameter optimization
python scripts/run_hyperopt.py my_strategy_name
```


