# Strategy Guide: Mathematical Foundations

This guide provides the mathematical foundation for each of the 12 orthogonal trading edges in Crypto-Strategy-Lab.

## 1. Lag-k Autocorrelation Momentum/Reversal Filter

**Equation:**## Implementation Notes

- Each strategy lives in either `src/strategies/<n>/signal.py` or `src/strategies/<n>/strategy.py`
- Configuration in `config/strategies/<n>.yaml`
- Mathematical proofs in `docs/pdf/<n>.pdf`
- Unit tests in `src/strategies/<n>/test_signal.py`
- Hyperparameter optimization via `src/pipelines/hyperopt.py`

## Framework Extension

The Crypto-Strategy-Lab is designed to be extensible beyond the 12 orthogonal trading edges implemented. The framework provides:

1. **Strategy Registry**: All strategies automatically register with the framework through module imports
2. **Common Interface**: Each strategy implements a standard interface for signal generation
3. **Validation Gates**: New strategies must pass statistical significance tests
4. **Hyperparameter Optimization**: Framework supports automated parameter tuning
5. **Composition Layer**: Strategies can be combined using weighted ensembles

### Adding New Strategies

To implement a new strategy:

1. Use `scripts/new_strategy.py` to generate the boilerplate
2. Implement the signal generation logic in `signal.py` or `strategy.py`
3. Create tests that validate statistical properties
4. Document the mathematical foundation in STRATEGY_GUIDE.md
5. Submit a pull request with comprehensive documentation

### Strategy Quality Standards

All strategies must meet these requirements:

1. **Statistical Significance**: DSR ≥ 0.95 across multiple tests
2. **Orthogonality**: Low correlation with existing strategies
3. **Implementation Quality**: Clean code with comprehensive tests
4. **Documentation**: Complete mathematical foundations
5. **Parameter Robustness**: Stable performance across parameter rangesΣ(r_t - r̄)(r_{t-k} - r̄) / Σ(r_t - r̄)²
```

**Edge:** Daily crypto returns exhibit statistically significant serial dependence (ρ(1) ≠ 0).

**Trade Rule:** Long if ρ(1) > 0 and p-value < 0.05; short/flat if ρ(1) < 0. Require |ρ| > 1.96/√N over a 250-day look-back to clear noise.

**Risk Hooks:** Position size proportional to t-statistic significance.

**Implementation:** `src/strategies/lag_autocorr/`

---

## 2. Variance-Ratio (VR) Drift Detector


---

## 3. Hurst-Exponent Regime Gauge


---

## 4. Cramér-Rao Drift-Significance Filter


---

## 5. Power-Spectral Peak Cycle Timer


---

## 6. Permutation-Entropy Predictability Index


---

## 7. Volatility-Clustering Regime Switch


---

## 8. Spectral-Entropy Collapse (SBPRF Core)


---

## 9. Volume-Synchronized Order-Flow Imbalance (VPIN)


---

## 10. Wavelet Energy-Ratio Breakout


---

## 11. Partial Autocorrelation "Spike" Detector


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


---




---




---




---




---




---




---


