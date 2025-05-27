# Strategy Guide: Mathematical Foundations

This guide provides the mathematical foundation for each of the 12 orthogonal trading edges in Crypto-Strategy-Lab.

## 1. Lag-k Autocorrelation Momentum/Reversal Filter

**Equation:**
```
ρ(k) = Σ(r_t - r̄)(r_{t-k} - r̄) / Σ(r_t - r̄)²
```

**Edge:** Daily crypto returns exhibit statistically significant serial dependence (ρ(1) ≠ 0).

**Trade Rule:** Long if ρ(1) > 0 and p-value < 0.05; short/flat if ρ(1) < 0. Require |ρ| > 1.96/√N over a 250-day look-back to clear noise.

**Risk Hooks:** Position size proportional to t-statistic significance.

**Implementation:** `src/strategies/lag_autocorr/`

---

## 2. Variance-Ratio (VR) Drift Detector

**Equation:**
```
VR(q) = Var[P_{t+q} - P_t] / (q × Var[P_{t+1} - P_t])
```

**Edge:** VR(q) > 1 signals persistent drift; < 1 signals mean-reverting noise.

**Trade Rule:** In a 60-day window trade momentum when VR(5) > 1+δ (δ=0.1) and Lo-MacKinlay Z > 2; trade contrarian when VR(5) < 1-δ.

**Risk Hooks:** Scale position by Z-statistic confidence.

**Implementation:** `src/strategies/variance_ratio/`

---

## 3. Hurst-Exponent Regime Gauge

**Equation:**
```
(R/S)_n ∝ n^H  ⟹  H = log((R/S)_n) / log(n)
```

**Edge:** H > 0.5 indicates trending persistence; H < 0.5 indicates anti-persistence.

**Trade Rule:** Compute DFA-based H_200 (≈9 months). If H > 0.55 use trend-following with wider stops; if H < 0.45 use mean-reversion with tighter targets.

**Risk Hooks:** Stop-loss width proportional to H estimate confidence.

**Implementation:** `src/strategies/hurst_exponent/`

---

## 4. Cramér-Rao Drift-Significance Filter

**Equation:**
```
t = μ̂/(σ/√N),  Var(μ̂) ≥ σ²/N
```

**Edge:** Only drifts with |t| > 2 on a 90-day window beat the noise floor.

**Trade Rule:** Position size ∝ |t|; skip trades when |t| ≤ 2.

**Risk Hooks:** Dynamic position sizing based on signal strength.

**Implementation:** `src/strategies/drift_significance/`

---

## 5. Power-Spectral Peak Cycle Timer

**Equation:**
```
S(f) = |ℱ{r_t}|²
```

**Edge:** Significant peaks reveal hidden periodicities (e.g., 4-year halving cycle).

**Trade Rule:** Band-pass ±15% around the dominant frequency; long at troughs, short at peaks when amplitude > 2× noise baseline (χ² test).

**Risk Hooks:** Position size scaled by spectral peak significance.

**Implementation:** `src/strategies/spectral_peaks/`

---

## 6. Permutation-Entropy Predictability Index

**Equation:**
```
H_PE(m) = -Σ_π p(π) ln p(π)
```

**Edge:** Low H_PE ⟹ high regularity ⟹ greater forecastability.

**Trade Rule:** With embedding m=5, trade only when H_PE is in the bottom decile; remain flat when in the top decile.

**Risk Hooks:** Position size inversely proportional to entropy level.

**Implementation:** `src/strategies/permutation_entropy/`

---

## 7. Volatility-Clustering Regime Switch

**Equation:**
```
σ²_{t+1} = ω + α r_t² + β σ_t²
```

**Edge:** Crypto vol clusters; calm regimes yield cleaner drift.

**Trade Rule:** Label Calm if 20-day realized vol < 20-percentile; Storm if > 80-percentile. Run trend-following only in Calm; reduce size or switch to mean-reversion in Storm.

**Risk Hooks:** Regime-dependent position sizing and strategy selection.

**Implementation:** `src/strategies/volatility_clustering/`

---

## 8. Spectral-Entropy Collapse (SBPRF Core)

**Equation:**
```
H_spec = -Σ_f (S(f)/ΣS(f)) ln[S(f)/ΣS(f)]
```

**Edge:** Entropy collapse ⟹ variance concentrates in few frequencies ⟹ temporary order.

**Trade Rule:** On a 64-day FFT window trade only when H_spec < 25-percentile and the dominant peak > 3× median power.

**Risk Hooks:** Signal strength proportional to entropy collapse magnitude.

**Implementation:** `src/strategies/spectral_entropy/`

---

## 9. Volume-Synchronized Order-Flow Imbalance (VPIN)

**Equation:**
```
VPIN = (1/N) Σ_{i=1}^N |B_i - S_i|
```

**Edge:** High VPIN marks toxic flow preceding whipsaws; low VPIN indicates balanced flow.

**Trade Rule:** Using 50 equal-volume buckets, go flat/contrarian when VPIN > 95-percentile; favor continuation when < 10-percentile.

**Risk Hooks:** Reduce position size during high VPIN periods.

**Implementation:** `src/strategies/vpin/`

---

## 10. Wavelet Energy-Ratio Breakout

**Equation:**
```
E_j = Σ_t |W_{j,t}|²,  WER = (Σ_{j≤J_trend} E_j) / (Σ_{j>J_trend} E_j)
```

**Edge:** Surge in low-frequency energy signals emerging macro moves.

**Trade Rule:** When WER > one-year median +1σ, classify "trend" and allow momentum; when below median -1σ, trade mean-reversion.

**Risk Hooks:** Position size scaled by energy ratio deviation.

**Implementation:** `src/strategies/wavelet_energy/`

---

## 11. Partial Autocorrelation "Spike" Detector

**Equation:**
```
φ_kk = PACF(k),  k = 1,2,...
```

**Edge:** A lone significant PACF spike at lag k implies exploitable AR(k) structure.

**Trade Rule:** On a 180-day window, if |φ_kk| > 2×SE and higher lags are insignificant: buy k days after a down-day when φ_kk > 0; sell when φ_kk < 0.

**Risk Hooks:** Hold period limited to k×2 days maximum.

**Implementation:** `src/strategies/pacf_spike/`

---

## 12. True-Range Divergence Mean-Reversion

**Equation:**
```
TR_t = max{H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|}
```

**Edge:** Extremely high true-range unconfirmed by net direction often snaps back.

**Trade Rule:** If 20-day z-score of TR > 2 and |C_t - O_t| < 0.2×TR_t, fade the move and target reversion to the 5-day mean.

**Risk Hooks:** Stop-loss at 1.5× true range from entry.

**Implementation:** `src/strategies/true_range_divergence/`

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

## 12. Variance Ratio

**Equation:**
```
[Insert equation here]
```

**Edge:** [Insert statistical hypothesis]

**Trade Rule:** [Insert trade rule]

**Risk Hooks:** [Insert risk management details]

**Implementation:** `src/strategies/variance_ratio/`
