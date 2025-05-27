# Crypto-Strategy-Lab

A research/deployment workbench that glues twelve orthogonal trading edges to a statistically bullet-proof back-testing core.

Everything—data ingestion, signal logic, cross-validated hyper-search, DSR gate-keeping, position sizing, shadow trading, and live monitoring—mirrors the master recipe and the two-layer blueprint.

**One repo → repeatable research → auditable production.**

## Quick Start

```bash
git clone https://github.com/your-org/Crypto-Strategy-Lab.git
cd Crypto-Strategy-Lab

# Install dependencies
poetry install                # or pip -r requirements.txt
docker compose build          # reproducible research/live env

# 1. Pull & preprocess data
python scripts/fetch_data.py
python scripts/build_features.py

# 2. Hyper-opt all strategies
for yml in config/strategies/*.yaml; do
  python scripts/run_hyperopt.py "${yml##*/}"
done

# 3. View reports
open reports/index.html
```

## Project Structure

```
Crypto-Strategy-Lab/
│  README.md
│  docker-compose.yml                 # frozen runtime for research + live
│
├─ docs/                              # human-readables only
│   BLUEPRINT.md                      # master recipe + layer 1 & 2 checklists
│   STRATEGY_GUIDE.md                 # links every strategy to its PDF
│   CONTRIBUTING.md                   # PR process, style, DSR gate
│   TEAM.md                           # Joel + collaborators
│   pdf/                              # strategy white-papers (math, edge, risk)
│
├─ config/                            # YAML knobs
│   base.yaml                         # fees, slippage, Kelly caps
│   data_sources.yaml                 # exchange URLS + lag offsets
│   strategies/                       # one YAML per strategy (θ grid, windows…)
│
├─ data/                              # DVC-tracked
│   raw/      processed/      reference/
│
├─ src/                               # **code that enforces the blueprint**
│   core/                             # engine shared by every strategy
│       features.py   backtest.py   position.py
│       metrics.py    validation.py execution.py
│   strategies/                       # plug-ins (one sub-package per edge)
│       lag_autocorr/      …          pacf_spike/
│   pipelines/                        # hyper-opt, walk-forward, live monitor
│   utils/
│
├─ scripts/                           # CLI wrappers (fetch_data, build_features…)
├─ tests/                             # unit, integration, regression, audit
└─ .github/workflows/                 # CI: lint → tests → build PDFs → DSR audit
```

## The 12 Orthogonal Trading Edges

1. **Lag-k Autocorrelation** - Serial dependence momentum/reversal
2. **Variance-Ratio** - Drift vs noise detection
3. **Hurst Exponent** - Persistence regime gauge
4. **Cramér-Rao Drift** - Statistical significance filter
5. **Power-Spectral Peak** - Cycle timing
6. **Permutation Entropy** - Predictability index
7. **Volatility Clustering** - Regime switching
8. **Spectral Entropy** - SBPRF core collapse detector
9. **VPIN** - Order flow imbalance
10. **Wavelet Energy-Ratio** - Breakout detection
11. **PACF Spike** - AR structure identification
12. **True-Range Divergence** - Mean reversion signals

## Philosophy

- **Mathematics first**: Every strategy requires a PDF proof before code
- **DSR gate**: All strategies must pass Deflated Sharpe Ratio ≥ 0.95
- **Separation of concerns**: Math → YAML → Signal class → Engine
- **Reproducible research**: Docker + DVC for data versioning
- **Auditable production**: CI/CD with statistical validation gates

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for the full process.

## Team

Lead: **Joel Saucedo** - Physics & Math undergrad (GCSU)
See [TEAM.md](docs/TEAM.md) for collaborators.
