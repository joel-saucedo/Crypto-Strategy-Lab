# Crypto Strategy Lab

A quantitative trading framework implementing orthogonal trading strategies with statistical validation and production deployment capabilities.

## Installation

```bash
git clone https://github.com/your-org/crypto-strategy-lab.git
cd crypto-strategy-lab

# Install dependencies
poetry install
docker compose build

# Data preparation
python scripts/fetch_data.py
python scripts/build_features.py

# Strategy optimization
for yml in config/strategies/*.yaml; do
  python scripts/run_hyperopt.py "${yml##*/}"
done

# View results
open reports/index.html
```

## Project Structure

```
crypto-strategy-lab/
├── docs/                     # Documentation
├── config/                   # Configuration files
├── data/                     # Data storage
├── src/                      # Source code
│   ├── core/                 # Core engine
│   ├── strategies/           # Trading strategies
│   ├── pipelines/            # Processing pipelines
│   └── utils/                # Utilities
├── scripts/                  # CLI tools
├── tests/                    # Test suite
└── .github/workflows/        # CI/CD
```

## Trading Strategies

1. **Lag-k Autocorrelation** - Serial dependence analysis
2. **Variance-Ratio** - Drift detection
3. **Hurst Exponent** - Persistence measurement
4. **Cramér-Rao Drift** - Statistical significance filtering
5. **Power-Spectral Peak** - Cycle identification
6. **Permutation Entropy** - Predictability measurement
7. **Volatility Clustering** - Regime detection
8. **Spectral Entropy** - Order measurement
9. **VPIN** - Order flow analysis
10. **Wavelet Energy-Ratio** - Breakout detection
11. **PACF Spike** - Autoregressive structure
12. **True-Range Divergence** - Mean reversion signals

## Requirements

- Mathematical validation for all strategies
- Deflated Sharpe Ratio ≥ 0.95 threshold
- Statistical significance testing
- Reproducible research environment
- Production deployment capabilities

## Documentation

- [Blueprint](docs/BLUEPRINT.md) - System architecture
- [Strategy Guide](docs/STRATEGY_GUIDE.md) - Strategy documentation
- [Contributing](docs/CONTRIBUTING.md) - Development guidelines
- [Team](docs/TEAM.md) - Project team

## Author

**Joel Saucedo** - Physics & Mathematics, Georgia College & State University
