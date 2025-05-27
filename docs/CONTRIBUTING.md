# Contributing to Crypto-Strategy-Lab

This document outlines the process for adding new strategies to the framework while maintaining statistical rigor and code quality.

## Quick Start: Adding a New Strategy

### 1. Generate Strategy Scaffold

```bash
python scripts/new_strategy.py my_new_edge
```

This generates:
- `config/strategies/my_new_edge.yaml` (parameter grid stub)
- `src/strategies/my_new_edge/` with templated `signal.py` and `test_signal.py`
- `docs/pdf_src/my_new_edge.tex` (LaTeX template)

### 2. Mathematical Foundation First

**CRITICAL:** Write the mathematical proof before any code.

1. Complete `docs/pdf_src/my_new_edge.tex` with:
   - Formal hypothesis statement
   - Statistical edge derivation
   - Expected performance bounds
   - Risk characteristics

2. Build PDF: `make -C docs/pdf_src my_new_edge.pdf`

3. Update `docs/STRATEGY_GUIDE.md` with equation and trade rule

### 3. Implementation Checklist

#### Mathematical Implementation
- [ ] Copy equations into `signal.py` header docstring
- [ ] Implement `generate(self, returns, **context)` method
- [ ] Ensure output is `pd.Series` with values in `{-1, 0, 1}`
- [ ] Use only parameters from YAML configuration
- [ ] No hardcoded constants or look-ahead bias

#### Testing Requirements
Write `test_signal.py` that verifies:
- [ ] **No NaN Output:** `test_no_nan_output()`
- [ ] **Index Alignment:** `test_index_alignment()`
- [ ] **Signal Range:** `test_signal_range()`
- [ ] **No Look-Ahead:** `test_no_look_ahead()`
- [ ] **Monte Carlo DSR:** `test_monte_carlo_dsr()` with synthetic data
- [ ] **Parameter Sensitivity:** `test_parameter_sensitivity()`

#### Configuration
- [ ] Define parameter grid in `config/strategies/my_new_edge.yaml`
- [ ] Include reasonable bounds for hyperoptimization
- [ ] Set appropriate risk limits
- [ ] Document parameter meanings

### 4. Validation Pipeline

#### Local Testing
```bash
# Run unit tests
pytest src/strategies/my_new_edge/test_signal.py -v

# Run hyperparameter optimization
python scripts/run_hyperopt.py my_new_edge

# Validate DSR threshold
python scripts/validate_strategy.py my_new_edge
```

#### Pull Request Requirements
All PRs must pass:
1. **Code Quality:** Linting, type hints, documentation
2. **Unit Tests:** All tests pass with 100% coverage
3. **Integration Tests:** Strategy works with backtest engine
4. **DSR Gate:** Pooled OOS DSR ≥ 0.95
5. **Stress Tests:** Passes bootstrap and permutation tests

## Code Style Guidelines

### Signal Class Template
```python
"""
Strategy Name: Brief mathematical description

Mathematical foundation:
[Insert main equation here]

Edge: Statistical hypothesis
Trade rule: Precise entry/exit logic
Risk hooks: Position sizing and limits
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import yaml

class StrategyNameSignal:
    """
    Brief description with equation reference.
    
    Implements the framework from docs/pdf/strategy_name.pdf
    """
    
    def __init__(self, config_path: str = "config/strategies/strategy_name.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.params = config['best_params']
        
    def generate(self, returns: pd.Series, **context) -> pd.Series:
        """
        Generate trading signals.
        
        Args:
            returns: Price return series
            **context: Additional market context (prices, volume, etc.)
            
        Returns:
            Signal series with values {-1, 0, 1}
        """
        # Implementation here
        pass
        
    def get_param_grid(self) -> Dict[str, Any]:
        """Return parameter grid for hyperoptimization."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['param_grid']
```

### Best Practices

#### Mathematical Accuracy
- Implement equations exactly as documented
- Use numerical stable algorithms
- Handle edge cases (zero variance, insufficient data)
- Validate against known analytical solutions when possible

#### Performance
- Vectorize operations using NumPy/Pandas
- Avoid loops for large datasets
- Cache expensive computations
- Profile critical paths

#### Robustness
- Handle missing data gracefully
- Validate input parameters
- Return meaningful error messages
- Test with edge cases (constant prices, extreme volatility)

## CI/CD Pipeline

### Automated Checks
The CI pipeline automatically:

1. **Lints Code:** `flake8`, `mypy`, `black`
2. **Runs Tests:** Unit, integration, and regression tests
3. **Builds Documentation:** LaTeX → PDF → strategy guide updates
4. **Hyperparameter Optimization:** Nested CV with DSR objective
5. **Stress Testing:** Bootstrap, permutation, White Reality Check
6. **DSR Validation:** Enforces DSR ≥ 0.95 gate

### Failure Modes
PRs are rejected if:
- Code quality checks fail
- Any unit test fails
- DSR < 0.95 on pooled OOS data
- Mathematical implementation doesn't match PDF
- Look-ahead bias detected
- Insufficient test coverage

## Integration with Core Framework

### Backtest Engine Integration
Strategies automatically integrate with:
- Position sizing (fractional Kelly)
- Risk management (VaR limits, drawdown caps)
- Execution simulation (latency, slippage)
- Performance attribution
- Live monitoring

### Data Pipeline
Access to:
- Price data (OHLCV)
- Volume profile
- Market microstructure
- Alternative data feeds
- Feature engineering pipeline

### Risk Framework
Automatic enforcement of:
- Maximum position limits
- Correlation limits
- Drawdown controls
- Regime-aware sizing
- Kill switches

## Collaboration Workflow

### For Contributors
1. Fork the repository
2. Create feature branch: `git checkout -b strategy/my-new-edge`
3. Follow the implementation checklist
4. Submit PR with DSR validation proof
5. Address review comments
6. Merge after CI passes

### For Maintainers
1. Review mathematical foundation first
2. Verify implementation matches equations
3. Check test coverage and quality
4. Validate DSR calculations
5. Ensure no architectural violations
6. Approve only after all gates pass

## Mathematical Review Process

### Peer Review
Each strategy requires:
- Independent mathematical verification
- Code review by senior contributor
- Backtesting validation on out-of-sample data
- Economic rationale assessment

### Documentation Standards
- LaTeX proofs with proper notation
- Clear statement of assumptions
- Derivation of expected performance
- Risk factor identification
- Unit test specifications

## Questions?

- **Technical Issues:** Open GitHub issue with `question` label
- **Mathematical Discussions:** Email team with PDF draft
- **Collaboration:** See `docs/TEAM.md` for contact information

Remember: Mathematics first, code second. The DSR gate ensures every strategy meets our statistical standards before going live.
