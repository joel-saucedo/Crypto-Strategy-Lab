# Contributing to Crypto Strategy Lab

This document outlines the process for contributing new strategies to the framework while maintaining statistical rigor and code quality.

## Quick Start

### 1. Generate Strategy Scaffold

```bash
python scripts/new_strategy.py my_new_strategy
```

This creates:
- `config/strategies/my_new_strategy.yaml` - Parameter configuration
- `src/strategies/my_new_strategy/` - Strategy implementation directory
- `docs/pdf_src/my_new_strategy.tex` - LaTeX documentation template

### 2. Mathematical Foundation

Write the mathematical proof before implementation:

1. Complete `docs/pdf_src/my_new_strategy.tex` with:
   - Formal hypothesis statement
   - Statistical edge derivation
   - Expected performance bounds
   - Risk characteristics

2. Build PDF: `make -C docs/pdf_src my_new_strategy.pdf`

3. Update `docs/STRATEGY_GUIDE.md` with strategy documentation

### 3. Implementation Requirements

#### Core Implementation
- Implement `generate(self, returns, **context)` method in `signal.py`
- Ensure output is `pd.Series` with values in `{-1, 0, 1}`
- Use only parameters from YAML configuration
- No hardcoded constants or look-ahead bias
- Include mathematical equations in docstring

#### Testing Requirements
Create `test_signal.py` with these tests:
- **No NaN Output:** Verify signal generation produces valid outputs
- **Index Alignment:** Confirm proper time series alignment
- **Signal Range:** Validate signal values are within expected range
- **No Look-Ahead:** Ensure no future information is used
- **Monte Carlo DSR:** Test with synthetic data for statistical significance
- **Parameter Sensitivity:** Validate robustness across parameter ranges

#### Configuration
- Define parameter grid in `config/strategies/my_new_strategy.yaml`
- Include reasonable bounds for optimization
- Set appropriate risk limits
- Document all parameter meanings

### 4. Validation Process

#### Local Testing
```bash
# Unit tests
pytest src/strategies/my_new_strategy/test_signal.py -v

# Hyperparameter optimization
python scripts/run_hyperopt.py my_new_strategy

# Strategy validation
python scripts/validate_strategy.py my_new_strategy
```

#### Pull Request Requirements
All contributions must pass:
1. **Code Quality:** Linting, type hints, comprehensive documentation
2. **Unit Tests:** 100% test coverage with all tests passing
3. **Integration Tests:** Strategy compatibility with backtest engine
4. **DSR Validation:** Out-of-sample DSR ≥ 0.95
5. **Stress Tests:** Bootstrap and permutation test validation

## Code Style Guidelines

### Strategy Class Template
```python
"""
Strategy Name: Brief mathematical description

Mathematical foundation:
[Insert main equation here]

Edge: Statistical hypothesis
Trade rule: Precise entry/exit logic
Risk management: Position sizing and limits
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

class MyNewStrategySignal:
    """
    Brief description with equation reference.
    
    This strategy implements [mathematical concept] to detect
    [market inefficiency] in cryptocurrency markets.
    """
    
    def __init__(self, **params):
        """Initialize strategy with parameters."""
        self.lookback_period = params.get('lookback_period', 20)
        self.significance_threshold = params.get('significance_threshold', 0.05)
    
    def generate(self, returns: pd.Series, **context) -> pd.Series:
        """
        Generate trading signals based on strategy logic.
        
        Args:
            returns: Price return series
            **context: Additional market data
            
        Returns:
            pd.Series: Trading signals (-1, 0, 1)
        """
        # Implementation details
        pass
```

### Documentation Standards
- Clear mathematical foundations in docstrings
- Comprehensive parameter documentation
- Usage examples with expected outputs
- Performance characteristics and limitations
- References to academic literature when applicable

### Testing Standards
```python
class TestMyNewStrategySignal:
    """Test suite for MyNewStrategySignal."""
    
    def test_no_nan_output(self):
        """Ensure strategy produces no NaN values."""
        pass
    
    def test_signal_range(self):
        """Verify signals are in valid range."""
        pass
    
    def test_monte_carlo_dsr(self):
        """Test statistical significance with synthetic data."""
        pass
```

## Review Process

### Code Review Checklist
- [ ] Mathematical foundation is clearly documented
- [ ] Implementation follows framework conventions
- [ ] All tests pass with adequate coverage
- [ ] Strategy demonstrates statistical significance
- [ ] Code is well-documented and readable
- [ ] Configuration is properly structured
- [ ] No hardcoded values or magic numbers

### Performance Review
- [ ] DSR ≥ 0.95 on out-of-sample data
- [ ] Strategy shows orthogonality to existing strategies
- [ ] Reasonable transaction costs are considered
- [ ] Risk management is properly implemented
- [ ] Performance is robust across different market conditions

## Community Guidelines

### Collaboration Standards
- Respectful and constructive feedback
- Focus on mathematical rigor and statistical significance
- Share knowledge and help other contributors
- Follow academic standards for research and validation
- Maintain open-source principles and transparency

### Communication
- Use GitHub issues for technical questions
- GitHub Discussions for methodology and research topics
- Provide detailed descriptions for pull requests
- Include performance metrics and validation results
- Reference relevant academic literature when applicable

## Getting Help

### Resources
- Framework documentation in `docs/` directory
- Example strategies in `src/strategies/` for reference
- Test examples for validation patterns
- Configuration templates for parameter setup

### Support Channels
- **GitHub Issues:** Technical questions and bug reports
- **GitHub Discussions:** Research methodology and strategy concepts
- **Code Review:** Detailed feedback on implementations
- **Documentation:** Comprehensive guides and examples

## Quality Assurance

### Continuous Integration
All contributions are automatically tested through:
- Automated code quality checks
- Comprehensive test suite execution
- Statistical validation of strategy performance
- Documentation building and validation
- Security and dependency scanning

### Manual Review
Experienced contributors review:
- Mathematical foundations and derivations
- Implementation correctness and efficiency
- Test coverage and validation approaches
- Documentation quality and completeness
- Overall contribution fit with framework goals

This process ensures all contributions maintain the high standards required for quantitative trading strategy development and deployment.
    
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
