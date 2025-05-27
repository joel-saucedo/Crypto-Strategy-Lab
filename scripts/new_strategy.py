#!/bin/bash

"""
Strategy scaffold generator for Crypto-Strategy-Lab.
Creates complete directory structure for new trading strategies.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml

def create_strategy_scaffold(strategy_name: str, base_dir: str = "."):
    """Create complete scaffold for a new strategy."""
    
    # Validate strategy name
    if not strategy_name.replace('_', '').isalnum():
        raise ValueError("Strategy name must contain only letters, numbers, and underscores")
    
    base_path = Path(base_dir)
    
    # Create directory structure
    strategy_dir = base_path / "src" / "strategies" / strategy_name
    strategy_dir.mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py
    init_content = f'"""Strategy module: {strategy_name}"""'
    (strategy_dir / "__init__.py").write_text(init_content)
    
    # Create signal.py template
    signal_template = f'''"""
{strategy_name.replace('_', ' ').title()} Strategy

Mathematical foundation:
[Insert equation here]

Edge: [Insert statistical hypothesis]
Trade rule: [Insert precise entry/exit logic]
Risk hooks: [Insert risk management details]
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import yaml

class {strategy_name.replace('_', '').title()}Signal:
    """
    {strategy_name.replace('_', ' ').title()} trading signal generator.
    
    Implements the framework from docs/pdf/{strategy_name}.pdf
    """
    
    def __init__(self, config_path: str = "config/strategies/{strategy_name}.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.params = config['best_params']
        
    def generate(self, returns: pd.Series, **context) -> pd.Series:
        """
        Generate trading signals based on {strategy_name.replace('_', ' ')}.
        
        Args:
            returns: Price return series
            **context: Additional market context
            
        Returns:
            Signal series with values {{-1, 0, 1}}
        """
        signals = pd.Series(0, index=returns.index)
        
        # TODO: Implement strategy logic here
        # Example template:
        # lookback = self.params['lookback_days']
        # threshold = self.params['threshold']
        #
        # for i in range(lookback, len(returns)):
        #     window_data = returns.iloc[i-lookback:i]
        #     signal_value = self._calculate_signal(window_data)
        #     signals.iloc[i] = signal_value
            
        return signals
        
    def _calculate_signal(self, data: pd.Series) -> int:
        """Calculate signal for a given data window."""
        # TODO: Implement signal calculation
        return 0
        
    def get_param_grid(self) -> Dict[str, Any]:
        """Return parameter grid for hyperoptimization."""
        with open(f"config/strategies/{strategy_name}.yaml", 'r') as f:
            config = yaml.safe_load(f)
        return config['param_grid']
'''
    
    (strategy_dir / "signal.py").write_text(signal_template)
    
    # Create test template
    test_template = f'''"""
Unit tests for {strategy_name} strategy.
Validates implementation against mathematical expectations.
"""

import pytest
import numpy as np
import pandas as pd
from src.strategies.{strategy_name}.signal import {strategy_name.replace('_', '').title()}Signal

class Test{strategy_name.replace('_', '').title()}Signal:
    
    def setup_method(self):
        """Setup test fixtures."""
        self.signal = {strategy_name.replace('_', '').title()}Signal()
        
    def test_no_nan_output(self):
        """Test that signal generation doesn't produce NaN values."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 300))
        
        signals = self.signal.generate(returns)
        
        assert not signals.isna().any(), "Signal contains NaN values"
        
    def test_signal_range(self):
        """Test that signals are in valid range {{-1, 0, 1}}."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 300))
        
        signals = self.signal.generate(returns)
        
        valid_signals = signals.isin([-1, 0, 1])
        assert valid_signals.all(), "Invalid signal values detected"
        
    def test_index_alignment(self):
        """Test that output index aligns with input (no look-ahead)."""
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        returns = pd.Series(np.random.normal(0, 0.02, 200), index=dates)
        
        signals = self.signal.generate(returns)
        
        assert signals.index.equals(returns.index), "Index alignment failed"
        
    def test_no_look_ahead(self):
        """Test that signals don't use future information."""
        # Create returns with known future pattern
        returns = pd.Series([0.01] * 100 + [0.05] * 100)
        
        signals = self.signal.generate(returns)
        
        # Signals in first 100 periods shouldn't know about future high returns
        early_signals = signals.iloc[:100]
        late_signals = signals.iloc[100:]
        
        # This test needs to be customized based on strategy logic
        # assert condition that ensures no look-ahead bias
        
    def test_parameter_sensitivity(self):
        """Test reasonable behavior across parameter ranges."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 300))
        
        # Test with different parameter values
        # This should be customized based on actual parameters
        original_params = self.signal.params.copy()
        
        # Modify parameters and test
        # self.signal.params['some_param'] = different_value
        # signals = self.signal.generate(returns)
        # assert some_reasonable_condition
        
        # Restore original parameters
        self.signal.params = original_params
        
    def test_monte_carlo_dsr(self):
        """Test that strategy has positive expected performance on synthetic data."""
        # This test should create synthetic data with known properties
        # that the strategy should be able to exploit
        
        # TODO: Implement based on strategy's theoretical edge
        pass
        
    def test_mathematical_consistency(self):
        """Test that implementation matches documented equations."""
        # TODO: Test specific mathematical properties
        # For example, if strategy calculates correlation:
        # manual_corr = np.corrcoef(x, y)[0,1]
        # strategy_corr = self.signal._calculate_correlation(x, y)
        # assert np.isclose(manual_corr, strategy_corr)
        pass
'''
    
    (strategy_dir / "test_signal.py").write_text(test_template)
    
    # Create configuration file
    config_template = {
        'strategy': {
            'name': strategy_name,
            'description': f'{strategy_name.replace("_", " ").title()} trading strategy',
            'version': '1.0'
        },
        'param_grid': {
            'lookback_days': {
                'min': 30,
                'max': 250,
                'step': 10
            },
            'threshold': {
                'min': 0.01,
                'max': 0.10,
                'step': 0.01
            }
        },
        'risk': {
            'max_position': 0.05,
            'stop_loss': 0.10,
            'take_profit': 0.20
        },
        'signal': {
            'rebalance_freq': 'daily'
        },
        'best_params': {
            'lookback_days': 90,
            'threshold': 0.05
        }
    }
    
    config_path = base_path / "config" / "strategies" / f"{strategy_name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False)
    
    # Create LaTeX template
    latex_template = f'''\\documentclass{{article}}
\\usepackage{{amsmath, amssymb, amsfonts}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}

\\title{{{strategy_name.replace('_', ' ').title()} Strategy: Mathematical Foundation}}
\\author{{Crypto-Strategy-Lab}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{Abstract}}

[Brief description of the trading edge and mathematical foundation]

\\section{{Mathematical Framework}}

\\subsection{{Core Equation}}

[Insert main equation here]

\\subsection{{Statistical Hypothesis}}

\\textbf{{Null Hypothesis:}} [Insert H0]

\\textbf{{Alternative Hypothesis:}} [Insert H1]

\\section{{Edge Derivation}}

[Detailed mathematical derivation of why this edge should exist]

\\subsection{{Market Microstructure Basis}}

[Explanation of market mechanism that creates the edge]

\\subsection{{Statistical Properties}}

[Expected distribution of returns, Sharpe ratio bounds, etc.]

\\section{{Implementation Details}}

\\subsection{{Signal Generation}}

[Step-by-step algorithm for generating trading signals]

\\subsection{{Parameter Optimization}}

[Description of parameter space and optimization objective]

\\section{{Risk Management}}

\\subsection{{Position Sizing}}

[Integration with Kelly framework and risk limits]

\\subsection{{Stop Loss Logic}}

[Mathematical basis for stop loss placement]

\\section{{Expected Performance}}

\\subsection{{Theoretical Bounds}}

[Theoretical Sharpe ratio and DSR bounds]

\\subsection{{Sensitivity Analysis}}

[Parameter sensitivity and regime dependence]

\\section{{Unit Test Specifications}}

\\subsection{{Synthetic Data Tests}}

[Description of synthetic data that should yield positive DSR]

\\subsection{{Mathematical Consistency}}

[Tests that verify implementation matches equations]

\\section{{References}}

[Academic references and prior work]

\\end{{document}}
'''
    
    pdf_dir = base_path / "docs" / "pdf_src"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    (pdf_dir / f"{strategy_name}.tex").write_text(latex_template)
    
    # Update strategy guide
    strategy_guide_path = base_path / "docs" / "STRATEGY_GUIDE.md"
    if strategy_guide_path.exists():
        with open(strategy_guide_path, 'a') as f:
            f.write(f'''

---

## {len(open(strategy_guide_path).readlines()) // 20 + 1}. {strategy_name.replace('_', ' ').title()}

**Equation:**
```
[Insert equation here]
```

**Edge:** [Insert statistical hypothesis]

**Trade Rule:** [Insert trade rule]

**Risk Hooks:** [Insert risk management details]

**Implementation:** `src/strategies/{strategy_name}/`
''')
    
    print(f"Created strategy scaffold for '{strategy_name}':")
    print(f"  - src/strategies/{strategy_name}/signal.py")
    print(f"  - src/strategies/{strategy_name}/test_signal.py") 
    print(f"  - config/strategies/{strategy_name}.yaml")
    print(f"  - docs/pdf_src/{strategy_name}.tex")
    print(f"  - Updated docs/STRATEGY_GUIDE.md")
    print()
    print("Next steps:")
    print("1. Complete the mathematical foundation in the LaTeX file")
    print("2. Implement the signal generation logic")
    print("3. Write comprehensive unit tests")
    print("4. Run hyperparameter optimization")
    print("5. Validate DSR ≥ 0.95 requirement")

def main():
    parser = argparse.ArgumentParser(description='Generate strategy scaffold')
    parser.add_argument('strategy_name', help='Name of the new strategy (use underscores)')
    parser.add_argument('--base-dir', default='.', help='Base directory (default: current)')
    
    args = parser.parse_args()
    
    try:
        create_strategy_scaffold(args.strategy_name, args.base_dir)
    except Exception as e:
        print(f"Error creating strategy scaffold: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
