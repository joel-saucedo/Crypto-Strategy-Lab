"""
Position sizing and risk management for Crypto Strategy Lab.
Implements robust fractional Kelly with VaR and drawdown caps.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy import stats
import yaml

class PositionSizer:
    """
    Robust position sizing that enforces risk limits from the blueprint.
    """
    
    def __init__(self, config_path: str = "config/base.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.position_limits = self.config['position_limits']
        
    def calculate_position(self, 
                          signal: int,
                          expected_return: float,
                          volatility: float,
                          current_portfolio_value: float,
                          current_drawdown: float = 0) -> float:
        """
        Calculate position size using robust Kelly framework.
        
        Args:
            signal: Trading signal {-1, 0, 1}
            expected_return: Expected return for the trade
            volatility: Expected volatility
            current_portfolio_value: Current portfolio value
            current_drawdown: Current drawdown from peak
            
        Returns:
            Position size as fraction of portfolio
        """
        if signal == 0 or volatility <= 0:
            return 0
            
        # Basic Kelly calculation
        kelly_fraction = self._kelly_fraction(expected_return, volatility)
        
        # Apply risk limits
        position_size = self._apply_risk_limits(kelly_fraction, current_drawdown)
        
        # Apply signal direction
        position_size *= signal
        
        return position_size
        
    def _kelly_fraction(self, expected_return: float, volatility: float) -> float:
        """Calculate Kelly fraction with robustness adjustments."""
        if volatility <= 0:
            return 0
            
        # Standard Kelly: f = μ/σ²
        raw_kelly = expected_return / (volatility ** 2)
        
        # Cap Kelly at configured maximum
        kelly_cap = self.position_limits['kelly_cap']
        capped_kelly = np.clip(raw_kelly, -kelly_cap, kelly_cap)
        
        return capped_kelly
        
    def _apply_risk_limits(self, kelly_fraction: float, current_drawdown: float) -> float:
        """Apply position and risk limits."""
        position_size = kelly_fraction
        
        # Maximum position size limit
        max_position = self.position_limits['max_position_size']
        position_size = np.clip(position_size, -max_position, max_position)
        
        # Reduce size during drawdowns
        if current_drawdown > 0.05:  # 5% drawdown threshold
            drawdown_adjustment = 1 - (current_drawdown * 2)  # Linear reduction
            position_size *= max(drawdown_adjustment, 0.1)  # Minimum 10% of normal size
            
        return position_size
        
    def calculate_var_limit(self, 
                           positions: Dict[str, float],
                           returns_history: pd.DataFrame,
                           confidence: float = 0.05) -> Dict[str, float]:
        """
        Calculate Value at Risk limits for current positions.
        
        Args:
            positions: Dictionary of strategy positions
            returns_history: Historical returns for each strategy
            confidence: VaR confidence level (default 5%)
            
        Returns:
            Dictionary of VaR limits for each strategy
        """
        var_limits = {}
        
        for strategy, position in positions.items():
            if strategy not in returns_history.columns:
                continue
                
            strategy_returns = returns_history[strategy].dropna()
            
            if len(strategy_returns) < 30:  # Need sufficient history
                var_limits[strategy] = 0
                continue
                
            # Calculate parametric VaR
            mean_return = strategy_returns.mean()
            vol_return = strategy_returns.std()
            
            var_limit = stats.norm.ppf(confidence, mean_return, vol_return)
            var_limits[strategy] = abs(var_limit) * position
            
        return var_limits
        
    def portfolio_heat(self, positions: Dict[str, float]) -> float:
        """
        Calculate total portfolio heat (sum of absolute positions).
        
        Args:
            positions: Dictionary of strategy positions
            
        Returns:
            Total portfolio heat
        """
        return sum(abs(pos) for pos in positions.values())
        
    def rebalance_positions(self, 
                          current_positions: Dict[str, float],
                          target_positions: Dict[str, float],
                          max_turnover: float = 0.5) -> Dict[str, float]:
        """
        Rebalance positions with turnover constraints.
        
        Args:
            current_positions: Current strategy positions
            target_positions: Target strategy positions
            max_turnover: Maximum turnover allowed
            
        Returns:
            Adjusted target positions
        """
        adjusted_positions = {}
        
        for strategy in target_positions:
            current_pos = current_positions.get(strategy, 0)
            target_pos = target_positions[strategy]
            
            position_change = target_pos - current_pos
            
            # Limit position change by max turnover
            if abs(position_change) > max_turnover:
                direction = np.sign(position_change)
                position_change = direction * max_turnover
                
            adjusted_positions[strategy] = current_pos + position_change
            
        return adjusted_positions
        
    def validate_positions(self, positions: Dict[str, float]) -> Dict[str, bool]:
        """
        Validate positions against risk limits.
        
        Returns:
            Dictionary of validation results
        """
        total_heat = self.portfolio_heat(positions)
        max_leverage = self.position_limits['max_leverage']
        
        validation = {
            'within_leverage_limit': total_heat <= max_leverage,
            'individual_position_limits': all(
                abs(pos) <= self.position_limits['max_position_size'] 
                for pos in positions.values()
            ),
            'no_extreme_concentration': max(abs(pos) for pos in positions.values()) < 0.5
        }
        
        return validation
