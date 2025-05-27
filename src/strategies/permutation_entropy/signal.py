"""
Permutation Entropy Strategy

Mathematical foundation:
H_PE(m) = -Σ_π p(π) ln p(π)

Edge: Low H_PE ⟹ high regularity ⟹ greater forecastability
Trade rule: With embedding m=5, trade only when H_PE is in the bottom decile; 
           remain flat when in the top decile
Risk hooks: Position size inversely proportional to entropy level
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import yaml
import math
from itertools import permutations
from scipy import stats

class PermutationentropySignal:
    """
    Permutation Entropy trading signal generator.
    
    Uses ordinal pattern analysis to measure predictability of price movements.
    Low entropy indicates high regularity and potential forecastability.
    """
    
    def __init__(self, config_path: str = "config/strategies/permutation_entropy.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.params = config['best_params']
        
    def generate(self, returns: pd.Series, **context) -> pd.Series:
        """
        Generate trading signals based on permutation entropy analysis.
        
        Args:
            returns: Price return series
            **context: Additional market context
            
        Returns:
            Signal series with values {-1, 0, 1}
        """
        signals = pd.Series(0, index=returns.index)
        
        window_size = self.params['window_size']
        embedding_dim = self.params['embedding_dim']
        entropy_threshold_low = self.params['entropy_threshold_low']
        entropy_threshold_high = self.params['entropy_threshold_high']
        
        for i in range(window_size, len(returns)):
            window_data = returns.iloc[i-window_size:i]
            signal_value = self._calculate_signal(
                window_data, 
                embedding_dim,
                entropy_threshold_low,
                entropy_threshold_high
            )
            signals.iloc[i] = signal_value
            
        return signals
        
    def _calculate_signal(
        self, 
        data: pd.Series, 
        embedding_dim: int,
        entropy_threshold_low: float,
        entropy_threshold_high: float
    ) -> int:
        """
        Calculate signal based on permutation entropy analysis.
        
        Args:
            data: Return series window
            embedding_dim: Embedding dimension for ordinal patterns
            entropy_threshold_low: Low entropy threshold (bottom decile)
            entropy_threshold_high: High entropy threshold (top decile)
            
        Returns:
            Signal value {-1, 0, 1}
        """
        if len(data) < embedding_dim + 10:  # Need sufficient data
            return 0
            
        # Calculate permutation entropy
        pe_value = self._calculate_permutation_entropy(data, embedding_dim)
        
        if np.isnan(pe_value):
            return 0
            
        # Normalize entropy (0 = perfectly predictable, 1 = random)
        max_entropy = np.log(math.factorial(embedding_dim))
        normalized_pe = pe_value / max_entropy if max_entropy > 0 else 0
        
        # Trading logic based on entropy levels
        if normalized_pe < entropy_threshold_low:
            # Low entropy = high predictability = follow trend
            recent_trend = np.sign(data.iloc[-5:].mean())
            return int(recent_trend)
        elif normalized_pe > entropy_threshold_high:
            # High entropy = low predictability = stay flat
            return 0
        else:
            # Medium entropy = uncertain regime
            return 0
            
    def _calculate_permutation_entropy(self, data: pd.Series, m: int) -> float:
        """
        Calculate permutation entropy using ordinal pattern analysis.
        
        Args:
            data: Time series data
            m: Embedding dimension (pattern length)
            
        Returns:
            Permutation entropy value
        """
        if len(data) < m:
            return np.nan
            
        # Extract ordinal patterns
        patterns = []
        
        for i in range(len(data) - m + 1):
            window = data.iloc[i:i+m].values
            
            # Get ordinal pattern (relative ranking)
            ordinal_pattern = tuple(np.argsort(np.argsort(window)))
            patterns.append(ordinal_pattern)
        
        if not patterns:
            return np.nan
            
        # Count pattern frequencies
        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
        # Calculate relative frequencies
        total_patterns = len(patterns)
        pattern_probs = [count / total_patterns for count in pattern_counts.values()]
        
        # Calculate permutation entropy
        pe = -sum(p * np.log(p) for p in pattern_probs if p > 0)
        
        return pe
        
    def _get_ordinal_pattern(self, window: np.ndarray) -> Tuple[int, ...]:
        """
        Convert a numeric window to its ordinal pattern.
        
        Args:
            window: Array of numeric values
            
        Returns:
            Tuple representing ordinal pattern
        """
        # Handle ties by adding small random noise
        if len(np.unique(window)) < len(window):
            noise = np.random.normal(0, 1e-10, len(window))
            window = window + noise
            
        # Get ranking (0 = smallest, m-1 = largest)
        ranks = np.argsort(np.argsort(window))
        
        return tuple(ranks)
        
    def calculate_predictability_score(self, data: pd.Series, embedding_dim: int = 5) -> float:
        """
        Calculate predictability score based on permutation entropy.
        
        Args:
            data: Time series data
            embedding_dim: Embedding dimension
            
        Returns:
            Predictability score (0 = random, 1 = perfectly predictable)
        """
        pe = self._calculate_permutation_entropy(data, embedding_dim)
        
        if np.isnan(pe):
            return 0.0
            
        # Normalize by maximum possible entropy
        max_entropy = np.log(math.factorial(embedding_dim))
        normalized_pe = pe / max_entropy if max_entropy > 0 else 0
        
        # Convert to predictability (inverse of entropy)
        predictability = 1 - normalized_pe
        
        return max(0, min(1, predictability))
        
    def get_entropy_percentiles(self, data: pd.Series, window_size: int = 200) -> Dict[str, float]:
        """
        Calculate entropy percentiles for threshold setting.
        
        Args:
            data: Return series
            window_size: Rolling window size
            
        Returns:
            Dictionary with entropy percentiles
        """
        embedding_dim = self.params['embedding_dim']
        entropies = []
        
        for i in range(window_size, len(data), 10):  # Sample every 10 days
            window_data = data.iloc[i-window_size:i]
            pe = self._calculate_permutation_entropy(window_data, embedding_dim)
            
            if not np.isnan(pe):
                max_entropy = np.log(math.factorial(embedding_dim))
                normalized_pe = pe / max_entropy if max_entropy > 0 else 0
                entropies.append(normalized_pe)
        
        if not entropies:
            return {'p10': 0.1, 'p90': 0.9}
            
        return {
            'p10': np.percentile(entropies, 10),
            'p25': np.percentile(entropies, 25),
            'p50': np.percentile(entropies, 50),
            'p75': np.percentile(entropies, 75),
            'p90': np.percentile(entropies, 90)
        }
        
    def get_param_grid(self) -> Dict[str, Any]:
        """Return parameter grid for hyperoptimization."""
        with open(f"config/strategies/permutation_entropy.yaml", 'r') as f:
            config = yaml.safe_load(f)
        return config['param_grid']
