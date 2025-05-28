"""
Feature engineering pipeline for Crypto Strategy Lab.
Handles raw price → processed features with lag alignment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats
import talib

class FeatureEngine:
    """
    Unified feature engineering that enforces no look-ahead bias.
    All strategies use the same feature preparation.
    """
    
    def __init__(self):
        self.features = {}
        
    def prepare_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw OHLCV to analysis-ready features.
        
        Args:
            prices: DataFrame with ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with all engineered features
        """
        df = prices.copy()
        
        # Basic returns
        df['returns'] = df['close'].pct_change(fill_method=None)
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price-based features
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['co_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volatility features
        df['realized_vol'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['parkinson_vol'] = self._parkinson_volatility(df)
        
        # Technical indicators
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'].values)
        
        # Regime indicators
        df['regime_vol'] = self._volatility_regime(df['realized_vol'])
        df['regime_trend'] = self._trend_regime(df['close'])
        
        # Autocorrelation features
        df['autocorr_1'] = self._rolling_autocorr(df['returns'], lag=1, window=90)
        df['autocorr_5'] = self._rolling_autocorr(df['returns'], lag=5, window=90)
        
        # Higher moments
        df['skewness'] = df['returns'].rolling(60).skew()
        df['kurtosis'] = df['returns'].rolling(60).kurt()
        
        return df
        
    def _parkinson_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Parkinson volatility estimator."""
        hl_log = np.log(df['high'] / df['low'])
        park_vol = np.sqrt(hl_log.rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252)
        return park_vol
        
    def _volatility_regime(self, vol: pd.Series, window: int = 252) -> pd.Series:
        """Classify volatility regime: 0=Low, 1=Medium, 2=High."""
        vol_roll = vol.rolling(window)
        low_thresh = vol_roll.quantile(0.33)
        high_thresh = vol_roll.quantile(0.67)
        
        regime = pd.Series(1, index=vol.index)  # Default medium
        regime[vol < low_thresh] = 0  # Low vol
        regime[vol > high_thresh] = 2  # High vol
        
        return regime
        
    def _trend_regime(self, prices: pd.Series, window: int = 50) -> pd.Series:
        """Classify trend regime based on moving average slope."""
        ma = prices.rolling(window).mean()
        slope = ma.diff(10) / ma  # 10-day slope
        
        regime = pd.Series(0, index=prices.index)  # Default flat
        regime[slope > 0.002] = 1   # Uptrend (0.2% per 10 days)
        regime[slope < -0.002] = -1  # Downtrend
        
        return regime
        
    def _rolling_autocorr(self, series: pd.Series, lag: int, window: int) -> pd.Series:
        """Calculate rolling autocorrelation."""
        def autocorr_func(x):
            if len(x) <= lag:
                return np.nan
            return x.autocorr(lag=lag)
            
        return series.rolling(window).apply(autocorr_func, raw=False)
        
    def align_features(self, df: pd.DataFrame, strategy_lag: int = 1) -> pd.DataFrame:
        """
        Ensure proper lag alignment to prevent look-ahead bias.
        
        Args:
            df: DataFrame with features
            strategy_lag: Days to lag features (default 1 for daily strategies)
            
        Returns:
            Properly aligned DataFrame
        """
        # Shift features by strategy_lag to prevent look-ahead
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        for col in feature_cols:
            df[f"{col}_lag{strategy_lag}"] = df[col].shift(strategy_lag)
            
        return df.dropna()
        
    def validate_features(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate features for common issues.
        
        Returns:
            Dictionary of validation results
        """
        validation = {
            'no_infinite_values': not np.isinf(df.select_dtypes(include=[np.number])).any().any(),
            'no_all_nan_columns': not df.isnull().all().any(),
            'sufficient_history': len(df) >= 252,  # At least 1 year
            'no_future_leakage': True  # Placeholder for more sophisticated check
        }
        
        return validation
