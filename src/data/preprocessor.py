"""
Data preprocessing module for cryptocurrency trading data.
Implements comprehensive feature engineering and data transformations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessing for cryptocurrency trading strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration dictionary
        """
        self.config = config or self._default_config()
        self.scalers = {}
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int, alpha: Optional[float] = None) -> pd.Series:
        """Exponential Moving Average."""
        if alpha is None:
            alpha = 2.0 / (period + 1.0)
        return series.ewm(alpha=alpha, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = DataPreprocessor.ema(series, fast)
        ema_slow = DataPreprocessor.ema(series, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = DataPreprocessor.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """True Range calculation."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range."""
        tr = DataPreprocessor.true_range(high, low, close)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return wr
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_dev = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (typical_price - sma_tp) / (0.015 * mean_dev)
        return cci
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index."""
        # Calculate True Range
        tr = DataPreprocessor.true_range(high, low, close)
        
        # Calculate Directional Movement
        plus_dm = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
        minus_dm = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=low.index)
        
        # Smooth the values
        tr_smooth = tr.rolling(window=period).mean()
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()
        
        # Calculate Directional Indicators
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
        
    def _default_config(self) -> Dict[str, Any]:
        """Default preprocessing configuration."""
        return {
            'remove_outliers': True,
            'outlier_method': 'iqr',  # 'iqr' or 'zscore'
            'outlier_threshold': 3.0,
            'fill_method': 'forward',  # 'forward', 'linear', 'spline'
            'min_periods': 100,  # Minimum periods for valid calculations
            'technical_indicators': {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bollinger_period': 20,
                'bollinger_std': 2,
                'atr_period': 14,
                'volume_sma_period': 20
            }
        }
    
    def clean_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean OHLCV data with comprehensive quality checks.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        initial_len = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove rows with any NaN values in OHLCV
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df.dropna(subset=required_cols)
        
        # OHLC relationship validation
        valid_ohlc = (
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close']) &
            (df['high'] >= df['low']) &
            (df['high'] > 0) &
            (df['low'] > 0) &
            (df['open'] > 0) &
            (df['close'] > 0) &
            (df['volume'] >= 0)
        )
        
        df = df[valid_ohlc].copy()
        
        logger.info(f"Data cleaning: {initial_len} -> {len(df)} rows ({len(df)/initial_len:.1%} retained)")
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Remove outliers using specified method.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers (default: price columns)
            
        Returns:
            DataFrame with outliers removed
        """
        if not self.config['remove_outliers']:
            return df
            
        if columns is None:
            columns = ['open', 'high', 'low', 'close']
            
        df = df.copy()
        initial_len = len(df)
        
        method = self.config['outlier_method']
        threshold = self.config['outlier_threshold']
        
        mask = pd.Series(True, index=df.index)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                col_mask = (df[col] >= lower) & (df[col] <= upper)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col]))
                col_mask = z_scores < threshold
                
            else:
                raise ValueError(f"Unknown outlier method: {method}")
                
            mask &= col_mask
        
        df_clean = df[mask].copy()
        
        removed_pct = (initial_len - len(df_clean)) / initial_len * 100
        logger.info(f"Outlier removal: {removed_pct:.1f}% of data removed")
        
        return df_clean
    
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add various return calculations.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with return columns added
        """
        df = df.copy()
        
        # Basic returns
        df['returns'] = df['close'].pct_change(fill_method=None)
        df['returns_1h'] = df['close'].pct_change(1, fill_method=None)
        df['returns_4h'] = df['close'].pct_change(4, fill_method=None)
        df['returns_24h'] = df['close'].pct_change(24, fill_method=None)
        
        # Log returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Intraday returns
        df['intraday_returns'] = (df['close'] - df['open']) / df['open']
        df['high_low_returns'] = (df['high'] - df['low']) / df['low']
        
        # Overnight gap
        df['gap_returns'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with technical indicators added
        """
        df = df.copy()
        config = self.config['technical_indicators']
        
        # Price-based indicators
        df['sma_20'] = self.sma(df['close'], period=20)
        df['ema_20'] = self.ema(df['close'], period=20)
        df['sma_50'] = self.sma(df['close'], period=50)
        df['ema_50'] = self.ema(df['close'], period=50)
        
        # RSI
        df['rsi'] = self.rsi(df['close'], period=config['rsi_period'])
        
        # MACD
        macd, macd_signal, macd_hist = self.macd(
            df['close'],
            fast=config['macd_fast'],
            slow=config['macd_slow'],
            signal=config['macd_signal']
        )
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.bollinger_bands(
            df['close'],
            period=config['bb_period'],
            std_dev=config['bb_std']
        )
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR (Average True Range)
        df['atr'] = self.atr(df['high'], df['low'], df['close'], period=config['atr_period'])
        
        # Volume indicators
        df['volume_sma'] = self.sma(df['volume'], period=config['volume_sma_period'])
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = self.stochastic(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = self.williams_r(df['high'], df['low'], df['close'])
        
        # Commodity Channel Index
        df['cci'] = self.cci(df['high'], df['low'], df['close'])
        
        # Average Directional Index
        df['adx'] = self.adx(df['high'], df['low'], df['close'])
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based features.
        
        Args:
            df: DataFrame with returns
            
        Returns:
            DataFrame with volatility features added
        """
        df = df.copy()
        
        # Rolling volatility (multiple windows)
        for window in [5, 10, 20, 50]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std() * np.sqrt(24)  # Annualized
            df[f'volatility_{window}_rank'] = df[f'volatility_{window}'].rolling(252).rank(pct=True)
        
        # Realized volatility (Garman-Klass)
        gk_vol = np.log(df['high'] / df['low']) ** 2 - (2 * np.log(2) - 1) * np.log(df['close'] / df['open']) ** 2
        df['gk_volatility'] = gk_vol.rolling(20).mean().apply(np.sqrt) * np.sqrt(24)
        
        # Parkinson volatility
        park_vol = np.log(df['high'] / df['low']) ** 2 / (4 * np.log(2))
        df['parkinson_volatility'] = park_vol.rolling(20).mean().apply(np.sqrt) * np.sqrt(24)
        
        # Rogers-Satchell volatility
        rs_vol = (np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) + 
                 np.log(df['low'] / df['close']) * np.log(df['low'] / df['open']))
        df['rs_volatility'] = rs_vol.rolling(20).mean().apply(np.sqrt) * np.sqrt(24)
        
        return df
    
    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add microstructure-based features.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with microstructure features added
        """
        df = df.copy()
        
        # Price impact measures
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        df['tick_direction'] = np.sign(df['close'] - df['close'].shift(1))
        
        # Volume-price relationship
        df['price_volume_trend'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * df['volume']
        df['volume_weighted_price'] = (df['volume'] * df['close']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Order flow imbalance proxy
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['money_flow_multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        df['money_flow_volume'] = df['money_flow_multiplier'] * df['volume']
        df['money_flow_index'] = df['money_flow_volume'].rolling(14).sum() / df['volume'].rolling(14).sum()
        
        # Accumulation/Distribution Line
        df['acc_dist_line'] = (df['money_flow_multiplier'] * df['volume']).cumsum()
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with time features added
        """
        df = df.copy()
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Market session indicators (assuming UTC timestamps)
        # US session: 13:30-20:00 UTC
        # EU session: 07:00-16:00 UTC  
        # Asia session: 23:00-08:00 UTC
        df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 20)).astype(int)
        df['eu_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
        df['asia_session'] = ((df['hour'] >= 23) | (df['hour'] < 8)).astype(int)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Add lagged features for specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features added
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
                
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
        """
        Add rolling statistical features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for window in windows:
                df[f"{col}_mean_{window}"] = df[col].rolling(window).mean()
                df[f"{col}_std_{window}"] = df[col].rolling(window).std()
                df[f"{col}_min_{window}"] = df[col].rolling(window).min()
                df[f"{col}_max_{window}"] = df[col].rolling(window).max()
                df[f"{col}_median_{window}"] = df[col].rolling(window).median()
                df[f"{col}_skew_{window}"] = df[col].rolling(window).skew()
                df[f"{col}_kurt_{window}"] = df[col].rolling(window).kurt()
                
                # Rank-based features
                df[f"{col}_rank_{window}"] = df[col].rolling(window).rank(pct=True)
                
        return df
    
    def create_target_variables(
        self, 
        df: pd.DataFrame, 
        horizons: List[int] = [1, 4, 12, 24]
    ) -> pd.DataFrame:
        """
        Create target variables for different prediction horizons.
        
        Args:
            df: OHLCV DataFrame
            horizons: List of forward-looking periods
            
        Returns:
            DataFrame with target variables added
        """
        df = df.copy()
        
        for h in horizons:
            # Future returns
            df[f'target_return_{h}h'] = df['close'].shift(-h).pct_change(fill_method=None)
            
            # Future volatility
            future_returns = df['returns'].shift(-h).rolling(h).std()
            df[f'target_volatility_{h}h'] = future_returns
            
            # Direction prediction
            df[f'target_direction_{h}h'] = (df[f'target_return_{h}h'] > 0).astype(int)
            
            # Quintile labels for regression
            df[f'target_quintile_{h}h'] = pd.qcut(
                df[f'target_return_{h}h'], 
                q=5, 
                labels=[0, 1, 2, 3, 4],
                duplicates='drop'
            ).astype(float)
            
        return df
    
    def scale_features(
        self, 
        df: pd.DataFrame, 
        feature_columns: List[str],
        method: str = 'robust',
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale features using specified method.
        
        Args:
            df: Input DataFrame
            feature_columns: Columns to scale
            method: Scaling method ('standard', 'robust')
            fit: Whether to fit the scaler (True for training, False for test)
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        if method not in self.scalers:
            if method == 'standard':
                self.scalers[method] = StandardScaler()
            elif method == 'robust':
                self.scalers[method] = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
        
        scaler = self.scalers[method]
        valid_cols = [col for col in feature_columns if col in df.columns]
        
        if fit:
            # Fit on non-NaN data
            valid_data = df[valid_cols].dropna()
            if len(valid_data) > 0:
                scaler.fit(valid_data)
        
        # Transform data
        df[valid_cols] = scaler.transform(df[valid_cols].fillna(0))
        
        return df
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        target_horizons: List[int] = [1, 4, 12, 24],
        lag_periods: List[int] = [1, 2, 3, 5, 10],
        rolling_windows: List[int] = [5, 10, 20, 50],
        scale_features: bool = True
    ) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Raw OHLCV DataFrame
            target_horizons: Prediction horizons for targets
            lag_periods: Lag periods for autoregressive features
            rolling_windows: Windows for rolling statistics
            scale_features: Whether to scale features
            
        Returns:
            Fully processed DataFrame with all features
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Clean data
        df = self.clean_ohlcv_data(df)
        df = self.remove_outliers(df)
        
        # Add basic features
        df = self.add_returns(df)
        df = self.add_technical_indicators(df)
        df = self.add_volatility_features(df)
        df = self.add_microstructure_features(df)
        df = self.add_time_features(df)
        
        # Add lag features for key variables
        key_features = ['returns', 'volume_ratio', 'rsi', 'volatility_20']
        df = self.add_lag_features(df, key_features, lag_periods)
        
        # Add rolling features
        rolling_features = ['returns', 'volume']
        df = self.add_rolling_features(df, rolling_features, rolling_windows)
        
        # Create targets
        df = self.create_target_variables(df, target_horizons)
        
        # Scale features if requested
        if scale_features:
            feature_cols = [col for col in df.columns 
                          if col not in ['timestamp', 'symbol', 'exchange', 'timeframe'] 
                          and not col.startswith('target_')]
            df = self.scale_features(df, feature_cols)
        
        # Remove initial periods with NaN values
        min_periods = self.config['min_periods']
        df = df.iloc[min_periods:].reset_index(drop=True)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        
        return df
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Return feature groups for analysis and selection.
        
        Returns:
            Dictionary mapping group names to feature lists
        """
        return {
            'price': ['open', 'high', 'low', 'close'],
            'returns': [col for col in ['returns', 'returns_1h', 'returns_4h', 'returns_24h', 
                                      'log_returns', 'intraday_returns', 'gap_returns']],
            'technical': [col for col in ['rsi', 'macd', 'macd_signal', 'macd_histogram',
                                        'bb_position', 'stoch_k', 'stoch_d', 'williams_r', 'cci', 'adx']],
            'volatility': [col for col in ['volatility_5', 'volatility_10', 'volatility_20', 'volatility_50',
                                         'gk_volatility', 'parkinson_volatility', 'rs_volatility', 'atr']],
            'volume': [col for col in ['volume', 'volume_ratio', 'volume_weighted_price', 
                                     'money_flow_index', 'acc_dist_line']],
            'microstructure': [col for col in ['spread_proxy', 'tick_direction', 'price_volume_trend']],
            'time': [col for col in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                                   'us_session', 'eu_session', 'asia_session', 'is_weekend']]
        }
