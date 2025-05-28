"""
Data validation and preprocessing utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
from datetime import datetime, timedelta


def validate_dataframe(df: pd.DataFrame, required_columns: List[str],
                      min_rows: int = 100) -> Dict[str, Any]:
    """
    Validate a dataframe for backtesting requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        Validation results dictionary
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    # Check if DataFrame is empty
    if df.empty:
        results['valid'] = False
        results['errors'].append("DataFrame is empty")
        return results
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        results['valid'] = False
        results['errors'].append(f"Missing required columns: {missing_cols}")
    
    # Check minimum rows
    if len(df) < min_rows:
        results['warnings'].append(f"DataFrame has only {len(df)} rows, minimum recommended: {min_rows}")
    
    # Check for missing values
    missing_data = df[required_columns].isnull().sum()
    if missing_data.any():
        results['warnings'].append(f"Missing values found: {missing_data[missing_data > 0].to_dict()}")
    
    # Check for duplicate indices
    if df.index.duplicated().any():
        results['warnings'].append("Duplicate indices found")
    
    # Check data types for OHLCV columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            results['errors'].append(f"Column '{col}' is not numeric")
            results['valid'] = False
    
    # Check OHLC relationships
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        if invalid_ohlc.any():
            results['warnings'].append(f"Invalid OHLC relationships in {invalid_ohlc.sum()} rows")
    
    # Store basic info
    results['info'] = {
        'rows': len(df),
        'columns': list(df.columns),
        'date_range': (df.index.min(), df.index.max()) if hasattr(df.index, 'min') else None,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    return results


def clean_ohlcv_data(df: pd.DataFrame, remove_outliers: bool = True,
                     outlier_threshold: float = 3.0) -> pd.DataFrame:
    """
    Clean OHLCV data by handling missing values and outliers.
    
    Args:
        df: Input DataFrame with OHLCV data
        remove_outliers: Whether to remove statistical outliers
        outlier_threshold: Z-score threshold for outlier detection
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Ensure index is datetime
    if not isinstance(df_clean.index, pd.DatetimeIndex):
        try:
            df_clean.index = pd.to_datetime(df_clean.index)
        except:
            warnings.warn("Could not convert index to datetime")
    
    # Sort by index
    df_clean = df_clean.sort_index()
    
    # Remove rows where all OHLCV values are NaN
    ohlcv_cols = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col in df_clean.columns]
    df_clean = df_clean.dropna(subset=ohlcv_cols, how='all')
    
    # Forward fill missing values (conservative approach)
    for col in ohlcv_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(method='ffill')
    
    # Remove remaining NaN rows
    df_clean = df_clean.dropna(subset=ohlcv_cols)
    
    # Fix OHLC inconsistencies
    if all(col in df_clean.columns for col in ['open', 'high', 'low', 'close']):
        # Ensure high is the maximum
        df_clean['high'] = df_clean[['open', 'high', 'low', 'close']].max(axis=1)
        # Ensure low is the minimum
        df_clean['low'] = df_clean[['open', 'high', 'low', 'close']].min(axis=1)
    
    # Remove outliers if requested
    if remove_outliers:
        df_clean = remove_price_outliers(df_clean, threshold=outlier_threshold)
    
    # Ensure volume is non-negative
    if 'volume' in df_clean.columns:
        df_clean['volume'] = df_clean['volume'].abs()
    
    return df_clean


def remove_price_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove statistical outliers from price data.
    
    Args:
        df: DataFrame with price data
        threshold: Z-score threshold for outlier detection
        
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    # Calculate returns for outlier detection
    if 'close' in df_clean.columns:
        returns = df_clean['close'].pct_change().dropna()
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        
        # Mark outliers
        outlier_mask = z_scores > threshold
        outlier_indices = returns[outlier_mask].index
        
        # Remove outlier rows
        df_clean = df_clean.drop(outlier_indices)
        
        if len(outlier_indices) > 0:
            warnings.warn(f"Removed {len(outlier_indices)} outlier observations")
    
    return df_clean


def resample_data(df: pd.DataFrame, freq: str, agg_dict: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Resample OHLCV data to different frequency.
    
    Args:
        df: Input DataFrame
        freq: Target frequency ('1H', '4H', '1D', etc.)
        agg_dict: Custom aggregation dictionary
        
    Returns:
        Resampled DataFrame
    """
    if agg_dict is None:
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
    
    # Only use columns that exist in the DataFrame
    available_agg = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    resampled = df.resample(freq).agg(available_agg)
    
    # Remove rows with NaN values (incomplete periods)
    resampled = resampled.dropna()
    
    return resampled


def fill_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Fill missing values in the DataFrame.
    
    Args:
        df: Input DataFrame
        method: Filling method ('ffill', 'bfill', 'interpolate')
        
    Returns:
        DataFrame with filled values
    """
    df_filled = df.copy()
    
    if method == 'ffill':
        df_filled = df_filled.fillna(method='ffill')
    elif method == 'bfill':
        df_filled = df_filled.fillna(method='bfill')
    elif method == 'interpolate':
        df_filled = df_filled.interpolate(method='linear')
    else:
        raise ValueError(f"Unknown filling method: {method}")
    
    return df_filled


def detect_outliers(series: pd.Series, method: str = 'zscore', 
                   threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers in a time series.
    
    Args:
        series: Input time series
        method: Detection method ('zscore', 'iqr', 'isolation')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean series indicating outliers
    """
    if method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    elif method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == 'isolation':
        try:
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=0.1, random_state=42)
            outliers = clf.fit_predict(series.values.reshape(-1, 1))
            return pd.Series(outliers == -1, index=series.index)
        except ImportError:
            warnings.warn("scikit-learn not available, falling back to z-score method")
            return detect_outliers(series, method='zscore', threshold=threshold)
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def calculate_data_quality_score(df: pd.DataFrame) -> float:
    """
    Calculate a data quality score for the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Quality score between 0 and 1
    """
    if df.empty:
        return 0.0
    
    score = 1.0
    
    # Penalize missing values
    missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    score -= missing_ratio * 0.3
    
    # Check for duplicate rows
    duplicate_ratio = df.duplicated().sum() / len(df)
    score -= duplicate_ratio * 0.2
    
    # Check temporal consistency (if datetime index)
    if isinstance(df.index, pd.DatetimeIndex):
        # Check for gaps in the time series
        expected_freq = pd.infer_freq(df.index[:50])  # Infer from first 50 observations
        if expected_freq:
            expected_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=expected_freq)
            missing_timestamps = len(expected_index) - len(df)
            gap_ratio = missing_timestamps / len(expected_index)
            score -= gap_ratio * 0.2
    
    # Check for constant values (lack of variation)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].nunique() == 1:  # All values are the same
            score -= 0.1 / len(numeric_cols)
    
    return max(0.0, min(1.0, score))


def validate_time_series_properties(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate time series properties of the data.
    
    Args:
        df: Input DataFrame with datetime index
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'is_datetime_index': isinstance(df.index, pd.DatetimeIndex),
        'is_sorted': False,
        'has_gaps': False,
        'frequency': None,
        'date_range': None,
        'total_days': None
    }
    
    if not results['is_datetime_index']:
        return results
    
    # Check if sorted
    results['is_sorted'] = df.index.is_monotonic_increasing
    
    # Get date range and total days
    results['date_range'] = (df.index.min(), df.index.max())
    results['total_days'] = (df.index.max() - df.index.min()).days
    
    # Try to infer frequency
    try:
        results['frequency'] = pd.infer_freq(df.index)
    except:
        results['frequency'] = None
    
    # Check for gaps
    if results['frequency']:
        try:
            expected_index = pd.date_range(
                start=df.index[0], 
                end=df.index[-1], 
                freq=results['frequency']
            )
            results['has_gaps'] = len(expected_index) != len(df)
        except:
            results['has_gaps'] = None
    
    return results
