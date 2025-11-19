"""
Volume Surprise / Liquidity Stress Factor (Factor 3)

This module implements the volume surprise and liquidity stress factors
as defined in the project requirements.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_log_volume_zscore(
    df: pd.DataFrame,
    volume_col: str = 'volume',
    lookback: int = 50,
    eps: float = 1e-10
) -> pd.Series:
    """
    Compute z-score of log(volume).
    
    Args:
        df: DataFrame with volume data
        volume_col: Name of volume column
        lookback: Rolling window for mean/std calculation
        eps: Small constant to avoid log(0)
    
    Returns:
        Series with z_vol values
    """
    # Ensure volume is positive
    vol = df[volume_col].copy()
    vol = vol.clip(lower=eps)
    
    # Log transform
    log_vol = np.log(vol)
    
    # Rolling mean and std
    log_vol_mean = log_vol.rolling(window=lookback, min_periods=lookback//2).mean()
    log_vol_std = log_vol.rolling(window=lookback, min_periods=lookback//2).std()
    
    # Z-score
    z_vol = (log_vol - log_vol_mean) / (log_vol_std + eps)
    
    return z_vol


def compute_true_range(
    df: pd.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close'
) -> pd.Series:
    """
    Compute True Range (TR).
    
    TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    
    Args:
        df: DataFrame with OHLC data
        high_col: Name of high column
        low_col: Name of low column
        close_col: Name of close column
    
    Returns:
        Series with TR values
    """
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr


def compute_atr(
    df: pd.DataFrame,
    lookback: int = 50,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close'
) -> pd.Series:
    """
    Compute Average True Range (ATR).
    
    Args:
        df: DataFrame with OHLC data
        lookback: Rolling window for ATR calculation
        high_col: Name of high column
        low_col: Name of low column
        close_col: Name of close column
    
    Returns:
        Series with ATR values
    """
    tr = compute_true_range(df, high_col, low_col, close_col)
    atr = tr.rolling(window=lookback, min_periods=lookback//2).mean()
    
    return atr


def compute_liq_stress(
    df: pd.DataFrame,
    atr: Optional[pd.Series] = None,
    lookback_atr: int = 50,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    eps: float = 1e-10
) -> pd.Series:
    """
    Compute liquidity stress as range / ATR.
    
    Args:
        df: DataFrame with OHLC data
        atr: Pre-computed ATR (if None, will compute)
        lookback_atr: Lookback for ATR if not provided
        high_col: Name of high column
        low_col: Name of low column
        close_col: Name of close column
        eps: Small constant to avoid division by zero
    
    Returns:
        Series with liq_stress values
    """
    if atr is None:
        atr = compute_atr(df, lookback_atr, high_col, low_col, close_col)
    
    range_val = df[high_col] - df[low_col]
    liq_stress = range_val / (atr + eps)
    
    return liq_stress


def compute_liq_stress_zscore(
    liq_stress: pd.Series,
    lookback: int = 50,
    eps: float = 1e-10
) -> pd.Series:
    """
    Compute z-score of liquidity stress.
    
    Args:
        liq_stress: Series with liq_stress values
        lookback: Rolling window for mean/std calculation
        eps: Small constant to avoid division by zero
    
    Returns:
        Series with z_liq_stress values
    """
    liq_mean = liq_stress.rolling(window=lookback, min_periods=lookback//2).mean()
    liq_std = liq_stress.rolling(window=lookback, min_periods=lookback//2).std()
    
    z_liq_stress = (liq_stress - liq_mean) / (liq_std + eps)
    
    return z_liq_stress


def add_vol_liq_factors(
    df: pd.DataFrame,
    lookback_vol: int = 50,
    lookback_atr: int = 50,
    lookback_liq_z: int = 50,
    weight_vol: float = 0.5,
    weight_liq: float = 0.5,
    volume_col: str = 'volume',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Add all volume/liquidity factors to DataFrame.
    
    This is the main entry point for adding Factor 3 to a dataset.
    
    Args:
        df: DataFrame with OHLCV data
        lookback_vol: Lookback window for volume z-score
        lookback_atr: Lookback window for ATR
        lookback_liq_z: Lookback window for liq_stress z-score
        weight_vol: Weight for z_vol in combined score
        weight_liq: Weight for z_liq_stress in combined score
        volume_col: Name of volume column
        high_col: Name of high column
        low_col: Name of low column
        close_col: Name of close column
        inplace: Whether to modify df in place
    
    Returns:
        DataFrame with added columns:
        - z_vol: Volume surprise (z-score of log volume)
        - TR: True Range
        - ATR: Average True Range
        - liq_stress: Range / ATR
        - z_liq_stress: Z-score of liq_stress
        - VolLiqScore: Combined factor (weighted sum of z_vol and z_liq_stress)
    """
    if not inplace:
        df = df.copy()
    
    # Check required columns
    required_cols = [volume_col, high_col, low_col, close_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Computing volume/liquidity factors with lookback_vol={lookback_vol}, "
                f"lookback_atr={lookback_atr}, lookback_liq_z={lookback_liq_z}")
    
    # 1. Volume surprise
    df['z_vol'] = compute_log_volume_zscore(df, volume_col, lookback_vol)
    
    # 2. True Range and ATR
    df['TR'] = compute_true_range(df, high_col, low_col, close_col)
    df['ATR'] = compute_atr(df, lookback_atr, high_col, low_col, close_col)
    
    # 3. Liquidity stress
    df['liq_stress'] = compute_liq_stress(df, df['ATR'], lookback_atr, 
                                          high_col, low_col, close_col)
    df['z_liq_stress'] = compute_liq_stress_zscore(df['liq_stress'], lookback_liq_z)
    
    # 4. Combined score
    df['VolLiqScore'] = weight_vol * df['z_vol'] + weight_liq * df['z_liq_stress']
    
    logger.info(f"Added {len([c for c in df.columns if c in ['z_vol', 'TR', 'ATR', 'liq_stress', 'z_liq_stress', 'VolLiqScore']])} factor columns")
    
    return df


def add_forward_behavior_indicators(
    df: pd.DataFrame,
    horizons: list = [1, 2, 4],
    k_tail: float = 2.0,
    close_col: str = 'close',
    atr_col: str = 'ATR',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Add forward-looking behavior indicators for analysis.
    
    Args:
        df: DataFrame with price and ATR data
        horizons: List of forward horizons (in bars)
        k_tail: Threshold multiplier for tail events (e.g., 2.0 means 2*ATR)
        close_col: Name of close price column
        atr_col: Name of ATR column
        inplace: Whether to modify df in place
    
    Returns:
        DataFrame with added columns for each horizon H:
        - ret_fwd_H: Forward H-bar log return
        - move_abs_H: Absolute value of ret_fwd_H
        - tail_flag_H: Binary flag for |ret_fwd_H| > k * ATR
    """
    if not inplace:
        df = df.copy()
    
    close = df[close_col]
    atr = df[atr_col]
    
    for H in horizons:
        # Forward return
        ret_fwd = np.log(close.shift(-H) / close)
        df[f'ret_fwd_{H}'] = ret_fwd
        
        # Absolute move
        df[f'move_abs_{H}'] = ret_fwd.abs()
        
        # Tail flag
        df[f'tail_flag_{H}'] = (ret_fwd.abs() > k_tail * atr).astype(int)
    
    return df


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic data
    np.random.seed(42)
    n = 1000
    test_df = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=n, freq='4H'),
        'open': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(n) * 0.5) + np.abs(np.random.randn(n)),
        'low': 100 + np.cumsum(np.random.randn(n) * 0.5) - np.abs(np.random.randn(n)),
        'close': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'volume': np.abs(np.random.randn(n) * 1000 + 5000)
    })
    
    # Add factors
    test_df = add_vol_liq_factors(test_df)
    
    print("\nSample output:")
    print(test_df[['timestamp', 'close', 'volume', 'z_vol', 'liq_stress', 'z_liq_stress', 'VolLiqScore']].tail(10))
    
    print("\nFactor statistics:")
    print(test_df[['z_vol', 'liq_stress', 'z_liq_stress', 'VolLiqScore']].describe())
