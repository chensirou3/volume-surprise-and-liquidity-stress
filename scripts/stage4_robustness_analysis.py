"""
Stage 4: Cross-Symbol / Cross-Timeframe Robustness Analysis

Analyzes:
- Factor consistency across different symbols
- Factor consistency across different timeframes
- Comparative statistics
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict, List

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / 'data' / 'intermediate'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'stage4_robustness'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target symbols and timeframes
TARGET_SYMBOLS = ['BTCUSD', 'ETHUSD', 'EURUSD', 'USDJPY', 'XAGUSD', 'XAUUSD']
TARGET_TIMEFRAMES = ['5min', '15min', '30min', '1H', '2H', '4H', '8H']


def compute_factor_statistics(df: pd.DataFrame) -> Dict:
    """Compute comprehensive factor statistics"""
    if 'VolLiqScore' not in df.columns:
        return {}
    
    factor = df['VolLiqScore'].dropna()
    
    if len(factor) < 10:
        return {}
    
    stats = {
        'mean': factor.mean(),
        'std': factor.std(),
        'median': factor.median(),
        'min': factor.min(),
        'max': factor.max(),
        'skew': factor.skew(),
        'kurtosis': factor.kurtosis(),
        'p25': factor.quantile(0.25),
        'p75': factor.quantile(0.75),
        'p90': factor.quantile(0.90),
        'p95': factor.quantile(0.95),
        'p99': factor.quantile(0.99),
        'iqr': factor.quantile(0.75) - factor.quantile(0.25),
        'n_obs': len(factor),
        'n_positive': (factor > 0).sum(),
        'n_negative': (factor < 0).sum(),
        'pct_positive': (factor > 0).sum() / len(factor) * 100,
        'pct_extreme_high': (factor > factor.quantile(0.95)).sum() / len(factor) * 100,
        'pct_extreme_low': (factor < factor.quantile(0.05)).sum() / len(factor) * 100,
    }
    
    return stats


def analyze_single_file(file_path: Path) -> Dict:
    """Analyze a single file"""
    try:
        # Extract symbol and timeframe from filename
        filename = file_path.name
        parts = filename.split('_')
        symbol = parts[0] if len(parts) > 0 else 'unknown'
        timeframe = parts[1] if len(parts) > 1 else 'unknown'
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Compute statistics
        stats = compute_factor_statistics(df)
        
        if not stats:
            return {
                'file': filename,
                'symbol': symbol,
                'timeframe': timeframe,
                'status': 'skipped',
                'reason': 'no valid data'
            }
        
        result = {
            'file': filename,
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'success',
            **stats
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing {file_path.name}: {e}")
        return {
            'file': file_path.name,
            'symbol': 'unknown',
            'timeframe': 'unknown',
            'status': 'error',
            'reason': str(e)
        }


def main():
    """Main analysis loop"""
    
    logger.info("=" * 80)
    logger.info("Stage 4: Cross-Symbol / Cross-Timeframe Robustness Analysis")
    logger.info("=" * 80)
    
    # Find all merged files (complete historical data)
    all_files = sorted(DATA_DIR.glob('*_merged_bars_with_ofi.csv'))
    
    logger.info(f"Found {len(all_files)} merged files to analyze")
    logger.info(f"Target symbols: {TARGET_SYMBOLS}")
    logger.info(f"Target timeframes: {TARGET_TIMEFRAMES}")
    logger.info("=" * 80)
    logger.info("")
    
    # Analyze each file
    results = []
    for i, file_path in enumerate(all_files, 1):
        logger.info(f"[{i}/{len(all_files)}] Analyzing {file_path.name}...")
        result = analyze_single_file(file_path)
        results.append(result)
        
        if result['status'] == 'success':
            logger.info(f"  Symbol: {result['symbol']}, Timeframe: {result['timeframe']}")
            logger.info(f"  N: {result['n_obs']}, Mean: {result['mean']:.4f}, Std: {result['std']:.4f}")
        logger.info("")
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / 'robustness_analysis.csv'
    results_df.to_csv(output_path, index=False)
    
    # Generate summary statistics
    success_df = results_df[results_df['status'] == 'success'].copy()
    
    if len(success_df) > 0:
        # Summary by symbol
        logger.info("=" * 80)
        logger.info("Summary by Symbol")
        logger.info("=" * 80)
        
        summary_by_symbol = success_df.groupby('symbol').agg({
            'mean': ['mean', 'std'],
            'std': ['mean', 'std'],
            'skew': ['mean', 'std'],
            'kurtosis': ['mean', 'std'],
            'p95': ['mean', 'std'],
            'n_obs': 'sum'
        }).round(4)
        
        print(summary_by_symbol)
        summary_by_symbol.to_csv(OUTPUT_DIR / 'summary_by_symbol.csv')
        
        # Summary by timeframe
        logger.info("")
        logger.info("=" * 80)
        logger.info("Summary by Timeframe")
        logger.info("=" * 80)
        
        summary_by_tf = success_df.groupby('timeframe').agg({
            'mean': ['mean', 'std'],
            'std': ['mean', 'std'],
            'skew': ['mean', 'std'],
            'kurtosis': ['mean', 'std'],
            'p95': ['mean', 'std'],
            'n_obs': 'sum'
        }).round(4)
        
        # Sort by timeframe order
        tf_order = ['5min', '15min', '30min', '1H', '2H', '4H', '8H']
        summary_by_tf = summary_by_tf.reindex([tf for tf in tf_order if tf in summary_by_tf.index])
        
        print(summary_by_tf)
        summary_by_tf.to_csv(OUTPUT_DIR / 'summary_by_timeframe.csv')
        
        # Cross-tabulation: mean by symbol x timeframe
        logger.info("")
        logger.info("=" * 80)
        logger.info("Factor Mean: Symbol × Timeframe")
        logger.info("=" * 80)
        
        pivot_mean = success_df.pivot_table(
            values='mean',
            index='symbol',
            columns='timeframe',
            aggfunc='mean'
        ).round(4)
        
        # Reorder columns
        pivot_mean = pivot_mean[[tf for tf in tf_order if tf in pivot_mean.columns]]
        
        print(pivot_mean)
        pivot_mean.to_csv(OUTPUT_DIR / 'cross_tab_mean.csv')
        
        # Cross-tabulation: std by symbol x timeframe
        logger.info("")
        logger.info("=" * 80)
        logger.info("Factor Std: Symbol × Timeframe")
        logger.info("=" * 80)
        
        pivot_std = success_df.pivot_table(
            values='std',
            index='symbol',
            columns='timeframe',
            aggfunc='mean'
        ).round(4)
        
        # Reorder columns
        pivot_std = pivot_std[[tf for tf in tf_order if tf in pivot_std.columns]]
        
        print(pivot_std)
        pivot_std.to_csv(OUTPUT_DIR / 'cross_tab_std.csv')
        
        # Coefficient of variation analysis
        logger.info("")
        logger.info("=" * 80)
        logger.info("Coefficient of Variation Analysis")
        logger.info("=" * 80)
        
        # CV across symbols (for each timeframe)
        cv_by_tf = success_df.groupby('timeframe').apply(
            lambda x: x['mean'].std() / x['mean'].mean() if x['mean'].mean() != 0 else np.nan
        ).round(4)
        
        logger.info("CV across symbols (by timeframe):")
        print(cv_by_tf)
        
        # CV across timeframes (for each symbol)
        cv_by_symbol = success_df.groupby('symbol').apply(
            lambda x: x['mean'].std() / x['mean'].mean() if x['mean'].mean() != 0 else np.nan
        ).round(4)
        
        logger.info("\nCV across timeframes (by symbol):")
        print(cv_by_symbol)
        
        # Save CV analysis
        cv_df = pd.DataFrame({
            'cv_across_symbols': cv_by_tf,
            'cv_across_timeframes': cv_by_symbol
        })
        cv_df.to_csv(OUTPUT_DIR / 'coefficient_of_variation.csv')
    
    # Print final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Processing Summary")
    logger.info("=" * 80)
    logger.info(f"Total files: {len(results)}")
    logger.info(f"Success: {sum(1 for r in results if r['status'] == 'success')}")
    logger.info(f"Errors: {sum(1 for r in results if r['status'] == 'error')}")
    logger.info(f"Skipped: {sum(1 for r in results if r['status'] == 'skipped')}")
    logger.info("")
    logger.info(f"Saved detailed results to: {output_path}")
    logger.info(f"Saved summary by symbol to: {OUTPUT_DIR / 'summary_by_symbol.csv'}")
    logger.info(f"Saved summary by timeframe to: {OUTPUT_DIR / 'summary_by_timeframe.csv'}")
    logger.info(f"Saved cross-tabulations to: {OUTPUT_DIR / 'cross_tab_*.csv'}")
    logger.info("=" * 80)
    logger.info("Stage 4 Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
