"""
Stage 2: Time Structure & Frequency Analysis

Analyzes:
- Factor autocorrelation (ACF)
- Factor persistence
- High-value event frequency
- Time series characteristics
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
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'stage2_time_structure'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
ACF_LAGS = [1, 2, 3, 5, 10, 20, 50]
PERCENTILE_THRESHOLDS = [90, 95, 99]


def compute_acf(series: pd.Series, lags: List[int]) -> Dict[int, float]:
    """Compute autocorrelation for specified lags"""
    acf_results = {}
    for lag in lags:
        if len(series) > lag:
            corr = series.autocorr(lag=lag)
            acf_results[lag] = corr if not pd.isna(corr) else 0.0
        else:
            acf_results[lag] = np.nan
    return acf_results


def compute_persistence_metrics(series: pd.Series) -> Dict[str, float]:
    """Compute persistence metrics"""
    # Half-life of autocorrelation
    acf_1 = series.autocorr(lag=1)
    if pd.notna(acf_1) and acf_1 > 0:
        half_life = -np.log(2) / np.log(acf_1) if acf_1 < 1 else np.inf
    else:
        half_life = np.nan
    
    # Mean reversion speed (1 - ACF(1))
    mean_reversion_speed = 1 - acf_1 if pd.notna(acf_1) else np.nan
    
    return {
        'acf_1': acf_1,
        'half_life': half_life,
        'mean_reversion_speed': mean_reversion_speed
    }


def compute_event_frequency(series: pd.Series, thresholds: List[int]) -> Dict[str, float]:
    """Compute frequency of high-value events"""
    results = {}
    valid_series = series.dropna()
    
    for pct in thresholds:
        threshold = np.percentile(valid_series, pct)
        event_count = (valid_series > threshold).sum()
        event_freq = event_count / len(valid_series) * 100
        
        results[f'p{pct}_threshold'] = threshold
        results[f'p{pct}_event_freq_pct'] = event_freq
        results[f'p{pct}_event_count'] = event_count
    
    return results


def analyze_single_file(file_path: Path) -> Dict:
    """Analyze time structure for a single file"""
    try:
        logger.info(f"Analyzing {file_path.name}...")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Check if VolLiqScore exists
        if 'VolLiqScore' not in df.columns:
            logger.warning(f"Skipping {file_path.name}: no VolLiqScore column")
            return {'file': file_path.name, 'status': 'skipped', 'reason': 'no VolLiqScore'}
        
        # Get valid factor values
        factor = df['VolLiqScore'].dropna()
        
        if len(factor) < 100:
            logger.warning(f"Skipping {file_path.name}: insufficient data ({len(factor)} rows)")
            return {'file': file_path.name, 'status': 'skipped', 'reason': 'insufficient data'}
        
        # Compute ACF
        acf_results = compute_acf(factor, ACF_LAGS)
        
        # Compute persistence metrics
        persistence = compute_persistence_metrics(factor)
        
        # Compute event frequency
        events = compute_event_frequency(factor, PERCENTILE_THRESHOLDS)
        
        # Combine results
        result = {
            'file': file_path.name,
            'status': 'success',
            'n_obs': len(factor),
            'n_total': len(df),
            **{f'acf_lag_{lag}': val for lag, val in acf_results.items()},
            **persistence,
            **events
        }
        
        logger.info(f"  Success: {len(factor)} observations")
        logger.info(f"  ACF(1): {persistence['acf_1']:.4f}, Half-life: {persistence['half_life']:.2f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing {file_path.name}: {e}")
        return {'file': file_path.name, 'status': 'error', 'reason': str(e)}


def main():
    """Main analysis loop"""
    
    logger.info("=" * 80)
    logger.info("Stage 2: Time Structure & Frequency Analysis")
    logger.info("=" * 80)
    
    # Find all intermediate files
    all_files = sorted(DATA_DIR.glob('*_bars_with_ofi.csv'))
    
    logger.info(f"Found {len(all_files)} files to analyze")
    logger.info(f"ACF lags: {ACF_LAGS}")
    logger.info(f"Percentile thresholds: {PERCENTILE_THRESHOLDS}")
    logger.info("=" * 80)
    logger.info("")
    
    # Analyze each file
    results = []
    for i, file_path in enumerate(all_files, 1):
        logger.info(f"[{i}/{len(all_files)}] {file_path.name}")
        result = analyze_single_file(file_path)
        results.append(result)
        logger.info("")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / 'time_structure_analysis.csv'
    results_df.to_csv(output_path, index=False)
    
    # Generate summary statistics
    success_df = results_df[results_df['status'] == 'success'].copy()
    
    if len(success_df) > 0:
        # Summary by timeframe
        def extract_timeframe(filename):
            parts = filename.split('_')
            return parts[1] if len(parts) > 1 else 'unknown'
        
        success_df['timeframe'] = success_df['file'].apply(extract_timeframe)
        
        summary_by_tf = success_df.groupby('timeframe').agg({
            'acf_lag_1': ['mean', 'std', 'min', 'max'],
            'half_life': ['mean', 'median', 'std'],
            'mean_reversion_speed': ['mean', 'std'],
            'p95_event_freq_pct': ['mean', 'std']
        }).round(4)
        
        summary_path = OUTPUT_DIR / 'summary_by_timeframe.csv'
        summary_by_tf.to_csv(summary_path)
        
        logger.info("=" * 80)
        logger.info("Summary by Timeframe")
        logger.info("=" * 80)
        print(summary_by_tf)
    
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
    logger.info(f"Saved results to: {output_path}")
    logger.info(f"Saved summary to: {summary_path}")
    logger.info("=" * 80)
    logger.info("Stage 2 Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
