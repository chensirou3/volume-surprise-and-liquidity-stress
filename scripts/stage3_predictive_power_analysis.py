"""
Stage 3: Factor vs Future Volatility/Tail Risk Analysis

Analyzes:
- Factor's relationship with future realized volatility
- Factor's ability to predict tail events
- Forward-looking predictive power
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
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'stage3_predictive_power'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
FORWARD_WINDOWS = [1, 3, 5, 10, 20]  # Forward-looking windows
FACTOR_QUANTILES = [0.8, 0.9, 0.95, 0.99]  # High factor value thresholds
TAIL_THRESHOLDS = [0.95, 0.99]  # Tail event thresholds


def compute_forward_volatility(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Compute forward realized volatility"""
    result_df = df.copy()
    
    for window in windows:
        # Forward returns
        fwd_returns = df['close'].pct_change().shift(-window)
        # Rolling forward volatility
        fwd_vol = df['close'].pct_change().rolling(window).std().shift(-window)
        
        result_df[f'fwd_ret_{window}'] = fwd_returns
        result_df[f'fwd_vol_{window}'] = fwd_vol
        
        # Forward absolute return (proxy for volatility)
        result_df[f'fwd_abs_ret_{window}'] = fwd_returns.abs()
    
    return result_df


def analyze_factor_vs_volatility(df: pd.DataFrame, windows: List[int]) -> Dict:
    """Analyze correlation between factor and future volatility"""
    results = {}
    
    for window in windows:
        vol_col = f'fwd_vol_{window}'
        abs_ret_col = f'fwd_abs_ret_{window}'
        
        if vol_col in df.columns and 'VolLiqScore' in df.columns:
            # Correlation with forward volatility
            valid_data = df[['VolLiqScore', vol_col]].dropna()
            if len(valid_data) > 10:
                corr_vol = valid_data['VolLiqScore'].corr(valid_data[vol_col])
                results[f'corr_fwd_vol_{window}'] = corr_vol
            else:
                results[f'corr_fwd_vol_{window}'] = np.nan
        
        if abs_ret_col in df.columns and 'VolLiqScore' in df.columns:
            # Correlation with forward absolute return
            valid_data = df[['VolLiqScore', abs_ret_col]].dropna()
            if len(valid_data) > 10:
                corr_abs = valid_data['VolLiqScore'].corr(valid_data[abs_ret_col])
                results[f'corr_fwd_abs_ret_{window}'] = corr_abs
            else:
                results[f'corr_fwd_abs_ret_{window}'] = np.nan
    
    return results


def analyze_tail_prediction(df: pd.DataFrame, 
                            factor_quantiles: List[float],
                            tail_thresholds: List[float],
                            windows: List[int]) -> Dict:
    """Analyze factor's ability to predict tail events"""
    results = {}
    
    for window in windows:
        abs_ret_col = f'fwd_abs_ret_{window}'
        
        if abs_ret_col not in df.columns or 'VolLiqScore' not in df.columns:
            continue
        
        valid_data = df[['VolLiqScore', abs_ret_col]].dropna()
        
        if len(valid_data) < 100:
            continue
        
        # Define tail events
        for tail_pct in tail_thresholds:
            tail_threshold = valid_data[abs_ret_col].quantile(tail_pct)
            is_tail_event = valid_data[abs_ret_col] > tail_threshold
            
            # Check if high factor values predict tail events
            for factor_q in factor_quantiles:
                factor_threshold = valid_data['VolLiqScore'].quantile(factor_q)
                is_high_factor = valid_data['VolLiqScore'] > factor_threshold
                
                # Conditional probability: P(tail | high_factor)
                if is_high_factor.sum() > 0:
                    prob_tail_given_high = (is_tail_event & is_high_factor).sum() / is_high_factor.sum()
                else:
                    prob_tail_given_high = np.nan
                
                # Baseline probability: P(tail)
                prob_tail = is_tail_event.sum() / len(valid_data)
                
                # Lift: P(tail | high_factor) / P(tail)
                lift = prob_tail_given_high / prob_tail if prob_tail > 0 else np.nan
                
                key = f'tail_p{int(tail_pct*100)}_factor_q{int(factor_q*100)}_window_{window}'
                results[f'{key}_prob'] = prob_tail_given_high
                results[f'{key}_lift'] = lift
    
    return results


def analyze_single_file(file_path: Path) -> Dict:
    """Analyze predictive power for a single file"""
    try:
        logger.info(f"Analyzing {file_path.name}...")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Check if VolLiqScore exists
        if 'VolLiqScore' not in df.columns:
            logger.warning(f"Skipping {file_path.name}: no VolLiqScore column")
            return {'file': file_path.name, 'status': 'skipped', 'reason': 'no VolLiqScore'}
        
        if 'close' not in df.columns:
            logger.warning(f"Skipping {file_path.name}: no close column")
            return {'file': file_path.name, 'status': 'skipped', 'reason': 'no close'}
        
        # Compute forward volatility
        df = compute_forward_volatility(df, FORWARD_WINDOWS)
        
        # Analyze factor vs volatility
        vol_results = analyze_factor_vs_volatility(df, FORWARD_WINDOWS)
        
        # Analyze tail prediction
        tail_results = analyze_tail_prediction(df, FACTOR_QUANTILES, TAIL_THRESHOLDS, FORWARD_WINDOWS)
        
        # Combine results
        result = {
            'file': file_path.name,
            'status': 'success',
            'n_obs': len(df),
            **vol_results,
            **tail_results
        }
        
        logger.info(f"  Success: {len(df)} observations")
        if 'corr_fwd_vol_5' in vol_results:
            logger.info(f"  Corr(Factor, FwdVol_5): {vol_results['corr_fwd_vol_5']:.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing {file_path.name}: {e}")
        return {'file': file_path.name, 'status': 'error', 'reason': str(e)}


def main():
    """Main analysis loop"""
    
    logger.info("=" * 80)
    logger.info("Stage 3: Predictive Power Analysis")
    logger.info("=" * 80)
    
    # Find all intermediate files
    all_files = sorted(DATA_DIR.glob('*_bars_with_ofi.csv'))
    
    logger.info(f"Found {len(all_files)} files to analyze")
    logger.info(f"Forward windows: {FORWARD_WINDOWS}")
    logger.info(f"Factor quantiles: {FACTOR_QUANTILES}")
    logger.info(f"Tail thresholds: {TAIL_THRESHOLDS}")
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
    output_path = OUTPUT_DIR / 'predictive_power_analysis.csv'
    results_df.to_csv(output_path, index=False)
    
    # Generate summary statistics
    success_df = results_df[results_df['status'] == 'success'].copy()
    
    if len(success_df) > 0:
        # Summary by timeframe
        def extract_timeframe(filename):
            parts = filename.split('_')
            return parts[1] if len(parts) > 1 else 'unknown'
        
        success_df['timeframe'] = success_df['file'].apply(extract_timeframe)
        
        # Select key metrics for summary
        corr_cols = [c for c in success_df.columns if c.startswith('corr_fwd_')]
        if corr_cols:
            summary_by_tf = success_df.groupby('timeframe')[corr_cols].agg(['mean', 'std', 'min', 'max']).round(4)
            summary_path = OUTPUT_DIR / 'summary_by_timeframe.csv'
            summary_by_tf.to_csv(summary_path)
            
            logger.info("=" * 80)
            logger.info("Summary by Timeframe (Correlations)")
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
    if len(success_df) > 0 and corr_cols:
        logger.info(f"Saved summary to: {summary_path}")
    logger.info("=" * 80)
    logger.info("Stage 3 Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
