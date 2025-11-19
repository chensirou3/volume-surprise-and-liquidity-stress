"""
Stage 1b: Add Volume/Liquidity Factors to Short Timeframe Datasets (5min, 15min, 30min)

This script processes the shorter timeframes that were not included in Stage 1.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from joint_factors.vol_liq_factor import add_vol_liq_factors

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / 'data' / 'bars_with_ofi'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'intermediate'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target symbols and timeframes
TARGET_SYMBOLS = ['BTCUSD', 'XAUUSD', 'EURUSD', 'XAGUSD', 'ETHUSD', 'USDJPY']
TARGET_TIMEFRAMES = ['5min', '15min', '30min']  # Short timeframes

# Factor parameters
LOOKBACK_VOL = 50
LOOKBACK_ATR = 50
LOOKBACK_LIQ_Z = 50


def process_single_file(file_path: Path) -> dict:
    """
    Process a single bars_with_ofi file.
    
    Returns:
        dict with processing stats
    """
    try:
        # Load data
        logger.info(f"Loading {file_path.name}...")
        df = pd.read_csv(file_path)
        
        original_rows = len(df)
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Skipping {file_path.name}: missing columns {missing_cols}")
            return {
                'file': file_path.name,
                'status': 'skipped',
                'reason': f'missing columns: {missing_cols}'
            }
        
        # Add factors
        logger.info(f"Computing factors for {file_path.name}...")
        df = add_vol_liq_factors(
            df,
            lookback_vol=LOOKBACK_VOL,
            lookback_atr=LOOKBACK_ATR,
            lookback_liq_z=LOOKBACK_LIQ_Z,
            inplace=False
        )
        
        # Save to intermediate
        output_path = OUTPUT_DIR / file_path.name
        logger.info(f"Saving to {output_path.name}...")
        df.to_csv(output_path, index=False)
        
        # Count valid factor rows (non-NaN VolLiqScore)
        valid_factor_rows = df['VolLiqScore'].notna().sum()
        
        logger.info(f"  Status: success")
        logger.info(f"  Valid factor rows: {valid_factor_rows}/{len(df)}")
        logger.info("")
        
        return {
            'file': file_path.name,
            'status': 'success',
            'original_rows': original_rows,
            'output_rows': len(df),
            'valid_factor_rows': valid_factor_rows,
            'output_path': str(output_path)
        }
        
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")
        return {
            'file': file_path.name,
            'status': 'error',
            'reason': str(e)
        }


def main():
    """Main processing loop"""
    
    # Find all target files
    all_files = sorted(DATA_DIR.glob('*_bars_with_ofi.csv'))
    
    # Filter for short timeframes and target symbols
    target_files = []
    for f in all_files:
        # Check if any target timeframe is in filename
        has_target_tf = any(tf in f.name for tf in TARGET_TIMEFRAMES)
        # Check if any target symbol is in filename
        has_target_symbol = any(sym in f.name for sym in TARGET_SYMBOLS)
        
        if has_target_tf and has_target_symbol:
            target_files.append(f)
    
    logger.info("=" * 80)
    logger.info(f"Stage 1b: Processing Short Timeframes")
    logger.info("=" * 80)
    logger.info(f"Found {len(target_files)} files to process")
    logger.info(f"Timeframes: {TARGET_TIMEFRAMES}")
    logger.info(f"Symbols: {TARGET_SYMBOLS}")
    logger.info("=" * 80)
    logger.info("")
    
    # Process each file
    results = []
    for i, file_path in enumerate(target_files, 1):
        logger.info(f"[{i}/{len(target_files)}] Processing {file_path.name}")
        result = process_single_file(file_path)
        results.append(result)
    
    # Save summary
    summary_df = pd.DataFrame(results)
    summary_path = PROJECT_ROOT / 'results' / 'stats' / 'stage1b_short_timeframes_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    # Print summary
    logger.info("=" * 80)
    logger.info("Processing Summary")
    logger.info("=" * 80)
    logger.info(f"Total files: {len(results)}")
    logger.info(f"Success: {sum(1 for r in results if r['status'] == 'success')}")
    logger.info(f"Errors: {sum(1 for r in results if r['status'] == 'error')}")
    logger.info(f"Skipped: {sum(1 for r in results if r['status'] == 'skipped')}")
    logger.info("")
    logger.info(f"Saved processing summary to: {summary_path}")
    logger.info("=" * 80)
    logger.info("Stage 1b Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
