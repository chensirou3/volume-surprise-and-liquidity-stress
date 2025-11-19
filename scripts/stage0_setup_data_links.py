"""
Stage 0: Setup data links and inspect data availability
This script creates symlinks to existing data from Order-Flow-Imbalance-analysis project
and generates a summary of available data.
"""

import os
import pandas as pd
from pathlib import Path
import glob

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
OFI_PROJECT_ROOT = PROJECT_ROOT.parent / "Order-Flow-Imbalance-analysis"
OFI_RESULTS_DIR = OFI_PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "bars_with_ofi"
RESULTS_DIR = PROJECT_ROOT / "results" / "stats"

# Target symbols and timeframes for this research
TARGET_SYMBOLS = ['BTCUSD', 'XAUUSD', 'EURUSD', 'XAGUSD', 'ETHUSD', 'USDJPY']
TARGET_TIMEFRAMES = ['1H', '2H', '4H', '8H']

def main():
    print("=" * 80)
    print("Stage 0: Data Setup and Availability Check")
    print("=" * 80)
    
    # Create necessary directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all bars_with_ofi files
    ofi_files = list(OFI_RESULTS_DIR.glob("*_bars_with_ofi.csv"))
    print(f"\nFound {len(ofi_files)} bars_with_ofi files in OFI project")
    
    # Create symlinks for relevant files
    linked_files = []
    for ofi_file in ofi_files:
        # Parse filename to extract symbol and timeframe
        filename = ofi_file.name
        parts = filename.replace("_bars_with_ofi.csv", "").split("_")
        
        # Extract symbol (first part before timeframe)
        symbol = parts[0]
        
        # Check if this is a target symbol
        if symbol in TARGET_SYMBOLS:
            # Create symlink
            link_path = DATA_DIR / filename
            if not link_path.exists():
                try:
                    link_path.symlink_to(ofi_file)
                    linked_files.append((symbol, filename, ofi_file))
                    print(f"  Linked: {filename}")
                except Exception as e:
                    print(f"  Error linking {filename}: {e}")
    
    print(f"\nCreated {len(linked_files)} symlinks")
    
    # Inspect data availability
    print("\n" + "=" * 80)
    print("Inspecting Data Availability")
    print("=" * 80)
    
    availability_records = []
    
    for link_file in DATA_DIR.glob("*.csv"):
        try:
            # Read first few rows to check columns
            df = pd.read_csv(link_file, nrows=10)
            
            # Parse filename
            filename = link_file.name
            parts = filename.replace("_bars_with_ofi.csv", "").split("_")
            symbol = parts[0]
            
            # Try to identify timeframe
            timeframe = "unknown"
            for part in parts[1:]:
                if any(tf in part.upper() for tf in ['H', 'MIN', 'D']):
                    timeframe = part
                    break
            
            # Check for required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            has_required = all(col in df.columns for col in required_cols)
            
            # Check for OFI columns
            has_ofi = 'OFI_z' in df.columns or 'OFI_raw' in df.columns
            
            # Check for forward returns
            fwd_ret_cols = [col for col in df.columns if 'fut_ret' in col or 'ret_fwd' in col]
            
            # Get full row count
            df_full = pd.read_csv(link_file)
            n_rows = len(df_full)
            
            # Get date range
            df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
            start_date = df_full['timestamp'].min()
            end_date = df_full['timestamp'].max()
            
            availability_records.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'filename': filename,
                'n_rows': n_rows,
                'start_date': start_date,
                'end_date': end_date,
                'has_required_cols': has_required,
                'has_ofi': has_ofi,
                'n_fwd_ret_cols': len(fwd_ret_cols),
                'fwd_ret_cols': ','.join(fwd_ret_cols[:5]),  # First 5
                'all_columns': ','.join(df.columns.tolist())
            })
            
            print(f"\n{symbol} {timeframe}:")
            print(f"  Rows: {n_rows:,}")
            print(f"  Date range: {start_date.date()} to {end_date.date()}")
            print(f"  Has OHLCV: {has_required}")
            print(f"  Has OFI: {has_ofi}")
            print(f"  Forward return columns: {len(fwd_ret_cols)}")
            
        except Exception as e:
            print(f"\nError processing {link_file.name}: {e}")
    
    # Save availability summary
    if availability_records:
        summary_df = pd.DataFrame(availability_records)
        summary_path = RESULTS_DIR / "data_availability_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'=' * 80}")
        print(f"Saved availability summary to: {summary_path}")
        print(f"Total datasets available: {len(summary_df)}")
        print(f"\nSymbols: {sorted(summary_df['symbol'].unique())}")
        print(f"Timeframes: {sorted(summary_df['timeframe'].unique())}")
    
    print("\n" + "=" * 80)
    print("Stage 0 Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
