"""
Master script to run all analysis stages (2-4)

Executes:
- Stage 2: Time Structure & Frequency Analysis
- Stage 3: Predictive Power Analysis  
- Stage 4: Cross-Symbol/Cross-Timeframe Robustness
"""

import sys
from pathlib import Path
import subprocess
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'


def run_stage(stage_name: str, script_path: Path) -> bool:
    """Run a single stage script"""
    logger.info("=" * 80)
    logger.info(f"Starting {stage_name}")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            ['python3', str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result.returncode == 0:
            logger.info(f"✓ {stage_name} completed successfully in {duration:.1f}s")
            return True
        else:
            logger.error(f"✗ {stage_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"✗ {stage_name} failed with exception: {e}")
        return False


def main():
    """Run all stages"""
    
    logger.info("=" * 80)
    logger.info("RUNNING ALL ANALYSIS STAGES (2-4)")
    logger.info("=" * 80)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Start time: {datetime.now()}")
    logger.info("=" * 80)
    logger.info("")
    
    overall_start = datetime.now()
    
    stages = [
        ("Stage 2: Time Structure Analysis", SCRIPTS_DIR / 'stage2_time_structure_analysis.py'),
        ("Stage 3: Predictive Power Analysis", SCRIPTS_DIR / 'stage3_predictive_power_analysis.py'),
        ("Stage 4: Robustness Analysis", SCRIPTS_DIR / 'stage4_robustness_analysis.py'),
    ]
    
    results = {}
    
    for stage_name, script_path in stages:
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            results[stage_name] = False
            continue
        
        success = run_stage(stage_name, script_path)
        results[stage_name] = success
        logger.info("")
    
    overall_end = datetime.now()
    overall_duration = (overall_end - overall_start).total_seconds()
    
    # Print final summary
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total duration: {overall_duration:.1f}s ({overall_duration/60:.1f} minutes)")
    logger.info("")
    
    for stage_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{status}: {stage_name}")
    
    logger.info("")
    logger.info("=" * 80)
    
    all_success = all(results.values())
    if all_success:
        logger.info("🎉 ALL STAGES COMPLETED SUCCESSFULLY!")
    else:
        logger.info("⚠️  SOME STAGES FAILED - CHECK LOGS ABOVE")
    
    logger.info("=" * 80)
    
    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())
