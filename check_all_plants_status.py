#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check completion status of all plant experiments
Scan and display which plants are complete, in progress, or not started
"""

import os
import pandas as pd
import glob
from pathlib import Path


def check_plant_status(plant_id: str, script_dir: str = '.') -> dict:
    """
    Check status of a single plant
    
    Returns:
        Status dictionary
    """
    # Find all result files for this plant
    pattern = os.path.join(script_dir, f"results_{plant_id}_*.csv")
    result_files = glob.glob(pattern)
    
    if not result_files:
        return {
            'plant_id': plant_id,
            'status': 'NOT_STARTED',
            'completed': 0,
            'remaining': 284,
            'progress': 0.0,
            'result_file': None
        }
    
    # Use the latest result file
    result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_file = result_files[0]
    
    try:
        df = pd.read_csv(latest_file)
        
        # Count successful experiments
        if 'status' in df.columns:
            completed = len(df[df['status'] == 'SUCCESS'])
        else:
            completed = len(df[df['experiment_name'].notna()])
        
        progress = completed / 284 * 100
        
        if completed >= 284:
            status = 'COMPLETE'
        elif completed > 0:
            status = 'IN_PROGRESS'
        else:
            status = 'NOT_STARTED'
        
        return {
            'plant_id': plant_id,
            'status': status,
            'completed': completed,
            'remaining': 284 - completed,
            'progress': progress,
            'result_file': os.path.basename(latest_file)
        }
    
    except Exception as e:
        return {
            'plant_id': plant_id,
            'status': 'ERROR',
            'completed': 0,
            'remaining': 284,
            'progress': 0.0,
            'result_file': os.path.basename(latest_file),
            'error': str(e)
        }


def scan_all_plants(output_dir=None):
    """
    Scan status of all plants
    
    Args:
        output_dir: Directory to check for results (default: current directory)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if output_dir is None:
        output_dir = script_dir
    
    print(f"Checking results in: {output_dir}")
    
    print("=" * 100)
    print("All Plants Experiment Status Scan")
    print("=" * 100)
    
    # Find all configuration files
    config_dir = os.path.join(script_dir, 'config', 'plants')
    if not os.path.exists(config_dir):
        print(f"[ERROR] Config directory not found: {config_dir}")
        return
    
    config_files = glob.glob(os.path.join(config_dir, 'Plant*.yaml'))
    
    if not config_files:
        print("[ERROR] No plant configuration files found")
        print(f"Please run: python batch_create_configs.py")
        return
    
    # Extract plant IDs
    plant_ids = []
    for f in config_files:
        basename = os.path.basename(f)
        # Plant1140.yaml -> 1140
        plant_id = basename.replace('Plant', '').replace('.yaml', '')
        plant_ids.append(plant_id)
    
    plant_ids.sort()
    
    print(f"\nFound {len(plant_ids)} plant configurations")
    print("Scanning status...\n")
    
    # Check each plant's status
    statuses = []
    for plant_id in plant_ids:
        status = check_plant_status(plant_id, output_dir)
        statuses.append(status)
    
    # Classify by status
    complete = [s for s in statuses if s['status'] == 'COMPLETE']
    in_progress = [s for s in statuses if s['status'] == 'IN_PROGRESS']
    not_started = [s for s in statuses if s['status'] == 'NOT_STARTED']
    error = [s for s in statuses if s['status'] == 'ERROR']
    
    # Overall statistics
    total_exps = len(statuses) * 284
    completed_exps = sum(s['completed'] for s in statuses)
    
    print("=" * 100)
    print("Overall Statistics")
    print("=" * 100)
    print(f"  Total plants:     {len(statuses)}")
    print(f"  [COMPLETE]:       {len(complete)} plants ({len(complete)/len(statuses)*100:.1f}%)")
    print(f"  [IN_PROGRESS]:    {len(in_progress)} plants")
    print(f"  [NOT_STARTED]:    {len(not_started)} plants")
    if error:
        print(f"  [ERROR]:          {len(error)} plants")
    print(f"\n  Total experiments:     {total_exps} (284 x {len(statuses)})")
    print(f"  Completed experiments: {completed_exps} ({completed_exps/total_exps*100:.1f}%)")
    print(f"  Remaining experiments: {total_exps - completed_exps}")
    
    # Estimate remaining time (assuming 2.5 hours per plant)
    remaining_plants_work = len(not_started) + len(in_progress) * 0.5
    estimated_hours = remaining_plants_work * 2.5
    print(f"\n  Estimated remaining time: {estimated_hours:.1f} hours (assuming 2.5h per plant)")
    
    # Detailed list
    print("\n" + "=" * 100)
    print("Detailed Status List")
    print("=" * 100)
    print(f"{'Plant ID':<10} {'Status':<15} {'Progress':<12} {'Done/Total':<15} {'Result File':<45}")
    print("-" * 100)
    
    for status in statuses:
        if status['status'] == 'COMPLETE':
            status_display = '[COMPLETE]'
        elif status['status'] == 'IN_PROGRESS':
            status_display = '[IN_PROGRESS]'
        elif status['status'] == 'ERROR':
            status_display = '[ERROR]'
        else:
            status_display = '[NOT_STARTED]'
        
        progress_str = f"{status['progress']:.1f}%"
        completed_str = f"{status['completed']}/284"
        result_file = status['result_file'] if status['result_file'] else 'N/A'
        
        print(f"{status['plant_id']:<10} {status_display:<15} {progress_str:<12} "
              f"{completed_str:<15} {result_file:<45}")
    
    # Categorized display
    if in_progress:
        print("\n" + "=" * 100)
        print("Plants IN PROGRESS (need to continue)")
        print("=" * 100)
        for status in sorted(in_progress, key=lambda x: x['completed'], reverse=True):
            print(f"  Plant {status['plant_id']}: {status['completed']}/284 done "
                  f"({status['progress']:.1f}%), {status['remaining']} remaining")
    
    if not_started:
        print("\n" + "=" * 100)
        print("Plants NOT STARTED")
        print("=" * 100)
        plant_ids_str = ', '.join([s['plant_id'] for s in not_started[:20]])
        if len(not_started) > 20:
            plant_ids_str += f" ... (total {len(not_started)} plants)"
        print(f"  {plant_ids_str}")
    
    if error:
        print("\n" + "=" * 100)
        print("Plants with ERROR (need to check)")
        print("=" * 100)
        for status in error:
            print(f"  Plant {status['plant_id']}: {status.get('error', 'Unknown error')}")
    
    # Suggested actions
    print("\n" + "=" * 100)
    print("Suggested Actions")
    print("=" * 100)
    
    if complete and len(complete) == len(statuses):
        print("  [OK] All plant experiments completed!")
    elif in_progress or not_started:
        print("  Suggested commands to continue experiments:")
        print()
        
        if in_progress:
            in_progress_ids = [s['plant_id'] for s in in_progress]
            print(f"  # Continue in-progress plants ({len(in_progress)} plants)")
            if len(in_progress_ids) <= 5:
                print(f"  python run_experiments_multi_plant.py --plants {' '.join(in_progress_ids)}")
            else:
                print(f"  python run_experiments_multi_plant.py --plants {' '.join(in_progress_ids[:5])} ...")
        
        if not_started:
            print(f"\n  # Run not-started plants ({len(not_started)} plants)")
            if len(not_started) <= 25:
                print(f"  python run_experiments_multi_plant.py --max-plants {len(not_started)}")
            else:
                print(f"  # Run in batches (25 plants per batch)")
                print(f"  python run_experiments_multi_plant.py --skip 0 --max-plants 25   # Batch 1")
                print(f"  python run_experiments_multi_plant.py --skip 25 --max-plants 25  # Batch 2")
                print(f"  python run_experiments_multi_plant.py --skip 50 --max-plants 25  # Batch 3")
        
        print(f"\n  # Or run all (auto-resume supported)")
        print(f"  python run_experiments_multi_plant.py")
    
    print("\n" + "=" * 100)
    
    # Save status report
    status_df = pd.DataFrame(statuses)
    status_file = 'all_plants_status_report.csv'
    status_df.to_csv(status_file, index=False, encoding='utf-8-sig')
    print(f"[OK] Status report saved to: {status_file}")
    print("=" * 100)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Check status of all plant experiments')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to check for results (default: current directory). '
                            'For Colab/Drive (use quotes): "/content/drive/MyDrive/Solar PV electricity/results"')
    
    args = parser.parse_args()
    
    scan_all_plants(output_dir=args.output_dir)

