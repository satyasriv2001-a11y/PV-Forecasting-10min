#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-Click Script to Run All Sensitivity Analysis Experiments on Google Colab

This script automatically:
1. Detects if running on Colab
2. Mounts Google Drive (if on Colab)
3. Checks GPU availability
4. Runs all 8 sensitivity analysis experiments
5. Saves results to Google Drive: /content/drive/MyDrive/Solar PV electricity/sensitivity_analysis_results

Usage:
    # On Google Colab:
    !python run_sensitivity_analysis_colab.py
    
    # Or specify custom data/output directories:
    !python run_sensitivity_analysis_colab.py --data-dir data --custom-output "/path/to/output"
    
    # Or run specific experiments only:
    !python run_sensitivity_analysis_colab.py --experiments 1 3 5
"""

import os
import sys
import argparse
from datetime import datetime
import subprocess


def is_colab():
    """Check if running on Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def mount_drive():
    """Mount Google Drive on Colab"""
    if not is_colab():
        print("Not running on Colab, skipping Drive mount")
        return False
    
    try:
        from google.colab import drive
        
        # Check if already mounted
        if os.path.exists('/content/drive/MyDrive'):
            print("[OK] Google Drive already mounted")
            return True
        
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
        print("[OK] Google Drive mounted successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to mount Google Drive: {e}")
        return False


def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[OK] GPU available: {gpu_name}")
            return True
        else:
            print("[WARNING] No GPU detected. Training will use CPU (slower)")
            return False
    except ImportError:
        print("[WARNING] PyTorch not found, cannot check GPU")
        return False


def check_data_directory(data_dir):
    """Check if data directory exists and count CSV files"""
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        return False
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if len(csv_files) == 0:
        print(f"[ERROR] No CSV files found in {data_dir}")
        return False
    
    print(f"[OK] Found {len(csv_files)} CSV files in {data_dir}")
    return True


def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[OK] Output directory ready: {output_dir}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create output directory: {e}")
        return False


def run_experiment(exp_num, exp_name, exp_module, data_dir, output_dir, local_output_dir):
    """Run a single sensitivity analysis experiment"""
    print("\n" + "=" * 80)
    print(f"Experiment {exp_num}/8: {exp_name}")
    print("=" * 80)
    
    script_path = f"sensitivity_analysis/{exp_module}.py"
    
    if not os.path.exists(script_path):
        print(f"[ERROR] Script not found: {script_path}")
        return False
    
    # Show command being run
    print(f"Running command: python {script_path} --data-dir {data_dir} --output-dir {output_dir} --local-output {local_output_dir}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = datetime.now()
    
    try:
        # Run experiment using subprocess with real-time output
        result = subprocess.run(
            ['python', script_path, '--data-dir', data_dir, '--output-dir', output_dir, '--local-output', local_output_dir],
            capture_output=False,  # Show real-time output
            text=True
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        if result.returncode == 0:
            print(f"\n[OK] Experiment {exp_num} completed successfully in {duration:.1f} minutes")
            return True
        else:
            print(f"\n[ERROR] Experiment {exp_num} failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Experiment {exp_num} failed: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='One-click script to run all sensitivity analysis experiments on Colab',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all experiments (default):
    python run_sensitivity_analysis_colab.py
    
    # Run specific experiments:
    python run_sensitivity_analysis_colab.py --experiments 1 3 5
    
    # Use custom paths:
    python run_sensitivity_analysis_colab.py --data-dir /path/to/data --custom-output /path/to/output
        """
    )
    
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing plant CSV files (default: data)')
    
    parser.add_argument('--custom-output', type=str, default=None,
                       help='Custom output directory (default: Google Drive path)')
    
    parser.add_argument('--experiments', type=int, nargs='+', choices=range(1, 9),
                       help='Experiment numbers to run (1-8). If not specified, run all.')
    
    parser.add_argument('--skip-mount', action='store_true',
                       help='Skip Google Drive mounting (use if already mounted)')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("SENSITIVITY ANALYSIS - ONE-CLICK RUNNER")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Check environment
    on_colab = is_colab()
    print(f"\nEnvironment: {'Google Colab' if on_colab else 'Local'}")
    
    # Mount Google Drive (if on Colab)
    if on_colab and not args.skip_mount:
        if not mount_drive():
            print("\n[ERROR] Failed to mount Google Drive. Exiting.")
            return
    
    # Check GPU
    check_gpu()
    
    # Determine output directory
    if args.custom_output:
        output_dir = args.custom_output
    elif on_colab:
        output_dir = "/content/drive/MyDrive/Solar PV electricity/sensitivity_analysis_results"
    else:
        output_dir = "sensitivity_analysis/results"
    
    # Always set up local backup directory
    local_output_dir = "sensitivity_analysis/results"
    
    print(f"\nData directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Local backup directory: {local_output_dir}")
    
    # Check data directory
    print("\n" + "=" * 80)
    print("Checking data directory...")
    print("=" * 80)
    if not check_data_directory(args.data_dir):
        print("\n[ERROR] Data directory check failed. Exiting.")
        return
    
    # Create output directory
    print("\n" + "=" * 80)
    print("Creating output directory...")
    print("=" * 80)
    if not create_output_directory(output_dir):
        print("\n[ERROR] Failed to create output directory. Exiting.")
        return
    
    # Define all experiments
    all_experiments = {
        1: ('Seasonal Effect', 'seasonal_effect'),
        2: ('Hourly Effect', 'hourly_effect'),
        3: ('Weather Feature Adoption', 'weather_feature_adoption'),
        4: ('Lookback Window Length', 'lookback_window'),
        5: ('Model Complexity', 'model_complexity'),
        6: ('Training Dataset Scale', 'training_scale'),
        7: ('No Shuffle Training', 'no_shuffle'),
        8: ('Dataset Extension (Hourly Sliding Windows)', 'dataset_extension')
    }
    
    # Determine which experiments to run
    if args.experiments is None:
        experiments_to_run = list(all_experiments.keys())
    else:
        experiments_to_run = sorted(args.experiments)
    
    print("\n" + "=" * 80)
    print(f"Running {len(experiments_to_run)}/{len(all_experiments)} experiments:")
    print("=" * 80)
    for exp_num in experiments_to_run:
        print(f"  {exp_num}. {all_experiments[exp_num][0]}")
    print()
    
    # Run experiments
    results = {}
    successful = 0
    failed = 0
    
    for i, exp_num in enumerate(experiments_to_run, 1):
        exp_name, exp_module = all_experiments[exp_num]
        
        print(f"\n[{i}/{len(experiments_to_run)}] Starting: {exp_name}...")
        
        success = run_experiment(exp_num, exp_name, exp_module, args.data_dir, output_dir, local_output_dir)
        
        results[exp_num] = {
            'name': exp_name,
            'success': success
        }
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTotal experiments: {len(experiments_to_run)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("\nExperiment Results:")
    print("-" * 80)
    
    for exp_num in experiments_to_run:
        result = results[exp_num]
        status = "[OK]" if result['success'] else "[FAILED]"
        print(f"{status} Experiment {exp_num}: {result['name']}")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {output_dir}")
    print("=" * 80)
    
    if failed == 0:
        print("\n[SUCCESS] All experiments completed successfully!")
    else:
        print(f"\n[WARNING] {failed} experiment(s) failed. Check logs above for details.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[FATAL ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

