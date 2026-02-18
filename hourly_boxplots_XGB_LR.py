#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10-Minute Resolution Error Box Plots for XGB and LR Models

Creates box plots showing average error distribution by prediction start hour (0-23).
Runs for both XGB and Linear Regression models at 10-minute resolution only.

Usage:
    python hourly_boxplots_XGB_LR.py --data-path data/Project1140.csv
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime, timedelta
import warnings
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, MinuteLocator, DateFormatter
import seaborn as sns
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

from data.data_utils import preprocess_features, create_sliding_windows
from train.train_dl import train_dl_model
from train.train_ml import train_ml_model

# Import config creation functions from multi_resolution_predictions_1140
sys.path.insert(0, script_dir)
from multi_resolution_predictions_1140 import create_config_from_args, make_prediction_at_time
from rmse_boxplots_1140 import calculate_average_error_for_prediction, run_predictions_and_calculate_error, create_error_boxplots


def run_boxplots_for_model(data_path, model, complexity, scenario, lookback, use_time_encoding, output_base_dir):
    """
    Run box plots for a specific model configuration.
    
    Args:
        data_path: Path to data CSV file
        model: Model name ('XGB' or 'Linear')
        complexity: Model complexity ('low' or 'high')
        scenario: Feature scenario (e.g., 'PV+NWP')
        lookback: Lookback window in hours
        use_time_encoding: Whether to use time encoding
        output_base_dir: Base output directory
    """
    print("\n" + "=" * 80)
    print(f"RUNNING BOX PLOTS FOR {model.upper()} MODEL")
    print("=" * 80)
    
    # Create config
    config = create_config_from_args(
        data_path, model, complexity, scenario,
        lookback, use_time_encoding
    )
    
    # Create output directory for this model
    model_suffix = 'LR' if model == 'Linear' else 'XGB'
    output_dir = os.path.join(output_base_dir, f"{model_suffix}_{complexity}_{scenario}_{lookback}h")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Model: {model} {complexity} {scenario} {lookback}h")
    print(f"Time encoding: {use_time_encoding}")
    print("=" * 80)
    
    # Define resolution: 10-minute only
    resolutions = [
        (10, 144, "10-minute")   # 10 minutes, 144 intervals per 24h
    ]
    
    # Store error data for summary table
    all_error_data = {}
    
    for resolution_minutes, test_intervals, resolution_name in resolutions:
        print(f"\n{'='*80}")
        print(f"PROCESSING {resolution_name.upper()} RESOLUTION")
        print(f"{'='*80}")
        
        try:
            error_by_datetime, _, _ = run_predictions_and_calculate_error(
                data_path, config, resolution_minutes, test_intervals
            )
            
            if len(error_by_datetime) > 0:
                # Store error data for summary table
                all_error_data[resolution_name] = error_by_datetime
                
                # Create box plots grouped by hour (0-23)
                create_error_boxplots(error_by_datetime, output_dir, 
                                     config.get('experiment_name', 'Model'), 
                                     resolution_name)
                
                print(f"\n[SUCCESS] {resolution_name} resolution error box plots completed")
            else:
                print(f"\n[WARNING] No error data generated for {resolution_name} resolution")
                
        except Exception as e:
            print(f"\n[ERROR] Failed to process {resolution_name} resolution: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary table comparing average error across resolutions
    if len(all_error_data) > 0:
        print(f"\n{'='*80}")
        print("CREATING SUMMARY TABLE")
        print(f"{'='*80}")
        
        summary_data = []
        for resolution_name, error_data in all_error_data.items():
            if len(error_data) > 0:
                errors = [err for _, err in error_data]
                errors = [e for e in errors if not np.isnan(e)]
                if len(errors) > 0:
                    avg_error = np.mean(errors)
                    summary_data.append({
                        'Resolution': resolution_name,
                        'Average_Error_Percentage_Points': avg_error,
                        'Num_Predictions': len(errors)
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(output_dir, f"average_error_hourly_{model_suffix}.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"\n[SUCCESS] Summary table saved: {summary_path}")
            print(summary_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print(f"[SUCCESS] {model.upper()} MODEL BOX PLOTS COMPLETED")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Generate hourly error box plots for XGB and LR models')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to data CSV file')
    parser.add_argument('--complexity', type=str, default='high', choices=['low', 'high'],
                        help='Model complexity (default: high)')
    parser.add_argument('--scenario', type=str, default='PV+NWP',
                        choices=['PV', 'PV+HW', 'PV+NWP', 'PV+NWP+', 'NWP', 'NWP+'],
                        help='Feature scenario (default: PV+NWP)')
    parser.add_argument('--lookback', type=int, default=24, choices=[24, 72],
                        help='Lookback window in hours (default: 24)')
    parser.add_argument('--use-time-encoding', action='store_true',
                        help='Use time encoding')
    parser.add_argument('--no-time-encoding', dest='use_time_encoding', action='store_false',
                        help='Do not use time encoding (default)')
    parser.set_defaults(use_time_encoding=False)
    parser.add_argument('--output-dir', type=str, default='outputs/hourly_boxplots_XGB_LR',
                        help='Output directory (default: outputs/hourly_boxplots_XGB_LR)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("HOURLY ERROR BOX PLOTS FOR XGB AND LR MODELS")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Complexity: {args.complexity}")
    print(f"Scenario: {args.scenario}")
    print(f"Lookback: {args.lookback}h")
    print(f"Time encoding: {args.use_time_encoding}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Run for both models
    models = ['XGB', 'Linear']
    
    for model in models:
        try:
            run_boxplots_for_model(
                args.data_path, model, args.complexity, args.scenario,
                args.lookback, args.use_time_encoding, args.output_dir
            )
        except Exception as e:
            print(f"\n[ERROR] Failed to run box plots for {model}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("[SUCCESS] ALL MODELS COMPLETED")
    print(f"{'='*80}")
    print(f"\nResults saved to: {args.output_dir}")
    print("\nGenerated files for each model:")
    print("  - error_boxplot_hourly_by_hour.png (Error box plot by hour 0-23)")
    print("  - average_error_hourly_[XGB|LR].csv (Summary table)")


if __name__ == '__main__':
    main()

