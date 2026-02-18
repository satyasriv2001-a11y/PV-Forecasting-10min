#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10-Minute RMSE Box Plots for XGB and LR Models

Creates box plots showing RMSE distribution for each prediction start time.
Runs for both XGB and Linear Regression models at 10-minute resolution only.

Usage:
    python hourly_10min_boxplots_XGB_LR.py --data-path data/Project1140.csv
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
from rmse_boxplots_1140 import calculate_average_error_for_prediction, run_predictions_and_calculate_error, create_error_boxplots, create_overlay_plot_error


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
    
    # Store error data for each resolution for summary table
    all_error_data = {}
    all_predictions_data = {}  # Store prediction data for overlay plots
    
    for resolution_minutes, test_intervals, resolution_name in resolutions:
        print(f"\n{'='*80}")
        print(f"PROCESSING {resolution_name.upper()} RESOLUTION")
        print(f"{'='*80}")
        
        try:
            error_by_datetime, _, all_predictions = run_predictions_and_calculate_error(
                data_path, config, resolution_minutes, test_intervals
            )
            
            if len(error_by_datetime) > 0:
                # Store error data for summary table
                all_error_data[resolution_name] = error_by_datetime
                # Store prediction data for overlay plots
                all_predictions_data[resolution_name] = all_predictions
                
                # Create box plots grouped by hour (0-23)
                create_error_boxplots(error_by_datetime, output_dir, 
                                     config.get('experiment_name', 'Model'), 
                                     resolution_name)
                
                # Create overlay plot (predicted - actual) for 10-minute resolution
                if resolution_name == "10-minute" and len(all_predictions) > 0:
                    overlay_path = os.path.join(output_dir, f"overlay_plot_{resolution_name.lower().replace('-', '_')}.png")
                    create_overlay_plot_error(all_predictions, overlay_path, 
                                            config.get('experiment_name', 'Model'), 
                                            resolution_name, max_predictions_per_plot=12)
                
                print(f"\n[SUCCESS] {resolution_name} resolution error box plots completed")
            else:
                print(f"\n[WARNING] No error data generated for {resolution_name} resolution")
        
        except Exception as e:
            print(f"\n[ERROR] Failed {resolution_name} resolution: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary table for this model
    print(f"\n{'='*80}")
    print(f"Creating Summary Table for {model.upper()} Model")
    print(f"{'='*80}")
    
    error_summary = []
    
    for resolution_name, error_by_datetime in all_error_data.items():
        if len(error_by_datetime) > 0:
            # Extract error values
            error_values = [err for _, err in error_by_datetime]
            error_values = [e for e in error_values if not np.isnan(e)]
            
            if len(error_values) > 0:
                avg_error = np.mean(error_values)
                median_error = np.median(error_values)
                std_error = np.std(error_values)
                min_error = np.min(error_values)
                max_error = np.max(error_values)
                n_predictions = len(error_values)
                
                error_summary.append({
                    'Resolution': resolution_name,
                    'Average_Error_Percentage_Points': avg_error,
                    'Median_Error_Percentage_Points': median_error,
                    'Std_Error_Percentage_Points': std_error,
                    'Min_Error_Percentage_Points': min_error,
                    'Max_Error_Percentage_Points': max_error,
                    'Number_of_Predictions': n_predictions
                })
    
    # Save and display summary table
    if len(error_summary) > 0:
        error_summary_df = pd.DataFrame(error_summary)
        error_summary_csv_path = os.path.join(output_dir, f'average_error_boxplot_{model_suffix}_by_resolution.csv')
        error_summary_df.to_csv(error_summary_csv_path, index=False)
        print(f"\n  Average Error Summary saved: {error_summary_csv_path}")
        print(f"\n  Average Error by Resolution ({model}):")
        print(error_summary_df.to_string(index=False))
    else:
        print(f"\n  [WARNING] No error data available for summary table")
    
    return all_rmse_data


def main():
    parser = argparse.ArgumentParser(
        description='Create hourly and 10-minute RMSE box plots for XGB and LR models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to data CSV file (e.g., data/Project1140.csv)')
    parser.add_argument('--complexity', type=str, default='high',
                       choices=['low', 'high'],
                       help='Model complexity (default: high)')
    parser.add_argument('--scenario', type=str, default='PV+NWP',
                       choices=['PV', 'PV+HW', 'PV+NWP', 'PV+NWP+', 'NWP', 'NWP+'],
                       help='Feature scenario (default: PV+NWP)')
    parser.add_argument('--lookback', type=int, default=24,
                       choices=[24, 72],
                       help='Lookback window in hours (default: 24)')
    parser.add_argument('--use-time-encoding', action='store_true', default=False,
                       help='Use time encoding features (default: False)')
    parser.add_argument('--no-time-encoding', dest='use_time_encoding', action='store_false',
                       help='Disable time encoding features')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: ./hourly_10min_boxplots_XGB_LR)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("10-Minute RMSE Box Plots for XGB and LR Models")
    print("=" * 80)
    print(f"Data file: {args.data_path}")
    print(f"Complexity: {args.complexity}")
    print(f"Scenario: {args.scenario}")
    print(f"Lookback: {args.lookback}h")
    print(f"Time encoding: {args.use_time_encoding}")
    print("=" * 80 + "\n")
    
    if args.output_dir is None:
        output_base_dir = os.path.join(script_dir, "hourly_10min_boxplots_XGB_LR")
    else:
        output_base_dir = args.output_dir
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Run for both models
    models = ['XGB', 'Linear']
    all_results = {}
    
    for model in models:
        try:
            rmse_data = run_boxplots_for_model(
                args.data_path, model, args.complexity, args.scenario,
                args.lookback, args.use_time_encoding, output_base_dir
            )
            all_results[model] = rmse_data
        except Exception as e:
            print(f"\n[ERROR] Failed to run {model} model: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create combined summary table
    print(f"\n{'='*80}")
    print("Creating Combined Summary Table")
    print(f"{'='*80}")
    
    combined_summary = []
    
    for model, error_data in all_results.items():
        model_suffix = 'LR' if model == 'Linear' else 'XGB'
        for resolution_name, error_by_datetime in error_data.items():
            if len(error_by_datetime) > 0:
                error_values = [err for _, err in error_by_datetime]
                error_values = [e for e in error_values if not np.isnan(e)]
                
                if len(error_values) > 0:
                    avg_error = np.mean(error_values)
                    median_error = np.median(error_values)
                    std_error = np.std(error_values)
                    min_error = np.min(error_values)
                    max_error = np.max(error_values)
                    n_predictions = len(error_values)
                    
                    combined_summary.append({
                        'Model': model_suffix,
                        'Resolution': resolution_name,
                        'Average_Error_Percentage_Points': avg_error,
                        'Median_Error_Percentage_Points': median_error,
                        'Std_Error_Percentage_Points': std_error,
                        'Min_Error_Percentage_Points': min_error,
                        'Max_Error_Percentage_Points': max_error,
                        'Number_of_Predictions': n_predictions
                    })
    
    # Save combined summary
    if len(combined_summary) > 0:
        combined_df = pd.DataFrame(combined_summary)
        combined_csv_path = os.path.join(output_base_dir, 'average_error_boxplot_combined_XGB_LR.csv')
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\n  Combined Summary saved: {combined_csv_path}")
        print(f"\n  Combined Average Error by Model and Resolution:")
        print(combined_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("[SUCCESS] All Box Plot Generation Completed!")
    print(f"Output directory: {output_base_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

