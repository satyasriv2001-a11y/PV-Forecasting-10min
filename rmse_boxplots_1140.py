#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RMSE Box and Whisker Plots for Project1140

Creates box plots showing RMSE distribution for each prediction start time.
For each 24-hour sliding window prediction, calculates RMSE across all forecasted points
and displays as box plots grouped by datetime.

Usage:
    python rmse_boxplots_1140.py --data-path data/Project1140.csv --model XGB --complexity high --scenario PV+NWP
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
from scipy.interpolate import PchipInterpolator
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


def calculate_average_error_for_prediction(preds, gt):
    """
    Calculate average absolute error for a single prediction (24-hour forecast).
    
    Args:
        preds: Predicted values (array) - in percentage points (0-100)
        gt: Ground truth values (array) - in percentage points (0-100)
    
    Returns:
        Average absolute error in percentage points (float) or np.nan if insufficient valid data
    """
    # Filter out NaN values
    valid_mask = ~(np.isnan(preds) | np.isnan(gt))
    
    if np.sum(valid_mask) < 1:  # Need at least 1 point
        return np.nan
    
    preds_valid = preds[valid_mask]
    gt_valid = gt[valid_mask]
    
    # Calculate absolute error in percentage points
    abs_errors = np.abs(preds_valid - gt_valid)
    avg_error = np.mean(abs_errors)
    
    return avg_error


def run_predictions_and_calculate_error(data_path, config, resolution_minutes, test_intervals):
    """
    Run predictions and calculate average error for each prediction start time.
    
    Args:
        data_path: Path to data CSV file
        config: Configuration dictionary
        resolution_minutes: Resolution in minutes (60, 30, 15, or 10)
        test_intervals: Number of intervals to test
    
    Returns:
        List of (prediction_start_datetime, average_error_value) tuples
    """
    print("=" * 80)
    resolution_name = f"{resolution_minutes}-minute" if resolution_minutes < 60 else "hourly"
    print(f"Running {resolution_name.upper()} Predictions for Error Box Plots")
    print("=" * 80)
    print(f"Data file: {data_path}")
    print(f"Model: {config['experiment_name']}")
    print(f"Resolution: {resolution_minutes} minutes")
    print(f"Test intervals: {test_intervals}")
    print("=" * 80)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    has_minute_column = 'Minute' in df.columns
    
    if has_minute_column:
        df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        print(f"  Detected {resolution_minutes}-minute resolution data (Minute column found)")
    else:
        df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
        print(f"  Detected hourly resolution data (no Minute column)")
        print(f"  Interpolating to {resolution_minutes}-minute resolution...")
        
        df = df.drop_duplicates(subset='Datetime', keep='first')
        df = df.sort_values('Datetime').reset_index(drop=True)
        df_indexed = df.set_index('Datetime')
        
        # Resample to target resolution
        freq_map = {60: 'H', 30: '30T', 15: '15T', 10: '10T'}
        freq = freq_map.get(resolution_minutes, 'H')
        df_resampled = df_indexed.resample(freq).asfreq()
        
        # Interpolate numeric columns using Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)
        t_orig = df_indexed.index.astype(np.float64)
        t_new = df_resampled.index.astype(np.float64)
        numeric_cols = df_resampled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            y_orig = df_indexed[col].values
            if len(np.unique(t_orig)) < 2:
                df_resampled[col] = df_resampled[col].ffill().bfill()
                continue
            pchip = PchipInterpolator(t_orig, y_orig)
            df_resampled[col] = pchip(t_new)
        
        non_numeric_cols = df_resampled.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            if col not in ['Datetime']:
                df_resampled[col] = df_resampled[col].ffill()
        
        df = df_resampled.reset_index()
        df['Year'] = df['Datetime'].dt.year
        df['Month'] = df['Datetime'].dt.month
        df['Day'] = df['Datetime'].dt.day
        df['Hour'] = df['Datetime'].dt.hour
        df['Minute'] = df['Datetime'].dt.minute
        
        has_minute_column = True
        print(f"  Interpolated to {len(df)} {resolution_minutes}-minute points")
    
    print("\n[1/4] Preprocessing data...")
    df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = preprocess_features(df, config)
    
    print("\n[2/4] Creating sliding windows and splitting data...")
    past_hours = int(config.get('past_hours', 24))
    future_hours = int(config.get('future_hours', 24))
    
    # Convert hours to intervals based on resolution
    intervals_per_hour = 60 // resolution_minutes
    past_intervals = past_hours * intervals_per_hour
    future_intervals = future_hours * intervals_per_hour
    
    X_hist, X_fcst, y, hours, dates = create_sliding_windows(
        df_clean, past_intervals, future_intervals, hist_feats, fcst_feats, no_hist_power
    )
    
    total_samples = len(X_hist)
    indices = np.arange(total_samples)
    
    if config.get('shuffle_split', True):
        np.random.seed(config.get('random_seed', 42))
        np.random.shuffle(indices)
    
    train_size = int(total_samples * config['train_ratio'])
    val_size = int(total_samples * config['val_ratio'])
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    print(f"  Train samples: {len(train_idx)}")
    print(f"  Val samples: {len(val_idx)}")
    print(f"  Test samples: {len(test_idx)}")
    
    X_hist_train, y_train = X_hist[train_idx], y[train_idx]
    X_hist_val, y_val = X_hist[val_idx], y[val_idx]
    X_hist_test, y_test = X_hist[test_idx], y[test_idx]
    
    if X_fcst is not None:
        X_fcst_train, X_fcst_val, X_fcst_test = X_fcst[train_idx], X_fcst[val_idx], X_fcst[test_idx]
    else:
        X_fcst_train = X_fcst_val = X_fcst_test = None
    
    train_hours = np.array([hours[int(i)] for i in train_idx])
    val_hours = np.array([hours[int(i)] for i in val_idx])
    test_hours_array = np.array([hours[int(i)] for i in test_idx])
    
    train_data = (X_hist_train, X_fcst_train, y_train, train_hours, [])
    val_data = (X_hist_val, X_fcst_val, y_val, val_hours, [])
    test_data = (X_hist_test, X_fcst_test, y_test, test_hours_array, [])
    scalers = (scaler_hist, scaler_fcst, scaler_target)
    
    print("\n[3/4] Training model...")
    import torch
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    start_time = time.time()
    if config['model'] in ['LSTM', 'GRU', 'Transformer', 'TCN']:
        model, metrics = train_dl_model(config, train_data, val_data, test_data, scalers)
    else:
        model, metrics = train_ml_model(config, train_data, val_data, test_data, scalers)
    
    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.1f} seconds")
    print(f"  MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
    
    print("\n[4/4] Making predictions and calculating RMSE...")
    
    if len(test_idx) == 0:
        raise ValueError("No test samples available.")
    
    if isinstance(test_idx, np.ndarray):
        test_idx_list = test_idx.tolist()
    else:
        test_idx_list = list(test_idx)
    
    first_test_sample_idx = int(test_idx_list[0])
    intervals_per_hour = 60 // resolution_minutes
    past_intervals = past_hours * intervals_per_hour
    first_test_start_in_df = int(past_intervals) + first_test_sample_idx
    
    first_test_datetime = df_clean.iloc[first_test_start_in_df]['Datetime']
    target_year = first_test_datetime.year
    
    target_date = pd.Timestamp(year=target_year, month=6, day=20, hour=0, minute=0)
    
    start_idx = None
    for idx in range(len(df_clean)):
        if df_clean.iloc[idx]['Datetime'] >= target_date:
            start_idx = idx
            break
    
    if start_idx is None or start_idx < first_test_start_in_df:
        print(f"  Warning: Could not find June 20, {target_year} 00:00 in test data.")
        print(f"  Using first test sample start instead: {df_clean.iloc[first_test_start_in_df]['Datetime']}")
        start_idx = first_test_start_in_df
    else:
        actual_start_date = df_clean.iloc[start_idx]['Datetime']
        print(f"  Starting predictions from: {actual_start_date.strftime('%Y-%m-%d %H:%M')}")
    
    test_time_indices = []
    for i in range(test_intervals):
        time_idx = int(start_idx + i)
        if time_idx >= 0 and time_idx < len(df_clean) - future_intervals:
            test_time_indices.append(time_idx)
        else:
            break
    
    if len(test_time_indices) < test_intervals:
        print(f"  Warning: Only found {len(test_time_indices)} valid intervals (requested {test_intervals})")
    
    print(f"  Making predictions for {len(test_time_indices)} intervals...")
    
    error_by_datetime = []
    all_predictions = []  # Store prediction data for overlay plots
    
    for pred_num, time_idx in enumerate(test_time_indices, 1):
        try:
            preds, gt, pred_datetime, future_dt = make_prediction_at_time(
                model, config, df_clean, hist_feats, fcst_feats, 
                scaler_hist, scaler_fcst, scaler_target, no_hist_power,
                time_idx, past_intervals, future_intervals, resolution_minutes
            )
            
            # Calculate average error for this prediction
            avg_error = calculate_average_error_for_prediction(preds, gt)
            
            if not np.isnan(avg_error):
                error_by_datetime.append((pred_datetime, avg_error))
            
            # Store prediction data for overlay plots
            pred_df = pd.DataFrame({
                'Datetime': future_dt,
                'Predicted_Capacity_Factor': preds[:len(future_dt)],
                'Ground_Truth_Capacity_Factor': gt[:len(future_dt)]
            })
            all_predictions.append((pred_df, pred_num, pred_datetime))
            
            if pred_num % 10 == 0 or pred_num == len(test_time_indices):
                print(f"  [{pred_num}/{len(test_time_indices)}] Completed")
        
        except Exception as e:
            print(f"  [ERROR] Prediction {pred_num} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"  Calculated average error for {len(error_by_datetime)} predictions")
    
    return error_by_datetime, resolution_name, all_predictions


def create_overlay_plot_error(all_predictions, output_path, model_name, resolution_name, max_predictions_per_plot=12):
    """
    Create overlay plots showing (prediction - actual) differences.
    Similar to multi_resolution_predictions_1140.py but with large fonts.
    
    Args:
        all_predictions: List of (pred_df, pred_num, pred_datetime) tuples
        output_path: Base path for output plots
        model_name: Name of the model
        resolution_name: Resolution name (e.g., "Hourly", "10-minute")
        max_predictions_per_plot: Maximum number of predictions to show per plot (default: 12)
    
    Returns:
        List of output file paths
    """
    num_predictions = len(all_predictions)
    num_plots = (num_predictions + max_predictions_per_plot - 1) // max_predictions_per_plot
    
    output_paths = []
    
    for plot_idx in range(num_plots):
        start_idx = plot_idx * max_predictions_per_plot
        end_idx = min(start_idx + max_predictions_per_plot, num_predictions)
        plot_predictions = all_predictions[start_idx:end_idx]
        
        if num_plots > 1:
            base_path = os.path.splitext(output_path)[0]
            ext = os.path.splitext(output_path)[1]
            plot_output_path = f"{base_path}_part{plot_idx + 1}{ext}"
        else:
            plot_output_path = output_path
        
        plt.figure(figsize=(24, 10))
        plt.rcParams.update({'font.size': 27})
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(plot_predictions)))
        plot_datetimes = set()
        
        for idx, (pred_df, pred_num, pred_datetime) in enumerate(plot_predictions):
            differences = pred_df['Predicted_Capacity_Factor'] - pred_df['Ground_Truth_Capacity_Factor']
            valid_mask = ~(pred_df['Predicted_Capacity_Factor'].isna() | pred_df['Ground_Truth_Capacity_Factor'].isna())
            valid_datetimes = pred_df.loc[valid_mask, 'Datetime']
            valid_differences = differences.loc[valid_mask]
            
            plt.plot(valid_datetimes, valid_differences, 
                    linewidth=3.0, alpha=0.85, color=colors[idx])
            plot_datetimes.update(valid_datetimes)
        
        plt.axhline(y=0, color='black', linestyle='--', linewidth=2.5, alpha=0.8)
        
        plot_datetimes_sorted = sorted(list(plot_datetimes))
        if plot_datetimes_sorted:
            x_min = min(plot_datetimes_sorted)
            x_max = max(plot_datetimes_sorted)
            x_range = x_max - x_min
            plt.xlim(x_min - 0.02 * x_range, x_max + 0.02 * x_range)
        
        ax = plt.gca()
        if plot_datetimes_sorted:
            ax.xaxis.set_major_locator(HourLocator(interval=6))
            ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:00'))
            ax.xaxis.set_minor_locator(HourLocator(interval=1))
        
        plt.xlabel('Datetime', fontsize=33, fontweight='bold')
        plt.ylabel('Prediction Error (Predicted - Actual) (%)', fontsize=33, fontweight='bold')
        
        if num_plots > 1:
            title = f'Prediction Error Overlay - {model_name} ({resolution_name}) - Part {plot_idx + 1}/{num_plots}\n'
            title += f'Predictions {start_idx + 1}-{end_idx} of {num_predictions} (24-hour ahead forecasts)'
        else:
            title = f'Prediction Error Overlay - {model_name} ({resolution_name})\n'
            title += f'{num_predictions} prediction intervals (24-hour ahead forecasts)'
        
        plt.title(title, fontsize=36, fontweight='bold')
        plt.grid(True, alpha=0.3, which='both')
        plt.xticks(rotation=45, ha='right', fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(plot_output_path) if os.path.dirname(plot_output_path) else '.', exist_ok=True)
        plt.savefig(plot_output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Overlay plot saved: {plot_output_path} (Predictions {start_idx + 1}-{end_idx})")
        output_paths.append(plot_output_path)
    
    return output_paths


def create_error_boxplots(error_by_datetime_list, output_dir, model_name, resolution_name):
    """
    Create box and whisker plots of average error values grouped by prediction start hour (0-23).
    
    Args:
        error_by_datetime_list: List of (datetime, average_error) tuples
        output_dir: Output directory for plots
        model_name: Name of the model
        resolution_name: Resolution name (e.g., "Hourly", "30-minute", "15-minute", "10-minute")
    """
    if len(error_by_datetime_list) == 0:
        print(f"  [WARNING] No error data to plot for {resolution_name}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(error_by_datetime_list, columns=['Datetime', 'Average_Error'])
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Extract hour (0-23) from datetime
    df['Hour'] = df['Datetime'].dt.hour
    
    # Group by hour (0-23)
    # Remove hours with too few data points (less than 2)
    hour_counts = df.groupby('Hour').size()
    valid_hours = hour_counts[hour_counts >= 2].index
    df_filtered = df[df['Hour'].isin(valid_hours)]
    
    if len(df_filtered) == 0:
        print(f"  [WARNING] No valid hours with sufficient data points for {resolution_name}")
        return
    
    # Create box plot
    plt.figure(figsize=(16, 8))
    
    # Prepare data for box plot - group by hour (0-23)
    hours = sorted(df_filtered['Hour'].unique())
    error_data_by_hour = [df_filtered[df_filtered['Hour'] == h]['Average_Error'].values for h in hours]
    
    # Create box plot
    bp = plt.boxplot(error_data_by_hour, labels=hours, patch_artist=True, widths=0.6)
    
    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Format x-axis - show hours 0-23
    ax = plt.gca()
    ax.set_xlim(0.5, 23.5)
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    
    plt.xlabel('Prediction Start Hour (0-23)', fontsize=14, fontweight='bold')
    plt.ylabel('Average Error (Percentage Points)', fontsize=14, fontweight='bold')
    plt.title(f'Average Error Distribution by Prediction Start Hour - {model_name} ({resolution_name})\n'
              f'Box plot shows average error distribution for 24-hour forecasts starting at each hour',
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"error_boxplot_{resolution_name.lower().replace('-', '_')}_by_hour.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Error box plot saved: {output_path}")


def run_error_boxplots(data_path, config, output_dir, resolutions=None):
    """
    Run error box plots for multiple resolutions.
    
    Args:
        data_path: Path to data CSV file
        config: Configuration dictionary
        output_dir: Output directory for plots
        resolutions: List of (resolution_minutes, test_intervals, resolution_name) tuples. If None, uses defaults
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if resolutions is None:
        resolutions = [
            (10, 144, "10-minute")   # 10 minutes, 144 intervals per 24h
        ]
    
    # Store error data for each resolution for summary table
    all_error_data = {}
    
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
                
                # Create box plots grouped by hour (0-23)
                create_error_boxplots(error_by_datetime, output_dir, 
                                     config.get('experiment_name', 'Model'), 
                                     resolution_name)
                
                # Create overlay plot (predicted - actual)
                if len(all_predictions) > 0:
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
    
    # =============================================================================
    # CREATE SUMMARY TABLE: Average Error by Resolution
    # =============================================================================
    print(f"\n{'='*80}")
    print("Creating Summary Table: Average Error by Resolution")
    print(f"{'='*80}")
    
    error_summary = []
    
    for resolution_name, error_by_datetime in all_error_data.items():
        if len(error_by_datetime) > 0:
            # Extract error values (ignore datetime, just get error)
            error_values = [err for _, err in error_by_datetime]
            error_values = [e for e in error_values if not np.isnan(e)]  # Remove NaN values
            
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
        error_summary_csv_path = os.path.join(output_dir, 'average_error_boxplot_by_resolution.csv')
        error_summary_df.to_csv(error_summary_csv_path, index=False)
        print(f"\n  Average Error Summary saved: {error_summary_csv_path}")
        print(f"\n  Average Error by Resolution (from box plot data):")
        print(f"  Note: Average error across all prediction start times for each resolution")
        print(error_summary_df.to_string(index=False))
    else:
        print(f"\n  [WARNING] No error data available for summary table")
    
    print(f"\n{'='*80}")
    print("[SUCCESS] Error Box Plot Generation Completed!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")


# =============================================================================
# MAIN ENTRY
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create error box plots for multi-resolution predictions (grouped by hour 0-23)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to data CSV file (e.g., data/Project1140.csv)')
    parser.add_argument('--model', type=str, default='LSTM',
                       choices=['LSTM', 'GRU', 'Transformer', 'TCN', 'RF', 'XGB', 'LGBM', 'Linear'],
                       help='Model to use (default: LSTM)')
    parser.add_argument('--complexity', type=str, default='high',
                       choices=['low', 'high'],
                       help='Model complexity (default: high)')
    parser.add_argument('--scenario', type=str, default='PV+NWP',
                       choices=['PV', 'PV+HW', 'PV+NWP', 'PV+NWP+', 'NWP', 'NWP+'],
                       help='Feature scenario (default: PV+NWP)')
    parser.add_argument('--lookback', type=int, default=24,
                       choices=[24, 72],
                       help='Lookback window in hours (default: 24)')
    parser.add_argument('--use-time-encoding', action='store_true', default=True,
                       help='Use time encoding features (default: True)')
    parser.add_argument('--no-time-encoding', dest='use_time_encoding', action='store_false',
                       help='Disable time encoding features')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: ./error_boxplots_<model>_<scenario>)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("MODE: Error Box and Whisker Plots (Grouped by Hour 0-23)")
    print(f"Algorithm: {args.model} {args.complexity} {args.scenario}")
    print("=" * 80 + "\n")
    
    config = create_config_from_args(
        args.data_path, args.model, args.complexity, args.scenario,
        args.lookback, args.use_time_encoding
    )
    
    if args.output_dir is None:
        output_dir = os.path.join(script_dir, f"error_boxplots_{args.model}_{args.scenario}")
    else:
        output_dir = args.output_dir
    
    try:
        run_error_boxplots(
            args.data_path, config, output_dir
        )
    except Exception as e:
        print(f"\n[ERROR] Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


