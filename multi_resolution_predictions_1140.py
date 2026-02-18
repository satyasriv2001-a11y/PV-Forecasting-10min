#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Multi-Resolution Predictions for Project1140



Runs predictions at 10-minute resolution only.

Generates overlay plots and hourly RMSE plots.



Usage:

    python multi_resolution_predictions_1140.py --data-path data/Project1140.csv --model XGB --complexity high --scenario PV+NWP

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

warnings.filterwarnings('ignore')



script_dir = os.path.dirname(os.path.abspath(__file__))

os.chdir(script_dir)

sys.path.append(script_dir)



from data.data_utils import preprocess_features, create_sliding_windows

from train.train_dl import train_dl_model

from train.train_ml import train_ml_model





# =============================================================================

# CONFIG CREATION

# =============================================================================

def create_config(data_path, model, complexity, lookback, feat_combo, use_te, is_nwp_only):

    """Create a single experiment configuration"""

    config = {

        'data_path': data_path,

        'model': model,

        'model_complexity': complexity,

        'use_pv': feat_combo['use_pv'],

        'use_hist_weather': feat_combo['use_hist_weather'],

        'use_forecast': feat_combo['use_forecast'],

        'use_ideal_nwp': feat_combo['use_ideal_nwp'],

        'use_time_encoding': use_te,

        'weather_category': 'medium_weather',

        'future_hours': 24,

        'start_date': '2022-01-01',

        'end_date': '2024-09-28',

        'save_options': {

            'save_model': False,

            'save_predictions': False,

            'save_excel_results': False,

            'save_training_log': False

        }

    }



    if is_nwp_only:

        config['past_hours'] = 0

        config['past_days'] = 0

        config['no_hist_power'] = True

        feat_name = feat_combo['name']

    else:

        config['past_hours'] = lookback

        config['past_days'] = lookback // 24

        config['no_hist_power'] = False

        feat_name = f"{feat_combo['name']}_{lookback}h"



    config.update({

        'train_ratio': 0.8, 

        'val_ratio': 0.1, 

        'test_ratio': 0.1,

        'shuffle_split': False,

        'random_seed': 42

    })



    # Model-specific hyperparams

    if model in ['LSTM', 'GRU', 'Transformer', 'TCN']:

        if complexity == 'low':

            config.update({

                'train_params': {'epochs': 20, 'batch_size': 64, 'learning_rate': 0.001,

                                 'patience': 10, 'min_delta': 0.001, 'weight_decay': 1e-4},

                'model_params': {'d_model': 16, 'hidden_dim': 8, 'num_heads': 2, 'num_layers': 1,

                                 'dropout': 0.1, 'tcn_channels': [8, 16], 'kernel_size': 3}

            })

        else:

            config.update({

                'train_params': {'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001,

                                 'patience': 10, 'min_delta': 0.001, 'weight_decay': 1e-4},

                'model_params': {'d_model': 32, 'hidden_dim': 16, 'num_heads': 2, 'num_layers': 2,

                                 'dropout': 0.1, 'tcn_channels': [16, 32], 'kernel_size': 3}

            })

    elif model == 'Linear':

        config['model_params'] = {}

    else:

        if complexity == 'low':

            config['model_params'] = {'n_estimators': 10, 'max_depth': 1, 'learning_rate': 0.2,

                                      'random_state': 42, 'verbosity': -1}

        else:

            config['model_params'] = {'n_estimators': 30, 'max_depth': 3, 'learning_rate': 0.1,

                                      'random_state': 42, 'verbosity': -1}



    te_suffix = 'TE' if use_te else 'noTE'

    config['experiment_name'] = f"{model}_{feat_name}_{te_suffix}" if model == 'Linear' else f"{model}_{complexity}_{feat_name}_{te_suffix}"

    config['save_dir'] = f'results/{config["experiment_name"]}'

    return config





def create_config_from_args(data_path, model, complexity, scenario, lookback, use_time_encoding):

    """Create config from command line arguments"""

    feature_combos_pv = [

        {'name': 'PV', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': False, 'use_ideal_nwp': False},

        {'name': 'PV+HW', 'use_pv': True, 'use_hist_weather': True, 'use_forecast': False, 'use_ideal_nwp': False},

        {'name': 'PV+NWP', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': False},

        {'name': 'PV+NWP+', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': True},

    ]



    feature_combos_nwp = [

        {'name': 'NWP', 'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': False},

        {'name': 'NWP+', 'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': True},

    ]



    feat_combo = None

    is_nwp_only = False

    

    for combo in feature_combos_pv:

        if combo['name'] == scenario:

            feat_combo = combo

            break

    

    if feat_combo is None:

        for combo in feature_combos_nwp:

            if combo['name'] == scenario:

                feat_combo = combo

                is_nwp_only = True

                break

    

    if feat_combo is None:

        raise ValueError(f"Unknown scenario: {scenario}")

    

    return create_config(data_path, model, complexity, lookback, feat_combo, use_time_encoding, is_nwp_only)





# =============================================================================

# PREDICTION FUNCTIONS

# =============================================================================

def make_prediction_at_time(model, config, df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, 

                            scaler_target, no_hist_power, time_idx, past_intervals, future_intervals, resolution_minutes):

    """

    Make a prediction for the next 24 hours starting from a specific index.

    

    Args:

        resolution_minutes: Resolution in minutes (60, 30, 15, or 10)

    """

    import torch

    

    time_idx = int(time_idx)

    past_intervals = int(past_intervals)

    future_intervals = int(future_intervals)

    

    # Get historical data

    hist_start = max(0, time_idx - past_intervals)

    hist_end = time_idx

    hist_data = df_clean.iloc[hist_start:hist_end].copy()

    

    # Get future data

    fut_start = time_idx

    fut_end = min(len(df_clean), time_idx + future_intervals)

    fut_data = df_clean.iloc[fut_start:fut_end].copy()

    

    hist_feats_list = list(hist_feats) if hist_feats else []

    n_hist_feats = len(hist_feats_list)

    

    if len(hist_data) < past_intervals and not no_hist_power:

        if len(hist_data) > 0 and n_hist_feats > 0:

            padding = np.zeros((int(past_intervals - len(hist_data)), n_hist_feats))

            hist_array = np.vstack([padding, hist_data[hist_feats_list].values])

        else:

            hist_array = np.zeros((int(past_intervals), n_hist_feats))

    elif no_hist_power:

        hist_array = np.zeros((int(past_intervals) if past_intervals > 0 else 1, n_hist_feats))

    else:

        if n_hist_feats > 0:

            hist_array = hist_data[hist_feats_list].values

        else:

            hist_array = np.zeros((len(hist_data), 0))

    

    fcst_feats_list = list(fcst_feats) if fcst_feats else []

    n_fcst_feats = len(fcst_feats_list)

    

    if fcst_feats_list and n_fcst_feats > 0:

        if len(fut_data) < future_intervals:

            if len(fut_data) > 0:

                last_row = fut_data[fcst_feats_list].iloc[-1:].values

            else:

                last_row = np.zeros((1, n_fcst_feats))

            padding = np.tile(last_row, (int(future_intervals - len(fut_data)), 1))

            if len(fut_data) > 0:

                fcst_array = np.vstack([fut_data[fcst_feats_list].values, padding])

            else:

                fcst_array = padding

        else:

            fcst_array = fut_data[fcst_feats_list].values[:int(future_intervals)]

    else:

        fcst_array = None

    

    # Get ground truth

    if len(fut_data) < future_intervals:

        gt = np.full(future_intervals, np.nan)

        if len(fut_data) > 0:

            gt[:len(fut_data)] = fut_data['Capacity Factor'].values

    else:

        gt = fut_data['Capacity Factor'].values[:future_intervals]

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    past_timesteps = hist_array.shape[0] if len(hist_array.shape) > 0 else past_intervals

    future_timesteps = fcst_array.shape[0] if fcst_array is not None and len(fcst_array.shape) > 0 else future_intervals

    

    X_hist = hist_array.reshape(1, past_timesteps, -1)

    if fcst_array is not None:

        X_fcst = fcst_array.reshape(1, future_timesteps, -1)

    else:

        X_fcst = None

    

    prediction_datetime = df_clean.iloc[time_idx]['Datetime']

    

    # Make prediction

    if config['model'] in ['LSTM', 'GRU', 'Transformer', 'TCN']:

        model.eval()

        with torch.no_grad():

            X_hist_tensor = torch.tensor(X_hist, dtype=torch.float32).to(device)

            

            if X_fcst is not None:

                X_fcst_tensor = torch.tensor(X_fcst, dtype=torch.float32).to(device)

                preds = model(X_hist_tensor, X_fcst_tensor)

            else:

                preds = model(X_hist_tensor)

            

            preds_np = preds.cpu().numpy().flatten()

    else:

        if X_fcst is not None:

            X_flat = np.concatenate([X_hist.reshape(1, -1), X_fcst.reshape(1, -1)], axis=1)

        else:

            X_flat = X_hist.reshape(1, -1)

        

        preds_np = model.predict(X_flat).flatten()

    

    # Inverse transform

    if scaler_target is not None:

        preds_inv = scaler_target.inverse_transform(preds_np.reshape(-1, 1)).flatten()

        gt_inv = scaler_target.inverse_transform(gt.reshape(-1, 1)).flatten() if not np.isnan(gt).all() else gt

    else:

        preds_inv = preds_np

        gt_inv = gt

    

    preds_inv = np.clip(preds_inv, 0, 100)

    

    # Create future datetimes based on resolution

    freq_map = {60: 'H', 30: '30T', 15: '15T', 10: '10T'}

    freq = freq_map.get(resolution_minutes, 'H')

    future_datetimes = pd.date_range(start=prediction_datetime, periods=future_intervals, freq=freq)

    

    return preds_inv, gt_inv, prediction_datetime, future_datetimes





# =============================================================================

# PLOTTING FUNCTIONS

# =============================================================================

def create_overlay_plot(all_predictions, output_path, model_name, resolution_name, max_predictions_per_plot=12):

    """

    Create overlay plots for all predictions showing (prediction - actual) differences.

    If there are more than max_predictions_per_plot predictions, creates multiple plots.

    

    Args:

        all_predictions: List of (pred_df, pred_num, pred_datetime) tuples

        output_path: Base path for output plots (will append _part1, _part2, etc. if needed)

        model_name: Name of the model

        resolution_name: Resolution name (e.g., "Hourly", "30-minute", "15-minute")

        max_predictions_per_plot: Maximum number of predictions to show per plot (default: 12)

    

    Returns:

        List of output file paths

    """

    num_predictions = len(all_predictions)

    num_plots = (num_predictions + max_predictions_per_plot - 1) // max_predictions_per_plot  # Ceiling division

    

    output_paths = []

    

    # Create each plot

    for plot_idx in range(num_plots):

        start_idx = plot_idx * max_predictions_per_plot

        end_idx = min(start_idx + max_predictions_per_plot, num_predictions)

        plot_predictions = all_predictions[start_idx:end_idx]

        

        # Determine output path

        if num_plots > 1:

            base_path = os.path.splitext(output_path)[0]

            ext = os.path.splitext(output_path)[1]

            plot_output_path = f"{base_path}_part{plot_idx + 1}{ext}"

        else:

            plot_output_path = output_path

        

        plt.figure(figsize=(24, 10))

        

        # Set larger font sizes globally for this figure (50% increase)

        plt.rcParams.update({'font.size': 27})

        

        colors = plt.cm.tab20(np.linspace(0, 1, len(plot_predictions)))

        

        # Collect datetimes for this specific plot only

        plot_datetimes = set()

        

        # Plot (prediction - actual) differences for this chunk

        for idx, (pred_df, pred_num, pred_datetime) in enumerate(plot_predictions):

            # Calculate prediction - actual difference

            differences = pred_df['Predicted_Capacity_Factor'] - pred_df['Ground_Truth_Capacity_Factor']

            # Only plot where both values are valid

            valid_mask = ~(pred_df['Predicted_Capacity_Factor'].isna() | pred_df['Ground_Truth_Capacity_Factor'].isna())

            valid_datetimes = pred_df.loc[valid_mask, 'Datetime']

            valid_differences = differences.loc[valid_mask]

            

            plt.plot(valid_datetimes, valid_differences, 

                    linewidth=3.0, alpha=0.85, color=colors[idx])

            plot_datetimes.update(valid_datetimes)

        

        # Add horizontal line at y=0 for reference

        plt.axhline(y=0, color='black', linestyle='--', linewidth=2.5, alpha=0.8)

        

        # Set x-axis limits based on this plot's data only

        plot_datetimes_sorted = sorted(list(plot_datetimes))

        if plot_datetimes_sorted:

            x_min = min(plot_datetimes_sorted)

            x_max = max(plot_datetimes_sorted)

            x_range = x_max - x_min

            plt.xlim(x_min - 0.02 * x_range, x_max + 0.02 * x_range)

        

        # Format x-axis

        ax = plt.gca()

        if plot_datetimes_sorted:

            ax.xaxis.set_major_locator(HourLocator(interval=6))

            ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:00'))

            ax.xaxis.set_minor_locator(HourLocator(interval=1))

        

        plt.xlabel('Datetime', fontsize=33, fontweight='bold')

        plt.ylabel('Prediction Error (Predicted - Actual) (%)', fontsize=33, fontweight='bold')

        

        # Title indicates which part this is

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

        

        # Ensure output directory exists

        os.makedirs(os.path.dirname(plot_output_path) if os.path.dirname(plot_output_path) else '.', exist_ok=True)

        plt.savefig(plot_output_path, dpi=200, bbox_inches='tight')

        plt.close()

        print(f"  Overlay plot saved: {plot_output_path} (Predictions {start_idx + 1}-{end_idx})")

        output_paths.append(plot_output_path)

    

    return output_paths





def create_big_overlay_plot(all_predictions, output_path, model_name, resolution_name):

    """

    Create one big overlay plot showing all predictions (prediction - actual) differences at once.

    

    Args:

        all_predictions: List of (pred_df, pred_num, pred_datetime) tuples

        output_path: Path to save the plot

        model_name: Name of the model

        resolution_name: Resolution name (e.g., "Hourly", "30-minute", "15-minute")

    

    Returns:

        Output file path

    """

    num_predictions = len(all_predictions)

    

    plt.figure(figsize=(32, 14))

    

    # Set larger font sizes globally for this figure (50% increase)

    plt.rcParams.update({'font.size': 27})

    

    colors = plt.cm.tab20(np.linspace(0, 1, num_predictions))

    

    # Collect all datetimes

    all_datetimes = set()

    

    # Plot (prediction - actual) differences for all predictions

    for idx, (pred_df, pred_num, pred_datetime) in enumerate(all_predictions):

        # Calculate prediction - actual difference

        differences = pred_df['Predicted_Capacity_Factor'] - pred_df['Ground_Truth_Capacity_Factor']

        # Only plot where both values are valid

        valid_mask = ~(pred_df['Predicted_Capacity_Factor'].isna() | pred_df['Ground_Truth_Capacity_Factor'].isna())

        valid_datetimes = pred_df.loc[valid_mask, 'Datetime']

        valid_differences = differences.loc[valid_mask]

        

        plt.plot(valid_datetimes, valid_differences, 

                linewidth=3.0, alpha=0.85, color=colors[idx])

        all_datetimes.update(valid_datetimes)

    

    # Add horizontal line at y=0 for reference

    plt.axhline(y=0, color='black', linestyle='--', linewidth=3.0, alpha=0.9)

    

    # Set x-axis limits based on all data

    all_datetimes_sorted = sorted(list(all_datetimes))

    if all_datetimes_sorted:

        x_min = min(all_datetimes_sorted)

        x_max = max(all_datetimes_sorted)

        x_range = x_max - x_min

        plt.xlim(x_min - 0.02 * x_range, x_max + 0.02 * x_range)

    

    # Format x-axis

    ax = plt.gca()

    if all_datetimes_sorted:

        ax.xaxis.set_major_locator(HourLocator(interval=6))

        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:00'))

        ax.xaxis.set_minor_locator(HourLocator(interval=1))

    

    plt.xlabel('Datetime', fontsize=33, fontweight='bold')

    plt.ylabel('Prediction Error (Predicted - Actual) (%)', fontsize=33, fontweight='bold')

    

    title = f'All Prediction Errors Overlay - {model_name} ({resolution_name})\n'

    title += f'All {num_predictions} prediction intervals (24-hour ahead forecasts)'

    

    plt.title(title, fontsize=36, fontweight='bold')

    plt.grid(True, alpha=0.3, which='both')

    plt.xticks(rotation=45, ha='right', fontsize=24)

    plt.yticks(fontsize=24)

    plt.tight_layout()

    

    # Ensure output directory exists

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')

    plt.close()

    print(f"  Big overlay plot saved: {output_path} (All {num_predictions} predictions)")

    

    return output_path





def create_ground_truth_plot(all_predictions, output_path, model_name, resolution_name):

    """

    Create a separate plot showing ground truth values.

    

    Args:

        all_predictions: List of (pred_df, pred_num, pred_datetime) tuples

        output_path: Path to save the plot

        model_name: Name of the model

        resolution_name: Resolution name (e.g., "Hourly", "30-minute", "15-minute")

    

    Returns:

        Output file path

    """

    plt.figure(figsize=(24, 10))

    

    # Set larger font sizes globally for this figure (50% increase)

    plt.rcParams.update({'font.size': 27})

    

    # Collect ground truth (shared across all predictions)

    gt_dict = {}

    for pred_df, _, _ in all_predictions:

        if not pred_df['Ground_Truth_Capacity_Factor'].isna().all():

            for dt, gt_val in zip(pred_df['Datetime'], pred_df['Ground_Truth_Capacity_Factor']):

                if not pd.isna(gt_val):

                    if dt not in gt_dict or pd.isna(gt_dict[dt]):

                        gt_dict[dt] = gt_val

    

    if not gt_dict:

        print(f"  [WARNING] No ground truth data to plot for {resolution_name}")

        return None

    

    # Sort by datetime

    sorted_datetimes = sorted(gt_dict.keys())

    sorted_values = [gt_dict[dt] for dt in sorted_datetimes]

    

    # Plot ground truth

    plt.plot(sorted_datetimes, sorted_values, 

            linewidth=4.5, color='black', marker='o', 

            markersize=9, alpha=1.0)

    

    # Set x-axis limits

    if sorted_datetimes:

        x_min = min(sorted_datetimes)

        x_max = max(sorted_datetimes)

        x_range = x_max - x_min

        plt.xlim(x_min - 0.02 * x_range, x_max + 0.02 * x_range)

    

    # Format x-axis

    ax = plt.gca()

    if sorted_datetimes:

        ax.xaxis.set_major_locator(HourLocator(interval=6))

        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:00'))

        ax.xaxis.set_minor_locator(HourLocator(interval=1))

    

    plt.xlabel('Datetime', fontsize=33, fontweight='bold')

    plt.ylabel('Capacity Factor (%)', fontsize=33, fontweight='bold')

    plt.title(f'Ground Truth - {model_name} ({resolution_name})\n'

              f'Actual capacity factor values across all prediction intervals', 

              fontsize=36, fontweight='bold')

    plt.grid(True, alpha=0.3, which='both')

    plt.xticks(rotation=45, ha='right', fontsize=24)

    plt.yticks(fontsize=24)

    plt.tight_layout()

    

    # Ensure output directory exists

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')

    plt.close()

    print(f"  Ground truth plot saved: {output_path}")

    

    return output_path





def create_hourly_rmse_plot(hourly_rmse_data, output_path, model_name, resolution_name, y_axis_limits=None):

    """

    Create a connected scatter plot showing hourly RMSE values.

    

    Args:

        hourly_rmse_data: List of tuples (hour_datetime, rmse_value)

        output_path: Path to save the plot

        model_name: Name of the model

        resolution_name: Resolution name (e.g., "Hourly", "30-minute", "15-minute")

        y_axis_limits: Tuple (ymin, ymax) for y-axis limits. If None, auto-scales.

    """

    if len(hourly_rmse_data) == 0:

        print(f"  [WARNING] No hourly RMSE data to plot for {resolution_name}")

        return

    

    plt.figure(figsize=(14, 8))

    

    # Set larger font sizes globally for this figure (50% increase)

    plt.rcParams.update({'font.size': 27})

    

    # Sort by datetime

    hourly_rmse_data = sorted(hourly_rmse_data, key=lambda x: x[0])

    hours = [item[0] for item in hourly_rmse_data]

    rmse_values = [item[1] for item in hourly_rmse_data]

    

    # Create connected scatter plot

    plt.plot(hours, rmse_values, marker='o', linewidth=3.0, markersize=12, 

             color='steelblue', markerfacecolor='lightblue', markeredgecolor='darkblue', 

             markeredgewidth=2.25, alpha=0.9)

    

    # Set y-axis limits if provided

    if y_axis_limits is not None:

        plt.ylim(y_axis_limits[0], y_axis_limits[1])

    

    # Format x-axis

    ax = plt.gca()

    ax.xaxis.set_major_locator(HourLocator(interval=2))

    ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:00'))

    ax.xaxis.set_minor_locator(HourLocator(interval=1))

    

    plt.xlabel('Hour', fontsize=33, fontweight='bold')

    plt.ylabel('RMSE', fontsize=33, fontweight='bold')

    plt.title(f'Hourly RMSE Values - {model_name} ({resolution_name})\n'

              f'RMSE calculated for each hour across all predictions', 

              fontsize=36, fontweight='bold')

    plt.grid(True, alpha=0.3, which='both')

    plt.xticks(rotation=45, ha='right', fontsize=24)

    plt.yticks(fontsize=24)

    plt.tight_layout()

    

    # Ensure output directory exists

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')

    plt.close()

    print(f"  Hourly RMSE plot saved: {output_path}")





# =============================================================================

# MAIN PREDICTION FUNCTION

# =============================================================================

def run_predictions_at_resolution(data_path, config, output_dir, resolution_minutes, 

                                  test_intervals, plant_name=None):

    """

    Run predictions at a specific resolution.

    

    Args:

        resolution_minutes: Resolution in minutes (60, 30, 15, or 10)

        test_intervals: Number of intervals to test (e.g., 24 for hourly, 48 for 30-min, 96 for 15-min, 144 for 10-min)

    

    Returns:

        all_predictions: List of (pred_df, pred_num, pred_datetime) tuples

        hourly_rmse_data: List of (hour_datetime, rmse_value) tuples

    """

    print("=" * 80)

    resolution_name = f"{resolution_minutes}-minute" if resolution_minutes < 60 else "hourly"

    print(f"Running {resolution_name.upper()} Predictions")

    print("=" * 80)

    print(f"Data file: {data_path}")

    print(f"Model: {config['experiment_name']}")

    print(f"Resolution: {resolution_minutes} minutes")

    print(f"Test intervals: {test_intervals}")

    print(f"Output directory: {output_dir}")

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

    

    # create_sliding_windows expects "hours" but actually uses it as number of data points

    # So for 30-min data, we pass 48 "hours" to get 48 30-min intervals (24 hours)

    # For 15-min data, we pass 96 "hours" to get 96 15-min intervals (24 hours)

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

    

    print("\n[4/4] Making predictions...")

    

    if len(test_idx) == 0:

        raise ValueError("No test samples available.")

    

    if isinstance(test_idx, np.ndarray):

        test_idx_list = test_idx.tolist()

    else:

        test_idx_list = list(test_idx)

    

    # Map test sample indices to dataframe time indices (start of each prediction window)
    intervals_per_hour = 60 // resolution_minutes
    past_intervals = past_hours * intervals_per_hour
    intervals_per_day = 24 * intervals_per_hour  # e.g. 144 for 10-min

    # All valid test time indices (one per test sample, where we have room for future_intervals)
    all_valid = []
    for s in test_idx_list:
        time_idx = int(past_intervals + s)
        if time_idx >= 0 and time_idx < len(df_clean) - future_intervals:
            all_valid.append(time_idx)

    # Subsample: one prediction per day so we get 10 AM-6 PM RMSE for every date without running 100k predictions
    stride = max(1, intervals_per_day)
    test_time_indices = all_valid[::stride]

    if len(test_time_indices) == 0:
        print(f"  Warning: No valid test time indices (check data length and test split).")
    else:
        first_dt = df_clean.iloc[test_time_indices[0]]['Datetime']
        last_dt = df_clean.iloc[test_time_indices[-1]]['Datetime']
        print(f"  Predictions from {first_dt.strftime('%Y-%m-%d')} to {last_dt.strftime('%Y-%m-%d')} ({len(test_time_indices)} days)")

    

    print(f"  Making predictions for {len(test_time_indices)} intervals...")

    

    all_predictions = []

    all_pred_data = []  # For hourly RMSE calculation

    

    for pred_num, time_idx in enumerate(test_time_indices, 1):

        try:

            preds, gt, pred_datetime, future_dt = make_prediction_at_time(

                model, config, df_clean, hist_feats, fcst_feats, 

                scaler_hist, scaler_fcst, scaler_target, no_hist_power,

                time_idx, past_intervals, future_intervals, resolution_minutes

            )

            

            # Interpolate ground truth for smooth plotting

            gt_df = pd.DataFrame({

                'Datetime': future_dt,

                'Ground_Truth_Capacity_Factor': gt

            })

            gt_df = gt_df.set_index('Datetime')

            gt_df['Ground_Truth_Capacity_Factor'] = gt_df['Ground_Truth_Capacity_Factor'].interpolate(method='pchip', limit_direction='both')

            gt_df_interpolated = gt_df.reset_index()

            

            pred_df = pd.DataFrame({

                'Datetime': gt_df_interpolated['Datetime'],

                'Predicted_Capacity_Factor': preds[:len(gt_df_interpolated)],

                'Ground_Truth_Capacity_Factor': gt_df_interpolated['Ground_Truth_Capacity_Factor']

            })

            

            all_predictions.append((pred_df, pred_num, pred_datetime))

            

            # Store for hourly RMSE calculation

            for dt, pred_val, gt_val in zip(pred_df['Datetime'], pred_df['Predicted_Capacity_Factor'], pred_df['Ground_Truth_Capacity_Factor']):

                if not (np.isnan(pred_val) or np.isnan(gt_val)):

                    all_pred_data.append((dt, pred_val, gt_val))

            

            if pred_num % 10 == 0 or pred_num == len(test_time_indices):

                print(f"  [{pred_num}/{len(test_time_indices)}] Completed")

        

        except Exception as e:

            print(f"  [ERROR] Prediction {pred_num} failed: {str(e)}")

            import traceback

            traceback.print_exc()

            continue

    

    # Calculate hourly RMSE

    hourly_rmse_data = []

    if len(all_pred_data) > 0:

        pred_df_all = pd.DataFrame(all_pred_data, columns=['Datetime', 'Predicted', 'Ground_Truth'])

        pred_df_all['Hour'] = pred_df_all['Datetime'].dt.floor('H')

        

        # Group by hour and calculate RMSE for each hour

        for hour, group in pred_df_all.groupby('Hour'):

            if len(group) > 0:

                preds_hour = group['Predicted'].values

                gt_hour = group['Ground_Truth'].values

                valid_mask = ~(np.isnan(preds_hour) | np.isnan(gt_hour))

                if np.sum(valid_mask) > 0:

                    preds_valid = preds_hour[valid_mask]

                    gt_valid = gt_hour[valid_mask]

                    mse = np.mean((preds_valid - gt_valid) ** 2)

                    rmse = np.sqrt(mse)

                    hourly_rmse_data.append((hour, rmse))

        

        hourly_rmse_data = sorted(hourly_rmse_data, key=lambda x: x[0])

        print(f"  Calculated hourly RMSE for {len(hourly_rmse_data)} hours")

    

    return all_predictions, hourly_rmse_data, all_pred_data  # Also return all_pred_data for 10AM-6PM RMSE calculation





# =============================================================================

# MAIN FUNCTION

# =============================================================================

def run_multi_resolution_predictions(data_path, config, output_dir, plant_name=None):

    """

    Run predictions at 10-minute resolution only.

    Generate overlay plots and hourly RMSE plots.

    """

    # Ensure output directory exists

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput directory created: {output_dir}")

    

    resolutions = [

        (10, 144, "10-minute")   # 10 minutes, 144 intervals per 24h

    ]

    

    all_results = {}

    all_hourly_rmse_data = []  # Collect all RMSE data to determine common y-axis scale

    

    for resolution_minutes, test_intervals, resolution_name in resolutions:

        print(f"\n{'='*80}")

        print(f"PROCESSING {resolution_name.upper()} RESOLUTION")

        print(f"{'='*80}")

        

        try:

            all_predictions, hourly_rmse_data, all_pred_data = run_predictions_at_resolution(

                data_path, config, output_dir, resolution_minutes, test_intervals, plant_name

            )

            

            if len(all_predictions) > 0:

                # Create overlay plot(s) - may create multiple if more than 12 predictions

                overlay_path = os.path.join(output_dir, f"overlay_plot_{resolution_name.lower().replace('-', '_')}.png")

                big_overlay_path = os.path.join(output_dir, f"big_overlay_plot_{resolution_name.lower().replace('-', '_')}.png")

                ground_truth_path = os.path.join(output_dir, f"ground_truth_plot_{resolution_name.lower().replace('-', '_')}.png")

                model_name = config.get('experiment_name', 'Model')

                overlay_paths = create_overlay_plot(all_predictions, overlay_path, model_name, resolution_name, max_predictions_per_plot=12)

                big_overlay_plot_path = create_big_overlay_plot(all_predictions, big_overlay_path, model_name, resolution_name)

                ground_truth_plot_path = create_ground_truth_plot(all_predictions, ground_truth_path, model_name, resolution_name)

                

                # Store RMSE data for later (we'll create plots after determining common y-axis scale)

                all_results[resolution_name] = {

                    'predictions': all_predictions,

                    'hourly_rmse': hourly_rmse_data,

                    'overlay_paths': overlay_paths,

                    'big_overlay_path': big_overlay_plot_path,

                    'ground_truth_path': ground_truth_plot_path,

                    'all_pred_data': all_pred_data  # Store prediction data for 10AM-6PM RMSE

                }

                all_hourly_rmse_data.append((resolution_name, hourly_rmse_data))

                

                print(f"\n[SUCCESS] {resolution_name} resolution predictions completed")

                if len(overlay_paths) == 1:

                    print(f"  Overlay plot: {overlay_paths[0]}")

                else:

                    print(f"  Overlay plots ({len(overlay_paths)} parts):")

                    for path in overlay_paths:

                        print(f"    - {path}")

                if big_overlay_plot_path:

                    print(f"  Big overlay plot (all predictions): {big_overlay_plot_path}")

                if ground_truth_plot_path:

                    print(f"  Ground truth plot: {ground_truth_plot_path}")

            else:

                print(f"\n[WARNING] No predictions generated for {resolution_name} resolution")

        

        except Exception as e:

            print(f"\n[ERROR] Failed {resolution_name} resolution: {str(e)}")

            import traceback

            traceback.print_exc()

            continue

    

    # Calculate common y-axis limits for RMSE plots

    all_rmse_values = []

    for resolution_name, hourly_rmse_data in all_hourly_rmse_data:

        if len(hourly_rmse_data) > 0:

            rmse_vals = [item[1] for item in hourly_rmse_data]

            all_rmse_values.extend(rmse_vals)

    

    if len(all_rmse_values) > 0:

        y_min = min(all_rmse_values)

        y_max = max(all_rmse_values)

        # Add 5% padding

        y_range = y_max - y_min

        y_axis_limits = (max(0, y_min - 0.05 * y_range), y_max + 0.05 * y_range)

        print(f"\n{'='*80}")

        print("Creating hourly RMSE plots with common y-axis scale")

        print(f"Y-axis range: {y_axis_limits[0]:.4f} to {y_axis_limits[1]:.4f}")

        print(f"{'='*80}")

    else:

        y_axis_limits = None

        print(f"\n[WARNING] No RMSE data found for any resolution")

    

    # Create RMSE plots with common y-axis scale

    for resolution_name, hourly_rmse_data in all_hourly_rmse_data:

        if len(hourly_rmse_data) > 0:

            rmse_plot_path = os.path.join(output_dir, f"hourly_rmse_plot_{resolution_name.lower().replace('-', '_')}.png")

            model_name = config.get('experiment_name', 'Model')

            create_hourly_rmse_plot(hourly_rmse_data, rmse_plot_path, model_name, resolution_name, y_axis_limits=y_axis_limits)

            print(f"  Hourly RMSE plot saved: {rmse_plot_path}")

    

    # Calculate and save RMSE for 10 AM - 6 PM for every date that has prediction data

    print(f"\n{'='*80}")

    print("Calculating RMSE for 10 AM - 6 PM for each date (each resolution)")

    print(f"{'='*80}")

    

    rmse_results = []

    

    for resolution_name, resolution_data in all_results.items():

        all_pred_data = resolution_data.get('all_pred_data', [])

        

        if len(all_pred_data) == 0:

            print(f"  [WARNING] No prediction data for {resolution_name}")

            continue

        

        # Convert to DataFrame

        pred_df = pd.DataFrame(all_pred_data, columns=['Datetime', 'Predicted', 'Ground_Truth'])

        pred_df['Datetime'] = pd.to_datetime(pred_df['Datetime'])

        # Restrict to 10 AM - 6 PM window for all dates
        window = pred_df[
            (pred_df['Datetime'].dt.hour >= 10) &
            (pred_df['Datetime'].dt.hour <= 18)
        ].copy()
        window = window.drop_duplicates(subset=['Datetime'], keep='first')

        # Unique dates that have at least one point in 10 AM - 6 PM
        unique_dates = sorted(window['Datetime'].dt.normalize().unique())

        if len(unique_dates) == 0:
            print(f"  [WARNING] No 10 AM - 6 PM data for {resolution_name}")
            rmse_results.append({
                'Resolution': resolution_name,
                'Date': 'N/A',
                'RMSE_10AM_6PM': np.nan,
                'MAE_10AM_6PM': np.nan,
                'Number_of_Samples_10AM_6PM': 0
            })
            continue

        for day_date in unique_dates:
            date_start = pd.Timestamp(day_date).replace(hour=10, minute=0, second=0, microsecond=0)
            date_end = pd.Timestamp(day_date).replace(hour=18, minute=0, second=0, microsecond=0)

            filtered_df = window[
                (window['Datetime'] >= date_start) &
                (window['Datetime'] <= date_end)
            ].copy()

            valid_mask = ~(filtered_df['Predicted'].isna() | filtered_df['Ground_Truth'].isna())

            if valid_mask.sum() > 0:
                preds_valid = filtered_df.loc[valid_mask, 'Predicted'].values
                gt_valid = filtered_df.loc[valid_mask, 'Ground_Truth'].values

                preds_valid_abs = preds_valid / 100.0
                gt_valid_abs = gt_valid / 100.0

                mse = np.mean((preds_valid_abs - gt_valid_abs) ** 2)
                rmse_10am_6pm = np.sqrt(mse)
                mae_10am_6pm = np.mean(np.abs(preds_valid_abs - gt_valid_abs))
                n_samples = len(preds_valid)

                date_str = pd.Timestamp(day_date).strftime('%Y-%m-%d')
                print(f"  {resolution_name} {date_str}: RMSE = {rmse_10am_6pm:.4f}, MAE = {mae_10am_6pm:.4f}, Samples = {n_samples}")

                rmse_results.append({
                    'Resolution': resolution_name,
                    'Date': date_str,
                    'RMSE_10AM_6PM': rmse_10am_6pm,
                    'MAE_10AM_6PM': mae_10am_6pm,
                    'Number_of_Samples_10AM_6PM': n_samples
                })
            else:
                date_str = pd.Timestamp(day_date).strftime('%Y-%m-%d')
                rmse_results.append({
                    'Resolution': resolution_name,
                    'Date': date_str,
                    'RMSE_10AM_6PM': np.nan,
                    'MAE_10AM_6PM': np.nan,
                    'Number_of_Samples_10AM_6PM': 0
                })

    

    # Save RMSE results to CSV (always write so downstream scripts find the file)
    rmse_csv_path = os.path.join(output_dir, 'rmse_10am_6pm_by_resolution.csv')
    os.makedirs(os.path.dirname(rmse_csv_path) if os.path.dirname(rmse_csv_path) else '.', exist_ok=True)

    if len(rmse_results) > 0:
        rmse_df = pd.DataFrame(rmse_results)
        rmse_df.to_csv(rmse_csv_path, index=False)
        print(f"\n  RMSE CSV saved: {rmse_csv_path}")
        print(f"\n  RMSE Results Summary:")
        print(rmse_df.to_string(index=False))
    else:
        # No results (e.g. no predictions for June 20/21 or all failed) – write placeholder so file exists
        rmse_df = pd.DataFrame([{
            'Resolution': '10-minute',
            'Date': 'N/A',
            'RMSE_10AM_6PM': np.nan,
            'MAE_10AM_6PM': np.nan,
            'Number_of_Samples_10AM_6PM': 0
        }])
        rmse_df.to_csv(rmse_csv_path, index=False)
        print(f"\n  [WARNING] No RMSE results (no valid 10 AM–6 PM data); placeholder CSV saved: {rmse_csv_path}")

    

    # =============================================================================

    # CREATE SUMMARY TABLES

    # =============================================================================

    print(f"\n{'='*80}")

    print("Creating Summary Tables")

    print(f"{'='*80}")

    

    # Summary Table 1: Average Error % by Resolution

    error_summary = []

    

    for resolution_name, resolution_data in all_results.items():

        all_pred_data = resolution_data.get('all_pred_data', [])

        

        if len(all_pred_data) == 0:

            continue

        

        # Convert to DataFrame

        pred_df = pd.DataFrame(all_pred_data, columns=['Datetime', 'Predicted', 'Ground_Truth'])

        pred_df['Datetime'] = pd.to_datetime(pred_df['Datetime'])

        

        # Remove duplicates to avoid counting overlapping predictions multiple times

        pred_df = pred_df.drop_duplicates(subset=['Datetime'], keep='first')

        

        # Calculate absolute error in percentage points (matches what plots show)

        valid_mask = ~(pred_df['Predicted'].isna() | pred_df['Ground_Truth'].isna())

        if valid_mask.sum() > 0:

            preds_valid = pred_df.loc[valid_mask, 'Predicted'].values

            gt_valid = pred_df.loc[valid_mask, 'Ground_Truth'].values

            

            # Calculate absolute error in percentage points (not relative %)

            # This matches what the plots show: |Predicted - Ground_Truth| in percentage points

            # Since capacity factor is already 0-100%, this is the absolute difference

            abs_errors = np.abs(preds_valid - gt_valid)

            

            if len(abs_errors) > 0:

                avg_error_pct = np.mean(abs_errors)

                median_error_pct = np.median(abs_errors)

                std_error_pct = np.std(abs_errors)

                n_samples = len(abs_errors)

                

                error_summary.append({

                    'Resolution': resolution_name,

                    'Average_Error_Percentage_Points': avg_error_pct,

                    'Median_Error_Percentage_Points': median_error_pct,

                    'Std_Error_Percentage_Points': std_error_pct,

                    'Number_of_Samples': n_samples

                })

    

    # Save and display Average Error % summary

    if len(error_summary) > 0:

        error_df = pd.DataFrame(error_summary)

        error_csv_path = os.path.join(output_dir, 'average_error_percentage_points_by_resolution.csv')

        error_df.to_csv(error_csv_path, index=False)

        print(f"\n  Average Error (Percentage Points) Summary saved: {error_csv_path}")

        print(f"\n  Average Error (Percentage Points) by Resolution:")

        print(f"  Note: This shows absolute error in percentage points (matches plot values)")

        print(error_df.to_string(index=False))

    else:

        print(f"\n  [WARNING] No error data available for summary table")

    

    # Summary Table 2: Average RMSE by Resolution

    rmse_summary = []

    

    for resolution_name, hourly_rmse_data in all_hourly_rmse_data:

        if len(hourly_rmse_data) > 0:

            rmse_vals = [item[1] for item in hourly_rmse_data]

            avg_rmse = np.mean(rmse_vals)

            median_rmse = np.median(rmse_vals)

            std_rmse = np.std(rmse_vals)

            min_rmse = np.min(rmse_vals)

            max_rmse = np.max(rmse_vals)

            n_hours = len(rmse_vals)

            

            rmse_summary.append({

                'Resolution': resolution_name,

                'Average_RMSE': avg_rmse,

                'Median_RMSE': median_rmse,

                'Std_RMSE': std_rmse,

                'Min_RMSE': min_rmse,

                'Max_RMSE': max_rmse,

                'Number_of_Hours': n_hours

            })

    

    # Save and display Average RMSE summary

    if len(rmse_summary) > 0:

        rmse_summary_df = pd.DataFrame(rmse_summary)

        rmse_summary_csv_path = os.path.join(output_dir, 'average_rmse_by_resolution.csv')

        rmse_summary_df.to_csv(rmse_summary_csv_path, index=False)

        print(f"\n  Average RMSE Summary saved: {rmse_summary_csv_path}")

        print(f"\n  Average RMSE by Resolution:")

        print(rmse_summary_df.to_string(index=False))

    else:

        print(f"\n  [WARNING] No RMSE data available for summary table")

    

    print(f"\n{'='*80}")

    print("[SUCCESS] Multi-Resolution Predictions Completed!")

    print(f"Output directory: {output_dir}")

    print(f"{'='*80}")

    

    return all_results





# =============================================================================

# MAIN ENTRY

# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(

        description='Run 10-minute resolution predictions for Project1140',

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

                       help='Output directory for plots (default: ./multi_resolution_predictions_<model>_<scenario>)')

    

    args = parser.parse_args()

    

    print("\n" + "=" * 80)

    print("MODE: 10-Minute Resolution Predictions")

    print(f"Algorithm: {args.model} {args.complexity} {args.scenario}")

    print("=" * 80 + "\n")

    

    config = create_config_from_args(

        args.data_path, args.model, args.complexity, args.scenario,

        args.lookback, args.use_time_encoding

    )

    

    if args.output_dir is None:

        output_dir = os.path.join(script_dir, f"multi_resolution_predictions_{args.model}_{args.scenario}")

    else:

        output_dir = args.output_dir

    

    try:

        run_multi_resolution_predictions(

            args.data_path, config, output_dir

        )

    except Exception as e:

        print(f"\n[ERROR] Failed: {str(e)}")

        import traceback

        traceback.print_exc()

        sys.exit(1)
