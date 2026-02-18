#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10-Minute Resolution Predictions for 24 Hours of Test Data

For each 10-minute interval in 24 hours of testing data, predict the next 24 hours' capacity factor.
Each 10-minute interval's prediction is saved as a separate CSV file with plots.

Usage:
    python halfhourly_predictions_48h.py --data-path data/Project1140.csv --model LSTM --complexity high --scenario PV+NWP
    python halfhourly_predictions_48h.py --data-path data/Project1140.csv --model XGB --complexity high --scenario PV+NWP
    
Note: This script uses 10-minute resolution only. If data is hourly, it is interpolated to 10-minute using PCHIP.
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

# Google Sheets imports
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    print("Warning: gspread not available. Install with: pip install gspread google-auth")

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
        'shuffle_split': False,   # Sequential split for temporal evaluation
        'random_seed': 42         # Fixed seed for reproducibility
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

    # Find matching feature combo
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
        raise ValueError(f"Unknown scenario: {scenario}. Must be one of: PV, PV+HW, PV+NWP, PV+NWP+, NWP, NWP+")
    
    return create_config(data_path, model, complexity, lookback, feat_combo, use_time_encoding, is_nwp_only)


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================
def make_prediction_at_halfhour(model, config, df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, 
                                scaler_target, no_hist_power, hour_idx, past_hours, future_hours):
    """
    Make a prediction for the next 24 hours starting from a specific index.
    
    Args:
        model: Trained model
        config: Model configuration
        df_clean: Preprocessed dataframe
        hist_feats: Historical feature columns
        fcst_feats: Forecast feature columns
        scaler_hist: Historical features scaler
        scaler_fcst: Forecast features scaler
        scaler_target: Target scaler
        no_hist_power: Whether to use historical power
        hour_idx: Index in df_clean to start prediction from
        past_hours: Lookback window size (in hours)
        future_hours: Number of hours to predict (24)
    
    Returns:
        predictions: Array of predicted capacity factors for next 24 hours
        ground_truth: Array of actual capacity factors for next 24 hours
        prediction_datetime: Datetime of the prediction
        future_datetimes: Array of datetimes for the 24 predicted hours
    """
    import torch
    
    # Ensure hour_idx is a scalar integer
    hour_idx = int(hour_idx)
    past_hours = int(past_hours)
    future_hours = int(future_hours)
    
    # Get historical data (past_hours before hour_idx)
    # Data is now always 10-minute resolution after interpolation
    # Convert hours to 10-minute intervals: past_hours * 6
    past_halfhours = past_hours * 6
    hist_start = max(0, hour_idx - past_halfhours)
    hist_end = hour_idx
    hist_data = df_clean.iloc[hist_start:hist_end].copy()
    
    # Get future data (24 hours starting from hour_idx)
    # Convert hours to 10-minute intervals: future_hours * 6
    future_halfhours = future_hours * 6
    fut_start = hour_idx
    fut_end = min(len(df_clean), hour_idx + future_halfhours)
    fut_data = df_clean.iloc[fut_start:fut_end].copy()
    
    # Check if we have enough data
    hist_feats_list = list(hist_feats) if hist_feats else []
    n_hist_feats = len(hist_feats_list)
    
    if len(hist_data) < past_halfhours and not no_hist_power:
        # Pad with zeros if needed
        if len(hist_data) > 0 and n_hist_feats > 0:
            padding = np.zeros((int(past_halfhours - len(hist_data)), n_hist_feats))
            hist_array = np.vstack([padding, hist_data[hist_feats_list].values])
        else:
            hist_array = np.zeros((int(past_halfhours), n_hist_feats))
    elif no_hist_power:
        hist_array = np.zeros((int(past_halfhours) if past_halfhours > 0 else 1, n_hist_feats))
    else:
        if n_hist_feats > 0:
            hist_array = hist_data[hist_feats_list].values
        else:
            hist_array = np.zeros((len(hist_data), 0))
    
    # Get forecast features for future period
    fcst_feats_list = list(fcst_feats) if fcst_feats else []
    n_fcst_feats = len(fcst_feats_list)
    
    if fcst_feats_list and n_fcst_feats > 0:
        if len(fut_data) < future_halfhours:
            # Pad with last available values
            if len(fut_data) > 0:
                last_row = fut_data[fcst_feats_list].iloc[-1:].values
            else:
                last_row = np.zeros((1, n_fcst_feats))
            padding = np.tile(last_row, (int(future_halfhours - len(fut_data)), 1))
            if len(fut_data) > 0:
                fcst_array = np.vstack([fut_data[fcst_feats_list].values, padding])
            else:
                fcst_array = padding
        else:
            fcst_array = fut_data[fcst_feats_list].values[:int(future_halfhours)]
    else:
        fcst_array = None
    
    # Get ground truth
    future_halfhours_int = int(future_halfhours)
    if len(fut_data) < future_halfhours_int:
        # Pad with NaN if not enough data
        gt = np.full(future_halfhours_int, np.nan)
        if len(fut_data) > 0:
            gt[:len(fut_data)] = fut_data['Capacity Factor'].values
    else:
        gt = fut_data['Capacity Factor'].values[:future_halfhours_int]
    
    # Prepare input for model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reshape for model input
    # After interpolation, data is 10-minute resolution
    # The model was trained with past_halfhours and future_halfhours timesteps (since we passed those to create_sliding_windows)
    # So we use the actual number of timesteps we have
    past_timesteps = hist_array.shape[0] if len(hist_array.shape) > 0 else past_halfhours
    future_timesteps = fcst_array.shape[0] if fcst_array is not None and len(fcst_array.shape) > 0 else future_halfhours
    
    X_hist = hist_array.reshape(1, past_timesteps, -1)
    if fcst_array is not None:
        X_fcst = fcst_array.reshape(1, future_timesteps, -1)
    else:
        X_fcst = None
    
    # Get prediction datetime
    prediction_datetime = df_clean.iloc[hour_idx]['Datetime']
    
    # Make prediction
    if config['model'] in ['LSTM', 'GRU', 'Transformer', 'TCN']:
        # Deep learning model
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
        # Machine learning model
        # Flatten features
        if X_fcst is not None:
            X_flat = np.concatenate([X_hist.reshape(1, -1), X_fcst.reshape(1, -1)], axis=1)
        else:
            X_flat = X_hist.reshape(1, -1)
        
        preds_np = model.predict(X_flat).flatten()
    
    # Inverse transform predictions
    if scaler_target is not None:
        preds_inv = scaler_target.inverse_transform(preds_np.reshape(-1, 1)).flatten()
        gt_inv = scaler_target.inverse_transform(gt.reshape(-1, 1)).flatten() if not np.isnan(gt).all() else gt
    else:
        preds_inv = preds_np
        gt_inv = gt
    
    # Clip predictions to reasonable range [0, 100]
    preds_inv = np.clip(preds_inv, 0, 100)
    
    # Create future datetimes (24 hours = 144 10-minute intervals)
    future_halfhours_int = int(future_halfhours)
    future_datetimes = pd.date_range(start=prediction_datetime, periods=future_halfhours_int, freq='10T')
    
    return preds_inv, gt_inv, prediction_datetime, future_datetimes


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def save_to_google_sheets(df, sheet_name, credentials_path=None, spreadsheet_name=None):
    """
    Save DataFrame to Google Sheets.
    
    Args:
        df: DataFrame to save
        sheet_name: Name of the sheet
        credentials_path: Path to Google service account JSON file
        spreadsheet_name: Name of the Google Spreadsheet (will be created if doesn't exist)
    
    Returns:
        URL of the created/updated sheet
    """
    if not GSPREAD_AVAILABLE:
        raise ImportError("gspread is required. Install with: pip install gspread google-auth")
    
    if credentials_path is None:
        # Try to find credentials in common locations
        possible_paths = [
            'credentials.json',
            'service_account.json',
            os.path.expanduser('~/.config/gspread/service_account.json')
        ]
        credentials_path = None
        for path in possible_paths:
            if os.path.exists(path):
                credentials_path = path
                break
        
        if credentials_path is None:
            raise FileNotFoundError(
                "Google Sheets credentials not found. Please provide --credentials-path or "
                "place credentials.json in the current directory."
            )
    
    # Authenticate
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_file(credentials_path, scopes=scope)
    client = gspread.authorize(creds)
    
    # Get or create spreadsheet
    if spreadsheet_name is None:
        spreadsheet_name = "PV_Forecasting_Predictions"
    
    try:
        spreadsheet = client.open(spreadsheet_name)
    except gspread.SpreadsheetNotFound:
        spreadsheet = client.create(spreadsheet_name)
    
    # Create or get worksheet
    try:
        worksheet = spreadsheet.worksheet(sheet_name)
        # Clear existing data
        worksheet.clear()
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=20)
    
    # Update with new data
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())
    
    return spreadsheet.url


def create_prediction_plot(pred_df, output_path, halfhour_num, pred_datetime):
    """
    Create a plot showing predicted vs actual capacity factor.
    
    Args:
        pred_df: DataFrame with Datetime, Predicted_Capacity_Factor, Ground_Truth_Capacity_Factor
        output_path: Path to save the plot
        halfhour_num: Half-hour number for title
        pred_datetime: Datetime of the prediction
    """
    plt.figure(figsize=(12, 6))
    
    # Plot predictions and ground truth
    plt.plot(pred_df['Datetime'], pred_df['Predicted_Capacity_Factor'], 
             label='Predicted', marker='o', linewidth=2, markersize=4)
    
    # Only plot ground truth if available (not all NaN)
    if not pred_df['Ground_Truth_Capacity_Factor'].isna().all():
        plt.plot(pred_df['Datetime'], pred_df['Ground_Truth_Capacity_Factor'], 
                 label='Ground Truth', marker='s', linewidth=2, markersize=4, alpha=0.7)
    
    plt.xlabel('Datetime', fontsize=12)
    plt.ylabel('Capacity Factor (%)', fontsize=12)
    plt.title(f'Half-Hour {halfhour_num}: 24-Hour Capacity Factor Prediction\n'
              f'Prediction made at: {pred_datetime.strftime("%Y-%m-%d %H:%M")}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_overlay_plot(all_predictions, output_path, model_name):
    """
    Create an overlay plot showing all predictions and ground truth for a single plant.
    
    Args:
        all_predictions: List of tuples (pred_df, halfhour_num, pred_datetime)
        output_path: Path to save the plot
        model_name: Name of the model for the title
    """
    # Increase figure size to stretch out x-axis for better visibility
    plt.figure(figsize=(24, 10))
    
    # Collect all unique datetimes and create a comprehensive timeline
    all_datetimes = set()
    for pred_df, _, _ in all_predictions:
        all_datetimes.update(pred_df['Datetime'])
    
    all_datetimes = sorted(list(all_datetimes))
    
    # Plot each half-hour's prediction with different colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_predictions)))
    
    for idx, (pred_df, halfhour_num, pred_datetime) in enumerate(all_predictions):
        # Plot predictions with transparency
        plt.plot(pred_df['Datetime'], pred_df['Predicted_Capacity_Factor'], 
                label=f'HalfHour {halfhour_num} ({pred_datetime.strftime("%m-%d %H:%M")})',
                linewidth=1.5, alpha=0.6, color=colors[idx])
    
    # Collect and plot ground truth (combine all ground truth values)
    gt_dict = {}  # datetime -> capacity_factor
    for pred_df, _, _ in all_predictions:
        if not pred_df['Ground_Truth_Capacity_Factor'].isna().all():
            for dt, gt_val in zip(pred_df['Datetime'], pred_df['Ground_Truth_Capacity_Factor']):
                if not pd.isna(gt_val):
                    if dt not in gt_dict or pd.isna(gt_dict[dt]):
                        gt_dict[dt] = gt_val
    
    # Plot ground truth if available
    if gt_dict:
        gt_datetimes = sorted(gt_dict.keys())
        gt_values = [gt_dict[dt] for dt in gt_datetimes]
        plt.plot(gt_datetimes, gt_values, 
                label='Ground Truth', linewidth=3, color='black', marker='o', 
                markersize=6, alpha=0.9, zorder=100)
    
    # Set x-axis limits to span all datetimes with some padding
    if all_datetimes:
        x_min = min(all_datetimes)
        x_max = max(all_datetimes)
        x_range = x_max - x_min
        plt.xlim(x_min - 0.02 * x_range, x_max + 0.02 * x_range)
    
    # Format x-axis with more frequent ticks for better readability
    ax = plt.gca()
    if all_datetimes:
        # Set major ticks every 6 hours
        ax.xaxis.set_major_locator(HourLocator(interval=6))
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:00'))
        # Set minor ticks every hour
        ax.xaxis.set_minor_locator(HourLocator(interval=1))
    
    plt.xlabel('Datetime', fontsize=14, fontweight='bold')
    plt.ylabel('Capacity Factor (%)', fontsize=14, fontweight='bold')
    plt.title(f'All Predictions Overlay - {model_name}\n'
              f'{len(all_predictions)} half-hour intervals (24-hour ahead forecasts)', 
              fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
    plt.grid(True, alpha=0.3, which='both')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Overlay plot saved: {output_path}")


def run_halfhourly_predictions(data_path, config, output_dir, test_halfhours=144, 
                               credentials_path=None, spreadsheet_name=None, save_plots=True, 
                               plant_name=None, return_predictions=False):
    """
    Run 10-minute resolution predictions for test_halfhours consecutive 10-minute intervals from test data.
    Each 10-minute interval's prediction (next 24 hours) is saved as a separate CSV file with plot.
    
    Args:
        data_path: Path to data CSV file
        config: Model configuration
        output_dir: Directory to save prediction plots
        test_halfhours: Number of consecutive 10-minute intervals from test data to use (default: 144 = 24 hours)
        credentials_path: Path to Google service account JSON file
        spreadsheet_name: Name of the Google Spreadsheet
        save_plots: Whether to save plots (default: True)
        plant_name: Name of the plant (for labeling)
        return_predictions: If True, return all_predictions list for overlay plotting
    
    Returns:
        all_predictions list if return_predictions=True, else None
    """
    # Ensure test_halfhours is a Python int
    if isinstance(test_halfhours, np.ndarray):
        if test_halfhours.size == 1:
            test_halfhours = int(test_halfhours.item())
        else:
            raise ValueError(f"test_halfhours must be a scalar, got array of size {test_halfhours.size}")
    else:
        test_halfhours = int(test_halfhours)
    
    print("=" * 80)
    print("10-Minute Resolution Predictions for Test Data")
    print("=" * 80)
    print(f"Data file: {data_path}")
    print(f"Model: {config['experiment_name']}")
    print(f"Test intervals: {test_halfhours} (={test_halfhours*10/60:.1f} hours at 10-min resolution)")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Load and preprocess data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Check if Minute column exists (10-minute or other sub-hourly resolution data)
    has_minute_column = 'Minute' in df.columns
    
    if has_minute_column:
        df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        print("  Detected 10-minute resolution data (Minute column found)")
    else:
        df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
        print("  Detected hourly resolution data (no Minute column)")
        print("  Interpolating to 10-minute resolution...")
        
        # Set Datetime as index for resampling
        df_indexed = df.set_index('Datetime')
        
        # Resample to 10-minute frequency and interpolate using PCHIP
        df_resampled = df_indexed.resample('10T').asfreq()
        
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
        
        # Forward fill non-numeric columns (like Year, Month, Day, Hour)
        non_numeric_cols = df_resampled.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            if col not in ['Datetime']:
                df_resampled[col] = df_resampled[col].ffill()
        
        # Reset index to get Datetime back as a column
        df = df_resampled.reset_index()
        
        # Update Year, Month, Day, Hour, Minute columns from the new Datetime
        df['Year'] = df['Datetime'].dt.year
        df['Month'] = df['Datetime'].dt.month
        df['Day'] = df['Datetime'].dt.day
        df['Hour'] = df['Datetime'].dt.hour
        df['Minute'] = df['Datetime'].dt.minute
        
        # Now treat as 10-minute data
        has_minute_column = True
        print(f"  Interpolated from {len(df_indexed)} hourly points to {len(df)} 10-minute points")
    
    print("\n[1/4] Preprocessing data...")
    df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = preprocess_features(df, config)
    
    # Create sliding windows for train/val/test split
    print("\n[2/4] Creating sliding windows and splitting data...")
    # After interpolation, data is 10-minute resolution
    # Convert hours to 10-minute intervals for sliding windows
    past_hours = int(config.get('past_hours', 24))
    future_hours = int(config.get('future_hours', 24))
    past_halfhours = past_hours * 6  # 24 hours = 144 10-minute intervals
    future_halfhours = future_hours * 6  # 24 hours = 144 10-minute intervals
    
    # Create sliding windows using 10-minute intervals
    # create_sliding_windows uses the interval counts as window size in dataframe rows
    
    X_hist, X_fcst, y, hours, dates = create_sliding_windows(
        df_clean, past_halfhours, future_halfhours, hist_feats, fcst_feats, no_hist_power
    )
    
    total_samples = len(X_hist)
    indices = np.arange(total_samples)
    
    # Sequential split (no shuffle)
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
    
    # Prepare training data
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
    
    # Train model
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
    
    print("\n[4/4] Making 10-minute resolution predictions...")
    
    # Find the start date: June 20th at 00:00
    if len(test_idx) == 0:
        raise ValueError("No test samples available. Check data split ratios.")
    
    if isinstance(test_idx, np.ndarray):
        test_idx_list = test_idx.tolist()
    else:
        test_idx_list = list(test_idx)
    
    first_test_sample_idx = int(test_idx_list[0])
    first_test_start_in_df = int(past_halfhours) + first_test_sample_idx
    
    first_test_datetime = df_clean.iloc[first_test_start_in_df]['Datetime']
    target_year = first_test_datetime.year
    
    # Find June 20th at 00:00 in the dataframe (data is now always 10-minute after interpolation)
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
    
    # After interpolation, data is always 10-minute resolution
    # Each index represents a 10-minute interval
    future_hours_int = int(future_hours)
    # 24 hours = 144 10-minute intervals
    future_halfhours = future_hours_int * 6
    
    test_halfhour_indices = []
    
    for i in range(test_halfhours):
        # 10-minute data: increment by 1 index per 10-minute interval
        halfhour_idx = int(start_idx + i)
        
        # Make sure we have enough data after this index for prediction
        # Need enough data for the lookback window and future predictions
        if halfhour_idx >= 0 and halfhour_idx < len(df_clean) - future_halfhours:
            test_halfhour_indices.append(halfhour_idx)
        else:
            break
    
    if len(test_halfhour_indices) < test_halfhours:
        print(f"  Warning: Only found {len(test_halfhour_indices)} valid 10-minute intervals (requested {test_halfhours})")
        print(f"  Using available intervals: {len(test_halfhour_indices)}")
    
    print(f"  Making predictions for {len(test_halfhour_indices)} 10-minute intervals...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all predictions for overlay plot
    all_predictions = []
    
    # Make prediction for each half-hour
    for halfhour_num, halfhour_idx in enumerate(test_halfhour_indices, 1):
        try:
            preds, gt, pred_datetime, future_dt = make_prediction_at_halfhour(
                model, config, df_clean, hist_feats, fcst_feats, 
                scaler_hist, scaler_fcst, scaler_target, no_hist_power,
                halfhour_idx, past_hours, future_hours
            )
            
            # Create DataFrame for this half-hour's prediction
            pred_df = pd.DataFrame({
                'Datetime': future_dt,
                'Predicted_Capacity_Factor': preds,
                'Ground_Truth_Capacity_Factor': gt
            })
            
            # Format datetime for display (data is now always 10-minute after interpolation)
            timestamp_str = pred_datetime.strftime('%Y-%m-%d_%H%M')
            
            sheet_name = f"HalfHour_{halfhour_num:03d}_{timestamp_str}"
            
            # Save to CSV (always save to CSV for 10-minute predictions)
            output_file = os.path.join(output_dir, f"predictions_halfhour_{halfhour_num:03d}_{timestamp_str}.csv")
            pred_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # Try to save to Google Sheets as well
            try:
                sheet_url = save_to_google_sheets(
                    pred_df, sheet_name, credentials_path, spreadsheet_name
                )
                if halfhour_num == 1:
                    print(f"  Google Spreadsheet: {sheet_url}")
            except Exception as e:
                # Silently fail for Google Sheets, CSV is the primary output
                pass
            
            # Store for overlay plot
            all_predictions.append((pred_df, halfhour_num, pred_datetime))
            
            # Create and save plot
            if save_plots:
                plot_file = os.path.join(output_dir, f"plot_halfhour_{halfhour_num:03d}_{timestamp_str}.png")
                create_prediction_plot(pred_df, plot_file, halfhour_num, pred_datetime)
            
            if halfhour_num % 10 == 0 or halfhour_num == len(test_halfhour_indices):
                print(f"  [{halfhour_num}/{len(test_halfhour_indices)}] Completed: {sheet_name}")
        
        except Exception as e:
            print(f"  [ERROR] Half-hour {halfhour_num} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create overlay plot with all predictions
    if save_plots and len(all_predictions) > 0:
        overlay_plot_path = os.path.join(output_dir, "all_predictions_overlay.png")
        model_name = config.get('experiment_name', 'Model')
        create_overlay_plot(all_predictions, overlay_plot_path, model_name)
    
    print(f"\n{'='*80}")
    print(f"[SUCCESS] Completed predictions for {len(test_halfhour_indices)} 10-minute intervals")
    if save_plots:
        print(f"Individual plots saved to: {output_dir}")
        if len(all_predictions) > 0:
            print(f"Overlay plot saved to: {os.path.join(output_dir, 'all_predictions_overlay.png')}")
    print(f"{'='*80}")
    
    if return_predictions:
        return all_predictions
    return None


# =============================================================================
# MAIN ENTRY
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Make 10-minute resolution predictions for test data. Each 10-minute interval predicts next 24 hours.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model (LSTM high complexity, PV+NWP scenario) on single plant
  python halfhourly_predictions_48h.py --data-path data/Project1140.csv
  
  # Specify model and scenario on single plant
  python halfhourly_predictions_48h.py --data-path data/Project1140.csv --model LSTM --complexity high --scenario PV+NWP
  
  # Use different model on single plant
  python halfhourly_predictions_48h.py --data-path data/Project1140.csv --model XGB --complexity high --scenario PV+NWP
  
  # Use different number of test intervals (default: 144 = 24 hours at 10-min resolution)
  python halfhourly_predictions_48h.py --data-path data/Project1140.csv --test-halfhours 144
        """
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
    parser.add_argument('--test-halfhours', type=int, default=144,
                       help='Number of consecutive 10-minute intervals from test data to use (default: 144 = 24 hours)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for prediction plots (default: ./halfhourly_predictions_<model>_<scenario>)')
    parser.add_argument('--credentials-path', type=str, default=None,
                       help='Path to Google service account JSON file (default: looks for credentials.json)')
    parser.add_argument('--spreadsheet-name', type=str, default=None,
                       help='Name of Google Spreadsheet (default: PV_Forecasting_Predictions)')
    parser.add_argument('--no-plots', dest='save_plots', action='store_false', default=True,
                       help='Disable saving plots')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("MODE: 10-minute resolution predictions")
    print(f"Algorithm: {args.model} {args.complexity} {args.scenario}")
    print("=" * 80 + "\n")
    
    # Create config
    config = create_config_from_args(
        args.data_path, args.model, args.complexity, args.scenario,
        args.lookback, args.use_time_encoding
    )
    
    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join(script_dir, f"halfhourly_predictions_{args.model}_{args.scenario}")
    else:
        output_dir = args.output_dir
    
    # Run predictions
    try:
        run_halfhourly_predictions(
            args.data_path, config, output_dir, args.test_halfhours,
            credentials_path=args.credentials_path,
            spreadsheet_name=args.spreadsheet_name,
            save_plots=args.save_plots
        )
    except Exception as e:
        print(f"\n[ERROR] Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

