#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run all 284 experiments for the first dataset (Project1140.csv) and create
individual CSV files for each experiment with 48-hour test period predictions.

Each file contains:
- Datetime: Hourly timestamps for 48-hour period
- Ground_Truth: Actual capacity factor values
- Predicted: Model predicted capacity factor values
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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
# CONFIG GENERATION (same as run_main_experiments.py)
# =============================================================================
def generate_all_configs():
    """
    Generate all experiment configurations
    Total: 284 (DL=160 + ML=120 + Linear=4)
    """
    configs = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "Project1140.csv")

    dl_models = ['LSTM', 'GRU', 'Transformer', 'TCN']
    ml_models = ['RF', 'XGB', 'LGBM']
    complexities = ['low', 'high']
    lookbacks = [24, 72]
    te_options = [True, False]

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

    # === 1. DL models: PV-based experiments ===
    for model in dl_models:
        for complexity in complexities:
            for lookback in lookbacks:
                for feat_combo in feature_combos_pv:
                    for use_te in te_options:
                        configs.append(create_config(data_path, model, complexity, lookback, feat_combo, use_te, False))

    # === 2. DL models: NWP-only experiments ===
    for model in dl_models:
        for complexity in complexities:
            for feat_combo in feature_combos_nwp:
                for use_te in te_options:
                    configs.append(create_config(data_path, model, complexity, 0, feat_combo, use_te, True))

    # === 3. ML models ===
    for model in ml_models:
        for complexity in complexities:
            for lookback in lookbacks:
                for feat_combo in feature_combos_pv:
                    for use_te in te_options:
                        configs.append(create_config(data_path, model, complexity, lookback, feat_combo, use_te, False))
            for feat_combo in feature_combos_nwp:
                for use_te in te_options:
                    configs.append(create_config(data_path, model, complexity, 0, feat_combo, use_te, True))

    # === 4. Linear model ===
    for feat_combo in feature_combos_nwp:
        for use_te in te_options:
            configs.append(create_config(data_path, 'Linear', None, 0, feat_combo, use_te, True))

    print(f"\nConfiguration summary:")
    print(f"  DL models: {4 * 2 * (16 + 4)} experiments = 160")
    print(f"  ML models: {3 * 2 * (16 + 4)} experiments = 120")
    print(f"  Linear model: 4 experiments")
    print(f"  Total: {len(configs)} experiments")
    return configs


def create_config(data_path, model, complexity, lookback, feat_combo, use_te, is_nwp_only):
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


# =============================================================================
# EXTRACT 48-HOUR PERIOD FROM TEST DATA
# =============================================================================
def extract_48hour_period(predictions_all, y_true_all, test_dates, df_clean, future_hours=24):
    """
    Extract a 48-hour period from test data and create datetime index.
    
    In sliding windows:
    - Each sample at position i in df_clean predicts hours i to i+23
    - test_dates[i] contains the LAST datetime of the prediction window (i+23)
    - So the prediction starts at test_dates[i] - 23 hours
    
    Args:
        predictions_all: (n_samples, 24) - full predictions for all test samples
        y_true_all: (n_samples, 24) - full ground truth for all test samples
        test_dates: List of dates corresponding to test samples (last hour of prediction window)
        df_clean: Cleaned dataframe with Datetime column
        future_hours: Number of hours predicted per sample (default 24)
    
    Returns:
        result_df: DataFrame with columns [Datetime, Ground_Truth, Predicted]
    """
    if len(test_dates) == 0:
        raise ValueError("No test dates available")
    
    # Convert test_dates to datetime if needed
    test_dates_dt = []
    for d in test_dates:
        if isinstance(d, str):
            test_dates_dt.append(pd.to_datetime(d))
        elif hasattr(d, 'to_pydatetime'):
            test_dates_dt.append(d)
        elif hasattr(d, 'strftime'):
            test_dates_dt.append(pd.to_datetime(d))
        else:
            test_dates_dt.append(pd.to_datetime(str(d)))
    
    # The test_dates represent the LAST hour of each 24-hour prediction window
    # So for the first test sample, the prediction starts at test_dates[0] - 23 hours
    # We'll extract 48 hours starting from the first sample
    
    if len(predictions_all) < 2:
        # If we only have one sample, use 24 hours
        first_end_date = test_dates_dt[0]
        start_datetime = first_end_date - pd.Timedelta(hours=future_hours-1)
        hours_to_extract = min(future_hours, predictions_all.shape[1])
        
        # Extract from first sample
        preds = predictions_all[0, :hours_to_extract]
        gt = y_true_all[0, :hours_to_extract]
    else:
        # Use first 2 samples for 48 hours
        # First sample: predictions from test_dates[0] - 23h to test_dates[0]
        first_end_date = test_dates_dt[0]
        first_start_date = first_end_date - pd.Timedelta(hours=future_hours-1)
        
        # Second sample: predictions from test_dates[1] - 23h to test_dates[1]
        second_end_date = test_dates_dt[1]
        second_start_date = second_end_date - pd.Timedelta(hours=future_hours-1)
        
        # Check if samples are consecutive (second should start right after first ends)
        # If they overlap or have gaps, we'll use the first sample's start as reference
        start_datetime = first_start_date
        
        # Extract first 24 hours from first sample
        preds_1 = predictions_all[0, :future_hours]
        gt_1 = y_true_all[0, :future_hours]
        
        # Extract next 24 hours from second sample
        preds_2 = predictions_all[1, :future_hours]
        gt_2 = y_true_all[1, :future_hours]
        
        # Concatenate
        preds = np.concatenate([preds_1, preds_2])
        gt = np.concatenate([gt_1, gt_2])
        hours_to_extract = 48
    
    # Create hourly datetime sequence
    datetimes = pd.date_range(start=start_datetime, periods=len(preds), freq='H')
    
    # Create DataFrame
    result_df = pd.DataFrame({
        'Datetime': datetimes,
        'Ground_Truth': gt,
        'Predicted': preds
    })
    
    return result_df


# =============================================================================
# MAIN LOOP
# =============================================================================
def run_all_experiments_with_predictions(output_dir=None):
    """
    Run all experiments and save individual prediction files for 48-hour period
    """
    print("=" * 80)
    print("PV Forecasting: Running 284 Experiments (First Dataset Only)")
    print("Creating 48-hour prediction files for each experiment")
    print("=" * 80)

    all_configs = generate_all_configs()
    print(f"Total configurations generated: {len(all_configs)}")

    import torch
    data_path = os.path.join(script_dir, "data", "Project1140.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(script_dir, "predictions_48h")
    else:
        output_dir = os.path.join(output_dir, "predictions_48h")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # === main loop ===
    for idx, config in enumerate(all_configs, 1):
        exp_name = config['experiment_name']
        
        # Check if prediction file already exists
        prediction_file = os.path.join(output_dir, f"{exp_name}_predictions_48h.csv")
        if os.path.exists(prediction_file):
            print(f"[{idx}/{len(all_configs)}] SKIP: {exp_name} (prediction file already exists)")
            continue

        print(f"\n{'='*80}")
        print(f"Experiment {idx}/{len(all_configs)}: {exp_name}")
        print(f"{'='*80}")

        try:
            start_time = time.time()
            df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = preprocess_features(df, config)

            # Use 24-hour sliding windows
            past_hours = config.get('past_hours', 24)
            X_hist, X_fcst, y, hours, dates = create_sliding_windows(
                df_clean, past_hours, config['future_hours'], hist_feats, fcst_feats, no_hist_power
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

            X_hist_train, y_train = X_hist[train_idx], y[train_idx]
            X_hist_val, y_val = X_hist[val_idx], y[val_idx]
            X_hist_test, y_test = X_hist[test_idx], y[test_idx]

            if X_fcst is not None:
                X_fcst_train, X_fcst_val, X_fcst_test = X_fcst[train_idx], X_fcst[val_idx], X_fcst[test_idx]
            else:
                X_fcst_train = X_fcst_val = X_fcst_test = None

            # Split hours and dates
            train_hours = np.array([hours[i] for i in train_idx])
            val_hours = np.array([hours[i] for i in val_idx])
            test_hours = np.array([hours[i] for i in test_idx])
            test_dates = [dates[i] for i in test_idx]

            train_data = (X_hist_train, X_fcst_train, y_train, train_hours, [])
            val_data = (X_hist_val, X_fcst_val, y_val, val_hours, [])
            test_data = (X_hist_test, X_fcst_test, y_test, test_hours, test_dates)
            scalers = (scaler_hist, scaler_fcst, scaler_target)

            # Train model
            if config['model'] in ['LSTM', 'GRU', 'Transformer', 'TCN']:
                model, metrics = train_dl_model(config, train_data, val_data, test_data, scalers)
            else:
                model, metrics = train_ml_model(config, train_data, val_data, test_data, scalers)

            training_time = time.time() - start_time
            
            print(f"  [OK] MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
            
            # Extract 48-hour period predictions
            predictions_all = metrics.get('predictions_all', None)
            y_true_all = metrics.get('y_true_all', None)
            
            if predictions_all is None or y_true_all is None:
                print(f"  [WARNING] No predictions_all or y_true_all in metrics, using predictions/y_true")
                predictions_all = metrics.get('predictions', None)
                y_true_all = metrics.get('y_true', None)
                
                # Reshape if needed (1D to 2D)
                if predictions_all is not None and len(predictions_all.shape) == 1:
                    # Need to reshape - but we don't know the original shape
                    # Try to infer from test data
                    n_test = len(test_idx)
                    future_hours = config.get('future_hours', 24)
                    try:
                        predictions_all = predictions_all.reshape(n_test, future_hours)
                        y_true_all = y_true_all.reshape(n_test, future_hours)
                    except:
                        print(f"  [ERROR] Cannot reshape predictions, skipping file creation")
                        continue
            
            # Extract 48-hour period
            try:
                future_hours = config.get('future_hours', 24)
                prediction_df = extract_48hour_period(
                    predictions_all, y_true_all, test_dates, df_clean, future_hours
                )
                
                # Save to CSV
                prediction_df.to_csv(prediction_file, index=False, encoding='utf-8-sig')
                print(f"  [SAVED] Prediction file: {prediction_file}")
                print(f"  [INFO] 48-hour period: {prediction_df['Datetime'].iloc[0]} to {prediction_df['Datetime'].iloc[-1]}")
                
            except Exception as e:
                print(f"  [ERROR] Failed to extract 48-hour period: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        except Exception as e:
            print(f"  [ERROR] {exp_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("[OK] All Experiments Completed!")
    print(f"Prediction files saved to: {output_dir}")
    print(f"{'='*80}")


# =============================================================================
# MAIN ENTRY
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run all 284 experiments for first dataset and create 48-hour prediction files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save prediction files (default: ./predictions_48h)')
    
    args = parser.parse_args()
    
    run_all_experiments_with_predictions(output_dir=args.output_dir)

