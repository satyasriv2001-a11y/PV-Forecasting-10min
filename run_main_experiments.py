#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run all 284 experiments for PV forecasting (DL + ML + Linear, with resume support)
"""

import pandas as pd
import numpy as np
import yaml
import os
import sys
import time
from datetime import datetime
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
# CONFIG GENERATION
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


# =============================================================================
# CONFIG CREATION
# =============================================================================
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
                                      'random_state': 42, 'verbosity': -1}  # Low complexity (underfit baseline)
        else:
            config['model_params'] = {'n_estimators': 30, 'max_depth': 3, 'learning_rate': 0.1,
                                      'random_state': 42, 'verbosity': -1}  # High complexity (validated optimal)

    te_suffix = 'TE' if use_te else 'noTE'
    config['experiment_name'] = f"{model}_{feat_name}_{te_suffix}" if model == 'Linear' else f"{model}_{complexity}_{feat_name}_{te_suffix}"
    config['save_dir'] = f'results/{config["experiment_name"]}'
    return config


# =============================================================================
# MAIN LOOP WITH RESUME
# =============================================================================
def run_all_experiments(output_dir=None):
    """
    Run all experiments with optional output directory
    
    Args:
        output_dir: Directory to save results (default: current directory)
    """
    print("=" * 80)
    print("PV Forecasting: Running 284 Experiments (with resume support)")
    print("=" * 80)

    all_configs = generate_all_configs()
    print(f"Total configurations generated: {len(all_configs)}")

    import torch
    data_path = os.path.join(script_dir, "data", "Project1140.csv")
    df = pd.read_csv(data_path)
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Set output directory
    if output_dir is None:
        output_dir = script_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")

    # === check for existing result CSV ===
    output_file = os.path.join(output_dir, "all_experiments_results.csv")
    
    if os.path.exists(output_file):
        print(f"Found existing result file: {output_file}")
        results_df = pd.read_csv(output_file)
        done_experiments = set(results_df["experiment_name"].tolist())
    else:
        results_df = pd.DataFrame(columns=[
            'experiment_name', 'model', 'complexity', 'feature_combo',
            'lookback_hours', 'use_time_encoding', 'mae', 'rmse', 'r2',
            'train_time_sec', 'test_samples', 'best_epoch', 'param_count',
            'predicted_values', 'mean_predicted_value', 'experiment_datetime', 'test_period_start', 'test_period_end'
        ])
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        done_experiments = set()
        print(f"Created new result file: {output_file}")

    print(f"[OK] Already completed: {len(done_experiments)}")
    print(f"[INFO] Remaining: {len(all_configs) - len(done_experiments)}")

    # === main loop ===
    for idx, config in enumerate(all_configs, 1):
        exp_name = config['experiment_name']
        if exp_name in done_experiments:
            print(f"[SKIP] {exp_name} already completed.")
            continue

        print(f"\n{'='*80}")
        print(f"Experiment {idx}/{len(all_configs)}: {exp_name}")
        print(f"{'='*80}")

        try:
            start_time = time.time()
            df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = preprocess_features(df, config)

            # Use 24-hour sliding windows (one prediction per hour)
            # Creates samples by sliding a 24-hour window across the time series
            # Supports variable lookback (24h, 72h)
            past_hours = config.get('past_hours', 24)
            X_hist, X_fcst, y, hours, dates = create_sliding_windows(
                df_clean, past_hours, config['future_hours'], hist_feats, fcst_feats, no_hist_power
            )

            total_samples = len(X_hist)
            indices = np.arange(total_samples)
            
            # Random shuffle for robust evaluation (covers all seasons)
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

            # choose training function
            if config['model'] in ['LSTM', 'GRU', 'Transformer', 'TCN']:
                model, metrics = train_dl_model(config, train_data, val_data, test_data, scalers)
            else:
                model, metrics = train_ml_model(config, train_data, val_data, test_data, scalers)

            training_time = time.time() - start_time
            
            # Parse feature combination name (scenario) from config
            # Scenario should only be: PV, PV+HW, PV+NWP, PV+NWP+, NWP, NWP+
            use_pv = config.get('use_pv', False)
            use_hist_weather = config.get('use_hist_weather', False)
            use_forecast = config.get('use_forecast', False)
            use_ideal_nwp = config.get('use_ideal_nwp', False)
            
            if use_pv and use_hist_weather:
                feat_name_str = 'PV+HW'
            elif use_pv and use_forecast and use_ideal_nwp:
                feat_name_str = 'PV+NWP+'
            elif use_pv and use_forecast:
                feat_name_str = 'PV+NWP'
            elif use_pv:
                feat_name_str = 'PV'
            elif use_forecast and use_ideal_nwp:
                feat_name_str = 'NWP+'
            elif use_forecast:
                feat_name_str = 'NWP'
            else:
                feat_name_str = 'Unknown'

            # Serialize predictions to comma-separated string for CSV storage
            predictions = metrics.get('predictions', np.array([]))
            if isinstance(predictions, np.ndarray):
                predicted_values_str = ','.join([f'{val:.6f}' for val in predictions.flatten()])
                # Calculate mean predicted value (representative value corresponding to RMSE)
                mean_predicted_value = float(np.mean(predictions))
            else:
                predicted_values_str = ''
                mean_predicted_value = np.nan
            
            # Get experiment run timestamp
            experiment_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Extract test period date range from metrics
            test_dates = metrics.get('dates', [])
            test_period_start = ''
            test_period_end = ''
            if test_dates:
                # Convert dates to strings and sort
                test_dates_str = []
                for d in test_dates:
                    if isinstance(d, str):
                        test_dates_str.append(d)
                    elif hasattr(d, 'strftime'):
                        test_dates_str.append(d.strftime('%Y-%m-%d'))
                    else:
                        test_dates_str.append(str(d))
                
                if test_dates_str:
                    test_dates_sorted = sorted(test_dates_str)
                    test_period_start = test_dates_sorted[0]
                    test_period_end = test_dates_sorted[-1]
            
            result = {
                'experiment_name': exp_name,
                'model': config['model'],
                'complexity': config.get('model_complexity', 'N/A'),
                'feature_combo': feat_name_str,
                'lookback_hours': config['past_hours'],
                'use_time_encoding': config['use_time_encoding'],
                'mae': metrics.get('mae', 0.0),
                'rmse': metrics.get('rmse', 0.0),
                'r2': metrics.get('r2', 0.0),
                'train_time_sec': round(training_time, 2),
                'test_samples': metrics.get('samples_count', 0),
                'best_epoch': int(metrics.get('best_epoch', 0)) if not pd.isna(metrics.get('best_epoch', 0)) else 0,
                'param_count': int(metrics.get('param_count', 0)),
                'predicted_values': predicted_values_str,
                'mean_predicted_value': mean_predicted_value,
                'experiment_datetime': experiment_datetime,
                'test_period_start': test_period_start,
                'test_period_end': test_period_end
            }

            print(f"  [OK] MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
            pd.DataFrame([result]).to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
            
            # Update done_experiments to prevent duplicates in same run
            done_experiments.add(exp_name)

        except Exception as e:
            print(f"  [ERROR] {exp_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            pd.DataFrame([{
                'experiment_name': exp_name,
                'model': config['model'],
                'complexity': config['model_complexity'],
                'feature_combo': 'FAILED',
                'lookback_hours': config['past_hours'],
                'use_time_encoding': config['use_time_encoding'],
                'mae': np.nan, 'rmse': np.nan, 'r2': np.nan,
                'train_time_sec': 0, 'test_samples': 0,
                'best_epoch': 0, 'param_count': 0,
                'predicted_values': '',
                'mean_predicted_value': np.nan,
                'experiment_datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'test_period_start': '',
                'test_period_end': ''
            }]).to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
            continue

    print(f"\n{'='*80}")
    print("[OK] All Experiments Completed or Skipped!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")


# =============================================================================
# MAIN ENTRY
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run all 284 experiments for single plant')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save results (default: current directory). '
                            'For Colab/Drive (use quotes): "/content/drive/MyDrive/Solar PV electricity/results"')
    
    args = parser.parse_args()
    
    success = run_all_experiments(output_dir=args.output_dir)
    if not success:
        sys.exit(1)
