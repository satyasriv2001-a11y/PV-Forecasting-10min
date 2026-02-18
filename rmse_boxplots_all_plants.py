#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RMSE Box and Whisker Plots for All Plants - Multi-Resolution (XGB and LR)

Creates box plots showing RMSE distribution across all plants for each prediction start hour.
Plots RMSE values for 10-minute resolution only.
Separate plots for XGB and LR models (8 plots total: 4 resolutions × 2 models).
The x-axis is the starting hour of the 24-hour sliding window (0-23), and y-axis is RMSE.

Similar to rmse_boxplots_1140.py but aggregated across all plants.

Usage:
    python rmse_boxplots_all_plants.py --predictions-dir /path/to/predictions --model both
    python rmse_boxplots_all_plants.py --predictions-dir ./all_plants_predictions --model XGB
    python rmse_boxplots_all_plants.py --predictions-dir ./all_plants_predictions --model LR
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
import argparse
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
import re

warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)


def calculate_rmse(preds, gt):
    """
    Calculate RMSE for a prediction window.
    
    Args:
        preds: Predicted capacity factor values (array)
        gt: Ground truth capacity factor values (array)
    
    Returns:
        RMSE (float) or np.nan if insufficient valid data
    """
    # Filter out NaN values
    valid_mask = ~(np.isnan(preds) | np.isnan(gt))
    
    if np.sum(valid_mask) < 1:  # Need at least 1 point
        return np.nan
    
    preds_valid = preds[valid_mask]
    gt_valid = gt[valid_mask]
    
    # Calculate RMSE
    mse = np.mean((preds_valid - gt_valid) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse


def detect_resolution(df, file_path):
    """
    Detect the resolution (hourly, 30-min, 15-min, or 10-min) from data or filename.
    
    Args:
        df: DataFrame with Datetime column
        file_path: Path to the file (for filename-based detection)
    
    Returns:
        Resolution name ('Hourly', '30-minute', '15-minute', '10-minute') and resolution_minutes
    """
    # First try filename-based detection
    filename = os.path.basename(file_path).lower()
    if 'hour' in filename or 'hourly' in filename:
        return 'Hourly', 60
    if '30min' in filename or 'halfhour' in filename or '30_min' in filename:
        return '30-minute', 30
    if '15min' in filename or '15_min' in filename:
        return '15-minute', 15
    if '10min' in filename or '10_min' in filename:
        return '10-minute', 10
    
    # Otherwise, detect from data frequency
    if 'Datetime' not in df.columns or len(df) < 2:
        return 'Hourly', 60  # Default
    
    df = df.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    # Calculate time differences
    time_diffs = df['Datetime'].diff().dropna()
    if len(time_diffs) == 0:
        return 'Hourly', 60  # Default
    
    median_diff_minutes = time_diffs.median().total_seconds() / 60.0
    
    # Determine resolution based on median time difference
    if median_diff_minutes <= 12:  # <= 12 minutes -> 10-minute
        return '10-minute', 10
    elif median_diff_minutes <= 20:  # <= 20 minutes -> 15-minute
        return '15-minute', 15
    elif median_diff_minutes <= 40:  # <= 40 minutes -> 30-minute
        return '30-minute', 30
    else:  # > 40 minutes -> hourly
        return 'Hourly', 60


def detect_model_from_path(path):
    """
    Detect model type (XGB or LR) from path or filename.
    
    Args:
        path: File path or directory path
    
    Returns:
        'XGB', 'LR', or None if cannot determine
    """
    path_lower = path.lower()
    
    # Check for XGB indicators
    if 'xgb' in path_lower or 'xgboost' in path_lower:
        return 'XGB'
    
    # Check for LR indicators
    if 'lr' in path_lower or 'linear' in path_lower:
        return 'LR'
    
    return None


def find_model_directories(predictions_dir, model_filter):
    """
    Find directories containing predictions for the specified model(s).
    
    Args:
        predictions_dir: Base directory to search
        model_filter: 'XGB', 'LR', or 'both'
    
    Returns:
        List of (model_name, directory_path) tuples
    """
    model_dirs = []
    
    if model_filter == 'both':
        # Look for directories with XGB or LR in the name
        print(f"\nSearching for model directories in: {predictions_dir}")
        all_items = os.listdir(predictions_dir)
        print(f"  Found {len(all_items)} item(s) in base directory")
        
        for item in all_items:
            item_path = os.path.join(predictions_dir, item)
            if os.path.isdir(item_path):
                detected_model = detect_model_from_path(item_path)
                print(f"    [DIR] {item} -> detected model: {detected_model}")
                if detected_model in ['XGB', 'LR']:
                    model_dirs.append((detected_model, item_path))
                    print(f"      ✓ Added {detected_model} directory: {item_path}")
            else:
                print(f"    [FILE] {item} (skipping)")
        
        # If no model-specific directories found, check if base directory contains model indicators
        if len(model_dirs) == 0:
            detected_model = detect_model_from_path(predictions_dir)
            if detected_model:
                model_dirs.append((detected_model, predictions_dir))
            else:
                # Assume both models might be in subdirectories, or use base directory for both
                # Try to find any subdirectory with model indicators
                for root, dirs, files in os.walk(predictions_dir):
                    for d in dirs:
                        dir_path = os.path.join(root, d)
                        detected_model = detect_model_from_path(dir_path)
                        if detected_model in ['XGB', 'LR']:
                            model_dirs.append((detected_model, dir_path))
    else:
        # Single model
        detected_model = detect_model_from_path(predictions_dir)
        if detected_model == model_filter:
            model_dirs.append((model_filter, predictions_dir))
        else:
            # Search for subdirectories with this model
            for item in os.listdir(predictions_dir):
                item_path = os.path.join(predictions_dir, item)
                if os.path.isdir(item_path):
                    detected = detect_model_from_path(item_path)
                    if detected == model_filter:
                        model_dirs.append((model_filter, item_path))
            
            # If still nothing found, use base directory anyway
            if len(model_dirs) == 0:
                model_dirs.append((model_filter, predictions_dir))
    
    return model_dirs


def load_predictions_from_dir(predictions_dir, model_name=None):
    """
    Load all prediction CSV files from plant directories and calculate RMSE.
    
    Args:
        predictions_dir: Base directory containing plant prediction folders
        model_name: Model name ('XGB' or 'LR') for labeling
    
    Returns:
        Dictionary: {resolution_name: {hour: [rmse_values]}}
    """
    if not os.path.exists(predictions_dir):
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    
    # Find all plant folders (directories in predictions_dir)
    plant_folders = []
    all_items = os.listdir(predictions_dir)
    print(f"  Scanning directory: {predictions_dir}")
    print(f"  Found {len(all_items)} item(s) in directory")
    
    for item in all_items:
        item_path = os.path.join(predictions_dir, item)
        if os.path.isdir(item_path):
            plant_folders.append(item_path)
        else:
            print(f"    [FILE] {item} (skipping - not a directory)")
    
    plant_folders.sort()
    
    if len(plant_folders) == 0:
        # Maybe predictions_dir itself contains CSV files directly
        print(f"[INFO] No plant folders found, checking for CSV files directly in {predictions_dir}")
        csv_files = glob.glob(os.path.join(predictions_dir, "*.csv"))
        if len(csv_files) > 0:
            print(f"[INFO] Found {len(csv_files)} CSV file(s) directly in directory")
            plant_folders = [predictions_dir]
        else:
            raise ValueError(f"No plant folders or CSV files found in {predictions_dir}")
    
    print(f"Found {len(plant_folders)} plant folder(s):")
    for i, folder in enumerate(plant_folders[:10], 1):
        print(f"  [{i}] {os.path.basename(folder)}")
    if len(plant_folders) > 10:
        print(f"  ... and {len(plant_folders) - 10} more plant folders")
    
    # Dictionary to store results: {resolution: {hour: [rmse, rmse, ...]}}
    results_by_resolution = defaultdict(lambda: defaultdict(list))
    
    # Process each plant folder
    for plant_folder in plant_folders:
        plant_name = os.path.basename(plant_folder)
        if plant_name == os.path.basename(predictions_dir):
            plant_name = "All_Plants"  # Use generic name if processing directory directly
        
        print(f"\nProcessing plant: {plant_name}")
        
        # Find all prediction CSV files in this plant's folder
        prediction_files = []
        patterns = [
            "predictions_*.csv",
            "prediction_*.csv",
            "*predictions*.csv",
            "*.csv"  # Last resort: all CSV files
        ]
        
        for pattern in patterns:
            found_files = glob.glob(os.path.join(plant_folder, pattern))
            if len(found_files) > 0:
                prediction_files.extend(found_files)
        
        # Remove duplicates
        prediction_files = list(set(prediction_files))
        
        # Filter out summary/result files
        prediction_files = [f for f in prediction_files 
                          if not any(x in os.path.basename(f).lower() 
                                   for x in ['summary', 'result', 'rmse', 'error', 'average', 'status'])]
        
        if len(prediction_files) == 0:
            print(f"  [WARNING] No prediction files found in {plant_folder}")
            continue
        
        print(f"  Found {len(prediction_files)} prediction file(s)")
        
        # Process each prediction file
        processed_count = 0
        for pred_file in sorted(prediction_files):
            try:
                df = pd.read_csv(pred_file)
                
                # Check required columns - try different possible column names
                datetime_col = None
                pred_col = None
                gt_col = None
                
                for col in df.columns:
                    col_lower = col.lower()
                    if datetime_col is None and ('datetime' in col_lower or 'date' in col_lower or 'time' in col_lower):
                        datetime_col = col
                    if pred_col is None and ('predicted' in col_lower and 'capacity' in col_lower):
                        pred_col = col
                    if gt_col is None and (('ground' in col_lower and 'truth' in col_lower) or 
                                         ('actual' in col_lower) or
                                         ('true' in col_lower and 'capacity' in col_lower)):
                        gt_col = col
                
                # Use exact matches if available
                if 'Datetime' in df.columns:
                    datetime_col = 'Datetime'
                if 'Predicted_Capacity_Factor' in df.columns:
                    pred_col = 'Predicted_Capacity_Factor'
                if 'Ground_Truth_Capacity_Factor' in df.columns:
                    gt_col = 'Ground_Truth_Capacity_Factor'
                
                if datetime_col is None or pred_col is None or gt_col is None:
                    print(f"    [SKIP] {os.path.basename(pred_file)}: Missing required columns")
                    continue
                
                # Convert datetime column to datetime
                df[datetime_col] = pd.to_datetime(df[datetime_col])
                df = df.sort_values(datetime_col).reset_index(drop=True)
                
                # Detect resolution
                resolution_name, resolution_minutes = detect_resolution(df, pred_file)
                
                # Calculate RMSE for this prediction window
                preds = df[pred_col].values
                gt = df[gt_col].values
                
                rmse = calculate_rmse(preds, gt)
                
                if np.isnan(rmse):
                    print(f"    [SKIP] {os.path.basename(pred_file)}: Invalid RMSE (insufficient data)")
                    continue
                
                # Extract starting hour from the first datetime
                start_datetime = df[datetime_col].iloc[0]
                start_hour = start_datetime.hour
                
                # Store result: {resolution: {hour: [rmse, rmse, ...]}}
                results_by_resolution[resolution_name][start_hour].append(rmse)
                processed_count += 1
                
                if processed_count <= 3 or processed_count % 10 == 0:
                    print(f"    [{processed_count}/{len(prediction_files)}] {os.path.basename(pred_file)}: {resolution_name}, Hour {start_hour:02d}, RMSE={rmse:.4f}")
                
            except Exception as e:
                print(f"  [ERROR] Error processing {os.path.basename(pred_file)}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"  Processed {processed_count}/{len(prediction_files)} files successfully")
    
    return results_by_resolution


def create_rmse_boxplots(results_by_resolution, output_dir, model_name=""):
    """
    Create RMSE box and whisker plots for each resolution.
    
    Args:
        results_by_resolution: Dictionary {resolution_name: {hour: [rmse_values]}}
        output_dir: Output directory for plots
        model_name: Model name ('XGB' or 'LR') to include in filename and title
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define resolution order (10-minute only)
    resolution_order = ['10-minute']
    
    # Filter to only resolutions we have data for
    available_resolutions = [r for r in resolution_order if r in results_by_resolution]
    
    if len(available_resolutions) == 0:
        print(f"[WARNING] No data found for any resolution (model: {model_name})")
        return
    
    print(f"\n{'='*80}")
    print(f"Creating RMSE Box Plots for {model_name if model_name else 'All Models'}")
    print(f"{'='*80}")
    
    # Create separate plot for each resolution
    for resolution_name in available_resolutions:
        print(f"\nProcessing {resolution_name} resolution ({model_name})...")
        
        resolution_data = results_by_resolution[resolution_name]
        
        # Prepare data for box plot - collect RMSE values for each hour (0-23)
        # Show all hours 0-23 on x-axis, even if some have no data
        rmse_data_by_hour = []
        hour_labels = list(range(24))  # All hours 0-23
        
        for hour in range(24):  # Hours 0-23
            if hour in resolution_data and len(resolution_data[hour]) > 0:
                # Only include hours with at least 2 data points (for box plot to work)
                if len(resolution_data[hour]) >= 2:
                    rmse_data_by_hour.append(resolution_data[hour])
                else:
                    rmse_data_by_hour.append([])  # Too few points, show as empty
            else:
                # Include empty list for hours with no data (will show as empty box)
                rmse_data_by_hour.append([])
        
        if len([d for d in rmse_data_by_hour if len(d) > 0]) == 0:
            print(f"  [WARNING] No valid hour data for {resolution_name}")
            continue
        
        # Count total predictions
        total_rmse_values = sum(len(vals) for vals in rmse_data_by_hour)
        
        # Create box plot - larger figure for better visibility
        plt.figure(figsize=(18, 10))
        plt.rcParams.update({'font.size': 14})
        
        # Create box plot for all 24 hours (hours with no data will show as empty)
        bp = plt.boxplot(rmse_data_by_hour, labels=hour_labels, patch_artist=True, widths=0.7)
        
        # Color the boxes with distinct colors
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
        
        # Style the whiskers and medians
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
        
        # Format x-axis - show all hours 0-23
        ax = plt.gca()
        ax.set_xlim(0.5, 24.5)
        ax.set_xticks(range(24))
        ax.set_xticklabels(range(24))
        
        # Set y-axis limits - collect all RMSE values from hours with data
        all_rmse = [val for hour_vals in rmse_data_by_hour for val in hour_vals]
        if len(all_rmse) > 0:
            y_min = max(0, np.min(all_rmse) * 0.9)
            y_max = np.max(all_rmse) * 1.1
            ax.set_ylim(y_min, y_max)
        
        model_label = f" ({model_name})" if model_name else ""
        plt.xlabel('Starting Hour of 24-Hour Sliding Window (0-23)', fontsize=16, fontweight='bold')
        plt.ylabel('RMSE (Capacity Factor)', fontsize=16, fontweight='bold')
        plt.title(f'RMSE Distribution Across All Plants by Prediction Start Hour - {resolution_name}{model_label}\n'
                  f'Box plot shows RMSE distribution for 24-hour forecasts starting at each hour (n={total_rmse_values} predictions)',
                  fontsize=18, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
        plt.xticks(rotation=0, fontsize=13)
        plt.yticks(fontsize=13)
        plt.tight_layout()
        
        # Save plot with high resolution
        resolution_file_name = resolution_name.lower().replace('-', '_')
        model_suffix = f"_{model_name.lower()}" if model_name else ""
        output_path = os.path.join(output_dir, f"rmse_boxplot_{resolution_file_name}_all_plants{model_suffix}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"  Box plot saved: {output_path}")
        
        # Print statistics
        hours_with_data = len([d for d in rmse_data_by_hour if len(d) > 0])
        print(f"  Statistics for {resolution_name} ({model_name}):")
        print(f"    Total predictions: {total_rmse_values}")
        print(f"    Hours with data: {hours_with_data}/24")
        if len(all_rmse) > 0:
            print(f"    Mean RMSE: {np.mean(all_rmse):.4f}")
            print(f"    Median RMSE: {np.median(all_rmse):.4f}")
            print(f"    Std RMSE: {np.std(all_rmse):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Create RMSE box and whisker plots for all plants across multiple resolutions (XGB and LR)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create plots for both XGB and LR (8 plots total)
  python rmse_boxplots_all_plants.py --predictions-dir ./all_plants_predictions --model both
  
  # Create plots for XGB only (4 plots)
  python rmse_boxplots_all_plants.py --predictions-dir ./all_plants_XGB --model XGB
  
  # Create plots for LR only (4 plots)
  python rmse_boxplots_all_plants.py --predictions-dir ./all_plants_LR --model LR
        """
    )
    
    parser.add_argument('--predictions-dir', type=str, required=True,
                       help='Base directory containing plant prediction folders')
    parser.add_argument('--model', type=str, default='both',
                       choices=['XGB', 'LR', 'both'],
                       help='Model to process: XGB, LR, or both (default: both)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: ./rmse_boxplots_all_plants)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RMSE Box and Whisker Plots - All Plants (Multi-Resolution, XGB & LR)")
    print("=" * 80)
    print(f"Predictions directory: {os.path.abspath(args.predictions_dir)}")
    print(f"Model(s): {args.model}")
    
    if args.output_dir is None:
        output_dir = os.path.join(script_dir, "rmse_boxplots_all_plants")
    else:
        output_dir = args.output_dir
    
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print("=" * 80)
    
    try:
        # Find model directories
        model_dirs = find_model_directories(args.predictions_dir, args.model)
        
        if len(model_dirs) == 0:
            print(f"\n[WARNING] No model-specific directories found. Using base directory for {args.model}.")
            if args.model == 'both':
                # Try to process as if it contains both
                model_dirs = [('XGB', args.predictions_dir), ('LR', args.predictions_dir)]
            else:
                model_dirs = [(args.model, args.predictions_dir)]
        
        print(f"\nFound {len(model_dirs)} model directory set(s) to process:")
        for model, dir_path in model_dirs:
            print(f"  - {model}: {dir_path}")
        
        all_results = {}
        summary_data = []
        
        # Process each model
        for model_name, model_dir in model_dirs:
            print(f"\n{'='*80}")
            print(f"Processing {model_name} Model")
            print(f"{'='*80}")
            
            # Load predictions from all plants for this model
            print(f"\n[1/2] Loading predictions from all plants ({model_name})...")
            results_by_resolution = load_predictions_from_dir(model_dir, model_name)
            
            if len(results_by_resolution) == 0:
                print(f"\n[WARNING] No prediction data found for {model_name}")
                continue
            
            print(f"\nFound data for {len(results_by_resolution)} resolution(s): {list(results_by_resolution.keys())}")
            all_results[model_name] = results_by_resolution
            
            # Create box plots for this model
            print(f"\n[2/2] Creating RMSE box plots for {model_name}...")
            create_rmse_boxplots(results_by_resolution, output_dir, model_name)
            
            # Collect summary data
            for resolution_name, resolution_data in results_by_resolution.items():
                for hour in range(24):
                    if hour in resolution_data and len(resolution_data[hour]) > 0:
                        for rmse in resolution_data[hour]:
                            summary_data.append({
                                'Model': model_name,
                                'Resolution': resolution_name,
                                'Start_Hour': hour,
                                'RMSE': rmse
                            })
        
        # Create summary CSV
        if len(summary_data) > 0:
            print(f"\n{'='*80}")
            print("Creating Summary CSV")
            print(f"{'='*80}")
            summary_df = pd.DataFrame(summary_data)
            summary_csv_path = os.path.join(output_dir, 'rmse_summary_all_plants.csv')
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"  Summary CSV saved: {summary_csv_path}")
            
            # Print summary statistics
            print(f"\nSummary Statistics:")
            for model_name in sorted(all_results.keys()):
                model_df = summary_df[summary_df['Model'] == model_name]
                if len(model_df) > 0:
                    print(f"\n  {model_name} Model:")
                    for resolution_name in ['Hourly', '30-minute', '15-minute', '10-minute']:
                        res_df = model_df[model_df['Resolution'] == resolution_name]
                        if len(res_df) > 0:
                            print(f"    {resolution_name}:")
                            print(f"      Total predictions: {len(res_df)}")
                            print(f"      Mean RMSE: {res_df['RMSE'].mean():.4f}")
                            print(f"      Median RMSE: {res_df['RMSE'].median():.4f}")
                            print(f"      Std RMSE: {res_df['RMSE'].std():.4f}")
        
        print(f"\n{'='*80}")
        print("[SUCCESS] RMSE Box Plot Generation Completed!")
        print(f"Output directory: {output_dir}")
        
        # Count total plots generated
        total_plots = 0
        for model_name, results in all_results.items():
            for resolution_name in results.keys():
                total_plots += 1
        
        print(f"Generated {total_plots} plot(s) total ({len(all_results)} model(s) × up to 4 resolutions)")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
