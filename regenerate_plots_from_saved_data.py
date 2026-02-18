#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Regenerate Plots from Saved Prediction Data with Larger Fonts

This script loads saved prediction data and regenerates all plots with larger, more readable fonts.

Usage:
    python regenerate_plots_from_saved_data.py --data-dir <directory_with_predictions_data.pkl> [--output-dir <output_directory>]
    
Example:
    python regenerate_plots_from_saved_data.py --data-dir ./multi_resolution_predictions_LSTM_PV+NWP --output-dir ./regenerated_plots_large_fonts
"""



import os

import sys

import argparse

import glob

import pickle

import pandas as pd

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.dates import HourLocator, DateFormatter

import warnings

warnings.filterwarnings('ignore')



# =============================================================================

# PLOTTING FUNCTIONS WITH LARGE FONTS

# =============================================================================

def create_overlay_plot_large_fonts(all_predictions, output_path, model_name, resolution_name, max_predictions_per_plot=12):

    """Create overlay plots with large fonts"""

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

        plt.rcParams.update({'font.size': 18})

        

        colors = plt.cm.tab20(np.linspace(0, 1, len(plot_predictions)))

        plot_datetimes = set()

        

        for idx, (pred_df, pred_num, pred_datetime) in enumerate(plot_predictions):

            differences = pred_df['Predicted_Capacity_Factor'] - pred_df['Ground_Truth_Capacity_Factor']

            valid_mask = ~(pred_df['Predicted_Capacity_Factor'].isna() | pred_df['Ground_Truth_Capacity_Factor'].isna())

            valid_datetimes = pred_df.loc[valid_mask, 'Datetime']

            valid_differences = differences.loc[valid_mask]

            

            plt.plot(valid_datetimes, valid_differences, 

                    label=f'Pred {pred_num} ({pred_datetime.strftime("%m-%d %H:%M")})',

                    linewidth=1.5, alpha=0.6, color=colors[idx])

            plot_datetimes.update(valid_datetimes)

        

        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Zero Error')

        

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

        

        plt.xlabel('Datetime', fontsize=22, fontweight='bold')

        plt.ylabel('Prediction Error (Predicted - Actual) (%)', fontsize=22, fontweight='bold')

        

        if num_plots > 1:

            title = f'Prediction Error Overlay - {model_name} ({resolution_name}) - Part {plot_idx + 1}/{num_plots}\n'

            title += f'Predictions {start_idx + 1}-{end_idx} of {num_predictions} (24-hour ahead forecasts)'

        else:

            title = f'Prediction Error Overlay - {model_name} ({resolution_name})\n'

            title += f'{num_predictions} prediction intervals (24-hour ahead forecasts)'

        

        plt.title(title, fontsize=24, fontweight='bold')

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, ncol=1)

        plt.grid(True, alpha=0.3, which='both')

        plt.xticks(rotation=45, ha='right', fontsize=16)

        plt.yticks(fontsize=16)

        plt.tight_layout()

        

        os.makedirs(os.path.dirname(plot_output_path) if os.path.dirname(plot_output_path) else '.', exist_ok=True)

        plt.savefig(plot_output_path, dpi=200, bbox_inches='tight')

        plt.close()

        print(f"  ✓ Regenerated overlay plot: {os.path.basename(plot_output_path)}")

        output_paths.append(plot_output_path)

    

    return output_paths





def create_big_overlay_plot_large_fonts(all_predictions, output_path, model_name, resolution_name):

    """Create big overlay plot with large fonts"""

    num_predictions = len(all_predictions)

    

    plt.figure(figsize=(32, 14))

    plt.rcParams.update({'font.size': 18})

    

    colors = plt.cm.tab20(np.linspace(0, 1, num_predictions))

    all_datetimes = set()

    

    for idx, (pred_df, pred_num, pred_datetime) in enumerate(all_predictions):

        differences = pred_df['Predicted_Capacity_Factor'] - pred_df['Ground_Truth_Capacity_Factor']

        valid_mask = ~(pred_df['Predicted_Capacity_Factor'].isna() | pred_df['Ground_Truth_Capacity_Factor'].isna())

        valid_datetimes = pred_df.loc[valid_mask, 'Datetime']

        valid_differences = differences.loc[valid_mask]

        

        plt.plot(valid_datetimes, valid_differences, 

                label=f'Pred {pred_num} ({pred_datetime.strftime("%m-%d %H:%M")})',

                linewidth=1.5, alpha=0.5, color=colors[idx])

        all_datetimes.update(valid_datetimes)

    

    plt.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Zero Error')

    

    all_datetimes_sorted = sorted(list(all_datetimes))

    if all_datetimes_sorted:

        x_min = min(all_datetimes_sorted)

        x_max = max(all_datetimes_sorted)

        x_range = x_max - x_min

        plt.xlim(x_min - 0.02 * x_range, x_max + 0.02 * x_range)

    

    ax = plt.gca()

    if all_datetimes_sorted:

        ax.xaxis.set_major_locator(HourLocator(interval=6))

        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:00'))

        ax.xaxis.set_minor_locator(HourLocator(interval=1))

    

    plt.xlabel('Datetime', fontsize=22, fontweight='bold')

    plt.ylabel('Prediction Error (Predicted - Actual) (%)', fontsize=22, fontweight='bold')

    

    title = f'All Prediction Errors Overlay - {model_name} ({resolution_name})\n'

    title += f'All {num_predictions} prediction intervals (24-hour ahead forecasts)'

    

    plt.title(title, fontsize=24, fontweight='bold')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, ncol=1)

    plt.grid(True, alpha=0.3, which='both')

    plt.xticks(rotation=45, ha='right', fontsize=16)

    plt.yticks(fontsize=16)

    plt.tight_layout()

    

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')

    plt.close()

    print(f"  ✓ Regenerated big overlay plot: {os.path.basename(output_path)}")

    return output_path





def create_ground_truth_plot_large_fonts(all_predictions, output_path, model_name, resolution_name):

    """Create ground truth plot with large fonts"""

    plt.figure(figsize=(24, 10))

    plt.rcParams.update({'font.size': 18})

    

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

    

    sorted_datetimes = sorted(gt_dict.keys())

    sorted_values = [gt_dict[dt] for dt in sorted_datetimes]

    

    plt.plot(sorted_datetimes, sorted_values, 

            label='Ground Truth', linewidth=3, color='black', marker='o', 

            markersize=6, alpha=0.9)

    

    if sorted_datetimes:

        x_min = min(sorted_datetimes)

        x_max = max(sorted_datetimes)

        x_range = x_max - x_min

        plt.xlim(x_min - 0.02 * x_range, x_max + 0.02 * x_range)

    

    ax = plt.gca()

    if sorted_datetimes:

        ax.xaxis.set_major_locator(HourLocator(interval=6))

        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:00'))

        ax.xaxis.set_minor_locator(HourLocator(interval=1))

    

    plt.xlabel('Datetime', fontsize=22, fontweight='bold')

    plt.ylabel('Capacity Factor (%)', fontsize=22, fontweight='bold')

    plt.title(f'Ground Truth - {model_name} ({resolution_name})\n'

              f'Actual capacity factor values across all prediction intervals', 

              fontsize=24, fontweight='bold')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)

    plt.grid(True, alpha=0.3, which='both')

    plt.xticks(rotation=45, ha='right', fontsize=16)

    plt.yticks(fontsize=16)

    plt.tight_layout()

    

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')

    plt.close()

    print(f"  ✓ Regenerated ground truth plot: {os.path.basename(output_path)}")

    return output_path





def create_hourly_rmse_plot_large_fonts(hourly_rmse_data, output_path, model_name, resolution_name, y_axis_limits=None):

    """Create hourly RMSE plot with large fonts"""

    if len(hourly_rmse_data) == 0:

        print(f"  [WARNING] No hourly RMSE data to plot for {resolution_name}")

        return

    

    plt.figure(figsize=(14, 8))

    plt.rcParams.update({'font.size': 18})

    

    hourly_rmse_data = sorted(hourly_rmse_data, key=lambda x: x[0])

    hours = [item[0] for item in hourly_rmse_data]

    rmse_values = [item[1] for item in hourly_rmse_data]

    

    plt.plot(hours, rmse_values, marker='o', linewidth=2, markersize=8, 

             color='steelblue', markerfacecolor='lightblue', markeredgecolor='darkblue', 

             markeredgewidth=1.5, alpha=0.8)

    

    if y_axis_limits is not None:

        plt.ylim(y_axis_limits[0], y_axis_limits[1])

    

    ax = plt.gca()

    ax.xaxis.set_major_locator(HourLocator(interval=2))

    ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:00'))

    ax.xaxis.set_minor_locator(HourLocator(interval=1))

    

    plt.xlabel('Hour', fontsize=22, fontweight='bold')

    plt.ylabel('RMSE', fontsize=22, fontweight='bold')

    plt.title(f'Hourly RMSE Values - {model_name} ({resolution_name})\n'

              f'RMSE calculated for each hour across all predictions', 

              fontsize=24, fontweight='bold')

    plt.grid(True, alpha=0.3, which='both')

    plt.xticks(rotation=45, ha='right', fontsize=16)

    plt.yticks(fontsize=16)

    plt.tight_layout()

    

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')

    plt.close()

    print(f"  ✓ Regenerated hourly RMSE plot: {os.path.basename(output_path)}")





# =============================================================================

# DATA LOADING AND REGENERATION

# =============================================================================

def load_predictions_data(data_dir):

    """Load prediction data from pickle files"""

    pickle_files = glob.glob(os.path.join(data_dir, '**', 'predictions_data.pkl'), recursive=True)

    pickle_files.extend(glob.glob(os.path.join(data_dir, 'predictions_data.pkl')))

    

    if len(pickle_files) == 0:

        return None

    

    # Try to load the first pickle file found

    try:

        with open(pickle_files[0], 'rb') as f:

            data = pickle.load(f)

        print(f"  Loaded data from: {pickle_files[0]}")

        return data

    except Exception as e:

        print(f"  [ERROR] Failed to load {pickle_files[0]}: {str(e)}")

        return None





def regenerate_all_plots(data_dir, output_dir):

    """Regenerate all plots from saved data"""

    data = load_predictions_data(data_dir)

    

    if data is None:

        print("\n[ERROR] No prediction data found!")

        print("\nTo use this script:")

        print("1. First, modify your prediction script to save data using modify_script_to_save_data.py")

        print("2. Run your prediction script (it will save predictions_data.pkl)")

        print("3. Then run this script to regenerate plots with larger fonts")

        return False

    

    # Extract data

    all_predictions = data.get('all_predictions', [])

    hourly_rmse_data = data.get('hourly_rmse_data', [])

    model_name = data.get('model_name', 'Model')

    resolution_name = data.get('resolution_name', 'Unknown')

    

    if len(all_predictions) == 0:

        print("[ERROR] No predictions found in saved data")

        return False

    

    print(f"\nRegenerating plots for {resolution_name} resolution...")

    print(f"Model: {model_name}")

    print(f"Number of predictions: {len(all_predictions)}")

    print("=" * 80)

    

    # Regenerate overlay plots

    overlay_path = os.path.join(output_dir, f"overlay_plot_{resolution_name.lower().replace('-', '_')}_large_fonts.png")

    overlay_paths = create_overlay_plot_large_fonts(all_predictions, overlay_path, model_name, resolution_name)

    
    # Regenerate big overlay plot

    big_overlay_path = os.path.join(output_dir, f"big_overlay_plot_{resolution_name.lower().replace('-', '_')}_large_fonts.png")

    create_big_overlay_plot_large_fonts(all_predictions, big_overlay_path, model_name, resolution_name)

    

    # Regenerate ground truth plot

    ground_truth_path = os.path.join(output_dir, f"ground_truth_plot_{resolution_name.lower().replace('-', '_')}_large_fonts.png")

    create_ground_truth_plot_large_fonts(all_predictions, ground_truth_path, model_name, resolution_name)

    

    # Regenerate hourly RMSE plot

    if len(hourly_rmse_data) > 0:

        # Calculate y-axis limits

        rmse_values = [item[1] for item in hourly_rmse_data]

        y_min = min(rmse_values)

        y_max = max(rmse_values)

        y_range = y_max - y_min

        y_axis_limits = (max(0, y_min - 0.05 * y_range), y_max + 0.05 * y_range)

        

        rmse_plot_path = os.path.join(output_dir, f"hourly_rmse_plot_{resolution_name.lower().replace('-', '_')}_large_fonts.png")

        create_hourly_rmse_plot_large_fonts(hourly_rmse_data, rmse_plot_path, model_name, resolution_name, y_axis_limits)

    

    print("\n" + "=" * 80)

    print("[SUCCESS] All plots regenerated with larger fonts!")

    print(f"Output directory: {output_dir}")

    print("=" * 80)

    

    return True





# =============================================================================

# MAIN FUNCTION

# =============================================================================

def main():

    parser = argparse.ArgumentParser(

        description='Regenerate plots with larger fonts from saved prediction data',

        formatter_class=argparse.RawDescriptionHelpFormatter

    )

    

    parser.add_argument('--data-dir', type=str, required=True,

                       help='Directory containing predictions_data.pkl file')

    parser.add_argument('--output-dir', type=str, default=None,

                       help='Output directory for regenerated plots (default: <data_dir>/regenerated_plots_large_fonts)')

    

    args = parser.parse_args()

    

    if not os.path.exists(args.data_dir):

        print(f"[ERROR] Data directory does not exist: {args.data_dir}")

        sys.exit(1)

    

    if args.output_dir is None:

        args.output_dir = os.path.join(args.data_dir, 'regenerated_plots_large_fonts')

    

    os.makedirs(args.output_dir, exist_ok=True)

    

    print("=" * 80)

    print("Regenerate Plots with Larger Fonts")

    print("=" * 80)

    print(f"Data directory: {args.data_dir}")

    print(f"Output directory: {args.output_dir}")

    print("=" * 80)

    

    if regenerate_all_plots(args.data_dir, args.output_dir):

        print("\n[SUCCESS] Plot regeneration completed!")

    else:

        print("\n[ERROR] Plot regeneration failed. See messages above.")

        sys.exit(1)





if __name__ == "__main__":

    main()

