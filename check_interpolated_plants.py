#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check how many plants have interpolated prediction data

This script examines prediction CSV files to determine:
- How many plants have predictions at different resolutions (30-min, 15-min, 10-min)
- Which resolutions are available for each plant
- Whether data appears to be interpolated (has Minute column, sub-hourly resolution)
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
import argparse
from collections import defaultdict

# Handle both script execution and notebook/Colab execution
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Running in notebook/Colab
    script_dir = os.getcwd()
os.chdir(script_dir)
sys.path.append(script_dir)


def detect_resolution_from_file(file_path, df):
    """
    Detect resolution from file data.
    
    Returns:
        (resolution_name, resolution_minutes, has_minute_column)
    """
    has_minute_column = 'Minute' in df.columns
    
    if not has_minute_column:
        return 'Hourly', 60, False
    
    if 'Datetime' not in df.columns:
        # Try to create datetime
        if all(col in df.columns for col in ['Year', 'Month', 'Day', 'Hour']):
            df = df.copy()
            if 'Minute' in df.columns:
                df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
            else:
                df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    
    if 'Datetime' in df.columns:
        df = df.copy()
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.sort_values('Datetime').reset_index(drop=True)
        
        # Calculate time differences
        time_diffs = df['Datetime'].diff().dropna()
        if len(time_diffs) > 0:
            median_diff_minutes = time_diffs.median().total_seconds() / 60.0
            
            if median_diff_minutes <= 12:
                return '10-minute', 10, True
            elif median_diff_minutes <= 20:
                return '15-minute', 15, True
            elif median_diff_minutes <= 40:
                return '30-minute', 30, True
    
    return 'Hourly', 60, has_minute_column


def check_predictions_directory(predictions_dir):
    """
    Check how many plants have interpolated data.
    
    Returns:
        Dictionary with statistics
    """
    if not os.path.exists(predictions_dir):
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    
    # Find all plant folders
    plant_folders = []
    for item in os.listdir(predictions_dir):
        item_path = os.path.join(predictions_dir, item)
        if os.path.isdir(item_path):
            plant_folders.append(item_path)
    
    plant_folders.sort()
    
    print(f"Found {len(plant_folders)} plant folder(s)")
    
    # Statistics
    plant_stats = {}
    resolution_counts = defaultdict(int)
    plants_by_resolution = defaultdict(set)
    
    for plant_folder in plant_folders:
        plant_name = os.path.basename(plant_folder)
        
        # Find prediction CSV files
        prediction_files = []
        patterns = ["predictions_*.csv", "prediction_*.csv", "*predictions*.csv", "*.csv"]
        
        for pattern in patterns:
            found_files = glob.glob(os.path.join(plant_folder, pattern))
            prediction_files.extend(found_files)
        
        prediction_files = list(set(prediction_files))
        
        # Filter out summary files
        prediction_files = [f for f in prediction_files 
                          if not any(x in os.path.basename(f).lower() 
                                   for x in ['summary', 'result', 'rmse', 'error', 'average', 'status'])]
        
        if len(prediction_files) == 0:
            continue
        
        # Check each prediction file
        resolutions_found = set()
        for pred_file in prediction_files:
            try:
                df = pd.read_csv(pred_file, nrows=100)  # Read first 100 rows to check structure
                
                resolution_name, resolution_minutes, has_minute = detect_resolution_from_file(pred_file, df)
                resolutions_found.add(resolution_name)
                resolution_counts[resolution_name] += 1
                plants_by_resolution[resolution_name].add(plant_name)
                
            except Exception as e:
                continue
        
        plant_stats[plant_name] = {
            'num_files': len(prediction_files),
            'resolutions': sorted(list(resolutions_found)),
            'has_interpolated': any(r in resolutions_found for r in ['30-minute', '15-minute', '10-minute'])
        }
    
    return {
        'plant_stats': plant_stats,
        'resolution_counts': dict(resolution_counts),
        'plants_by_resolution': {k: sorted(list(v)) for k, v in plants_by_resolution.items()},
        'total_plants': len(plant_stats)
    }


def main(predictions_dir=None):
    """
    Main function. Can be called with predictions_dir directly (for notebooks) 
    or use argparse (for command line).
    """
    # Handle both command-line and direct call (for notebooks/Colab)
    if predictions_dir is None:
        parser = argparse.ArgumentParser(
            description='Check how many plants have interpolated prediction data'
        )
        
        parser.add_argument('--predictions-dir', type=str, required=True,
                           help='Base directory containing plant prediction folders')
        
        args = parser.parse_args()
        predictions_dir = args.predictions_dir
    
    print("=" * 80)
    print("Checking Interpolated Plant Data")
    print("=" * 80)
    print(f"Predictions directory: {os.path.abspath(predictions_dir)}")
    print("=" * 80)
    
    try:
        stats = check_predictions_directory(predictions_dir)
        
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Total plants found: {stats['total_plants']}")
        print(f"\nResolution distribution (number of prediction files):")
        for resolution in ['Hourly', '30-minute', '15-minute', '10-minute']:
            count = stats['resolution_counts'].get(resolution, 0)
            num_plants = len(stats['plants_by_resolution'].get(resolution, set()))
            print(f"  {resolution:15s}: {count:4d} files, {num_plants:3d} plants")
        
        # Count plants with interpolated data
        plants_with_interpolated = sum(1 for p in stats['plant_stats'].values() if p['has_interpolated'])
        plants_hourly_only = stats['total_plants'] - plants_with_interpolated
        
        print(f"\nPlants with interpolated data (30-min, 15-min, or 10-min): {plants_with_interpolated}")
        print(f"Plants with hourly data only: {plants_hourly_only}")
        
        print(f"\n{'='*80}")
        print("DETAILED PLANT STATISTICS")
        print(f"{'='*80}")
        
        # Sort by plant name
        sorted_plants = sorted(stats['plant_stats'].items())
        
        for plant_name, plant_info in sorted_plants:
            interpolated_marker = "✓" if plant_info['has_interpolated'] else " "
            resolutions_str = ", ".join(plant_info['resolutions']) if plant_info['resolutions'] else "none"
            print(f"  {interpolated_marker} {plant_name:30s}: {plant_info['num_files']:3d} files, resolutions: {resolutions_str}")
        
        print(f"\n{'='*80}")
        print("PLANTS BY RESOLUTION")
        print(f"{'='*80}")
        
        for resolution in ['Hourly', '30-minute', '15-minute', '10-minute']:
            plants = stats['plants_by_resolution'].get(resolution, [])
            if len(plants) > 0:
                print(f"\n{resolution} ({len(plants)} plants):")
                # Print in columns
                for i in range(0, len(plants), 5):
                    print(f"  {', '.join(plants[i:i+5])}")
        
        print(f"\n{'='*80}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        # Don't exit in notebook mode
        if predictions_dir is not None:  # Called directly (notebook mode)
            raise
        else:  # Command line mode
            sys.exit(1)


if __name__ == "__main__":
    # For Colab/notebooks, you can call: main("/path/to/predictions")
    # For command line, use: python check_interpolated_plants.py --predictions-dir /path/to/predictions
    main()

