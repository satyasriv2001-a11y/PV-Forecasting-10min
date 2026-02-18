#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate RMSE between 10 AM and 6 PM for June 20 or 21, 2024 for each plant's predictions.

This script:
1. Finds all prediction CSV files from hourly predictions
2. Determines which date (June 20 or 21, 2024) has complete data for hours 10-18
3. Filters predictions to that date between 10 AM - 6 PM (hours 10-18) only
4. Calculates RMSE between predicted and ground truth for each plant
5. Saves all RMSE values to a single CSV file in the base predictions directory

Usage:
    python calculate_rmse_10am_6pm.py --predictions-dir /path/to/predictions
    python calculate_rmse_10am_6pm.py --predictions-dir /content/drive/MyDrive/.../all_plants_predictions
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import glob
from pathlib import Path


def calculate_rmse_10am_6pm(predictions_dir):
    """
    Calculate RMSE for June 20 or 21, 2024 (whichever has complete data) between 10 AM and 6 PM for each plant.
    
    Args:
        predictions_dir: Base directory containing plant prediction folders
    """
    print("=" * 80)
    print("Calculating RMSE (June 20 or 21, 2024, 10 AM - 6 PM) for All Plants")
    print("=" * 80)
    print(f"Predictions directory: {os.path.abspath(predictions_dir)}")
    print(f"Directory exists: {os.path.exists(predictions_dir)}")
    if os.path.exists(predictions_dir):
        contents = os.listdir(predictions_dir)
        print(f"Directory contents ({len(contents)} items): {contents[:10]}")
    print("=" * 80)
    
    if not os.path.exists(predictions_dir):
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    
    # Ensure we can write to the directory
    if not os.access(predictions_dir, os.W_OK):
        print(f"[WARNING] Directory may not be writable: {predictions_dir}")
    
    # Find all plant folders (directories in predictions_dir)
    plant_folders = []
    for item in os.listdir(predictions_dir):
        item_path = os.path.join(predictions_dir, item)
        if os.path.isdir(item_path):
            plant_folders.append(item_path)
    
    plant_folders.sort()
    
    if len(plant_folders) == 0:
        raise ValueError(f"No plant folders found in {predictions_dir}")
    
    print(f"\nFound {len(plant_folders)} plant folder(s):")
    for idx, folder in enumerate(plant_folders, 1):
        print(f"  [{idx}/{len(plant_folders)}] {os.path.basename(folder)}")
    
    # Determine which date (June 20 or 21, 2024) has all hours 10-18 for all plants
    # Check first plant's files to determine the date
    date_to_use = None
    required_hours = set(range(10, 19))  # Hours 10-18 inclusive
    
    if len(plant_folders) > 0:
        first_plant_files = glob.glob(os.path.join(plant_folders[0], "predictions_hour_*.csv"))
        if len(first_plant_files) == 0:
            first_plant_files = glob.glob(os.path.join(plant_folders[0], "predictions_halfhour_*.csv"))
        if len(first_plant_files) == 0:
            first_plant_files = glob.glob(os.path.join(plant_folders[0], "*.csv"))
        
        if len(first_plant_files) > 0:
            try:
                sample_df = pd.read_csv(first_plant_files[0])
                if 'Datetime' in sample_df.columns:
                    sample_df['Datetime'] = pd.to_datetime(sample_df['Datetime'])
                    june_2024 = sample_df[(sample_df['Datetime'].dt.year == 2024) & 
                                         (sample_df['Datetime'].dt.month == 6) &
                                         (sample_df['Datetime'].dt.day.isin([20, 21]))]
                    
                    for day in [20, 21]:
                        day_data = june_2024[june_2024['Datetime'].dt.day == day]
                        hours_present = set(day_data[day_data['Datetime'].dt.hour.between(10, 18)]['Datetime'].dt.hour.unique())
                        if required_hours.issubset(hours_present):
                            date_to_use = day
                            print(f"\n[INFO] Using June {day}, 2024 (has all hours 10-18)")
                            break
                    
                    if date_to_use is None:
                        # Use whichever date has more hours
                        for day in [20, 21]:
                            day_data = june_2024[june_2024['Datetime'].dt.day == day]
                            hours_present = set(day_data[day_data['Datetime'].dt.hour.between(10, 18)]['Datetime'].dt.hour.unique())
                            if len(hours_present) >= len(required_hours):
                                date_to_use = day
                                print(f"\n[INFO] Using June {day}, 2024 (has {len(hours_present)} hours in range)")
                                break
            except Exception as e:
                print(f"  [WARNING] Could not determine date automatically: {str(e)}")
    
    if date_to_use is None:
        date_to_use = 20  # Default to June 20
        print(f"\n[INFO] Defaulting to June {date_to_use}, 2024")
    
    # Store results for all plants
    all_plant_results = []
    
    # Define output file path early
    summary_file = os.path.join(predictions_dir, 'rmse_10am_6pm.csv')
    print(f"\nOutput file will be: {os.path.abspath(summary_file)}")
    
    # Process each plant folder
    for plant_folder in plant_folders:
        plant_name = os.path.basename(plant_folder)
        print(f"\n{'='*80}")
        print(f"Processing Plant: {plant_name}")
        print(f"{'='*80}")
        
        # Find all prediction CSV files in this plant's folder
        # Try multiple patterns to find prediction files (hourly and half-hourly)
        prediction_files = glob.glob(os.path.join(plant_folder, "predictions_hour_*.csv"))
        
        # Also try half-hourly pattern
        if len(prediction_files) == 0:
            prediction_files = glob.glob(os.path.join(plant_folder, "predictions_halfhour_*.csv"))
        
        # Also try looking for any CSV files if the pattern doesn't match
        if len(prediction_files) == 0:
            all_csvs = glob.glob(os.path.join(plant_folder, "*.csv"))
            print(f"  [DEBUG] No files matching 'predictions_hour_*.csv' pattern")
            print(f"  [DEBUG] Found {len(all_csvs)} CSV file(s) total in folder")
            if len(all_csvs) > 0:
                print(f"  [DEBUG] Sample files: {[os.path.basename(f) for f in all_csvs[:3]]}")
            prediction_files = all_csvs
        
        if len(prediction_files) == 0:
            print(f"  [WARNING] No prediction CSV files found in {plant_folder}")
            print(f"  [DEBUG] Folder contents: {os.listdir(plant_folder)[:10]}")
            continue
        
        print(f"  Found {len(prediction_files)} prediction file(s)")
        
        # Collect all predictions and ground truth (aligned by datetime)
        all_data = []  # List of (datetime, pred, gt) tuples - for total RMSE
        filtered_data = []  # List of (datetime, pred, gt) tuples - for 10 AM - 6 PM RMSE
        
        for pred_file in sorted(prediction_files):
            try:
                df = pd.read_csv(pred_file)
                
                # Check required columns
                if 'Datetime' not in df.columns:
                    continue
                
                if 'Predicted_Capacity_Factor' not in df.columns:
                    continue
                
                if 'Ground_Truth_Capacity_Factor' not in df.columns:
                    continue
                
                # Convert Datetime column to datetime if it's not already
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                
                # Collect ALL data points for total RMSE
                for _, row in df.iterrows():
                    dt = row['Datetime']
                    pred = row.get('Predicted_Capacity_Factor', np.nan)
                    gt = row.get('Ground_Truth_Capacity_Factor', np.nan)
                    
                    # Only include rows where both pred and gt are valid
                    if not (np.isnan(pred) or np.isnan(gt)):
                        all_data.append((dt, pred, gt))
                
                # Filter to June 20 or 21, 2024 between 10 AM - 6 PM (hours 10-18, inclusive)
                df_filtered = df[
                    (df['Datetime'].dt.year == 2024) &
                    (df['Datetime'].dt.month == 6) &
                    (df['Datetime'].dt.day == date_to_use) &
                    (df['Datetime'].dt.hour >= 10) & 
                    (df['Datetime'].dt.hour <= 18)
                ].copy()
                
                if len(df_filtered) > 0:
                    # Get predicted and ground truth values for filtered time window
                    for _, row in df_filtered.iterrows():
                        dt = row['Datetime']
                        pred = row.get('Predicted_Capacity_Factor', np.nan)
                        gt = row.get('Ground_Truth_Capacity_Factor', np.nan)
                        
                        # Only include rows where both pred and gt are valid
                        if not (np.isnan(pred) or np.isnan(gt)):
                            filtered_data.append((dt, pred, gt))
                
            except Exception as e:
                print(f"  [ERROR] Error reading {os.path.basename(pred_file)}: {str(e)}")
                continue
        
        print(f"  [DEBUG] Total valid data points (all): {len(all_data)}")
        print(f"  [DEBUG] Total valid data points (10 AM - 6 PM): {len(filtered_data)}")
        
        # Calculate Total RMSE if we have valid data
        total_rmse = np.nan
        total_mae = np.nan
        total_n_samples = 0
        
        if len(all_data) > 0:
            # Extract predictions and ground truth for total RMSE
            preds_array_total = np.array([item[1] for item in all_data])
            gt_array_total = np.array([item[2] for item in all_data])
            
            # Calculate Total RMSE
            mse_total = np.mean((preds_array_total - gt_array_total) ** 2)
            total_rmse = np.sqrt(mse_total)
            
            # Calculate additional metrics
            total_mae = np.mean(np.abs(preds_array_total - gt_array_total))
            total_n_samples = len(preds_array_total)
            
            print(f"  Total RMSE (all predictions): {total_rmse:.4f}")
            print(f"  Total MAE (all predictions): {total_mae:.4f}")
            print(f"  Total number of samples: {total_n_samples}")
        
        # Calculate 10 AM - 6 PM RMSE if we have valid data
        rmse_10am_6pm = np.nan
        mae_10am_6pm = np.nan
        n_samples_10am_6pm = 0
        
        if len(filtered_data) > 0:
            # Extract predictions and ground truth for 10 AM - 6 PM
            preds_array = np.array([item[1] for item in filtered_data])
            gt_array = np.array([item[2] for item in filtered_data])
            
            # Calculate RMSE
            mse = np.mean((preds_array - gt_array) ** 2)
            rmse_10am_6pm = np.sqrt(mse)
            
            # Calculate additional metrics
            mae_10am_6pm = np.mean(np.abs(preds_array - gt_array))
            n_samples_10am_6pm = len(preds_array)
            
            date_str = f"June {date_to_use}, 2024"
            print(f"  RMSE ({date_str}, 10 AM - 6 PM): {rmse_10am_6pm:.4f}")
            print(f"  MAE ({date_str}, 10 AM - 6 PM): {mae_10am_6pm:.4f}")
            print(f"  Number of samples (10 AM - 6 PM): {n_samples_10am_6pm}")
        else:
            date_str = f"June {date_to_use}, 2024"
            print(f"  [WARNING] No valid data points found for 10 AM - 6 PM RMSE calculation")
        
        # Store results (will be saved to single CSV at the end)
        if len(all_data) > 0 or len(filtered_data) > 0:
            result = {
                'Plant_Name': plant_name,
                'RMSE_Total': total_rmse,
                'MAE_Total': total_mae,
                'Number_of_Samples_Total': total_n_samples,
                'RMSE_10AM_6PM': rmse_10am_6pm,
                'MAE_10AM_6PM': mae_10am_6pm,
                'Number_of_Samples_10AM_6PM': n_samples_10am_6pm,
                'Date': date_str,
                'Time_Range': '10:00 - 18:00'
            }
            all_plant_results.append(result)
        else:
            print(f"  [WARNING] No valid data points found for RMSE calculation")
            date_str = f"June {date_to_use}, 2024"
            result = {
                'Plant_Name': plant_name,
                'RMSE_Total': np.nan,
                'MAE_Total': np.nan,
                'Number_of_Samples_Total': 0,
                'RMSE_10AM_6PM': np.nan,
                'MAE_10AM_6PM': np.nan,
                'Number_of_Samples_10AM_6PM': 0,
                'Date': date_str,
                'Time_Range': '10:00 - 18:00',
                'Status': 'No data found'
            }
            all_plant_results.append(result)
            
            # Save incrementally after each plant (append mode)
            try:
                if len(all_plant_results) == 1:
                    # First plant - create new file with header
                    df_temp = pd.DataFrame([result])
                    df_temp.to_csv(summary_file, index=False)
                    print(f"  [SAVED] Created RMSE file with first plant")
                else:
                    # Append to existing file
                    existing_df = pd.read_csv(summary_file)
                    new_df = pd.DataFrame([result])
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df.to_csv(summary_file, index=False)
                    print(f"  [SAVED] Updated RMSE file (total: {len(all_plant_results)} plants)")
            except Exception as e:
                print(f"  [WARNING] Could not save incrementally: {str(e)}")
    
    # Final save/update of CSV with all plants' RMSE values
    if len(all_plant_results) > 0:
        try:
            summary_df = pd.DataFrame(all_plant_results)
            print(f"\n{'='*80}")
            print(f"Creating RMSE CSV file: {summary_file}")
            print(f"{'='*80}")
            print(f"Directory exists: {os.path.exists(predictions_dir)}")
            print(f"Directory writable: {os.access(predictions_dir, os.W_OK)}")
            print(f"Number of plant results: {len(all_plant_results)}")
            
            # Ensure directory exists
            os.makedirs(predictions_dir, exist_ok=True)
            
            # Write CSV file
            summary_df.to_csv(summary_file, index=False)
            print(f"CSV file written to: {os.path.abspath(summary_file)}")
            
            # Verify summary file was created
            if os.path.exists(summary_file):
                file_size = os.path.getsize(summary_file)
                print(f"File exists: True")
                print(f"File size: {file_size} bytes")
                print(f"\n{'='*80}")
                print("[SUCCESS] RMSE Calculation Completed!")
                print(f"{'='*80}")
                print(f"\nSummary:")
                print(f"  Total plants processed: {len(all_plant_results)}")
                # Filter out NaN values for statistics
                valid_rmse_total = summary_df['RMSE_Total'].dropna()
                valid_rmse_10am_6pm = summary_df['RMSE_10AM_6PM'].dropna()
                
                if len(valid_rmse_total) > 0:
                    print(f"  Average Total RMSE: {valid_rmse_total.mean():.4f}")
                    print(f"  Min Total RMSE: {valid_rmse_total.min():.4f} ({summary_df.loc[summary_df['RMSE_Total'].idxmin(), 'Plant_Name']})")
                    print(f"  Max Total RMSE: {valid_rmse_total.max():.4f} ({summary_df.loc[summary_df['RMSE_Total'].idxmax(), 'Plant_Name']})")
                else:
                    print(f"  [WARNING] No valid Total RMSE values to calculate statistics")
                
                if len(valid_rmse_10am_6pm) > 0:
                    print(f"  Average RMSE (10 AM - 6 PM): {valid_rmse_10am_6pm.mean():.4f}")
                    print(f"  Min RMSE (10 AM - 6 PM): {valid_rmse_10am_6pm.min():.4f} ({summary_df.loc[summary_df['RMSE_10AM_6PM'].idxmin(), 'Plant_Name']})")
                    print(f"  Max RMSE (10 AM - 6 PM): {valid_rmse_10am_6pm.max():.4f} ({summary_df.loc[summary_df['RMSE_10AM_6PM'].idxmax(), 'Plant_Name']})")
                else:
                    print(f"  [WARNING] No valid RMSE (10 AM - 6 PM) values to calculate statistics")
                print(f"\nSummary file saved: {os.path.abspath(summary_file)}")
                print(f"[VERIFIED] Summary file exists: {os.path.exists(summary_file)}")
                print(f"{'='*80}")
                
                # Display summary table
                print("\nRMSE Summary Table:")
                print(summary_df.to_string(index=False))
                
                print(f"\n{'='*80}")
                print(f"RMSE CSV file created: {os.path.abspath(summary_file)}")
                print(f"Total plants: {len(summary_df)}")
                valid_rmse_total = summary_df['RMSE_Total'].dropna()
                valid_rmse_10am_6pm = summary_df['RMSE_10AM_6PM'].dropna()
                print(f"Plants with valid Total RMSE: {len(valid_rmse_total) if len(valid_rmse_total) > 0 else 0}")
                print(f"Plants with valid RMSE (10 AM - 6 PM): {len(valid_rmse_10am_6pm) if len(valid_rmse_10am_6pm) > 0 else 0}")
                print(f"{'='*80}")
            else:
                print(f"\n[ERROR] Summary file was not created: {summary_file}")
                print(f"  Directory exists: {os.path.exists(predictions_dir)}")
                print(f"  Directory writable: {os.access(predictions_dir, os.W_OK)}")
                # Try to create an empty file to verify write permissions
                try:
                    test_file = os.path.join(predictions_dir, 'test_write.txt')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    print(f"  Write permissions: OK")
                except Exception as e:
                    print(f"  Write permissions: FAILED - {str(e)}")
        except Exception as e:
            print(f"\n[ERROR] Failed to create summary file: {str(e)}")
            import traceback
            traceback.print_exc()
            # Try to create at least an empty CSV with headers
            try:
                empty_df = pd.DataFrame(columns=['Plant_Name', 'RMSE_Total', 'MAE_Total', 'Number_of_Samples_Total', 'RMSE_10AM_6PM', 'MAE_10AM_6PM', 'Number_of_Samples_10AM_6PM', 'Date', 'Time_Range'])
                empty_df.to_csv(summary_file, index=False)
                print(f"Created empty CSV file with headers: {summary_file}")
            except Exception as e2:
                print(f"Failed to create even empty CSV: {str(e2)}")
    else:
        print(f"\n[WARNING] No RMSE values calculated for any plants")
        print(f"  This could mean:")
        print(f"    - No prediction files found")
        print(f"    - No data on June 20 or 21, 2024 in 10 AM - 6 PM time range")
        print(f"    - Missing required columns in prediction files")
        # Still create an empty CSV file
        try:
            empty_df = pd.DataFrame(columns=['Plant_Name', 'RMSE_Total', 'MAE_Total', 'Number_of_Samples_Total', 'RMSE_10AM_6PM', 'MAE_10AM_6PM', 'Number_of_Samples_10AM_6PM', 'Date', 'Time_Range'])
            empty_df.to_csv(summary_file, index=False)
            print(f"\nCreated empty RMSE CSV file: {summary_file}")
            print(f"File location: {os.path.abspath(summary_file)}")
        except Exception as e:
            print(f"\n[ERROR] Failed to create empty CSV file: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Final verification - always check if file exists
    print(f"\n{'='*80}")
    print("FINAL FILE VERIFICATION:")
    print(f"{'='*80}")
    if os.path.exists(summary_file):
        file_size = os.path.getsize(summary_file)
        print(f"✓ RMSE CSV file exists: {os.path.abspath(summary_file)}")
        print(f"  File size: {file_size} bytes")
        if file_size > 0:
            # Try to read it back
            try:
                verify_df = pd.read_csv(summary_file)
                print(f"  File contains {len(verify_df)} rows")
                print(f"  Columns: {list(verify_df.columns)}")
            except Exception as e:
                print(f"  [WARNING] Could not read file back: {str(e)}")
        else:
            print(f"  [WARNING] File is empty (0 bytes)")
    else:
        print(f"✗ RMSE CSV file NOT FOUND: {summary_file}")
        print(f"  Expected location: {os.path.abspath(summary_file)}")
        print(f"  Directory exists: {os.path.exists(predictions_dir)}")
        print(f"  Directory contents: {os.listdir(predictions_dir)[:10] if os.path.exists(predictions_dir) else 'N/A'}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate RMSE for June 20, 2024 (6-20-2024) between 10 AM and 6 PM for each plant\'s predictions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate RMSE for predictions in default directory
  python calculate_rmse_10am_6pm.py --predictions-dir ./all_plants_predictions
  
  # Calculate RMSE for predictions in Drive
  python calculate_rmse_10am_6pm.py --predictions-dir /content/drive/MyDrive/.../all_plants_predictions
        """
    )
    
    parser.add_argument('--predictions-dir', type=str, required=True,
                       help='Base directory containing plant prediction folders (e.g., ./all_plants_predictions)')
    
    args = parser.parse_args()
    
    try:
        calculate_rmse_10am_6pm(args.predictions_dir)
    except Exception as e:
        print(f"\n[ERROR] Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

