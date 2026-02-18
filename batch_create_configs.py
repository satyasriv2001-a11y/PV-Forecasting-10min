#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch create configuration files
Auto-scan data directory for CSV files and generate corresponding config files
"""

import os
import glob
import yaml
import pandas as pd
from pathlib import Path
import re

def extract_plant_id(filename):
    """
    Extract plant ID from filename
    Supported formats: Project1140.csv, Plant1140.csv, 1140.csv, etc.
    """
    basename = os.path.basename(filename)
    
    # Try multiple patterns
    patterns = [
        r'Project(\d+)',
        r'Plant(\d+)',
        r'plant(\d+)',
        r'project(\d+)',
        r'^(\d+)',  # Pure numeric prefix
    ]
    
    for pattern in patterns:
        match = re.search(pattern, basename, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # If no match, use filename without extension
    return basename.replace('.csv', '').replace('.CSV', '')

def detect_date_range(csv_file):
    """
    Auto-detect date range of the dataset
    """
    try:
        # Read CSV file (only head and tail to save time)
        df_head = pd.read_csv(csv_file, nrows=10)
        df = pd.read_csv(csv_file)
        
        # Try to construct datetime
        if all(col in df.columns for col in ['Year', 'Month', 'Day', 'Hour']):
            df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
            start_date = df['Datetime'].min().strftime('%Y-%m-%d')
            end_date = df['Datetime'].max().strftime('%Y-%m-%d')
            return start_date, end_date, len(df)
        else:
            print(f"  [WARNING] {csv_file} missing time columns, using default dates")
            return '2022-01-01', '2024-09-28', len(df)
    except Exception as e:
        print(f"  [WARNING] Error reading {csv_file}: {str(e)}")
        return '2022-01-01', '2024-09-28', 0

def create_plant_config(csv_file, template_path='config/plant_template.yaml'):
    """
    Create configuration file for a single dataset
    """
    # Read template
    with open(template_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Extract plant ID
    plant_id = extract_plant_id(csv_file)
    
    # Detect date range
    start_date, end_date, data_length = detect_date_range(csv_file)
    
    # Update config
    config['plant_id'] = str(plant_id)
    config['plant_name'] = f"Project {plant_id}"
    # Convert path to forward slashes for cross-platform compatibility
    config['data_path'] = csv_file.replace('\\', '/')
    config['start_date'] = start_date
    config['end_date'] = end_date
    
    # Keep default settings from template
    # shuffle_split defaults to True (Random shuffle for robust evaluation)
    if 'shuffle_split' not in config:
        config['shuffle_split'] = True
    if 'random_seed' not in config:
        config['random_seed'] = 42
    if 'past_hours' not in config:
        config['past_hours'] = 24
    
    return plant_id, config, data_length

def batch_create_configs(data_dir='data', output_dir='config/plants', template='config/plant_template.yaml'):
    """
    Batch create configuration files
    """
    print("="*80)
    print("Batch Configuration File Creation")
    print("="*80)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(data_dir, '*.csv')) + glob.glob(os.path.join(data_dir, '*.CSV'))
    
    if not csv_files:
        print(f"\n[ERROR] No CSV files found in {data_dir} directory")
        print(f"Please place dataset files in {data_dir} directory")
        return
    
    print(f"\nFound {len(csv_files)} dataset files")
    print("-"*80)
    
    created_configs = []
    skipped_configs = []
    
    for i, csv_file in enumerate(csv_files, 1):
        filename = os.path.basename(csv_file)
        print(f"\n[{i}/{len(csv_files)}] Processing: {filename}")
        
        try:
            # Create configuration
            plant_id, config, data_length = create_plant_config(csv_file, template)
            
            # Save configuration file
            config_file = os.path.join(output_dir, f'Plant{plant_id}.yaml')
            
            # Check if exists
            if os.path.exists(config_file):
                print(f"  [SKIP] Config file already exists: {config_file}")
                skipped_configs.append(plant_id)
                continue
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            print(f"  [OK] Created config: {config_file}")
            print(f"    Plant ID: {plant_id}")
            print(f"    Date range: {config['start_date']} to {config['end_date']}")
            print(f"    Data length: {data_length} records")
            
            created_configs.append(plant_id)
            
        except Exception as e:
            print(f"  [ERROR] {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("Configuration File Creation Completed")
    print("="*80)
    print(f"[OK] Created: {len(created_configs)} config files")
    print(f"[SKIP] Skipped: {len(skipped_configs)} config files (already exist)")
    print(f"Output directory: {output_dir}")
    
    if created_configs:
        print(f"\nCreated plant IDs: {', '.join(created_configs[:10])}" + 
              (f" ... (total {len(created_configs)} plants)" if len(created_configs) > 10 else ""))
    
    print("\nNext steps:")
    print("  1. Check configs: ls config/plants/")
    print("  2. Test single plant: python run_main_experiments.py")
    print("  3. Run all plants: python run_experiments_multi_plant.py")
    print("="*80)

def verify_configs(config_dir='config/plants'):
    """
    Verify all configuration files
    """
    print("\nVerifying configuration files...")
    config_files = glob.glob(os.path.join(config_dir, '*.yaml'))
    
    valid_count = 0
    invalid_count = 0
    
    for config_file in config_files:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Check required fields
            required_fields = ['plant_id', 'data_path', 'start_date', 'end_date']
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                print(f"  [WARNING] {os.path.basename(config_file)}: Missing fields {missing_fields}")
                invalid_count += 1
            else:
                # Check if data file exists
                if not os.path.exists(config['data_path']):
                    print(f"  [WARNING] {os.path.basename(config_file)}: Data file not found {config['data_path']}")
                    invalid_count += 1
                else:
                    valid_count += 1
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(config_file)}: {str(e)}")
            invalid_count += 1
    
    print(f"\nVerification result: [OK] {valid_count} valid, [WARNING] {invalid_count} invalid")
    return valid_count, invalid_count

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch create configuration files')
    parser.add_argument('--data-dir', default='data', help='Dataset directory (default: data)')
    parser.add_argument('--output-dir', default='config/plants', help='Config output directory (default: config/plants)')
    parser.add_argument('--template', default='config/plant_template.yaml', help='Configuration template file')
    parser.add_argument('--verify', action='store_true', help='Verify created configuration files')
    parser.add_argument('--force', action='store_true', help='Force overwrite existing config files')
    
    args = parser.parse_args()
    
    # Check if template file exists
    if not os.path.exists(args.template):
        print(f"[ERROR] Template file not found: {args.template}")
        exit(1)
    
    # If only verifying
    if args.verify:
        verify_configs(args.output_dir)
        exit(0)
    
    # Batch create configs
    batch_create_configs(args.data_dir, args.output_dir, args.template)
    
    # Verify created configs
    print("\n")
    verify_configs(args.output_dir)

