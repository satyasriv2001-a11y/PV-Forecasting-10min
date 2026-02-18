#download dependencies

#!pip install -q torch pyyaml pandas scikit-learn matplotlib seaborn tqdm

# download XGBoost和LightGBM (GPU版本)

#!pip install xgboost lightgbm

#download装cuML GPU (CUDA 12)

#!pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com

# 1. Clone repository (remove if exists to avoid nesting)
# Navigate to /content first
%cd /content

# Clean up any existing directories using shell command
!rm -rf PV-Forecasting

# Clone repository
!git clone https://github.com/satyasriv2001-a11y/PV-Forecasting

# Navigate into the cloned repository
%cd /content/PV-Forecasting

# 2. Install dependencies

!pip install -q -r requirements.txt

# Install Google Sheets and plotting libraries

!pip install -q gspread google-auth matplotlib

# 3. Mount Google Drive and copy datasets

from google.colab import drive

drive.mount('/content/drive')

!cp /content/drive/MyDrive/AI_Models_for_Solar_Energy/*.csv data/

# 4. Generate configs for all plants (optional, for reference)

!python batch_create_configs.py

# 5. Run Multi-Resolution Predictions on Project1140 with XGB high PV+NWP 24h no TE
# This will process only Project1140.csv
# Results save directly to Drive!
# Generates 6 plots: 3 overlay plots (hourly, 30-min, 15-min) and 3 hourly RMSE plots

import subprocess
import os
import glob

# Set output directory in Drive
DRIVE_PATH = "/content/drive/MyDrive/Solar PV electricity/final_multi_resolution_predictions_1140"

# Get list of CSV files in data directory
data_dir = "data"
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
csv_files.sort()

# Filter to only Project1140
project1140_files = [f for f in csv_files if "1140" in os.path.basename(f).upper()]

if len(project1140_files) == 0:
    print("[ERROR] No Project1140 CSV file found!")
    print(f"Available files: {[os.path.basename(f) for f in csv_files[:10]]}")
else:
    print("=" * 80)
    print("Running Multi-Resolution Predictions on Project1140")
    print("=" * 80)
    print(f"Configuration: XGB high complexity, PV+NWP, 24h lookback, NO time encoding")
    print(f"Output directory: {DRIVE_PATH}")
    print("=" * 80)
    print(f"\nFound {len(project1140_files)} Project1140 file(s):")
    for i, f in enumerate(project1140_files, 1):
        print(f"  [{i}] {os.path.basename(f)}")
    print("=" * 80)
    
    # Process each Project1140 file
    for file_idx, data_path in enumerate(project1140_files, 1):
        plant_name = os.path.splitext(os.path.basename(data_path))[0]
        print(f"\n{'='*80}")
        print(f"Processing Project1140 File {file_idx}/{len(project1140_files)}: {plant_name}")
        print(f"{'='*80}")
        
        # Create output directory for this plant
        output_dir = os.path.join(DRIVE_PATH, plant_name, "XGB_high_PV+NWP_24h_noTE")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Output: {output_dir}")
        
        # Get the script path
        script_path = os.path.join(os.getcwd(), 'multi_resolution_predictions_1140.py')
        
        if not os.path.exists(script_path):
            print(f"[ERROR] Script not found: {script_path}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in directory: {os.listdir('.')[:10]}")
            continue
        
        # Run multi-resolution predictions
        cmd = [
            'python', script_path,
            '--data-path', data_path,
            '--model', 'XGB',
            '--complexity', 'high',
            '--scenario', 'PV+NWP',
            '--lookback', '24',
            '--no-time-encoding',
            '--output-dir', output_dir
        ]
        
        print(f"\nRunning command:")
        print(f"  {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[OK] Completed successfully")
            if result.stdout:
                print("--- Standard Output ---")
                print(result.stdout[-2000:])  # Last 2000 chars
        else:
            print(f"[ERROR] Failed:")
            print("--- Standard Error ---")
            print(result.stderr)
            print("--- Standard Output ---")
            print(result.stdout[-2000:])  # Last 2000 chars
    
    print(f"\n{'='*80}")
    print("[SUCCESS] Multi-Resolution Predictions Completed!")
    print(f"Results saved to: {DRIVE_PATH}")
    print(f"{'='*80}")
    print("\nGenerated files:")
    print("  - overlay_plot_hourly.png (prediction error overlay: predicted - actual for hourly resolution)")
    print("  - overlay_plot_30_minute.png (prediction error overlay: predicted - actual for 30-minute resolution)")
    print("  - overlay_plot_15_minute.png (prediction error overlay: predicted - actual for 15-minute resolution)")
    print("  - big_overlay_plot_hourly.png (big overlay plot showing ALL predictions at once for hourly resolution)")
    print("  - big_overlay_plot_30_minute.png (big overlay plot showing ALL predictions at once for 30-minute resolution)")
    print("  - big_overlay_plot_15_minute.png (big overlay plot showing ALL predictions at once for 15-minute resolution)")
    print("  - ground_truth_plot_hourly.png (ground truth values for hourly resolution)")
    print("  - ground_truth_plot_30_minute.png (ground truth values for 30-minute resolution)")
    print("  - ground_truth_plot_15_minute.png (ground truth values for 15-minute resolution)")
    print("  - hourly_rmse_plot_hourly.png (hourly RMSE scatter plot for hourly resolution)")
    print("  - hourly_rmse_plot_30_minute.png (hourly RMSE scatter plot for 30-minute resolution)")
    print("  - hourly_rmse_plot_15_minute.png (hourly RMSE scatter plot for 15-minute resolution)")
    print("  - rmse_10am_6pm_by_resolution.csv (RMSE values for 10 AM-6 PM for each resolution)")

