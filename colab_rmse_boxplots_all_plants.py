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

!pip install -q gspread google-auth matplotlib seaborn

# 3. Mount Google Drive

from google.colab import drive

drive.mount('/content/drive')

# 4. Run RMSE Box and Whisker Plots for ALL PLANTS (XGB and LR)
# This script loads prediction data from all plants and creates RMSE box plots
# for hourly, 30-min, 15-min, and 10-min resolutions
# X-axis: Starting hour of 24-hour sliding window (0-23)
# Y-axis: RMSE (Capacity Factor)
# 
# Generates 8 plots total: 4 for XGB (one per resolution) + 4 for LR (one per resolution)
# Each plot is high-quality (18x10 inches, 300 DPI) with enhanced styling

import subprocess
import os

# Set paths - UPDATE THESE PATHS TO MATCH YOUR DRIVE STRUCTURE
# Option 1: Point to base directory containing both XGB and LR prediction folders
BASE_PREDICTIONS_DIR = "/content/drive/MyDrive/Solar PV electricity/final_hourly_predictions"

# Option 2: Point to specific model directory (uncomment to use)
# BASE_PREDICTIONS_DIR = "/content/drive/MyDrive/Solar PV electricity/final_hourly_predictions/all_plants_XGB_high_PV+NWP_24h_noTE"

# Alternative paths you might need (uncomment the one that matches your structure):
# BASE_PREDICTIONS_DIR = "/content/drive/MyDrive/Solar PV electricity/all_plants_predictions"
# BASE_PREDICTIONS_DIR = "/content/drive/MyDrive/Solar PV electricity/final_hourly_predictions"

# Model selection: 'XGB', 'LR', or 'both' (default: both for 8 plots total)
MODEL_SELECTION = "both"  # Change to "XGB" or "LR" if you only want one model

# Output directory (plots will be saved here)
OUTPUT_DIR = "/content/drive/MyDrive/Solar PV electricity/rmse_boxplots_all_plants"

print("=" * 80)
print("RMSE Box and Whisker Plots - ALL PLANTS (Multi-Resolution, XGB & LR)")
print("=" * 80)
print("Description: Creates individual high-quality box plots for each resolution")
print("             showing RMSE distribution across all plants")
print("             for each prediction start hour (0-23)")
print("Models: XGB and/or LR (specified below)")
print("Resolutions: Hourly, 30-minute, 15-minute, 10-minute")
print("Plot Quality: 18x10 inches, 300 DPI, enhanced styling with statistics")
print(f"Base predictions directory: {BASE_PREDICTIONS_DIR}")
print(f"Model selection: {MODEL_SELECTION}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Expected output: {8 if MODEL_SELECTION == 'both' else 4} plot(s) total")
print("=" * 80)

# Check if base predictions directory exists
if not os.path.exists(BASE_PREDICTIONS_DIR):
    print(f"\n[ERROR] Base predictions directory not found: {BASE_PREDICTIONS_DIR}")
    print("\nPlease update BASE_PREDICTIONS_DIR in the script to match your Drive structure.")
    print("\nCommon locations:")
    print("  - /content/drive/MyDrive/Solar PV electricity/final_hourly_predictions")
    print("  - /content/drive/MyDrive/Solar PV electricity/all_plants_predictions")
    print("\nListing Drive contents to help locate your predictions:")
    try:
        drive_base = "/content/drive/MyDrive"
        if os.path.exists(drive_base):
            items = os.listdir(drive_base)
            print(f"\nItems in {drive_base}:")
            for item in items[:20]:
                print(f"  - {item}")
    except:
        pass
else:
    # List what's in the base predictions directory
    print(f"\nBase predictions directory exists. Contents:")
    try:
        items = os.listdir(BASE_PREDICTIONS_DIR)
        print(f"  Found {len(items)} item(s)")
        for item in items[:10]:
            item_path = os.path.join(BASE_PREDICTIONS_DIR, item)
            if os.path.isdir(item_path):
                print(f"  [DIR]  {item}")
            else:
                print(f"  [FILE] {item}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more items")
    except Exception as e:
        print(f"  Could not list contents: {str(e)}")
    
    # Get the script path
    script_path = os.path.join(os.getcwd(), 'rmse_boxplots_all_plants.py')
    
    if not os.path.exists(script_path):
        print(f"\n[ERROR] Script not found: {script_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')[:10]}")
    else:
        print(f"\nScript found: {script_path}")
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Run RMSE box plots script
        cmd = [
            'python', script_path,
            '--predictions-dir', BASE_PREDICTIONS_DIR,
            '--model', MODEL_SELECTION,
            '--output-dir', OUTPUT_DIR
        ]
        
        print(f"\nRunning command:")
        print(f"  {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"\n[SUCCESS] RMSE Box Plot Generation Completed!")
            if result.stdout:
                print("\n--- Script Output ---")
                print(result.stdout)
            
            print(f"\n{'='*80}")
            print("Generated Output Files:")
            print(f"{'='*80}")
            print(f"Output directory: {OUTPUT_DIR}")
            
            if MODEL_SELECTION == 'both':
                print("\nOutput files (8 plots total - 4 for XGB, 4 for LR):")
                print("\nXGB Model plots:")
                print("  - rmse_boxplot_hourly_all_plants_xgb.png")
                print("  - rmse_boxplot_30_minute_all_plants_xgb.png")
                print("  - rmse_boxplot_15_minute_all_plants_xgb.png")
                print("  - rmse_boxplot_10_minute_all_plants_xgb.png")
                print("\nLR Model plots:")
                print("  - rmse_boxplot_hourly_all_plants_lr.png")
                print("  - rmse_boxplot_30_minute_all_plants_lr.png")
                print("  - rmse_boxplot_15_minute_all_plants_lr.png")
                print("  - rmse_boxplot_10_minute_all_plants_lr.png")
            else:
                print(f"\nOutput files (4 plots for {MODEL_SELECTION} model):")
                model_suffix = MODEL_SELECTION.lower()
                print(f"  - rmse_boxplot_hourly_all_plants_{model_suffix}.png")
                print(f"  - rmse_boxplot_30_minute_all_plants_{model_suffix}.png")
                print(f"  - rmse_boxplot_15_minute_all_plants_{model_suffix}.png")
                print(f"  - rmse_boxplot_10_minute_all_plants_{model_suffix}.png")
            
            print("\nAll plots:")
            print("  - High-quality: 18x10 inches, 300 DPI")
            print("  - X-axis: Starting hour of 24-hour sliding window (0-23)")
            print("  - Y-axis: RMSE (Capacity Factor)")
            print("  - Shows RMSE distribution across all plants for each starting hour")
            print("\nAdditional file:")
            print("  - rmse_summary_all_plants.csv (Summary data with all RMSE values by model, resolution, and hour)")
            print(f"{'='*80}")
            
            # List generated files
            try:
                if os.path.exists(OUTPUT_DIR):
                    generated_files = os.listdir(OUTPUT_DIR)
                    if len(generated_files) > 0:
                        print(f"\nGenerated files in output directory:")
                        for f in generated_files:
                            print(f"  - {f}")
            except:
                pass
        else:
            print(f"\n[ERROR] Script execution failed:")
            print("--- Standard Error ---")
            print(result.stderr)
            print("\n--- Standard Output ---")
            print(result.stdout)

print(f"\n{'='*80}")
print("Script Execution Complete")
print(f"{'='*80}")

