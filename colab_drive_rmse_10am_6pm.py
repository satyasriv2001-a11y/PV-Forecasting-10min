# =============================================================================
# Google Colab: Plant Data from Drive → 10-Minute Predictions → RMSE (10 AM–6 PM)
# =============================================================================
# Copy each section below into a separate Colab cell and run in order.
# Do not run the whole file as a script (magic commands require Colab/Jupyter).
# 1. Mount Drive, 2. Set paths (edit DATA_PATH_IN_DRIVE), 3. Clone + install,
# 4. Run predictions, 5. Display RMSE (10 AM–6 PM).
# =============================================================================

# --- CELL 1: Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')

# --- CELL 2: CONFIG – set your paths here ---
# Path to your plant CSV inside Google Drive (after mount it's /content/drive/MyDrive/...)
DATA_PATH_IN_DRIVE = "/content/drive/MyDrive/AI_Models_for_Solar_Energy/Project1140.csv"

# Where to save results (predictions + RMSE CSV). Use a Drive path to keep results.
OUTPUT_DIR_IN_DRIVE = "/content/drive/MyDrive/Solar PV electricity/rmse_10am_6pm_output"

# Repo to clone: must contain multi_resolution_predictions_1140.py and data/, train/ packages.
# If your 10-minute code is in a different repo or branch, change these (e.g. .../PV-Forecasting-10min).
REPO_URL = "https://github.com/satyasriv2001-a11y/PV-Forecasting"
CLONE_DIR = "PV-Forecasting"

# Model settings (optional to change)
MODEL = "XGB"           # or "Linear" for LR
COMPLEXITY = "high"
SCENARIO = "PV+NWP"
LOOKBACK_HOURS = "24"
USE_TIME_ENCODING = False  # set True to use time encoding

# --- CELL 3: Clone repo and install dependencies ---
%cd /content
!rm -rf {CLONE_DIR}
!git clone {REPO_URL} {CLONE_DIR}
%cd /content/{CLONE_DIR}
!pip install -q -r requirements.txt
# If using XGB and Colab has no XGBoost GPU: patch train_ml to use CPU (avoids "XGBoost GPU not available! Cannot train.")
# Run patch if present (add patch_xgb_cpu_for_colab.py to your repo so it's available after clone).
!python patch_xgb_cpu_for_colab.py . || true

# --- CELL 4: Run 10-minute predictions and generate RMSE (10 AM–6 PM) ---
import subprocess
import os
import pandas as pd

# Use same clone dir as Cell 3 (must match CLONE_DIR from Cell 2)
clone_dir = CLONE_DIR
repo_root = f"/content/{clone_dir}"
script_name = "multi_resolution_predictions_1140.py"
script_path = os.path.join(repo_root, script_name)

if not os.path.isfile(script_path):
    raise FileNotFoundError(
        f"Prediction script not found: {script_path}\n"
        "The cloned repo must contain multi_resolution_predictions_1140.py (and data/, train/ packages).\n"
        "Edit REPO_URL / CLONE_DIR in Cell 2 to point to the repo that has the 10-minute code."
    )

data_path = DATA_PATH_IN_DRIVE
output_dir = OUTPUT_DIR_IN_DRIVE
os.makedirs(output_dir, exist_ok=True)

if not os.path.isfile(data_path):
    raise FileNotFoundError(
        f"Plant data not found: {data_path}\n"
        "Edit DATA_PATH_IN_DRIVE in Cell 2 to point to your CSV in Google Drive."
    )

plant_name = os.path.splitext(os.path.basename(data_path))[0]
print("=" * 80)
print("10-Minute Resolution Predictions → RMSE (10 AM – 6 PM)")
print("=" * 80)
print(f"Data: {data_path}")
print(f"Plant: {plant_name}")
print(f"Output: {output_dir}")
print("=" * 80)

cmd = [
    "python", "multi_resolution_predictions_1140.py",
    "--data-path", data_path,
    "--model", MODEL,
    "--complexity", COMPLEXITY,
    "--scenario", SCENARIO,
    "--lookback", LOOKBACK_HOURS,
    "--output-dir", output_dir,
]
if not USE_TIME_ENCODING:
    cmd.append("--no-time-encoding")

result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)

if result.returncode != 0:
    print("STDERR:")
    print(result.stderr or "(empty)")
    print("\nSTDOUT:")
    out = result.stdout or ""
    if len(out) > 6000:
        print(out[:3000])
        print("\n... [truncated] ...\n")
        print(out[-3000:])
    else:
        print(out)
    print("\nFix the error above (e.g. missing data/train packages, wrong paths) then re-run this cell.")
else:
    print("\n[OK] Predictions completed.")

# --- CELL 5: Load and display RMSE (10 AM – 6 PM) ---
import pandas as pd
import os

rmse_file = os.path.join(output_dir, "rmse_10am_6pm_by_resolution.csv")

if os.path.isfile(rmse_file):
    df = pd.read_csv(rmse_file)
    print("\n" + "=" * 80)
    print("RMSE (10 AM – 6 PM) – 10-minute resolution")
    print("=" * 80)
    for _, row in df.iterrows():
        res = row.get("Resolution", "10-minute")
        date_used = row.get("Date", "N/A")
        rmse = row.get("RMSE_10AM_6PM", float("nan"))
        mae = row.get("MAE_10AM_6PM", float("nan"))
        n = int(row.get("Number_of_Samples_10AM_6PM", 0))
        print(f"  Resolution: {res}")
        print(f"  Date:      {date_used}")
        if pd.notna(rmse):
            print(f"  RMSE:      {rmse:.4f}")
            print(f"  MAE:       {mae:.4f}")
        else:
            print(f"  RMSE:      N/A (no valid samples)")
        print(f"  Samples:   {n}")
        print("=" * 80)
    # One-line summary for quick readout
    if "RMSE_10AM_6PM" in df.columns:
        valid = df["RMSE_10AM_6PM"].dropna()
        if len(valid) > 0:
            print("\n>>> RMSE (10 AM – 6 PM) =", f"{valid.iloc[0]:.4f}")
    print("\nFull table:")
    print(df.to_string(index=False))
else:
    print(f"[WARNING] RMSE file not found: {rmse_file}")
    print("Check Cell 4 output for errors.")
