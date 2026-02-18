# =============================================================================
# Google Colab: Plant Data from Drive → 10-Minute Predictions → RMSE (10 AM–6 PM)
# =============================================================================
# Copy each section below into a separate Colab cell and run in order.
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

# Model settings (optional to change)
MODEL = "XGB"           # or "Linear" for LR
COMPLEXITY = "high"
SCENARIO = "PV+NWP"
LOOKBACK_HOURS = "24"
USE_TIME_ENCODING = False  # set True to use time encoding

# --- CELL 3: Clone repo and install dependencies ---
%cd /content
!rm -rf PV-Forecasting
!git clone https://github.com/satyasriv2001-a11y/PV-Forecasting
%cd /content/PV-Forecasting
!pip install -q -r requirements.txt

# --- CELL 4: Run 10-minute predictions and generate RMSE (10 AM–6 PM) ---
import subprocess
import os
import pandas as pd

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

result = subprocess.run(cmd, capture_output=True, text=True, cwd="/content/PV-Forecasting")

if result.returncode != 0:
    print("STDERR:", result.stderr)
    print("STDOUT (last 3000 chars):", result.stdout[-3000:])
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
    valid = df["RMSE_10AM_6PM"].dropna()
    if len(valid) > 0:
        print("\n>>> RMSE (10 AM – 6 PM) =", f"{valid.iloc[0]:.4f}")
    print("\nFull table:")
    print(df.to_string(index=False))
else:
    print(f"[WARNING] RMSE file not found: {rmse_file}")
    print("Check Cell 4 output for errors.")
