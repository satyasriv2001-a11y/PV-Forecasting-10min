# Plot Enhancement Guide

This guide explains how to make your plots more readable with larger fonts.

## Quick Start

### Option 1: Enhance Existing PNG Images (No Re-running Models)

If you already have plot PNG files and don't want to re-run models:

```bash
python enhance_plots.py --input-dir ./multi_resolution_predictions_LSTM_PV+NWP --output-dir ./enhanced_plots
```

This will:
- Upscale all PNG images by 2x (making text appear larger)
- Apply sharpening to improve text clarity
- Save enhanced images to the output directory

**Note:** Font sizes cannot be changed in existing PNG files, but upscaling makes text appear larger when viewing.

---

### Option 2: Regenerate Plots with Larger Fonts (Requires Saved Data)

For best results, regenerate plots from saved prediction data:

#### Step 1: Modify Your Prediction Script to Save Data

Add this code to your prediction script (e.g., `multi_resolution_predictions_1140.py`) in the `run_predictions_at_resolution` function, right before the return statement:

```python
# Save predictions for replotting
SAVE_PREDICTIONS_FOR_REPLOTTING = True  # Set to False to disable

if SAVE_PREDICTIONS_FOR_REPLOTTING:
    import pickle
    import os
    
    predictions_data_path = os.path.join(output_dir, 'predictions_data.pkl')
    
    predictions_data = {
        'all_predictions': all_predictions,
        'hourly_rmse_data': hourly_rmse_data,
        'all_pred_data': all_pred_data if 'all_pred_data' in locals() else None,
        'model_name': config.get('experiment_name', 'Model'),
        'resolution_name': resolution_name if 'resolution_name' in locals() else f"{resolution_minutes}-minute"
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(predictions_data_path, 'wb') as f:
        pickle.dump(predictions_data, f)
    
    print(f"\n  [INFO] Saved prediction data for replotting: {predictions_data_path}")
```

Or use the helper script:

```bash
python modify_script_to_save_data.py --script-path multi_resolution_predictions_1140.py
```

#### Step 2: Run Your Prediction Script

Run your prediction script as normal. It will now save `predictions_data.pkl` in the output directory.

#### Step 3: Regenerate Plots with Larger Fonts

```bash
python regenerate_plots_from_saved_data.py --data-dir ./multi_resolution_predictions_LSTM_PV+NWP --output-dir ./regenerated_plots_large_fonts
```

This will:
- Load the saved prediction data
- Regenerate all plots with larger fonts:
  - X/Y axis labels: 22pt (was 14pt)
  - Titles: 24pt (was 16pt)
  - Legends: 14-16pt (was 9-12pt)
  - Tick labels: 16pt (was default)
- Save regenerated plots to the output directory

---

## Scripts Overview

### 1. `enhance_plots.py`
Enhances existing PNG images by upscaling and sharpening.

**Usage:**
```bash
python enhance_plots.py --input-dir <plots_directory> [--output-dir <output_directory>] [--scale-factor 2.0] [--no-sharpen]
```

**Options:**
- `--input-dir`: Directory containing PNG plot files
- `--output-dir`: Output directory (default: `<input_dir>/enhanced`)
- `--scale-factor`: Upscaling factor (default: 2.0)
- `--no-sharpen`: Disable sharpening filter

### 2. `regenerate_plots_from_saved_data.py`
Regenerates plots from saved prediction data with larger fonts.

**Usage:**
```bash
python regenerate_plots_from_saved_data.py --data-dir <directory_with_predictions_data.pkl> [--output-dir <output_directory>]
```

**Requirements:**
- `predictions_data.pkl` file in the data directory (created by modified prediction script)

### 3. `modify_script_to_save_data.py`
Automatically modifies your prediction script to save data.

**Usage:**
```bash
python modify_script_to_save_data.py --script-path <path_to_prediction_script.py> [--output-path <output_script.py>]
```

**Note:** Creates a backup of your original script before modifying.

### 4. `ADD_DATA_SAVING_TO_SCRIPT.py`
Reference file showing the code snippet to manually add to your script.

---

## Font Size Comparison

| Element | Original | Enhanced |
|---------|----------|----------|
| X/Y Axis Labels | 14pt | 22pt |
| Titles | 16pt | 24pt |
| Legends | 9-12pt | 14-16pt |
| Tick Labels | Default | 16pt |
| Global Font | Default | 18pt |

---

## Examples

### Example 1: Enhance Existing Plots

```bash
# Enhance all PNGs in a directory
python enhance_plots.py --input-dir ./results/multi_resolution_predictions_LSTM_PV+NWP

# Enhanced plots will be saved to: ./results/multi_resolution_predictions_LSTM_PV+NWP/enhanced/
```

### Example 2: Regenerate from Saved Data

```bash
# Step 1: Modify script (one-time setup)
python modify_script_to_save_data.py --script-path multi_resolution_predictions_1140.py

# Step 2: Run prediction script (saves data)
python multi_resolution_predictions_1140.py --data-path data/Project1140.csv --model LSTM --complexity high --scenario PV+NWP

# Step 3: Regenerate plots with larger fonts
python regenerate_plots_from_saved_data.py --data-dir ./multi_resolution_predictions_LSTM_PV+NWP --output-dir ./plots_large_fonts
```

---

## Troubleshooting

### "No prediction data files found"
- Make sure you've modified your prediction script to save data
- Check that `predictions_data.pkl` exists in the data directory
- Verify the script ran successfully and saved the data

### "No PNG files found"
- Check that the input directory path is correct
- Verify PNG files exist in the directory
- Try using absolute paths

### Enhanced images still have small text
- PNG images cannot have their fonts changed - only regenerated plots can
- Use `regenerate_plots_from_saved_data.py` for true font size changes
- Try increasing `--scale-factor` (e.g., 3.0 or 4.0) for more upscaling

---

## Best Practices

1. **For existing plots:** Use `enhance_plots.py` for quick improvement
2. **For future plots:** Modify your prediction script to save data, then use `regenerate_plots_from_saved_data.py`
3. **For presentations:** Use regenerated plots with larger fonts for better readability
4. **For publications:** Consider using even larger fonts (modify font sizes in the regeneration script)

---

## Questions?

If you encounter issues:
1. Check that all required files exist
2. Verify file paths are correct
3. Ensure you have write permissions in output directories
4. Check that pickle files were created successfully

