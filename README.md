# PV-Forecasting: Multi-Plant Solar Power Prediction

A comprehensive system for solar photovoltaic power forecasting using multiple deep learning and machine learning models, with support for batch processing multiple plants.

**Repository**: https://github.com/zhesun-0209/PV-Forecasting  
**Version**: 2.0 (2025-10-09)

---

## Key Features

- **8 Models**: LSTM, GRU, Transformer, TCN, Random Forest, XGBoost, LightGBM, Linear Regression
- **284 Experiments per Plant**: Comprehensive model and feature comparison
- **Multi-Plant Support**: Batch process 100+ plants with automatic configuration
- **Resume Support**: Automatically resume from interrupted experiments
- **GPU Accelerated**: 6/8 models use GPU by default (85% of experiments)
- **Sequential Split**: Time-series data split for realistic evaluation
- **Production Ready**: Optimized parameters and error handling

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/zhesun-0209/PV-Forecasting.git
cd PV-Forecasting
pip install -r requirements.txt
```

### 2. Prepare Data

Put your CSV files in the `data/` directory:
```
data/
├── Project1001.csv
├── Project1002.csv
└── ...
```

### 3. Generate Configurations

```bash
# Automatically create configs for all CSV files
python batch_create_configs.py
```

### 4. Run Experiments

```bash
# Single plant (284 experiments, ~2-3 hours on GPU)
python run_main_experiments.py

# Multiple plants (batch processing)
python run_experiments_multi_plant.py

# Check status of all plants
python check_all_plants_status.py
```

---

## Multi-Plant Batch Experiments

### Batch Processing Commands

```bash
# Run all plants (auto-resume supported)
python run_experiments_multi_plant.py

# Run first 25 plants
python run_experiments_multi_plant.py --max-plants 25

# Run plants 26-50 (skip first 25)
python run_experiments_multi_plant.py --skip 25 --max-plants 25

# Run specific plants
python run_experiments_multi_plant.py --plants 1001 1002 1003

# Check status only (no execution)
python run_experiments_multi_plant.py --status-only
```

### Resume Support

The system automatically detects completed experiments:
- Scans for existing `results_{plant_id}_*.csv` files
- Skips plants with 284/284 experiments complete
- Resumes incomplete plants from last checkpoint
- Works across Colab session interruptions

```bash
# If interrupted, simply re-run the same command
python run_experiments_multi_plant.py  # Will resume automatically
```

---

## Running on Google Colab

### Quick Setup (100 Plants)

```python
# 1. Clone repository
!git clone https://github.com/zhesun-0209/PV-Forecasting.git
%cd PV-Forecasting

# 2. Install dependencies
!pip install -q -r requirements.txt

# 3. Mount Google Drive and copy datasets
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/your_datasets_folder/*.csv data/

# 4. Generate configs for all plants
!python batch_create_configs.py

# 5. Run experiments - Results save directly to Drive!

# Method 1: Direct commands (recommended)
!python run_experiments_multi_plant.py --max-plants 25 --output-dir "/content/drive/MyDrive/Solar PV electricity/results"
!python run_experiments_multi_plant.py --skip 25 --max-plants 25 --output-dir "/content/drive/MyDrive/Solar PV electricity/results"
!python run_experiments_multi_plant.py --skip 50 --max-plants 25 --output-dir "/content/drive/MyDrive/Solar PV electricity/results"
!python run_experiments_multi_plant.py --skip 75 --max-plants 25 --output-dir "/content/drive/MyDrive/Solar PV electricity/results"

# Method 2: Using Python variable (for easier modification)
import subprocess
DRIVE_PATH = "/content/drive/MyDrive/Solar PV electricity/results"

subprocess.run(['python', 'run_experiments_multi_plant.py', '--max-plants', '25', '--output-dir', DRIVE_PATH])
subprocess.run(['python', 'run_experiments_multi_plant.py', '--skip', '25', '--max-plants', '25', '--output-dir', DRIVE_PATH])
subprocess.run(['python', 'run_experiments_multi_plant.py', '--skip', '50', '--max-plants', '25', '--output-dir', DRIVE_PATH])
subprocess.run(['python', 'run_experiments_multi_plant.py', '--skip', '75', '--max-plants', '25', '--output-dir', DRIVE_PATH])
```

### Check Status from Drive

```python
# Check what's completed in your Drive folder (use quotes for path with spaces!)
!python check_all_plants_status.py --output-dir "/content/drive/MyDrive/Solar PV electricity/results"

# Or check status in current directory
!python check_all_plants_status.py
```

**IMPORTANT**: Always use **quotes** around paths with spaces!

### Key Features

**Direct Drive Save**: Results save directly to Drive, no manual copying needed
**Auto Resume**: Automatically resume from Drive on session restart
**No Data Loss**: Each experiment result saved immediately to Drive

**For 100 Plants**:
- Split into 4 batches of 25 plants each
- All results save directly to Drive path
- Auto-resume works across Colab sessions
- Each batch: ~60-75 hours
- Total time: ~60-75 hours (with 4 parallel notebooks)

---

## Project Structure

```
PV-Forecasting/
├── data/                              # CSV data files (one per plant)
├── config/
│   ├── plant_template.yaml           # Configuration template
│   └── plants/                       # Auto-generated plant configs
├── models/                            # Model implementations
│   ├── ml_models.py                  # ML models
│   ├── rnn_models.py                 # LSTM, GRU
│   ├── transformer.py                # Transformer
│   └── tcn.py                        # TCN
├── train/                             # Training pipelines
├── eval/                              # Evaluation utilities
├── utils/                             # GPU utils
├── run_main_experiments.py           # Single plant runner
├── run_experiments_multi_plant.py    # Multi-plant batch runner
├── batch_create_configs.py           # Auto config generator
├── check_all_plants_status.py        # Status checker
└── README.md                          # This file
```

---

## Experiment Design

### 284 Experiments per Plant

**Deep Learning (160 experiments)**:
- Models: LSTM, GRU, Transformer, TCN (4)
- Complexity: low, high (2)
- Lookback: 24h, 72h (2)
- Time Encoding: yes, no (2)
- Feature Scenarios: PV, PV+HW, PV+NWP, PV+NWP+ (4)
- NWP-only: NWP, NWP+ (2)
- Total: 4 × 2 × (2 × 2 × 4 + 2 × 2) = 160

**Machine Learning (120 experiments)**:
- Models: RF, XGBoost, LightGBM (3)
- Same variables as DL
- Total: 3 × 2 × (2 × 2 × 4 + 2 × 2) = 120

**Linear (4 experiments)**:
- NWP scenarios with time encoding options
- Total: 2 × 2 = 4

**Grand Total**: 284 experiments per plant

---

## Feature Scenarios

1. **PV**: Historical power only
2. **PV+HW**: Historical power + Historical weather
3. **PV+NWP**: Historical power + Weather forecast
4. **PV+NWP+**: Historical power + Ideal forecast (actual weather)
5. **NWP**: Weather forecast only (no historical power)
6. **NWP+**: Ideal forecast only

---

## Model Parameters

### Deep Learning Models

**Low Complexity**:
- Epochs: 20
- d_model: 16
- num_layers: 1
- batch_size: 64

**High Complexity**:
- Epochs: 50
- d_model: 32
- num_layers: 2
- batch_size: 64

### Machine Learning Models

**Low Complexity**:
- n_estimators: 10
- max_depth: 1
- learning_rate: 0.2

**High Complexity**:
- n_estimators: 30
- max_depth: 3
- learning_rate: 0.1

---

## Data Requirements

### CSV Format

Required columns:
- `Year`, `Month`, `Day`, `Hour`: Timestamp information
- Power column: Actual PV generation (auto-detected)
- Weather columns: Temperature, irradiance, humidity, etc. (optional)

### Naming Convention

Supported formats (auto-detection):
- `Project1140.csv` -> Plant ID: 1140
- `Plant1140.csv` -> Plant ID: 1140
- `1140.csv` -> Plant ID: 1140

---

## Configuration System

### Template-Based Configuration

All plants share a base template (`config/plant_template.yaml`) with plant-specific overrides in `config/plants/Plant{ID}.yaml`.

**Auto-Generation**:
```bash
python batch_create_configs.py
```

This scans `data/` and creates:
- Plant-specific configs
- Auto-detected date ranges
- Standardized parameters

**Manual Editing**:
```yaml
# config/plants/Plant1140.yaml
plant_id: "1140"
data_path: "data/Project1140.csv"
start_date: "2022-01-01"
end_date: "2024-09-28"
shuffle_split: false  # Use sequential split
random_seed: 42
```

---

## Evaluation Metrics

- **MAE**: Mean Absolute Error (primary metric)
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination
- **Training Time**: Seconds per experiment
- **Model Size**: Number of parameters

All metrics calculated on test set (10% of data, sequential split).

---

## GPU Acceleration

| Model | GPU Library | Auto-Detection |
|-------|-------------|----------------|
| LSTM | PyTorch | Yes |
| GRU | PyTorch | Yes |
| Transformer | PyTorch | Yes |
| TCN | PyTorch | Yes |
| XGBoost | xgboost | Yes |
| LightGBM | lightgbm | Yes |
| Random Forest | sklearn | CPU (cuML optional) |
| Linear | sklearn | CPU (cuML optional) |

**Coverage**: 6/8 models (240/284 experiments, 85%) use GPU automatically.

---

## Advanced Usage

### Status Monitoring

```bash
# Check completion status of all plants
python check_all_plants_status.py

# Monitor single plant progress
python check_progress.py

# Real-time monitoring
python monitor_progress.py
```

### Parallel Execution (Colab)

Run multiple batches in parallel notebooks:

**Notebook 1**:
```python
!python run_experiments_multi_plant.py --max-plants 25
```

**Notebook 2**:
```python
!python run_experiments_multi_plant.py --skip 25 --max-plants 25
```

**Notebook 3**:
```python
!python run_experiments_multi_plant.py --skip 50 --max-plants 25
```

**Notebook 4**:
```python
!python run_experiments_multi_plant.py --skip 75 --max-plants 25
```

Total time: ~60-75 hours wall-clock time (parallel execution)

---

## Results Format

### Output Files

**Per Plant**:
- `results_{plant_id}_{timestamp}.csv`: 284 experiments × metrics

**Columns**:
- plant_id, experiment_name, model, complexity, scenario
- lookback_hours, use_time_encoding
- mae, rmse, r2
- train_time_sec, test_samples, best_epoch, param_count
- status (SUCCESS/FAILED)

### Aggregation

```python
import pandas as pd
import glob

# Load all results
all_results = []
for f in glob.glob('results_*.csv'):
    df = pd.read_csv(f)
    all_results.append(df)

combined = pd.concat(all_results, ignore_index=True)
print(f"Total experiments: {len(combined)}")
print(f"Best configuration: {combined.nsmallest(1, 'mae')}")
```

---

## Performance Benchmarks

**Sequential vs Shuffle Split** (Plant 1140):
- Sequential MAE: 4.85 (17% better)
- Shuffle MAE: 5.86
- Sequential R²: 0.84 (13% better)
- Shuffle R²: 0.75

**Best Configuration**:
- Model: LightGBM (high complexity)
- Features: PV+NWP
- Lookback: 24h
- MAE: 2.94, RMSE: 5.15, R²: 0.94

**Computational Cost** (per plant):
- Total time: ~2.5 hours (GPU)
- DL experiments: ~90% of time
- ML experiments: ~10% of time

---

## Troubleshooting

**Issue**: GPU OOM during prediction  
**Solution**: Batch prediction automatically enabled (batch_size=1000)

**Issue**: Colab session timeout  
**Solution**: Resume support - re-run same command to continue

**Issue**: Different data date ranges  
**Solution**: Edit `config/plants/Plant{ID}.yaml` manually or use batch_create_configs options

**Issue**: Missing dependencies  
**Solution**: `pip install -r requirements.txt`

---

## Citation

If you use this code, please cite:

```bibtex
@misc{pv-forecasting-2025,
  title={PV-Forecasting: Multi-Plant Solar Power Prediction System},
  author={Zhe Sun},
  year={2025},
  publisher={GitHub},
  url={https://github.com/zhesun-0209/PV-Forecasting}
}
```

---

## License

MIT License

---

## Contact

For questions or issues, please submit an Issue or Pull Request on GitHub.

---

**Last Updated**: 2025-10-09  
**Version**: 2.0 (Multi-Plant Optimized)
