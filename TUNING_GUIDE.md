# Hyperparameter Tuning Guide

## Overview
This system provides automated hyperparameter tuning for all three anomaly detection models to find optimal configurations for your specific dataset.

## Quick Start

### Option 1: Quick Tune (Recommended for First Run)
Fast tuning with reduced search space (~5-10 minutes):
```bash
python quick_tune.py
```

### Option 2: Full Tune
Comprehensive search with larger parameter grid (~30-60 minutes):
```bash
python tune_hyperparameters.py
```

### Option 3: Apply Previously Tuned Parameters
```bash
python apply_tuned_params.py
```

## Workflow

```
┌─────────────────────┐
│  1. Run Tuning      │
│  quick_tune.py      │
│  or                 │
│  tune_hyper...py    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  2. Review Results  │
│  tuning_results/    │
│  (JSON files)       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  3. Apply Params    │
│  apply_tuned_       │
│  params.py          │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  4. Train & Use     │
│  python main.py     │
└─────────────────────┘
```

## Files

### Core Scripts

1. **`tune_hyperparameters.py`** - Full hyperparameter tuning
   - Comprehensive parameter grid search
   - All three models: Isolation Forest, One-Class SVM, Autoencoder
   - Saves detailed results and optimal parameters
   - ~30-60 minutes runtime

2. **`quick_tune.py`** - Rapid tuning with reduced search space
   - Smaller parameter grids for faster results
   - Good starting point for experimentation
   - ~5-10 minutes runtime

3. **`apply_tuned_params.py`** - Apply tuned parameters to config
   - Automatically updates `config/settings.py`
   - Creates backup of original config
   - Prompts for confirmation before applying

### Results

All results are saved to `tuning_results/` directory:
- `tuning_results_YYYYMMDD_HHMMSS.json` - Full results with all metrics
- `optimized_config.py` or `quick_optimized_config.py` - Python code snippet

## Configuration Options

### Optimization Metric

Edit the script to choose which metric to optimize:

```python
METRIC_TO_OPTIMIZE = "auroc"  # Options:
# - "auroc": Area Under ROC Curve (default, best overall)
# - "precision_at_10": Precision in top 10% highest risk
# - "f1_score": F1 Score balance
```

### Dataset Source

```python
USE_SYNTHETIC = False  # False = use real dataset, True = synthetic data
```

## Parameter Grids

### Isolation Forest

**Quick Tune:**
- n_estimators: [100, 200]
- contamination: [0.1, 0.15]
- max_samples: ["auto", 512]
- Total: ~12 combinations

**Full Tune:**
- n_estimators: [50, 100, 200, 300]
- contamination: [0.05, 0.1, 0.15, 0.2]
- max_samples: ["auto", 256, 512, 1024]
- max_features: [0.5, 0.75, 1.0]
- bootstrap: [True, False]
- Total: ~192 combinations

### One-Class SVM

**Quick Tune:**
- kernel: ["rbf"]
- gamma: ["scale", "auto", 0.01]
- nu: [0.05, 0.1, 0.15]
- Total: ~9 combinations

**Full Tune:**
- kernel: ["rbf", "poly", "sigmoid"]
- gamma: ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
- nu: [0.05, 0.1, 0.15, 0.2, 0.25]
- degree: [2, 3, 4] (poly only)
- coef0: [0.0, 0.5, 1.0] (poly/sigmoid)
- Total: ~200+ combinations

### Autoencoder

**Quick Tune:**
- encoding_dim: [8, 16]
- hidden_dims: [[32, 16], [64, 32]]
- learning_rate: [0.001, 0.0005]
- Total: ~8 combinations

**Full Tune:**
- encoding_dim: [4, 8, 16]
- hidden_dims: [[16, 8], [32, 16], [64, 32], [32, 16, 8]]
- learning_rate: [0.001, 0.0005, 0.0001]
- batch_size: [16, 32, 64]
- epochs: [50, 100]
- Total: ~216 combinations

## Understanding Results

### Result File Structure

```json
{
  "isolation_forest": {
    "best_params": {
      "n_estimators": 200,
      "contamination": 0.15,
      ...
    },
    "best_score": 0.8543,
    "all_results": [...]
  },
  "one_class_svm": {...},
  "autoencoder": {...}
}
```

### Metrics Explained

- **AUROC**: Area Under ROC Curve (0-1, higher is better)
  - Best overall metric for anomaly detection
  - Measures model's ability to rank anomalies higher than normal samples
  - 0.5 = random, 1.0 = perfect

- **Precision@10%**: Precision in top 10% highest risk predictions
  - How many true anomalies are in the highest risk group
  - Important for clinical applications (focus on highest risk patients)

- **F1 Score**: Harmonic mean of precision and recall
  - Balance between false positives and false negatives

## Example: Complete Workflow

### Step 1: Run Quick Tuning
```bash
python quick_tune.py
```

Output:
```
==================================================================
QUICK HYPERPARAMETER TUNING (Reduced Search Space)
==================================================================

Testing 12 parameter combinations...
✓ [5/12] New best auroc: 0.8234
  Params: n_estimators=200, contamination=0.1, max_samples=512

✓ Best auroc: 0.8543
✓ Results saved to: tuning_results/tuning_results_20260210_143022.json
```

### Step 2: Review Results
```bash
# Check the JSON file
cat tuning_results/tuning_results_20260210_143022.json

# Or view the Python config snippet
cat tuning_results/quick_optimized_config.py
```

### Step 3: Apply Parameters
```bash
python apply_tuned_params.py
```

Output:
```
Using tuning results from: tuning_results/tuning_results_20260210_143022.json

Found optimized parameters for:
  - isolation_forest: 0.8543
  - one_class_svm: 0.8012
  - autoencoder: 0.7856

Update config/settings.py with these parameters? (y/n): y

✓ Backup created: config/settings_backup_20260210_143050.py
✓ Updated Isolation Forest (score: 0.8543)
✓ Updated One-Class SVM (score: 0.8012)
✓ Updated Autoencoder (score: 0.7856)
✓ Config file updated: config/settings.py
```

### Step 4: Train with Optimized Parameters
```bash
python main.py
```

## Tips & Best Practices

### 1. Start with Quick Tune
- Use `quick_tune.py` first to get baseline optimal parameters
- If results are satisfactory, use them directly
- If not, run full tuning for more comprehensive search

### 2. Choose the Right Metric
- **AUROC**: Best for general-purpose anomaly detection (recommended)
- **Precision@10%**: When you want to minimize false positives in highest risk
- **F1 Score**: When you need balance between precision and recall

### 3. Consider Your Dataset
- Small dataset (<500 samples): Use quick_tune.py
- Large dataset (>1000 samples): Full tuning is beneficial
- Imbalanced data: Ensure stratification is working correctly

### 4. Validation Strategy
- System uses 60% train / 20% validation / 20% test split
- Training done on normal samples only (unsupervised)
- Evaluation on mixed normal/anomaly validation set

### 5. Computational Resources
- **Isolation Forest**: Fast (~1-2 min for full grid)
- **One-Class SVM**: Medium (~5-10 min for full grid)
- **Autoencoder**: Slow (~20-40 min for full grid)
  - Requires training neural networks
  - Consider reducing epochs in grid for faster tuning

### 6. Retuning Frequency
- Retune when you get new data
- Retune if model performance degrades
- Consider retuning quarterly for production systems

## Custom Parameter Grids

You can modify parameter grids in the scripts:

```python
# In tune_hyperparameters.py or quick_tune.py

# Custom Isolation Forest grid
if_grid = {
    "n_estimators": [150, 250, 350],  # Your custom values
    "contamination": [0.08, 0.12, 0.16],
    "max_samples": [400, 800],
    # ... add more parameters
}

# Pass to tuner
if_params, if_score = tuner.tune_isolation_forest(if_grid)
```

## Troubleshooting

### Issue: Tuning takes too long
**Solution**: 
- Use `quick_tune.py` instead
- Reduce parameter grid size
- Reduce epochs for Autoencoder

### Issue: Out of memory errors
**Solution**:
- Reduce Autoencoder batch_size
- Reduce SHAP num_samples during evaluation
- Process fewer parameter combinations at once

### Issue: Poor tuning results
**Solution**:
- Check if you have enough validation data (need ~20% of dataset)
- Ensure data is properly preprocessed
- Try different optimization metrics
- Increase parameter grid search space

### Issue: Can't apply tuned parameters
**Solution**:
- Make sure tuning completed and saved results
- Check that tuning_results/ directory exists
- Verify JSON file is not corrupted

## Advanced: Manual Parameter Selection

If you want to manually select parameters from results:

1. Open `tuning_results/tuning_results_YYYYMMDD_HHMMSS.json`
2. Find `"best_params"` for each model
3. Manually edit `config/settings.py`:

```python
# In config/settings.py

ISOLATION_FOREST_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,        # From tuning results
    "contamination": 0.15,      # From tuning results
    "max_samples": 512,         # From tuning results
    "max_features": 0.75,       # From tuning results
    "bootstrap": True,          # From tuning results
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}
```

## Performance Expectations

Typical AUROC scores after tuning:
- **Isolation Forest**: 0.75 - 0.90
- **One-Class SVM**: 0.70 - 0.85
- **Autoencoder**: 0.65 - 0.85

If scores are lower:
- Check data quality
- Verify preprocessing
- Ensure sufficient normal samples for training
- Consider ensemble methods

---

**For Questions or Issues:**
- Review tuning logs in console output
- Check JSON files in tuning_results/
- Verify data loading and preprocessing steps
