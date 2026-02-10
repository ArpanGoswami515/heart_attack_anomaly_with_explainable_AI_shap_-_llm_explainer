# Tuning Results Directory

This directory stores hyperparameter tuning results.

## Files

- `tuning_results_YYYYMMDD_HHMMSS.json` - Full tuning results with all metrics
- `optimized_config.py` or `quick_optimized_config.py` - Python code snippets with optimal parameters

## Usage

After tuning completes, apply the parameters:
```bash
python apply_tuned_params.py
```

This will automatically use the latest tuning results from this directory.

## Result File Structure

```json
{
  "isolation_forest": {
    "best_params": {...},
    "best_score": 0.8543,
    "all_results": [...]
  },
  "one_class_svm": {...},
  "autoencoder": {...}
}
```

## Notes

- Results are timestamped to track tuning history
- Keep this directory in .gitignore (results are dataset-specific)
- Latest results are automatically used by apply_tuned_params.py
