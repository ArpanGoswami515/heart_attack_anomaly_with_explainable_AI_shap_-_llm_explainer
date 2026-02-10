"""
Quick hyperparameter tuning with reduced search space for faster results.
Use this for rapid experimentation before running full tuning.
"""

import numpy as np
import os
from typing import Dict, Any

from data.data_loader import HeartAttackDataLoader
from preprocessing.preprocess import DataPreprocessor
from tune_hyperparameters import HyperparameterTuner
from config.settings import Config


def main():
    """Quick tuning with smaller parameter grids."""
    print("="*70)
    print("QUICK HYPERPARAMETER TUNING (Reduced Search Space)")
    print("="*70)
    
    # Configuration
    USE_SYNTHETIC = False
    METRIC_TO_OPTIMIZE = "auroc"
    
    print(f"\nConfiguration:")
    print(f"  Optimization metric: {METRIC_TO_OPTIMIZE}")
    print(f"  Using synthetic data: {USE_SYNTHETIC}")
    print(f"  Mode: QUICK (reduced search space)")
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    if USE_SYNTHETIC:
        print("Generating synthetic dataset...")
        df = HeartAttackDataLoader.generate_sample_dataset(
            n_samples=1000,
            contamination=0.1,
            random_state=Config.RANDOM_SEED
        )
        data_loader = HeartAttackDataLoader(
            file_path="synthetic",
            random_state=Config.RANDOM_SEED
        )
        data_loader.data = df
    else:
        dataset_path = os.path.join(Config.DATA_DIR, "heart_attack_data.csv")
        data_loader = HeartAttackDataLoader(
            file_path=dataset_path,
            random_state=Config.RANDOM_SEED
        )
        data_loader.load_data()
    
    # Get features and labels
    X, y = data_loader.get_features_and_labels(target_column="target")
    feature_names = data_loader.get_feature_names()
    
    print(f"✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Convert y to numpy array for proper calculation
    if y is not None:
        y_vals = y.values if hasattr(y, 'values') else y
        anomaly_ratio = float(np.mean(np.asarray(y_vals)))  # type: ignore
        print(f"✓ Anomaly ratio: {anomaly_ratio:.2%}")
    
    # Split data
    X_train_val, X_test, y_train_val, y_test = data_loader.train_test_split(
        X, y, test_size=0.2, stratify=True
    )
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=Config.RANDOM_SEED
    )
    
    # Convert to numpy arrays
    y_train = y_train.values if hasattr(y_train, 'values') else y_train
    y_val = y_val.values if hasattr(y_val, 'values') else y_val
    
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Validation set: {len(X_val)} samples")
    
    # Preprocess
    preprocessor = DataPreprocessor(scaling_method="standard")
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    
    X_train_normal = X_train_scaled[y_train == 0]
    print(f"✓ Training on {len(X_train_normal)} normal samples")
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        X_train=X_train_normal,
        X_val=X_val_scaled,
        y_train=y_train,
        y_val=y_val,
        metric=METRIC_TO_OPTIMIZE,
        random_state=Config.RANDOM_SEED
    )
    
    # Reduced parameter grids for quick tuning
    
    # Isolation Forest - Quick grid (12 combinations)
    if_grid = {
        "n_estimators": [100, 200],
        "contamination": [0.1, 0.15],
        "max_samples": ["auto", 512],
        "max_features": [0.75, 1.0],
        "bootstrap": [True]
    }
    
    # One-Class SVM - Quick grid (18 combinations)
    svm_grid = {
        "kernel": ["rbf"],
        "gamma": ["scale", "auto", 0.01],
        "nu": [0.05, 0.1, 0.15],
        "degree": [3],
        "coef0": [0.0]
    }
    
    # Autoencoder - Quick grid (8 combinations)  
    ae_grid = {
        "encoding_dim": [8, 16],
        "hidden_dims": [[32, 16], [64, 32]],
        "learning_rate": [0.001, 0.0005],
        "batch_size": [32],
        "epochs": [50],
        "early_stopping_patience": [10]
    }
    
    print("\n" + "="*70)
    print("STARTING QUICK TUNING")
    print("="*70)
    
    # Tune models
    if_params, if_score = tuner.tune_isolation_forest(if_grid)
    svm_params, svm_score = tuner.tune_one_class_svm(svm_grid)
    ae_params, ae_score = tuner.tune_autoencoder(ae_grid)
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    results_path = tuner.save_results(output_dir="tuning_results")
    
    config_code = tuner.generate_config_code()
    config_path = os.path.join("tuning_results", "quick_optimized_config.py")
    with open(config_path, 'w') as f:
        f.write(config_code)
    
    print(f"✓ Config code saved to: {config_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("QUICK TUNING COMPLETE - SUMMARY")
    print("="*70)
    
    print(f"\nOptimal Parameters (optimized for {METRIC_TO_OPTIMIZE}):")
    print(f"\nIsolation Forest: {if_score:.4f}")
    print(f"One-Class SVM: {svm_score:.4f}")
    print(f"Autoencoder: {ae_score:.4f}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review results (saved to tuning_results/)")
    print("2. Apply parameters: python apply_tuned_params.py")
    print("3. Or run full tuning: python tune_hyperparameters.py")
    print("="*70)


if __name__ == "__main__":
    main()
