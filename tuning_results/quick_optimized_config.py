# Optimized hyperparameters from tuning
# Generated: 2026-02-10 11:54:09
# Optimization metric: auroc

# Isolation Forest - auroc: 0.8597
ISOLATION_FOREST_PARAMS: Dict[str, Any] = {
    "bootstrap": True,
    "contamination": 0.1,
    "max_features": 1.0,
    "max_samples": "auto",
    "n_estimators": 200,
    "random_state": 42,
    "n_jobs": -1,
}

# One-Class SVM - auroc: 0.7264
ONE_CLASS_SVM_PARAMS: Dict[str, Any] = {
    "gamma": "scale",
    "kernel": "rbf",
    "nu": 0.15,
}

# Autoencoder - auroc: 0.7708
AUTOENCODER_PARAMS: Dict[str, Any] = {
    "batch_size": 32,
    "early_stopping_patience": 10,
    "encoding_dim": 8,
    "epochs": 50,
    "hidden_dims": [32, 16],
    "learning_rate": 0.001,
}
