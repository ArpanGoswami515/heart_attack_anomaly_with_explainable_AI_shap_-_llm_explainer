# Optimized hyperparameters from tuning
# Generated: 2026-02-10 12:04:19
# Optimization metric: auroc

# Isolation Forest - auroc: 0.8611
ISOLATION_FOREST_PARAMS: Dict[str, Any] = {
    "bootstrap": True,
    "contamination": 0.05,
    "max_features": 1.0,
    "max_samples": "auto",
    "n_estimators": 300,
    "random_state": 42,
    "n_jobs": -1,
}

# One-Class SVM - auroc: 0.9264
ONE_CLASS_SVM_PARAMS: Dict[str, Any] = {
    "gamma": 0.001,
    "kernel": "poly",
    "nu": 0.25,
    "degree": 2,
    "coef0": 0.5,
}

# Autoencoder - auroc: 0.8444
AUTOENCODER_PARAMS: Dict[str, Any] = {
    "batch_size": 32,
    "early_stopping_patience": 10,
    "encoding_dim": 4,
    "epochs": 100,
    "hidden_dims": [32, 16],
    "learning_rate": 0.001,
}
