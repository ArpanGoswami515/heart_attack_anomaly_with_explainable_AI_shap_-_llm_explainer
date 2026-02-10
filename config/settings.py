"""
Configuration settings for the heart attack anomaly detection system.
"""

import os
from typing import Dict, Any


class Config:
    """Global configuration class for the project."""
    
    # Random seed for reproducibility
    RANDOM_SEED: int = 42
    
    # Data paths
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    MODELS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
    
    # Dataset settings
    TRAIN_TEST_SPLIT: float = 0.8
    VALIDATION_SPLIT: float = 0.2
    
    # Model hyperparameters
    ISOLATION_FOREST_PARAMS: Dict[str, Any] = {
        "bootstrap": True,
        "contamination": 0.05,
        "max_features": 1.0,
        "max_samples": "auto",
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
    }
    
    ONE_CLASS_SVM_PARAMS: Dict[str, Any] = {
        "gamma": 0.001,
        "kernel": "poly",
        "nu": 0.25,
        "degree": 2,
        "coef0": 0.5,
    }
    
    AUTOENCODER_PARAMS: Dict[str, Any] = {
        "batch_size": 32,
        "early_stopping_patience": 10,
        "encoding_dim": 4,
        "epochs": 100,
        "hidden_dims": [32, 16],
        "learning_rate": 0.001,
    }
    
    # Anomaly detection thresholds
    ANOMALY_THRESHOLDS: Dict[str, float] = {
        "low": 0.33,
        "medium": 0.66,
        "high": 1.0
    }
    
    # SHAP settings
    SHAP_NUM_SAMPLES: int = 100
    TOP_FEATURES_TO_EXPLAIN: int = 5
    
    # HuggingFace API settings
    HF_API_URL: str = "https://router.huggingface.co/models/"
    HF_MODEL: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    HF_API_TOKEN_ENV: str = "HUGGINGFACE_API_TOKEN"
    HF_MAX_TOKENS: int = 300
    HF_TEMPERATURE: float = 0.7
    
    # Evaluation settings
    PRECISION_AT_K: float = 0.1  # Top 10% highest risk
    
    @classmethod
    def get_hf_api_token(cls) -> str:
        """
        Get HuggingFace API token from environment variable.
        
        Returns:
            str: API token
            
        Raises:
            ValueError: If token is not set
        """
        token = os.getenv(cls.HF_API_TOKEN_ENV)
        if not token:
            raise ValueError(
                f"HuggingFace API token not found. "
                f"Please set {cls.HF_API_TOKEN_ENV} environment variable."
            )
        return token
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
    
    @classmethod
    def get_risk_level(cls, anomaly_score: float) -> str:
        """
        Convert anomaly score to risk level.
        
        Args:
            anomaly_score: Normalized anomaly score in [0, 1]
            
        Returns:
            str: Risk level (Low, Medium, or High)
        """
        if anomaly_score < cls.ANOMALY_THRESHOLDS["low"]:
            return "Low"
        elif anomaly_score < cls.ANOMALY_THRESHOLDS["medium"]:
            return "Medium"
        else:
            return "High"
