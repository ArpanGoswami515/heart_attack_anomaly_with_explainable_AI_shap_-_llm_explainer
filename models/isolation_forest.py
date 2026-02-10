"""
Isolation Forest anomaly detection model.
"""

import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from typing import Dict, Any
from .base_model import BaseAnomalyModel


class IsolationForest(BaseAnomalyModel):
    """Isolation Forest anomaly detection model using scikit-learn."""
    
    def __init__(self, **kwargs):
        """
        Initialize Isolation Forest model.
        
        Args:
            **kwargs: Parameters for sklearn IsolationForest
        """
        super().__init__(name="Isolation Forest")
        self.model = SklearnIsolationForest(**kwargs)
        self.params = kwargs
    
    def fit(self, X: np.ndarray) -> "IsolationForest":
        """
        Train the Isolation Forest model.
        
        Args:
            X: Training data
            
        Returns:
            self: Fitted model
        """
        self.model.fit(X)
        self.is_fitted = True
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for samples.
        
        Isolation Forest returns negative scores where more negative = more anomalous.
        We convert to [0, 1] where 1 = highly anomalous.
        
        Args:
            X: Input samples
            
        Returns:
            np.ndarray: Normalized anomaly scores in [0, 1]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring.")
        
        # Get decision scores (negative values, lower = more anomalous)
        raw_scores = self.model.decision_function(X)
        
        # Convert to [0, 1] where 1 = highly anomalous
        # More negative scores should become higher values
        normalized_scores = self._normalize_scores(raw_scores, reverse=True)
        
        return normalized_scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict if samples are anomalies.
        
        Args:
            X: Input samples
            
        Returns:
            np.ndarray: Predictions (-1 for anomaly, 1 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance from the model.
        Note: Isolation Forest doesn't have native feature importance,
        this is a placeholder for consistency.
        
        Returns:
            np.ndarray: Feature importance (uniform distribution)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        
        # Isolation Forest doesn't have direct feature importance
        # Return uniform distribution
        n_features = self.model.n_features_in_
        return np.ones(n_features) / n_features
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dict containing model metadata
        """
        info = super().get_model_info()
        info.update({
            "n_estimators": self.params.get("n_estimators", 100),
            "contamination": self.params.get("contamination", "auto"),
            "max_samples": self.params.get("max_samples", "auto")
        })
        
        if self.is_fitted:
            info["n_features"] = self.model.n_features_in_
        
        return info
