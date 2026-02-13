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
        
        # Use sklearn's score_samples which returns negative average path length
        # More negative = more anomalous
        raw_scores = self.model.score_samples(X)
        
        # Normalize to [0, 1] where 1 = highly anomalous
        # Based on empirical observation of Isolation Forest on typical data:
        # - Raw scores typically range from approximately -0.75 to -0.40
        # - We use the offset as the midpoint for anomaly/normal classification
        
        offset = self.model.offset_
        
        # Use a range centered on the offset
        # Empirically, scores span about ±3*std from mean
        # For heart attack data: std ≈ 0.07, so range ≈ ±0.21 from offset
        typical_std = 0.07
        score_range = 3 * typical_std  # ±0.21
        
        # Expected min/max based on offset
        expected_min = offset - score_range  # More anomalous
        expected_max = offset + score_range  # More normal
        
        # Normalize: map [expected_min, expected_max] → [1, 0]
        # (reverse because lower raw scores = higher anomaly)
        anomaly_scores = 1 - ((raw_scores - expected_min) / (expected_max - expected_min))
        
        # Clip to [0, 1]
        anomaly_scores = np.clip(anomaly_scores, 0, 1)
        
        return anomaly_scores
    
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
