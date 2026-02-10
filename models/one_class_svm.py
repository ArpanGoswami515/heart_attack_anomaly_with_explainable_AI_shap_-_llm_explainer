"""
One-Class SVM anomaly detection model.
"""

import numpy as np
from sklearn.svm import OneClassSVM as SklearnOneClassSVM
from typing import Dict, Any
from .base_model import BaseAnomalyModel


class OneClassSVM(BaseAnomalyModel):
    """One-Class SVM anomaly detection model using scikit-learn."""
    
    def __init__(self, **kwargs):
        """
        Initialize One-Class SVM model.
        
        Args:
            **kwargs: Parameters for sklearn OneClassSVM
        """
        super().__init__(name="One-Class SVM")
        self.model = SklearnOneClassSVM(**kwargs)
        self.params = kwargs
    
    def fit(self, X: np.ndarray) -> "OneClassSVM":
        """
        Train the One-Class SVM model.
        
        Args:
            X: Training data (normal samples)
            
        Returns:
            self: Fitted model
        """
        self.model.fit(X)
        self.is_fitted = True
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for samples.
        
        One-Class SVM returns decision function values where:
        - Positive values = inside the decision boundary (normal)
        - Negative values = outside the decision boundary (anomalous)
        
        We convert to [0, 1] where 1 = highly anomalous.
        
        Args:
            X: Input samples
            
        Returns:
            np.ndarray: Normalized anomaly scores in [0, 1]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring.")
        
        # Get decision scores (positive = normal, negative = anomalous)
        raw_scores = self.model.decision_function(X)
        
        # Convert to [0, 1] where 1 = highly anomalous
        # Negative scores should become higher values
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
    
    def get_decision_boundary_distance(self, X: np.ndarray) -> np.ndarray:
        """
        Get signed distance to decision boundary.
        
        Args:
            X: Input samples
            
        Returns:
            np.ndarray: Signed distances (negative = anomalous side)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        
        return self.model.decision_function(X)
    
    def get_support_vectors(self) -> np.ndarray:
        """
        Get support vectors from the trained model.
        
        Returns:
            np.ndarray: Support vectors
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        
        return self.model.support_vectors_
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dict containing model metadata
        """
        info = super().get_model_info()
        info.update({
            "kernel": self.params.get("kernel", "rbf"),
            "gamma": self.params.get("gamma", "scale"),
            "nu": self.params.get("nu", 0.5)
        })
        
        if self.is_fitted:
            info["n_support_vectors"] = len(self.model.support_vectors_)
            info["n_features"] = self.model.shape_fit_[1]
        
        return info
