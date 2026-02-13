"""
Base class for all anomaly detection models.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class BaseAnomalyModel(ABC):
    """Abstract base class for anomaly detection models."""
    
    def __init__(self, name: str):
        """
        Initialize base model.
        
        Args:
            name: Model name identifier
        """
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseAnomalyModel":
        """
        Train the anomaly detection model.
        
        Args:
            X: Training data (normal samples only for unsupervised methods)
            
        Returns:
            self: Fitted model
        """
        pass
    
    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for samples.
        
        Args:
            X: Input samples
            
        Returns:
            np.ndarray: Anomaly scores normalized to [0, 1]
                       where 0 = normal, 1 = highly anomalous
        """
        pass
    
    def predict_risk(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Predict risk levels for samples.
        
        Args:
            X: Input samples
            
        Returns:
            Dict containing:
                - scores: Anomaly scores
                - risk_levels: Risk categories (Low/Medium/High)
        """
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} is not fitted yet.")
        
        scores = self.score_samples(X)
        risk_levels = self._scores_to_risk_levels(scores)
        
        return {
            "scores": scores,
            "risk_levels": risk_levels
        }
    
    def _scores_to_risk_levels(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert anomaly scores to risk level categories.
        
        Args:
            scores: Anomaly scores in [0, 1]
            
        Returns:
            np.ndarray: Risk levels as strings
        """
        risk_levels = np.empty(len(scores), dtype=object)
        risk_levels[scores < 0.33] = "Low"
        risk_levels[(scores >= 0.33) & (scores < 0.66)] = "Medium"
        risk_levels[scores >= 0.66] = "High"
        
        return risk_levels
    
    def _normalize_scores(
        self, 
        scores: np.ndarray, 
        reverse: bool = False,
        score_range: tuple = None
    ) -> np.ndarray:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            scores: Raw scores
            reverse: If True, reverse the scores (higher becomes lower)
            score_range: Optional (min, max) tuple for normalization bounds.
                        If None, uses the actual min/max of the scores.
            
        Returns:
            np.ndarray: Normalized scores
        """
        if len(scores) == 0:
            return scores
        
        if score_range is not None:
            min_score, max_score = score_range
        else:
            min_score = np.min(scores)
            max_score = np.max(scores)
        
        if max_score - min_score == 0:
            # If scores are constant, return middle value
            # (This typically happens with single-sample normalization)
            return np.ones_like(scores) * 0.5
        
        # Clip scores to the specified range
        scores_clipped = np.clip(scores, min_score, max_score)
        
        normalized = (scores_clipped - min_score) / (max_score - min_score)
        
        if reverse:
            normalized = 1 - normalized
        
        return normalized
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dict containing model metadata
        """
        return {
            "name": self.name,
            "is_fitted": self.is_fitted,
            "type": self.__class__.__name__
        }
