"""
SHAP-based explainability for tree-based and kernel-based models.
"""

import numpy as np
import shap
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.svm import OneClassSVM as SklearnOneClassSVM


class SHAPExplainer:
    """SHAP explainer for Isolation Forest and One-Class SVM."""
    
    def __init__(
        self,
        model: Any,
        background_data: np.ndarray,
        feature_names: List[str],
        num_samples: int = 100
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained anomaly detection model (sklearn)
            background_data: Background dataset for SHAP (training data sample)
            feature_names: List of feature names
            num_samples: Number of samples for SHAP approximation
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self.num_samples = num_samples
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self) -> None:
        """Initialize appropriate SHAP explainer based on model type."""
        if isinstance(self.model, SklearnIsolationForest):
            # Use TreeExplainer for Isolation Forest
            self.explainer = shap.TreeExplainer(self.model)
        elif isinstance(self.model, SklearnOneClassSVM):
            # Use KernelExplainer for One-Class SVM
            # Create a wrapper function for the model
            def model_predict(X):
                return self.model.decision_function(X)
            
            # Sample background data if too large
            if len(self.background_data) > self.num_samples:
                indices = np.random.choice(
                    len(self.background_data), 
                    self.num_samples, 
                    replace=False
                )
                background_sample = self.background_data[indices]
            else:
                background_sample = self.background_data
            
            self.explainer = shap.KernelExplainer(
                model_predict, 
                background_sample
            )
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")
    
    def explain_sample(
        self, 
        X: np.ndarray, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Explain anomaly predictions for samples.
        
        Args:
            X: Samples to explain (n_samples, n_features)
            top_k: Number of top contributing features to return
            
        Returns:
            List of explanation dictionaries, one per sample
        """
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        explanations = []
        
        for i in range(len(X)):
            sample_shap_values = shap_values[i] if len(shap_values.shape) > 1 else shap_values
            
            # Get absolute SHAP values for ranking
            abs_shap_values = np.abs(sample_shap_values)
            
            # Get top k features
            top_indices = np.argsort(abs_shap_values)[-top_k:][::-1]
            
            # Create feature importance dictionary
            feature_importance = {}
            for idx in top_indices:
                feature_name = self.feature_names[idx]
                importance = float(sample_shap_values[idx])
                feature_importance[feature_name] = importance
            
            explanations.append({
                "feature_importance": feature_importance,
                "shap_values": sample_shap_values.tolist(),
                "base_value": float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0
            })
        
        return explanations
    
    def get_global_importance(self) -> Dict[str, float]:
        """
        Get global feature importance across background data.
        
        Returns:
            Dict mapping feature names to importance scores
        """
        shap_values = self.explainer.shap_values(self.background_data)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Compute mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Normalize to sum to 1
        mean_abs_shap = mean_abs_shap / np.sum(mean_abs_shap)
        
        importance_dict = {
            feature: float(importance)
            for feature, importance in zip(self.feature_names, mean_abs_shap)
        }
        
        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return importance_dict
    
    def get_feature_interactions(
        self, 
        X: np.ndarray, 
        feature_pairs: List[Tuple[int, int]] = None
    ) -> Dict[Tuple[str, str], float]:
        """
        Get feature interaction effects (if supported by explainer).
        
        Args:
            X: Samples to analyze
            feature_pairs: List of feature index pairs to analyze
            
        Returns:
            Dict mapping feature pairs to interaction scores
        """
        # Feature interactions are complex and computationally expensive
        # This is a placeholder for future implementation
        return {}
    
    def format_explanation_for_llm(
        self, 
        explanation: Dict[str, Any], 
        feature_values: np.ndarray
    ) -> str:
        """
        Format explanation in a structured way for LLM consumption.
        
        Args:
            explanation: Explanation dictionary from explain_sample
            feature_values: Original feature values for the sample
            
        Returns:
            Formatted string representation
        """
        feature_importance = explanation["feature_importance"]
        
        lines = []
        for feature, importance in feature_importance.items():
            feature_idx = self.feature_names.index(feature)
            feature_value = feature_values[feature_idx]
            
            direction = "increases" if importance > 0 else "decreases"
            lines.append(
                f"- {feature} (value: {feature_value:.2f}): "
                f"{direction} risk (importance: {abs(importance):.4f})"
            )
        
        return "\n".join(lines)
