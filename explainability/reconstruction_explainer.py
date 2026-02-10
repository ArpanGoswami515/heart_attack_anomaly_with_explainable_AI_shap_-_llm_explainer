"""
Reconstruction error-based explainability for autoencoder models.
"""

import numpy as np
from typing import Dict, List, Any


class ReconstructionExplainer:
    """Explainer for autoencoder models based on reconstruction error."""
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str]
    ):
        """
        Initialize reconstruction explainer.
        
        Args:
            model: Trained autoencoder model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
    
    def explain_sample(
        self, 
        X: np.ndarray, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Explain anomaly scores based on per-feature reconstruction errors.
        
        Args:
            X: Samples to explain (n_samples, n_features)
            top_k: Number of top contributing features to return
            
        Returns:
            List of explanation dictionaries, one per sample
        """
        # Get per-feature reconstruction errors
        reconstruction_errors = self.model.get_reconstruction_errors(X)
        
        explanations = []
        
        for i in range(len(X)):
            sample_errors = reconstruction_errors[i]
            
            # Get top k features with highest reconstruction error
            top_indices = np.argsort(sample_errors)[-top_k:][::-1]
            
            # Create feature importance dictionary
            feature_importance = {}
            for idx in top_indices:
                feature_name = self.feature_names[idx]
                error = float(sample_errors[idx])
                feature_importance[feature_name] = error
            
            # Normalize errors to sum to 1 for interpretability
            total_error = sum(feature_importance.values())
            if total_error > 0:
                feature_importance = {
                    k: v / total_error 
                    for k, v in feature_importance.items()
                }
            
            explanations.append({
                "feature_importance": feature_importance,
                "reconstruction_errors": sample_errors.tolist(),
                "total_reconstruction_error": float(np.mean(sample_errors))
            })
        
        return explanations
    
    def get_global_importance(self, X: np.ndarray) -> Dict[str, float]:
        """
        Get global feature importance based on average reconstruction errors.
        
        Args:
            X: Dataset to compute global importance on
            
        Returns:
            Dict mapping feature names to importance scores
        """
        # Get reconstruction errors for all samples
        reconstruction_errors = self.model.get_reconstruction_errors(X)
        
        # Compute mean error per feature
        mean_errors = np.mean(reconstruction_errors, axis=0)
        
        # Normalize to sum to 1
        mean_errors = mean_errors / np.sum(mean_errors)
        
        importance_dict = {
            feature: float(importance)
            for feature, importance in zip(self.feature_names, mean_errors)
        }
        
        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return importance_dict
    
    def get_reconstruction_quality(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Analyze reconstruction quality for samples.
        
        Args:
            X: Samples to analyze
            
        Returns:
            Dict containing reconstruction quality metrics
        """
        # Get reconstructed samples
        X_reconstructed = self.model.reconstruct(X)
        
        # Compute per-sample MSE
        mse_per_sample = np.mean((X - X_reconstructed) ** 2, axis=1)
        
        # Compute per-feature MSE
        mse_per_feature = np.mean((X - X_reconstructed) ** 2, axis=0)
        
        return {
            "mean_mse": float(np.mean(mse_per_sample)),
            "std_mse": float(np.std(mse_per_sample)),
            "min_mse": float(np.min(mse_per_sample)),
            "max_mse": float(np.max(mse_per_sample)),
            "feature_mse": {
                feature: float(mse)
                for feature, mse in zip(self.feature_names, mse_per_feature)
            }
        }
    
    def compare_original_reconstruction(
        self, 
        X: np.ndarray, 
        sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Compare original and reconstructed values for a specific sample.
        
        Args:
            X: Input samples
            sample_idx: Index of sample to analyze
            
        Returns:
            Dict containing comparison details
        """
        X_reconstructed = self.model.reconstruct(X)
        
        original = X[sample_idx]
        reconstructed = X_reconstructed[sample_idx]
        errors = (original - reconstructed) ** 2
        
        comparison = {}
        for i, feature in enumerate(self.feature_names):
            comparison[feature] = {
                "original": float(original[i]),
                "reconstructed": float(reconstructed[i]),
                "error": float(errors[i]),
                "percent_difference": float(
                    abs(original[i] - reconstructed[i]) / (abs(original[i]) + 1e-10) * 100
                )
            }
        
        return comparison
    
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
            
            lines.append(
                f"- {feature} (value: {feature_value:.2f}): "
                f"high reconstruction error (importance: {importance:.4f})"
            )
        
        return "\n".join(lines)
    
    def get_anomaly_contribution_breakdown(
        self, 
        X: np.ndarray
    ) -> List[Dict[str, float]]:
        """
        Break down each sample's anomaly score by feature contributions.
        
        Args:
            X: Samples to analyze
            
        Returns:
            List of dicts with per-feature contributions per sample
        """
        reconstruction_errors = self.model.get_reconstruction_errors(X)
        
        contributions = []
        for i in range(len(X)):
            sample_errors = reconstruction_errors[i]
            total_error = np.sum(sample_errors)
            
            if total_error > 0:
                feature_contributions = {
                    feature: float(error / total_error)
                    for feature, error in zip(self.feature_names, sample_errors)
                }
            else:
                feature_contributions = {
                    feature: 0.0
                    for feature in self.feature_names
                }
            
            contributions.append(feature_contributions)
        
        return contributions
