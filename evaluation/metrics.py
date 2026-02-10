"""
Evaluation metrics for anomaly detection models.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)


class AnomalyDetectionMetrics:
    """Comprehensive metrics for anomaly detection evaluation."""
    
    @staticmethod
    def compute_auroc(
        y_true: np.ndarray, 
        anomaly_scores: np.ndarray
    ) -> float:
        """
        Compute Area Under ROC Curve.
        
        Args:
            y_true: True labels (0=normal, 1=anomaly)
            anomaly_scores: Predicted anomaly scores [0, 1]
            
        Returns:
            AUROC score
        """
        try:
            return float(roc_auc_score(y_true, anomaly_scores))
        except ValueError as e:
            return 0.0
    
    @staticmethod
    def compute_precision_at_k(
        y_true: np.ndarray,
        anomaly_scores: np.ndarray,
        k: float = 0.1
    ) -> float:
        """
        Compute precision at top k% of highest risk predictions.
        
        Args:
            y_true: True labels (0=normal, 1=anomaly)
            anomaly_scores: Predicted anomaly scores [0, 1]
            k: Fraction of top predictions to consider (e.g., 0.1 = top 10%)
            
        Returns:
            Precision at k
        """
        n_samples = len(anomaly_scores)
        n_top = max(1, int(n_samples * k))
        
        # Get indices of top-k highest scores
        top_k_indices = np.argsort(anomaly_scores)[-n_top:]
        
        # Compute precision on top-k
        y_true_top_k = y_true[top_k_indices]
        
        if len(y_true_top_k) == 0:
            return 0.0
        
        precision = np.sum(y_true_top_k) / len(y_true_top_k)
        return float(precision)
    
    @staticmethod
    def compute_false_negative_rate(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute false negative rate (missed anomalies).
        
        Args:
            y_true: True labels (0=normal, 1=anomaly)
            y_pred: Predicted labels (0=normal, 1=anomaly)
            
        Returns:
            False negative rate
        """
        # Count actual anomalies
        n_anomalies = np.sum(y_true == 1)
        
        if n_anomalies == 0:
            return 0.0
        
        # Count missed anomalies (false negatives)
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        
        fnr = false_negatives / n_anomalies
        return float(fnr)
    
    @staticmethod
    def compute_confusion_matrix_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute confusion matrix and derived metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dict containing confusion matrix and metrics
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Extract values
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            "confusion_matrix": cm.tolist(),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "specificity": float(specificity)
        }
    
    @staticmethod
    def compute_average_precision(
        y_true: np.ndarray,
        anomaly_scores: np.ndarray
    ) -> float:
        """
        Compute average precision score.
        
        Args:
            y_true: True labels
            anomaly_scores: Predicted anomaly scores
            
        Returns:
            Average precision score
        """
        try:
            return float(average_precision_score(y_true, anomaly_scores))
        except ValueError:
            return 0.0
    
    @staticmethod
    def get_optimal_threshold(
        y_true: np.ndarray,
        anomaly_scores: np.ndarray,
        metric: str = "f1"
    ) -> Dict[str, float]:
        """
        Find optimal threshold for binary classification.
        
        Args:
            y_true: True labels
            anomaly_scores: Predicted anomaly scores
            metric: Metric to optimize ("f1", "precision", "recall")
            
        Returns:
            Dict with optimal threshold and corresponding metrics
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, anomaly_scores)
        
        if metric == "f1":
            # Compute F1 scores for each threshold
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            score = f1_scores[optimal_idx]
        elif metric == "precision":
            optimal_idx = np.argmax(precisions)
            score = precisions[optimal_idx]
        elif metric == "recall":
            optimal_idx = np.argmax(recalls)
            score = recalls[optimal_idx]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Handle case where threshold array is shorter
        if optimal_idx >= len(thresholds):
            optimal_idx = len(thresholds) - 1
        
        return {
            "optimal_threshold": float(thresholds[optimal_idx]),
            "precision": float(precisions[optimal_idx]),
            "recall": float(recalls[optimal_idx]),
            f"{metric}_score": float(score)
        }
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute essential metrics for model evaluation.
        Simplified version for hyperparameter tuning.
        
        Args:
            y_true: True labels (0=normal, 1=anomaly)
            y_scores: Predicted anomaly scores [0, 1]
            threshold: Threshold for binary classification
            
        Returns:
            Dict containing essential metrics
        """
        # Convert scores to binary predictions
        y_pred = (y_scores >= threshold).astype(int)
        
        # Compute essential metrics
        metrics = {
            "auroc": AnomalyDetectionMetrics.compute_auroc(y_true, y_scores),
            "precision_at_10": AnomalyDetectionMetrics.compute_precision_at_k(
                y_true, y_scores, k=0.1
            ),
            "precision_at_20": AnomalyDetectionMetrics.compute_precision_at_k(
                y_true, y_scores, k=0.2
            ),
        }
        
        # Add F1 score
        try:
            metrics["f1_score"] = float(f1_score(y_true, y_pred, zero_division=0.0))
        except:
            metrics["f1_score"] = 0.0
        
        # Add precision and recall
        try:
            metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0.0))
            metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0.0))
        except:
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
        
        # Add false negative rate
        metrics["false_negative_rate"] = AnomalyDetectionMetrics.compute_false_negative_rate(
            y_true, y_pred
        )
        
        return metrics
    
    @staticmethod
    def evaluate_model(
        y_true: np.ndarray,
        anomaly_scores: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            y_true: True labels (0=normal, 1=anomaly)
            anomaly_scores: Predicted anomaly scores [0, 1]
            threshold: Threshold for binary classification
            
        Returns:
            Dict containing all evaluation metrics
        """
        # Convert scores to binary predictions
        y_pred = (anomaly_scores >= threshold).astype(int)
        
        # Compute all metrics
        metrics = {
            "auroc": AnomalyDetectionMetrics.compute_auroc(y_true, anomaly_scores),
            "average_precision": AnomalyDetectionMetrics.compute_average_precision(
                y_true, anomaly_scores
            ),
            "precision_at_10": AnomalyDetectionMetrics.compute_precision_at_k(
                y_true, anomaly_scores, k=0.1
            ),
            "precision_at_20": AnomalyDetectionMetrics.compute_precision_at_k(
                y_true, anomaly_scores, k=0.2
            ),
            "false_negative_rate": AnomalyDetectionMetrics.compute_false_negative_rate(
                y_true, y_pred
            ),
            "threshold": float(threshold)
        }
        
        # Add confusion matrix metrics
        cm_metrics = AnomalyDetectionMetrics.compute_confusion_matrix_metrics(
            y_true, y_pred
        )
        metrics.update(cm_metrics)
        
        return metrics


class ModelComparison:
    """Compare multiple anomaly detection models."""
    
    @staticmethod
    def compare_models(
        y_true: np.ndarray,
        model_scores: Dict[str, np.ndarray],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Compare performance of multiple models.
        
        Args:
            y_true: True labels
            model_scores: Dict mapping model names to anomaly scores
            threshold: Threshold for binary classification
            
        Returns:
            Dict containing comparison results
        """
        comparison = {}
        
        for model_name, scores in model_scores.items():
            metrics = AnomalyDetectionMetrics.evaluate_model(
                y_true, scores, threshold
            )
            comparison[model_name] = metrics
        
        # Find best model for each metric
        best_models = ModelComparison._find_best_models(comparison)
        comparison["best_models"] = best_models
        
        return comparison
    
    @staticmethod
    def _find_best_models(comparison: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """
        Find best performing model for each metric.
        
        Args:
            comparison: Dict of model comparisons
            
        Returns:
            Dict mapping metric names to best model names
        """
        metrics_to_maximize = [
            "auroc", "average_precision", "precision", 
            "recall", "f1_score", "precision_at_10", "precision_at_20"
        ]
        
        metrics_to_minimize = ["false_negative_rate"]
        
        best_models = {}
        
        for metric in metrics_to_maximize:
            best_model = None
            best_score = -float('inf')
            
            for model_name, metrics_dict in comparison.items():
                if model_name == "best_models":
                    continue
                
                if metric in metrics_dict:
                    score = metrics_dict[metric]
                    if score > best_score:
                        best_score = score
                        best_model = model_name
            
            if best_model:
                best_models[metric] = best_model
        
        for metric in metrics_to_minimize:
            best_model = None
            best_score = float('inf')
            
            for model_name, metrics_dict in comparison.items():
                if model_name == "best_models":
                    continue
                
                if metric in metrics_dict:
                    score = metrics_dict[metric]
                    if score < best_score:
                        best_score = score
                        best_model = model_name
            
            if best_model:
                best_models[metric] = best_model
        
        return best_models
    
    @staticmethod
    def rank_models(
        comparison: Dict[str, Dict[str, float]],
        primary_metric: str = "auroc"
    ) -> List[tuple]:
        """
        Rank models by a primary metric.
        
        Args:
            comparison: Dict of model comparisons
            primary_metric: Metric to rank by
            
        Returns:
            List of (model_name, score) tuples, sorted by score
        """
        rankings = []
        
        for model_name, metrics in comparison.items():
            if model_name == "best_models":
                continue
            
            if primary_metric in metrics:
                rankings.append((model_name, metrics[primary_metric]))
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
