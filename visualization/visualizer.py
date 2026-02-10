"""
Model Visualization Module
Creates comprehensive comparative visualizations for anomaly detection models
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


class ModelVisualizer:
    """Creates comparative visualizations for multiple anomaly detection models"""
    
    def __init__(self, save_dir: str = "visualizations"):
        """
        Initialize the visualizer
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 5)
        plt.rcParams['font.size'] = 10
    
    def create_comparative_analysis(
        self,
        evaluation_results: Dict[str, Dict[str, float]],
        predictions: Dict[str, np.ndarray],
        feature_importance: Dict[str, Dict[str, float]],
        feature_names: List[str]
    ):
        """
        Create comprehensive comparative analysis with 3 separate plots
        
        Args:
            evaluation_results: Dict of model evaluation metrics
            predictions: Dict of model predictions on test set
            feature_importance: Dict of feature importance scores per model
            feature_names: List of feature names
        """
        output_paths = []
        
        # Plot 1: Model Performance Comparison
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        self._plot_performance_metrics(ax1, evaluation_results)
        plt.tight_layout()
        path1 = self.save_dir / "1_model_performance_comparison.png"
        plt.savefig(path1, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths.append(path1)
        print(f"✓ Plot 1 saved: {path1}")
        
        # Plot 2: Feature Importance Heatmap
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        self._plot_feature_importance_heatmap(ax2, feature_importance, feature_names)
        plt.tight_layout()
        path2 = self.save_dir / "2_feature_importance_heatmap.png"
        plt.savefig(path2, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths.append(path2)
        print(f"✓ Plot 2 saved: {path2}")
        
        # Plot 3: Anomaly Score Distributions
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        self._plot_score_distributions(ax3, predictions)
        plt.tight_layout()
        path3 = self.save_dir / "3_anomaly_score_distributions.png"
        plt.savefig(path3, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths.append(path3)
        print(f"✓ Plot 3 saved: {path3}")
        
        print(f"\n✓ All 3 visualizations saved to: {self.save_dir}")
        
        return output_paths
    
    def _plot_performance_metrics(self, ax, evaluation_results: Dict[str, Dict[str, float]]):
        """Plot comparative performance metrics as grouped bar chart"""
        # Select key metrics
        metrics_to_plot = ['auroc', 'f1_score', 'precision_at_10', 'false_negative_rate']
        metric_labels = ['AUROC', 'F1 Score', 'Precision@10%', 'FN Rate']
        
        model_names = list(evaluation_results.keys())
        x = np.arange(len(metrics_to_plot))
        width = 0.25
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for i, model_name in enumerate(model_names):
            values = [evaluation_results[model_name].get(metric, 0) for metric in metrics_to_plot]
            offset = (i - 1) * width
            ax.bar(x + offset, values, width, label=model_name.replace('_', ' ').title(), 
                   color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=15, ha='right')
        ax.legend(loc='upper right', frameon=True)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=8)
    
    def _plot_feature_importance_heatmap(self, ax, feature_importance: Dict[str, Dict[str, float]], 
                                          feature_names: List[str]):
        """Plot feature importance heatmap across models"""
        # Get top 8 features across all models
        all_features = set()
        for model_features in feature_importance.values():
            all_features.update(list(model_features.keys())[:8])
        
        top_features = sorted(all_features)[:10]  # Limit to 10 for readability
        
        # Create matrix
        model_names = list(feature_importance.keys())
        matrix = np.zeros((len(model_names), len(top_features)))
        
        for i, model_name in enumerate(model_names):
            for j, feature in enumerate(top_features):
                importance = feature_importance[model_name].get(feature, 0)
                matrix[i, j] = abs(importance)  # Use absolute values
        
        # Normalize by row (per model)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix_normalized = matrix / row_sums
        
        # Create heatmap
        sns.heatmap(matrix_normalized, annot=True, fmt='.2f', cmap='YlOrRd', 
                    xticklabels=top_features, 
                    yticklabels=[m.replace('_', ' ').title() for m in model_names],
                    cbar_kws={'label': 'Normalized Importance'},
                    ax=ax, linewidths=0.5)
        
        ax.set_title('Feature Importance Comparison', fontweight='bold', fontsize=12)
        ax.set_xlabel('Features', fontweight='bold')
        ax.set_ylabel('Models', fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    def _plot_score_distributions(self, ax, predictions: Dict[str, np.ndarray]):
        """Plot anomaly score distributions for each model"""
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for i, (model_name, scores) in enumerate(predictions.items()):
            # Plot histogram
            ax.hist(scores, bins=30, alpha=0.5, label=model_name.replace('_', ' ').title(),
                    color=colors[i], edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Anomaly Score', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Anomaly Score Distributions', fontweight='bold', fontsize=12)
        ax.legend(loc='upper right', frameon=True)
        ax.grid(axis='y', alpha=0.3)
        
        # Add vertical line at threshold (0.5)
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Threshold (0.5)')
    
    def create_single_model_detail(
        self,
        model_name: str,
        predictions: np.ndarray,
        y_true: np.ndarray,
        feature_importance: Dict[str, float],
        top_n: int = 10
    ):
        """
        Create detailed visualization for a single model
        
        Args:
            model_name: Name of the model
            predictions: Model prediction scores
            y_true: True labels
            feature_importance: Feature importance dict
            top_n: Number of top features to show
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Confusion matrix style visualization
        ax1 = axes[0]
        
        # Threshold at 0.5
        y_pred = (predictions > 0.5).astype(int)
        
        # Calculate confusion matrix values
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        confusion_matrix = np.array([[tn, fp], [fn, tp]])
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'],
                    ax=ax1, cbar=False, linewidths=2)
        
        ax1.set_title(f'{model_name.replace("_", " ").title()}\nConfusion Matrix', 
                     fontweight='bold', fontsize=12)
        ax1.set_xlabel('Predicted', fontweight='bold')
        ax1.set_ylabel('Actual', fontweight='bold')
        
        # Plot 2: Top features
        ax2 = axes[1]
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), 
                                key=lambda x: abs(x[1]), reverse=True)[:top_n]
        
        features, importances = zip(*sorted_features)
        
        colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in importances]
        
        y_pos = np.arange(len(features))
        ax2.barh(y_pos, [abs(imp) for imp in importances], color=colors, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(features, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel('Importance (Absolute Value)', fontweight='bold')
        ax2.set_title(f'Top {top_n} Feature Importance', fontweight='bold', fontsize=12)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.save_dir / f"{model_name}_detailed_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        
        return output_path
