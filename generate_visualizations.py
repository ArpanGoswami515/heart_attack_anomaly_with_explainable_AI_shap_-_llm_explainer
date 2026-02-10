"""
Standalone script to generate visualizations without full pipeline execution
"""

import numpy as np
import os
from config.settings import Config
from data.data_loader import HeartAttackDataLoader
from preprocessing.preprocess import DataPreprocessor
from models.isolation_forest import IsolationForest
from models.one_class_svm import OneClassSVM
from models.autoencoder import Autoencoder
from evaluation.metrics import AnomalyDetectionMetrics
from visualization.visualizer import ModelVisualizer
from explainability.shap_explainer import SHAPExplainer
from explainability.reconstruction_explainer import ReconstructionExplainer

def main():
    """Generate visualizations only."""
    print("="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Setup
    Config.create_directories()
    
    # Load and preprocess data
    print("\nLoading data...")
    dataset_path = os.path.join(Config.DATA_DIR, "heart_attack_data.csv")
    if not os.path.exists(dataset_path):
        dataset_path = "heart_attack_data.csv"
    
    loader = HeartAttackDataLoader(file_path=dataset_path)
    loader.load_data()
    X, y = loader.get_features_and_labels(target_column="target")
    feature_names = loader.get_feature_names()
    
    # Split into train/test
    X_train, X_test, y_train, y_test = loader.train_test_split(
        X, y, test_size=0.2, stratify=True
    )
    
    # Convert labels to numpy arrays
    y_train = y_train.values if hasattr(y_train, 'values') else y_train
    y_test = y_test.values if hasattr(y_test, 'values') else y_test
    
    # Initialize and apply preprocessor
    preprocessor = DataPreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    X_train_normal = X_train_scaled[y_train == 0]
    
    print(f"✓ Data loaded: {len(X_test)} test samples, {len(feature_names)} features")
    
    # Train models
    print("\nTraining models...")
    
    models = {}
    
    # Isolation Forest
    if_model = IsolationForest(**Config.ISOLATION_FOREST_PARAMS)
    if_model.fit(X_train_normal)
    models["isolation_forest"] = if_model
    print("✓ Isolation Forest trained")
    
    # One-Class SVM
    svm_model = OneClassSVM(**Config.ONE_CLASS_SVM_PARAMS)
    svm_model.fit(X_train_normal)
    models["one_class_svm"] = svm_model
    print("✓ One-Class SVM trained")
    
    # Autoencoder
    ae_model = Autoencoder(**Config.AUTOENCODER_PARAMS)
    ae_model.fit(X_train_normal)
    models["autoencoder"] = ae_model
    print("✓ Autoencoder trained")
    
    # Get predictions
    print("\nComputing predictions...")
    model_scores = {}
    for model_name, model in models.items():
        scores = model.score_samples(X_test_scaled)
        model_scores[model_name] = scores
    
    # Extract feature importance
    print("\nExtracting feature importance...")
    feature_importance = {}
    
    for model_name, model in models.items():
        if model_name == 'autoencoder':
            explainer = ReconstructionExplainer(
                model=model,
                feature_names=feature_names
            )
            explanations = explainer.explain_sample(X_test_scaled[0:1], top_k=10)
            importance_dict = explanations[0]['feature_importance']
        else:
            sklearn_model = model.model if hasattr(model, 'model') else model
            explainer = SHAPExplainer(
                model=sklearn_model, 
                background_data=X_train_scaled[:100],
                feature_names=feature_names
            )
            explanations = explainer.explain_sample(X_test_scaled[0:1], top_k=10)
            importance_dict = explanations[0]['feature_importance']
        
        feature_importance[model_name] = importance_dict
        print(f"  ✓ {model_name}")
    
    # Get evaluation results
    print("\nComputing evaluation metrics...")
    evaluation_results = {}
    for model_name in models.keys():
        scores = model_scores[model_name]
        metrics = AnomalyDetectionMetrics.compute_metrics(
            y_true=y_test,
            y_scores=scores,
            threshold=0.5
        )
        evaluation_results[model_name] = metrics
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualizer = ModelVisualizer(save_dir="visualizations")
    
    output_paths = visualizer.create_comparative_analysis(
        evaluation_results=evaluation_results,
        predictions=model_scores,
        feature_importance=feature_importance,
        feature_names=feature_names
    )
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\n✓ Generated {len(output_paths)} visualization plots")

if __name__ == "__main__":
    main()
