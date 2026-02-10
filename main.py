"""
Main entry point for heart attack anomaly detection system.
"""

import numpy as np
import os
import json
from dotenv import load_dotenv
from config.settings import Config

# Load environment variables from .env file
load_dotenv()
from data.data_loader import HeartAttackDataLoader
from preprocessing.preprocess import DataPreprocessor
from models.isolation_forest import IsolationForest
from models.one_class_svm import OneClassSVM
from models.autoencoder import Autoencoder
from pipeline.inference_pipeline import InferencePipeline
from evaluation.metrics import AnomalyDetectionMetrics, ModelComparison
from visualization.visualizer import ModelVisualizer
from explainability.shap_explainer import SHAPExplainer
from explainability.reconstruction_explainer import ReconstructionExplainer


def setup_environment() -> None:
    """Setup environment and create necessary directories."""
    Config.create_directories()
    print("✓ Environment setup complete")


def load_and_preprocess_data(use_synthetic: bool = True) -> tuple:
    """
    Load and preprocess heart attack dataset.
    
    Args:
        use_synthetic: Whether to use synthetic data for demo
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, preprocessor)
    """
    print("\n" + "="*60)
    print("LOADING AND PREPROCESSING DATA")
    print("="*60)
    
    # Initialize data loader
    if use_synthetic:
        print("Generating synthetic heart attack dataset...")
        df = HeartAttackDataLoader.generate_sample_dataset(
            n_samples=1000,
            contamination=0.1,
            random_state=Config.RANDOM_SEED
        )
        data_loader = HeartAttackDataLoader(
            file_path="synthetic",
            random_state=Config.RANDOM_SEED
        )
        data_loader.data = df
    else:
        # Load real dataset (update path as needed)
        dataset_path = os.path.join(Config.DATA_DIR, "heart_attack_data.csv")
        data_loader = HeartAttackDataLoader(
            file_path=dataset_path,
            random_state=Config.RANDOM_SEED
        )
        data_loader.load_data()
    
    # Get features and labels
    X, y = data_loader.get_features_and_labels(target_column="target")
    feature_names = data_loader.get_feature_names()
    
    print(f"✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✓ Anomaly ratio: {np.mean(y):.2%}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = data_loader.train_test_split(
        X, y, test_size=0.2, stratify=True
    )
    
    # Convert labels to numpy arrays to avoid index issues
    y_train = y_train.values if hasattr(y_train, 'values') else y_train
    y_test = y_test.values if hasattr(y_test, 'values') else y_test
    
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(scaling_method="standard")
    
    # Fit on training data and transform both sets
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    print("✓ Data preprocessing complete")
    
    # Get only normal samples for training (unsupervised anomaly detection)
    X_train_normal = X_train_scaled[y_train == 0]
    
    print(f"✓ Training on {len(X_train_normal)} normal samples")
    
    return (X_train_normal, X_test_scaled, y_train, y_test, 
            feature_names, preprocessor, X_train_scaled, X_test)


def train_models(X_train_normal: np.ndarray) -> dict:
    """
    Train all three anomaly detection models.
    
    Args:
        X_train_normal: Normal samples for training
        
    Returns:
        Dict of trained models
    """
    print("\n" + "="*60)
    print("TRAINING ANOMALY DETECTION MODELS")
    print("="*60)
    
    models = {}
    
    # Train Isolation Forest
    print("\n[1/3] Training Isolation Forest...")
    if_model = IsolationForest(**Config.ISOLATION_FOREST_PARAMS)
    if_model.fit(X_train_normal)
    models["isolation_forest"] = if_model
    print("✓ Isolation Forest trained")
    
    # Train One-Class SVM
    print("\n[2/3] Training One-Class SVM...")
    svm_model = OneClassSVM(**Config.ONE_CLASS_SVM_PARAMS)
    svm_model.fit(X_train_normal)
    models["one_class_svm"] = svm_model
    print("✓ One-Class SVM trained")
    
    # Train Autoencoder
    print("\n[3/3] Training Autoencoder...")
    ae_model = Autoencoder(**Config.AUTOENCODER_PARAMS)
    ae_model.fit(X_train_normal)
    models["autoencoder"] = ae_model
    print("✓ Autoencoder trained")
    
    print("\n✓ All models trained successfully")
    
    return models


def evaluate_models(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    Evaluate all models on test set.
    
    Args:
        models: Dict of trained models
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dict of evaluation results
    """
    print("\n" + "="*60)
    print("EVALUATING MODELS")
    print("="*60)
    
    # Collect scores from all models
    model_scores = {}
    for model_name, model in models.items():
        scores = model.score_samples(X_test)
        model_scores[model_name] = scores
    
    # Compare models
    comparison = ModelComparison.compare_models(
        y_true=y_test,
        model_scores=model_scores,
        threshold=0.5
    )
    
    # Print results
    print("\nModel Performance Comparison:")
    print("-" * 60)
    
    for model_name in models.keys():
        if model_name in comparison:
            metrics = comparison[model_name]
            print(f"\n{model_name.upper().replace('_', ' ')}:")
            print(f"  AUROC: {metrics.get('auroc', 0):.4f}")
            print(f"  Precision@10%: {metrics.get('precision_at_10', 0):.4f}")
            print(f"  F1 Score: {metrics.get('f1_score', 0):.4f}")
            print(f"  False Negative Rate: {metrics.get('false_negative_rate', 0):.4f}")
    
    if "best_models" in comparison:
        print("\nBest Models by Metric:")
        print("-" * 60)
        for metric, model_name in comparison["best_models"].items():
            print(f"  {metric}: {model_name}")
    
    return comparison, model_scores


def create_visualizations(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_scores: dict,
    feature_names: list,
    X_train: np.ndarray
) -> None:
    """
    Create comprehensive visualizations for model comparison.
    
    Args:
        models: Dict of trained models
        X_test: Test features
        y_test: Test labels
        model_scores: Dict of model prediction scores
        feature_names: List of feature names
        X_train: Training data for SHAP baseline
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    visualizer = ModelVisualizer(save_dir="visualizations")
    
    # Collect feature importance from each model using SHAP
    feature_importance = {}
    
    for model_name, model in models.items():
        print(f"\nExtracting feature importance for {model_name}...")
        
        # Use appropriate explainer based on model type
        if model_name == 'autoencoder':
            # Use reconstruction explainer for autoencoder
            explainer = ReconstructionExplainer(
                model=model,
                feature_names=feature_names
            )
            # Get explanation for first test sample
            explanations = explainer.explain_sample(X_test[0:1], top_k=10)
            importance_dict = explanations[0]['feature_importance']
        else:
            # Use SHAP explainer for tree/SVM models
            sklearn_model = model.model if hasattr(model, 'model') else model
            explainer = SHAPExplainer(
                model=sklearn_model, 
                background_data=X_train[:100],
                feature_names=feature_names
            )
            # Get SHAP explanation for first test sample
            explanations = explainer.explain_sample(X_test[0:1], top_k=10)
            importance_dict = explanations[0]['feature_importance']
        
        feature_importance[model_name] = importance_dict
    
    # Create comparative visualization
    print("\nGenerating comparative analysis plots...")
    
    # Get evaluation results for visualization
    evaluation_results = {}
    for model_name in models.keys():
        # Recalculate metrics for clean presentation
        scores = model_scores[model_name]
        metrics = AnomalyDetectionMetrics.compute_metrics(
            y_true=y_test,
            y_scores=scores,
            threshold=0.5
        )
        evaluation_results[model_name] = metrics
    
    visualizer.create_comparative_analysis(
        evaluation_results=evaluation_results,
        predictions=model_scores,
        feature_importance=feature_importance,
        feature_names=feature_names
    )
    
    print("\n✓ Visualizations complete")


def run_inference_demo(
    models: dict,
    preprocessor: DataPreprocessor,
    feature_names: list,
    X_test_scaled: np.ndarray,
    X_test_original: np.ndarray,
    X_train_background: np.ndarray
) -> None:
    """
    Run inference demo on sample data.
    
    Args:
        models: Dict of trained models
        preprocessor: Fitted preprocessor
        feature_names: List of feature names
        X_test_scaled: Scaled test data for models
        X_test_original: Original unscaled test data for LLM
        X_train_background: Background data for SHAP
    """
    print("\n" + "="*60)
    print("RUNNING INFERENCE DEMO")
    print("="*60)
    
    # Get HuggingFace API token (if available)
    try:
        hf_token = Config.get_hf_api_token()
        use_llm = True
        print("✓ HuggingFace API token found")
    except ValueError:
        hf_token = None
        use_llm = False
        print("⚠ HuggingFace API token not found - LLM explanations disabled")
        print("  Set HUGGINGFACE_API_TOKEN environment variable to enable")
    
    # Initialize pipeline
    pipeline = InferencePipeline(
        models=models,
        preprocessor=preprocessor,
        feature_names=feature_names,
        hf_api_token=hf_token,
        use_llm=use_llm
    )
    
    # Run inference on a single test sample
    print("\nAnalyzing sample patient data...")
    sample_scaled = X_test_scaled[0:1]  # Take first test sample (scaled for models)
    sample_original = X_test_original.iloc[0:1] if hasattr(X_test_original, 'iloc') else X_test_original[0:1]  # Original values for LLM
    
    results = pipeline.predict_single_sample(
        sample=sample_scaled[0],
        original_sample=sample_original,
        background_data=X_train_background[:100],  # Use subset for SHAP
        generate_llm_explanation=use_llm
    )
    
    # Print results in structured format
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    
    print(json.dumps(results, indent=2, default=str))
    
    # Save results to file
    results_path = os.path.join(Config.MODELS_DIR, "sample_inference_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {results_path}")


def save_models(
    models: dict,
    preprocessor: DataPreprocessor,
    feature_names: list
) -> None:
    """
    Save trained models and preprocessor.
    
    Args:
        models: Dict of trained models
        preprocessor: Fitted preprocessor
        feature_names: List of feature names
    """
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    # Create pipeline and save
    pipeline = InferencePipeline(
        models=models,
        preprocessor=preprocessor,
        feature_names=feature_names,
        use_llm=False  # Don't initialize LLM for saving
    )
    
    pipeline.save_pipeline(Config.MODELS_DIR)
    
    print(f"✓ Models saved to: {Config.MODELS_DIR}")


def main():
    """Main execution function."""
    print("="*60)
    print("HEART ATTACK ANOMALY DETECTION WITH XAI")
    print("="*60)
    
    # Setup
    setup_environment()
    
    # Load and preprocess data
    (X_train_normal, X_test_scaled, y_train, y_test, 
     feature_names, preprocessor, X_train_scaled, X_test_original) = load_and_preprocess_data(
        use_synthetic=False  # Set to False to use real dataset
    )
    
    # Train models
    models = train_models(X_train_normal)
    
    # Evaluate models
    evaluation_results, model_scores = evaluate_models(models, X_test_scaled, y_test)
    
    # Create visualizations
    create_visualizations(
        models=models,
        X_test=X_test_scaled,
        y_test=y_test,
        model_scores=model_scores,
        feature_names=feature_names,
        X_train=X_train_scaled
    )
    
    # Run inference demo
    run_inference_demo(
        models=models,
        preprocessor=preprocessor,
        feature_names=feature_names,
        X_test_scaled=X_test_scaled,
        X_test_original=X_test_original,
        X_train_background=X_train_scaled
    )
    
    # Save models
    save_models(models, preprocessor, feature_names)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Set HUGGINGFACE_API_TOKEN to enable LLM explanations")
    print("2. Replace synthetic data with real heart attack dataset")
    print("3. Tune model hyperparameters in config/settings.py")
    print("4. Deploy using the saved models in saved_models/")


if __name__ == "__main__":
    main()
