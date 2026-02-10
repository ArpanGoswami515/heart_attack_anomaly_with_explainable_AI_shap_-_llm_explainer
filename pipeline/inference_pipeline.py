"""
Unified inference pipeline for heart attack anomaly detection.
"""

import numpy as np
import pickle
import os
from typing import Dict, Any, List, Optional
from models.base_model import BaseAnomalyModel
from models.isolation_forest import IsolationForest
from models.one_class_svm import OneClassSVM
from models.autoencoder import Autoencoder
from explainability.shap_explainer import SHAPExplainer
from explainability.reconstruction_explainer import ReconstructionExplainer
from llm.hf_explainer import HuggingFaceExplainer
from preprocessing.preprocess import DataPreprocessor
from config.settings import Config


class InferencePipeline:
    """Complete inference pipeline for anomaly detection with explainability."""
    
    def __init__(
        self,
        models: Dict[str, BaseAnomalyModel],
        preprocessor: DataPreprocessor,
        feature_names: List[str],
        hf_api_token: Optional[str] = None,
        use_llm: bool = True
    ):
        """
        Initialize inference pipeline.
        
        Args:
            models: Dict mapping model names to trained model instances
            preprocessor: Fitted data preprocessor
            feature_names: List of feature names
            hf_api_token: HuggingFace API token (optional)
            use_llm: Whether to use LLM for explanations
        """
        self.models = models
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.use_llm = use_llm
        
        # Initialize explainers
        self.explainers = {}
        self._initialize_explainers()
        
        # Initialize LLM explainer if requested
        self.llm_explainer = None
        if use_llm and hf_api_token:
            try:
                self.llm_explainer = HuggingFaceExplainer(
                    api_token=hf_api_token,
                    model_name=Config.HF_MODEL,
                    max_tokens=Config.HF_MAX_TOKENS,
                    temperature=Config.HF_TEMPERATURE
                )
            except Exception as e:
                print(f"Warning: Could not initialize LLM explainer: {e}")
                self.llm_explainer = None
    
    def _initialize_explainers(self) -> None:
        """Initialize appropriate explainers for each model."""
        # Get background data (sample from training data if available)
        # For now, we'll initialize explainers on-demand during inference
        pass
    
    def predict(
        self,
        X: np.ndarray,
        background_data: Optional[np.ndarray] = None,
        top_k_features: int = 5
    ) -> Dict[str, Any]:
        """
        Run inference on input samples with full explainability.
        
        Args:
            X: Input samples (preprocessed or raw)
            background_data: Background data for SHAP (preprocessed)
            top_k_features: Number of top features to explain
            
        Returns:
            Dict containing predictions and explanations for all models
        """
        # Preprocess if needed
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Store original shape
        n_samples = X.shape[0]
        
        results = {}
        
        # Run each model
        for model_name, model in self.models.items():
            try:
                model_result = self._predict_single_model(
                    model=model,
                    model_name=model_name,
                    X=X,
                    background_data=background_data,
                    top_k_features=top_k_features
                )
                results[model_name] = model_result
            except Exception as e:
                results[model_name] = {
                    "error": str(e),
                    "status": "failed"
                }
        
        # Generate LLM summary if available
        if self.llm_explainer and n_samples == 1:
            results["llm_summary"] = self._generate_llm_summary(results, X)
        
        return results
    
    def _predict_single_model(
        self,
        model: BaseAnomalyModel,
        model_name: str,
        X: np.ndarray,
        background_data: Optional[np.ndarray] = None,
        top_k_features: int = 5
    ) -> Dict[str, Any]:
        """
        Run prediction and explanation for a single model.
        
        Args:
            model: Trained model
            model_name: Model identifier
            X: Input samples
            background_data: Background data for SHAP
            top_k_features: Number of top features to explain
            
        Returns:
            Dict containing predictions and explanations
        """
        # Get anomaly scores
        scores = model.score_samples(X)
        
        # Get risk levels
        risk_levels = model._scores_to_risk_levels(scores)
        
        # Generate explanations
        explanations = self._explain_predictions(
            model=model,
            model_name=model_name,
            X=X,
            background_data=background_data,
            top_k_features=top_k_features
        )
        
        # Compile results
        results = []
        for i in range(len(X)):
            sample_result = {
                "anomaly_score": float(scores[i]),
                "risk_level": str(risk_levels[i]),
                "explanation": explanations[i] if i < len(explanations) else {}
            }
            results.append(sample_result)
        
        # If single sample, unwrap list
        if len(results) == 1:
            return results[0]
        
        return {"samples": results}
    
    def _explain_predictions(
        self,
        model: BaseAnomalyModel,
        model_name: str,
        X: np.ndarray,
        background_data: Optional[np.ndarray] = None,
        top_k_features: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for predictions.
        
        Args:
            model: Trained model
            model_name: Model identifier
            X: Input samples
            background_data: Background data for SHAP
            top_k_features: Number of top features to explain
            
        Returns:
            List of explanation dicts
        """
        try:
            if isinstance(model, Autoencoder):
                # Use reconstruction explainer
                explainer = ReconstructionExplainer(
                    model=model,
                    feature_names=self.feature_names
                )
                explanations = explainer.explain_sample(X, top_k=top_k_features)
            else:
                # Use SHAP explainer for tree-based and SVM models
                if background_data is None:
                    # Use a subset of X as background if not provided
                    background_data = X[:min(100, len(X))]
                
                # Get the underlying sklearn model
                sklearn_model = model.model
                
                explainer = SHAPExplainer(
                    model=sklearn_model,
                    background_data=background_data,
                    feature_names=self.feature_names,
                    num_samples=Config.SHAP_NUM_SAMPLES
                )
                explanations = explainer.explain_sample(X, top_k=top_k_features)
            
            return explanations
        
        except Exception as e:
            # Return empty explanations if explanation fails
            return [{"error": str(e), "feature_importance": {}}] * len(X)
    
    def _generate_llm_summary(
        self,
        results: Dict[str, Any],
        X: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate LLM-based summary of all model results.
        
        Args:
            results: Dict containing results from all models
            X: Input sample data (for extracting feature values)
            
        Returns:
            LLM-generated summary text
        """
        if not self.llm_explainer:
            return "LLM explainer not available"
        
        try:
            # Extract feature values from input sample
            feature_values = {}
            if X is not None and len(X.shape) >= 2 and X.shape[0] > 0:
                sample_values = X[0]  # Get first sample
                for i, feature_name in enumerate(self.feature_names):
                    if i < len(sample_values):
                        feature_values[feature_name] = float(sample_values[i])
            
            # Prepare data for LLM
            model_summaries = {}
            
            for model_name, result in results.items():
                if "error" in result:
                    continue
                
                if "samples" in result:
                    # Handle batch results (take first sample)
                    result = result["samples"][0]
                
                model_summaries[model_name] = {
                    "risk_level": result.get("risk_level", "Unknown"),
                    "anomaly_score": result.get("anomaly_score", 0.0)
                }
            
            # Generate comparative summary with feature values
            if model_summaries:
                summary = self.llm_explainer.generate_comparative_summary(
                    model_summaries,
                    feature_values
                )
                return summary
            else:
                return "No valid model results to summarize"
        
        except Exception as e:
            return f"Error generating LLM summary: {str(e)}"
    
    def predict_single_sample(
        self,
        sample: np.ndarray,
        original_sample: Optional[np.ndarray] = None,
        background_data: Optional[np.ndarray] = None,
        generate_llm_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Convenience method for single sample prediction.
        
        Args:
            sample: Single input sample (1D array, scaled for models)
            original_sample: Original unscaled sample for LLM (optional)
            background_data: Background data for SHAP
            generate_llm_explanation: Whether to generate LLM explanation
            
        Returns:
            Dict containing comprehensive results
        """
        if len(sample.shape) == 1:
            sample = sample.reshape(1, -1)
        
        # Extract actual feature values from ORIGINAL unscaled sample for LLM
        all_feature_values = {}
        if original_sample is not None:
            # Use original unscaled values for LLM
            if hasattr(original_sample, 'iloc'):
                # It's a DataFrame/Series
                original_values = original_sample.iloc[0] if len(original_sample.shape) > 1 else original_sample
                for feature_name in self.feature_names:
                    if feature_name in original_values.index:
                        all_feature_values[feature_name] = float(original_values[feature_name])
            else:
                # It's a numpy array
                if len(original_sample.shape) == 1:
                    original_sample = original_sample.reshape(1, -1)
                original_values = original_sample[0]
                for i, feature_name in enumerate(self.feature_names):
                    if i < len(original_values):
                        all_feature_values[feature_name] = float(original_values[i])
        else:
            # Fallback to scaled values (not ideal but better than nothing)
            if len(sample.shape) >= 2 and sample.shape[0] > 0:
                sample_values = sample[0]
                for i, feature_name in enumerate(self.feature_names):
                    if i < len(sample_values):
                        all_feature_values[feature_name] = float(sample_values[i])
        
        # Get predictions from all models
        results = self.predict(
            X=sample,
            background_data=background_data,
            top_k_features=Config.TOP_FEATURES_TO_EXPLAIN
        )
        
        # Optionally generate detailed LLM explanations per model
        if generate_llm_explanation and self.llm_explainer and all_feature_values:
            print(f"\n[DEBUG] Generating LLM explanations for {len([k for k in results.keys() if k != 'llm_summary'])} models...")
            print(f"[DEBUG] Total feature values available: {len(all_feature_values)} (ORIGINAL UNSCALED VALUES)")
            
            for model_name, result in results.items():
                if model_name == "llm_summary" or "error" in result:
                    continue
                
                # Get feature importances
                explanation = result.get("explanation", {})
                feature_importance = explanation.get("feature_importance", {})
                
                print(f"[DEBUG] {model_name}: Found {len(feature_importance)} important features")
                
                if feature_importance:
                    # Extract only the feature values that are in feature_importance
                    # This ensures we only pass values for the top-k important features
                    relevant_feature_values = {
                        fname: all_feature_values[fname]
                        for fname in feature_importance.keys()
                        if fname in all_feature_values
                    }
                    
                    print(f"[DEBUG] {model_name}: Sending {len(relevant_feature_values)} feature values to LLM")
                    for fname, importance in list(feature_importance.items())[:3]:
                        value = relevant_feature_values.get(fname, "N/A")
                        print(f"         {fname}: value={value:.2f}, importance={importance:.3f}")
                    
                    # Generate detailed explanation for this model
                    try:
                        llm_explanation = self.llm_explainer.explain(
                            risk_level=result["risk_level"],
                            anomaly_score=result["anomaly_score"],
                            feature_importances=feature_importance,
                            feature_values=relevant_feature_values,
                            model_name=model_name
                        )
                        result["llm_explanation"] = llm_explanation
                        print(f"[DEBUG] {model_name}: LLM explanation generated successfully")
                    except Exception as e:
                        result["llm_explanation_error"] = str(e)
                        print(f"[DEBUG] {model_name}: LLM explanation failed - {e}")
                else:
                    print(f"[DEBUG] {model_name}: No feature importance found - skipping LLM explanation")
        
        return results
    
    def save_pipeline(self, save_dir: str) -> None:
        """
        Save pipeline components to disk.
        
        Args:
            save_dir: Directory to save pipeline components
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save preprocessor
        preprocessor_path = os.path.join(save_dir, "preprocessor.pkl")
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        # Save feature names
        feature_names_path = os.path.join(save_dir, "feature_names.pkl")
        with open(feature_names_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
    
    @classmethod
    def load_pipeline(
        cls,
        load_dir: str,
        hf_api_token: Optional[str] = None,
        use_llm: bool = True
    ) -> "InferencePipeline":
        """
        Load pipeline from disk.
        
        Args:
            load_dir: Directory containing saved pipeline components
            hf_api_token: HuggingFace API token
            use_llm: Whether to use LLM
            
        Returns:
            Loaded InferencePipeline
        """
        # Load preprocessor
        preprocessor_path = os.path.join(load_dir, "preprocessor.pkl")
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Load feature names
        feature_names_path = os.path.join(load_dir, "feature_names.pkl")
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)
        
        # Load models
        models = {}
        model_files = [f for f in os.listdir(load_dir) if f.endswith('.pkl') and f not in ['preprocessor.pkl', 'feature_names.pkl']]
        
        for model_file in model_files:
            model_name = model_file.replace('.pkl', '')
            model_path = os.path.join(load_dir, model_file)
            with open(model_path, 'rb') as f:
                models[model_name] = pickle.load(f)
        
        return cls(
            models=models,
            preprocessor=preprocessor,
            feature_names=feature_names,
            hf_api_token=hf_api_token,
            use_llm=use_llm
        )
