"""
Hyperparameter tuning for anomaly detection models.
Finds optimal parameters and updates configuration.
"""

import numpy as np
import json
import os
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import ParameterGrid
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data.data_loader import HeartAttackDataLoader
from preprocessing.preprocess import DataPreprocessor
from models.isolation_forest import IsolationForest
from models.one_class_svm import OneClassSVM
from models.autoencoder import Autoencoder
from evaluation.metrics import AnomalyDetectionMetrics
from config.settings import Config


class HyperparameterTuner:
    """Hyperparameter tuning for anomaly detection models."""
    
    def __init__(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        metric: str = "auroc",
        random_state: int = 42
    ):
        """
        Initialize tuner.
        
        Args:
            X_train: Training features (normal samples only)
            X_val: Validation features (mixed normal/anomaly)
            y_train: Training labels (should be mostly 0)
            y_val: Validation labels (mixed 0/1)
            metric: Primary metric to optimize ('auroc', 'precision_at_10', 'f1_score')
            random_state: Random seed
        """
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.metric = metric
        self.random_state = random_state
        self.results = {}
    
    def tune_isolation_forest(
        self,
        param_grid: Dict[str, List[Any]] = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tune Isolation Forest hyperparameters.
        
        Args:
            param_grid: Parameter grid to search (optional)
            
        Returns:
            Tuple of (best_params, best_score)
        """
        print("\n" + "="*70)
        print("TUNING ISOLATION FOREST")
        print("="*70)
        
        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200, 300],
                "max_samples": ["auto", 256, 512, 1024],
                "contamination": [0.05, 0.1, 0.15, 0.2],
                "max_features": [0.5, 0.75, 1.0],
                "bootstrap": [True, False]
            }
        
        best_score = -np.inf
        best_params = None
        all_results = []
        
        total_combinations = len(list(ParameterGrid(param_grid)))
        print(f"Testing {total_combinations} parameter combinations...")
        
        for i, params in enumerate(ParameterGrid(param_grid), 1):
            params["random_state"] = self.random_state
            params["n_jobs"] = -1
            
            try:
                # Train model
                model = IsolationForest(**params)
                model.fit(self.X_train)
                
                # Evaluate
                scores = model.score_samples(self.X_val)
                metrics = AnomalyDetectionMetrics.compute_metrics(
                    y_true=self.y_val,
                    y_scores=scores
                )
                
                score = metrics[self.metric]
                
                result = {
                    "params": params,
                    "score": score,
                    "all_metrics": metrics
                }
                all_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"✓ [{i}/{total_combinations}] New best {self.metric}: {score:.4f}")
                    print(f"  Params: n_estimators={params['n_estimators']}, "
                          f"contamination={params['contamination']}, "
                          f"max_samples={params['max_samples']}")
                else:
                    if i % 10 == 0:
                        print(f"  [{i}/{total_combinations}] {self.metric}: {score:.4f}")
                
            except Exception as e:
                print(f"✗ [{i}/{total_combinations}] Error: {e}")
                continue
        
        self.results["isolation_forest"] = {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": all_results
        }
        
        print(f"\n✓ Best {self.metric}: {best_score:.4f}")
        print(f"✓ Best parameters: {best_params}")
        
        return best_params, best_score
    
    def tune_one_class_svm(
        self,
        param_grid: Dict[str, List[Any]] = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tune One-Class SVM hyperparameters.
        
        Args:
            param_grid: Parameter grid to search (optional)
            
        Returns:
            Tuple of (best_params, best_score)
        """
        print("\n" + "="*70)
        print("TUNING ONE-CLASS SVM")
        print("="*70)
        
        if param_grid is None:
            param_grid = {
                "kernel": ["rbf", "poly", "sigmoid"],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
                "nu": [0.05, 0.1, 0.15, 0.2, 0.25],
                "degree": [2, 3, 4],  # Only for poly kernel
                "coef0": [0.0, 0.5, 1.0]  # For poly and sigmoid
            }
        
        best_score = -np.inf
        best_params = None
        all_results = []
        
        # Create filtered parameter combinations
        param_combinations = []
        base_params = {
            "kernel": param_grid["kernel"],
            "gamma": param_grid["gamma"],
            "nu": param_grid["nu"]
        }
        
        for params in ParameterGrid(base_params):
            if params["kernel"] == "poly":
                for degree in param_grid["degree"]:
                    for coef0 in param_grid["coef0"]:
                        p = params.copy()
                        p["degree"] = degree
                        p["coef0"] = coef0
                        param_combinations.append(p)
            elif params["kernel"] == "sigmoid":
                for coef0 in param_grid["coef0"]:
                    p = params.copy()
                    p["coef0"] = coef0
                    param_combinations.append(p)
            else:  # rbf
                param_combinations.append(params)
        
        total_combinations = len(param_combinations)
        print(f"Testing {total_combinations} parameter combinations...")
        
        for i, params in enumerate(param_combinations, 1):
            try:
                # Train model
                model = OneClassSVM(**params)
                model.fit(self.X_train)
                
                # Evaluate
                scores = model.score_samples(self.X_val)
                metrics = AnomalyDetectionMetrics.compute_metrics(
                    y_true=self.y_val,
                    y_scores=scores
                )
                
                score = metrics[self.metric]
                
                result = {
                    "params": params,
                    "score": score,
                    "all_metrics": metrics
                }
                all_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"✓ [{i}/{total_combinations}] New best {self.metric}: {score:.4f}")
                    print(f"  Params: kernel={params['kernel']}, "
                          f"gamma={params['gamma']}, nu={params['nu']}")
                else:
                    if i % 10 == 0:
                        print(f"  [{i}/{total_combinations}] {self.metric}: {score:.4f}")
                
            except Exception as e:
                print(f"✗ [{i}/{total_combinations}] Error: {e}")
                continue
        
        self.results["one_class_svm"] = {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": all_results
        }
        
        print(f"\n✓ Best {self.metric}: {best_score:.4f}")
        print(f"✓ Best parameters: {best_params}")
        
        return best_params, best_score
    
    def tune_autoencoder(
        self,
        param_grid: Dict[str, List[Any]] = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tune Autoencoder hyperparameters.
        
        Args:
            param_grid: Parameter grid to search (optional)
            
        Returns:
            Tuple of (best_params, best_score)
        """
        print("\n" + "="*70)
        print("TUNING AUTOENCODER")
        print("="*70)
        
        if param_grid is None:
            param_grid = {
                "encoding_dim": [4, 8, 16],
                "hidden_dims": [[16, 8], [32, 16], [64, 32], [32, 16, 8]],
                "learning_rate": [0.001, 0.0005, 0.0001],
                "batch_size": [16, 32, 64],
                "epochs": [50, 100],
                "early_stopping_patience": [5, 10, 15]
            }
        
        best_score = -np.inf
        best_params = None
        all_results = []
        
        total_combinations = len(list(ParameterGrid(param_grid)))
        print(f"Testing {total_combinations} parameter combinations...")
        print("(This may take a while as neural networks need to train...)")
        
        for i, params in enumerate(ParameterGrid(param_grid), 1):
            try:
                # Train model
                print(f"\n[{i}/{total_combinations}] Training with: "
                      f"encoding_dim={params['encoding_dim']}, "
                      f"hidden_dims={params['hidden_dims']}, "
                      f"lr={params['learning_rate']}")
                
                model = Autoencoder(**params)
                model.fit(self.X_train)
                
                # Evaluate
                scores = model.score_samples(self.X_val)
                metrics = AnomalyDetectionMetrics.compute_metrics(
                    y_true=self.y_val,
                    y_scores=scores
                )
                
                score = metrics[self.metric]
                
                result = {
                    "params": params,
                    "score": score,
                    "all_metrics": metrics
                }
                all_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"✓ New best {self.metric}: {score:.4f}")
                else:
                    print(f"  {self.metric}: {score:.4f}")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        
        self.results["autoencoder"] = {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": all_results
        }
        
        print(f"\n✓ Best {self.metric}: {best_score:.4f}")
        print(f"✓ Best parameters: {best_params}")
        
        return best_params, best_score
    
    def save_results(self, output_dir: str = "tuning_results") -> str:
        """
        Save tuning results to JSON file.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tuning_results_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Convert numpy types for JSON serialization
        results_serializable = {}
        for model_name, result in self.results.items():
            results_serializable[model_name] = {
                "best_params": self._convert_to_serializable(result["best_params"]),
                "best_score": float(result["best_score"]),
                "all_results": [
                    {
                        "params": self._convert_to_serializable(r["params"]),
                        "score": float(r["score"]),
                        "all_metrics": {k: float(v) for k, v in r["all_metrics"].items()}
                    }
                    for r in result["all_results"][:20]  # Save top 20 results
                ]
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\n✓ Results saved to: {filepath}")
        return filepath
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def generate_config_code(self) -> str:
        """
        Generate Python code to update config/settings.py with best parameters.
        
        Returns:
            Python code string
        """
        code_lines = [
            "# Optimized hyperparameters from tuning",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Optimization metric: {self.metric}",
            ""
        ]
        
        if "isolation_forest" in self.results:
            params = self.results["isolation_forest"]["best_params"]
            score = self.results["isolation_forest"]["best_score"]
            code_lines.append(f"# Isolation Forest - {self.metric}: {score:.4f}")
            code_lines.append("ISOLATION_FOREST_PARAMS: Dict[str, Any] = {")
            for key, value in params.items():
                if isinstance(value, str):
                    code_lines.append(f'    "{key}": "{value}",')
                else:
                    code_lines.append(f'    "{key}": {value},')
            code_lines.append("}")
            code_lines.append("")
        
        if "one_class_svm" in self.results:
            params = self.results["one_class_svm"]["best_params"]
            score = self.results["one_class_svm"]["best_score"]
            code_lines.append(f"# One-Class SVM - {self.metric}: {score:.4f}")
            code_lines.append("ONE_CLASS_SVM_PARAMS: Dict[str, Any] = {")
            for key, value in params.items():
                if isinstance(value, str):
                    code_lines.append(f'    "{key}": "{value}",')
                else:
                    code_lines.append(f'    "{key}": {value},')
            code_lines.append("}")
            code_lines.append("")
        
        if "autoencoder" in self.results:
            params = self.results["autoencoder"]["best_params"]
            score = self.results["autoencoder"]["best_score"]
            code_lines.append(f"# Autoencoder - {self.metric}: {score:.4f}")
            code_lines.append("AUTOENCODER_PARAMS: Dict[str, Any] = {")
            for key, value in params.items():
                if isinstance(value, str):
                    code_lines.append(f'    "{key}": "{value}",')
                elif isinstance(value, list):
                    code_lines.append(f'    "{key}": {value},')
                else:
                    code_lines.append(f'    "{key}": {value},')
            code_lines.append("}")
            code_lines.append("")
        
        return "\n".join(code_lines)


def main():
    """Main tuning execution."""
    print("="*70)
    print("HYPERPARAMETER TUNING FOR ANOMALY DETECTION MODELS")
    print("="*70)
    
    # Configuration
    USE_SYNTHETIC = False  # Set to False to use real dataset
    METRIC_TO_OPTIMIZE = "auroc"  # Options: 'auroc', 'precision_at_10', 'f1_score'
    
    print(f"\nConfiguration:")
    print(f"  Optimization metric: {METRIC_TO_OPTIMIZE}")
    print(f"  Using synthetic data: {USE_SYNTHETIC}")
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    if USE_SYNTHETIC:
        print("Generating synthetic dataset...")
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
        dataset_path = os.path.join(Config.DATA_DIR, "heart_attack_data.csv")
        data_loader = HeartAttackDataLoader(
            file_path=dataset_path,
            random_state=Config.RANDOM_SEED
        )
        data_loader.load_data()
    
    # Get features and labels
    X, y = data_loader.get_features_and_labels(target_column="target")
    print(f"✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✓ Anomaly ratio: {np.mean(y):.2%}")
    
    # Split data into train+val
    X_train_val, X_test, y_train_val, y_test = data_loader.train_test_split(
        X, y, test_size=0.2, stratify=True
    )
    
    # Further split train_val into train and validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.25,  # 0.25 * 0.8 = 0.2 of total data for validation
        stratify=y_train_val,
        random_state=Config.RANDOM_SEED
    )
    
    # Convert to numpy arrays
    y_train = y_train.values if hasattr(y_train, 'values') else y_train
    y_val = y_val.values if hasattr(y_val, 'values') else y_val
    y_test = y_test.values if hasattr(y_test, 'values') else y_test
    
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Validation set: {len(X_val)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Preprocess data
    preprocessor = DataPreprocessor(scaling_method="standard")
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Get normal samples for training
    X_train_normal = X_train_scaled[y_train == 0]
    print(f"✓ Training on {len(X_train_normal)} normal samples")
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        X_train=X_train_normal,
        X_val=X_val_scaled,
        y_train=y_train,
        y_val=y_val,
        metric=METRIC_TO_OPTIMIZE,
        random_state=Config.RANDOM_SEED
    )
    
    # Tune each model
    print("\n" + "="*70)
    print("STARTING HYPERPARAMETER TUNING")
    print("="*70)
    
    # Tune Isolation Forest
    if_params, if_score = tuner.tune_isolation_forest()
    
    # Tune One-Class SVM
    svm_params, svm_score = tuner.tune_one_class_svm()
    
    # Tune Autoencoder
    ae_params, ae_score = tuner.tune_autoencoder()
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    results_path = tuner.save_results()
    
    # Generate config code
    config_code = tuner.generate_config_code()
    
    config_path = os.path.join("tuning_results", "optimized_config.py")
    with open(config_path, 'w') as f:
        f.write(config_code)
    
    print(f"✓ Config code saved to: {config_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("TUNING COMPLETE - SUMMARY")
    print("="*70)
    
    print(f"\nOptimal Parameters (optimized for {METRIC_TO_OPTIMIZE}):")
    print("\n" + "-"*70)
    print(f"ISOLATION FOREST ({if_score:.4f}):")
    for key, value in if_params.items():
        if key not in ['random_state', 'n_jobs']:
            print(f"  {key}: {value}")
    
    print("\n" + "-"*70)
    print(f"ONE-CLASS SVM ({svm_score:.4f}):")
    for key, value in svm_params.items():
        print(f"  {key}: {value}")
    
    print("\n" + "-"*70)
    print(f"AUTOENCODER ({ae_score:.4f}):")
    for key, value in ae_params.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print(f"1. Review results in: {results_path}")
    print(f"2. Copy optimized parameters from: {config_path}")
    print("3. Update config/settings.py with the new parameters")
    print("4. Run main.py to train models with optimized parameters")
    print("="*70)


if __name__ == "__main__":
    main()
