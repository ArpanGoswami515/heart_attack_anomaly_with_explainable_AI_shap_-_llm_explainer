# Heart Attack Anomaly Detection with Explainable AI

A production-ready Python system for detecting heart attack risk using anomaly detection, explainable AI (SHAP), and generative AI (HuggingFace LLM).

## 🏗️ Architecture

```
heart_attack_anomaly_ai/
├── config/              # Configuration settings
├── data/                # Data loading utilities
├── preprocessing/       # Data preprocessing
├── models/              # Anomaly detection models
│   ├── isolation_forest.py
│   ├── one_class_svm.py
│   └── autoencoder.py
├── explainability/      # XAI modules
│   ├── shap_explainer.py
│   └── reconstruction_explainer.py
├── llm/                 # LLM integration
├── evaluation/          # Evaluation metrics
├── pipeline/            # Inference pipeline
└── main.py              # Main entry point
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set HuggingFace API Token (Optional)

```bash
# Linux/Mac
export HUGGINGFACE_API_TOKEN="your_token_here"

# Windows
set HUGGINGFACE_API_TOKEN=your_token_here
```

Or create a `.env` file:
```
HUGGINGFACE_API_TOKEN=your_token_here
```

### 3. Run the System

```bash
python main.py
```

### 4. (Optional) Optimize Hyperparameters

For best performance, tune hyperparameters for your dataset:

```bash
# Quick tuning (5-10 minutes)
python quick_tune.py

# Apply optimized parameters
python apply_tuned_params.py

# Or full tuning (30-60 minutes)
python tune_hyperparameters.py
```

See [TUNING_GUIDE.md](TUNING_GUIDE.md) for details.

## 🔬 Models Implemented

1. **Isolation Forest** (scikit-learn)
   - Tree-based anomaly detection
   - Fast training and inference
   - Normalized scores ∈ [0, 1]

2. **One-Class SVM** (scikit-learn)
   - RBF kernel
   - Decision boundary-based detection
   - Scaled anomaly scores

3. **Autoencoder** (PyTorch)
   - Neural network reconstruction
   - Trained on normal samples only
   - Reconstruction error as anomaly score

## 🔍 Explainability

### SHAP Explainer
- Used for Isolation Forest and One-Class SVM
- Feature importance based on Shapley values
- Model-agnostic explanations

### Reconstruction Explainer
- Used for Autoencoder
- Per-feature reconstruction error
- Identifies which features are anomalous

### LLM Explanations
- HuggingFace Inference API integration
- Clinical-safe explanations
- **Enhanced with actual feature values** (e.g., "Age: 63, Cholesterol: 295")
- Feature importance scores alongside real values
- Comparative model summaries

**Example LLM Input:**
```
Patient Key Features (with importance scores):
- Age: 63.00 (importance: 0.156)
- Cholesterol: 295.00 (importance: 0.234)
- Max Heart Rate: 142.00 (importance: 0.189)
- ST Depression: 3.20 (importance: 0.278)
```

This provides the LLM with both **what the values are** and **how much they contributed** to the anomaly detection.

## 📊 Evaluation Metrics

- AUROC (Area Under ROC Curve)
- Precision @ K (top 10%, 20%)
- False Negative Rate
- F1 Score
- Confusion Matrix

## ⚙️ Hyperparameter Tuning

Automated hyperparameter optimization for all models:

**Quick Tuning** (~5-10 minutes):
```bash
python quick_tune.py
python apply_tuned_params.py
```

**Full Tuning** (~30-60 minutes):
```bash
python tune_hyperparameters.py
python apply_tuned_params.py
```

**Features:**
- Grid search across optimal parameter ranges
- Multiple optimization metrics (AUROC, Precision@K, F1)
- Automatic config file updates with best parameters
- Detailed results saved as JSON
- Works with your specific dataset

See [TUNING_GUIDE.md](TUNING_GUIDE.md) for comprehensive documentation.

## 🎯 Usage Examples

### Basic Inference

```python
from pipeline.inference_pipeline import InferencePipeline

# Load saved pipeline
pipeline = InferencePipeline.load_pipeline(
    load_dir="saved_models",
    hf_api_token="your_token"
)

# Predict on new sample
results = pipeline.predict_single_sample(
    sample=patient_data,
    generate_llm_explanation=True
)

print(results["isolation_forest"]["anomaly_score"])
print(results["llm_summary"])
```

### Training Custom Models

```python
from models.isolation_forest import IsolationForest
from preprocessing.preprocess import DataPreprocessor

# Preprocess data
preprocessor = DataPreprocessor(scaling_method="standard")
X_scaled = preprocessor.fit_transform(X_train)

# Train model
model = IsolationForest(n_estimators=100, contamination=0.1)
model.fit(X_scaled)

# Get predictions
scores = model.score_samples(X_test)
```

### Model Comparison

```python
from evaluation.metrics import ModelComparison

comparison = ModelComparison.compare_models(
    y_true=y_test,
    model_scores={
        "isolation_forest": if_scores,
        "one_class_svm": svm_scores,
        "autoencoder": ae_scores
    }
)

print(comparison["best_models"])
```

## ⚙️ Configuration

Edit `config/settings.py` to customize:

- Model hyperparameters
- Risk thresholds
- SHAP settings
- HuggingFace model selection
- Data paths

## 🔒 Safety Features

✅ NO medical diagnosis claims
✅ NO direct LLM predictions
✅ Clinical-safe language
✅ Structured input validation
✅ Error handling and logging

## 📦 Output Format

```json
{
  "isolation_forest": {
    "anomaly_score": 0.78,
    "risk_level": "High",
    "explanation": {
      "feature_importance": {
        "age": 0.234,
        "cholesterol": 0.189,
        ...
      }
    }
  },
  "one_class_svm": {...},
  "autoencoder": {...},
  "llm_summary": "The models show consensus..."
}
```

## 🧪 Testing with Synthetic Data

The system includes synthetic data generation for testing:

```python
from data.data_loader import HeartAttackDataLoader

df = HeartAttackDataLoader.generate_sample_dataset(
    n_samples=1000,
    contamination=0.1
)
```

## 📝 Requirements

- Python 3.8+
- CPU: Any modern processor
- RAM: 4GB minimum
- GPU: Optional (for Autoencoder training speedup)

## 🤝 Contributing

This is a production-ready template. Customize for your specific use case:

1. Replace synthetic data with real heart attack dataset
2. Tune hyperparameters in `config/settings.py`
3. Add custom evaluation metrics
4. Extend with additional models

## 📄 License

This code is provided as-is for educational and research purposes.

## ⚠️ Disclaimer

**This system is for research purposes only and should not be used for medical diagnosis or treatment decisions without proper clinical validation and regulatory approval.**

## 🔗 References

- SHAP: https://github.com/slundberg/shap
- Isolation Forest: Scikit-learn
- HuggingFace Inference API: https://huggingface.co/docs/api-inference/
