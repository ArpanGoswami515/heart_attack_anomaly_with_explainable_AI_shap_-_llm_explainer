# Quick Start Guide

## 🚀 Installation & Setup (5 minutes)

### Step 1: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: (Optional) Set HuggingFace API Token

```bash
# Windows Command Prompt
set HUGGINGFACE_API_TOKEN=your_token_here

# Windows PowerShell
$env:HUGGINGFACE_API_TOKEN="your_token_here"

# Linux/Mac
export HUGGINGFACE_API_TOKEN=your_token_here
```

**Or** create `.env` file:
```
HUGGINGFACE_API_TOKEN=your_token_here
```

Get your token: https://huggingface.co/settings/tokens

### Step 4: Verify Setup

```bash
python setup_verify.py
```

### Step 5: Run the System

```bash
python main.py
```

## 📊 Expected Output

```
============================================================
HEART ATTACK ANOMALY DETECTION WITH XAI
============================================================

✓ Environment setup complete

============================================================
LOADING AND PREPROCESSING DATA
============================================================
Generating synthetic heart attack dataset...
✓ Dataset loaded: 1000 samples, 13 features
✓ Anomaly ratio: 10.00%
...

============================================================
TRAINING ANOMALY DETECTION MODELS
============================================================

[1/3] Training Isolation Forest...
✓ Isolation Forest trained

[2/3] Training One-Class SVM...
✓ One-Class SVM trained

[3/3] Training Autoencoder...
✓ Autoencoder trained

============================================================
EVALUATING MODELS
============================================================

Model Performance Comparison:
------------------------------------------------------------

ISOLATION FOREST:
  AUROC: 0.8534
  Precision@10%: 0.8200
  F1 Score: 0.7156
  False Negative Rate: 0.1500

...

============================================================
RUNNING INFERENCE DEMO
============================================================

Analyzing sample patient data...

============================================================
INFERENCE RESULTS
============================================================
{
  "isolation_forest": {
    "anomaly_score": 0.789,
    "risk_level": "High",
    "explanation": {
      "feature_importance": {
        "age": 0.234,
        "cholesterol": 0.189,
        ...
      }
    }
  },
  ...
}

✓ Results saved to: saved_models/sample_inference_results.json

============================================================
SAVING MODELS
============================================================
✓ Models saved to: saved_models

============================================================
PIPELINE COMPLETE
============================================================
```

## 🎯 Using Your Own Data

1. Prepare your CSV file with:
   - Features as columns
   - `target` column (0=normal, 1=heart attack)

2. Place file in `data/` directory

3. Update `main.py`:
```python
# Change this line
(X_train_normal, X_test, ...) = load_and_preprocess_data(
    use_synthetic=False  # Set to False
)
```

4. Update file path in `data_loader.py` if needed

## 🔧 Customization

### Adjust Model Parameters

Edit `config/settings.py`:

```python
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 200,  # More trees
    "contamination": 0.15,  # Adjust expected anomaly ratio
    ...
}
```

### Change Risk Thresholds

```python
ANOMALY_THRESHOLDS = {
    "low": 0.4,    # Adjust thresholds
    "medium": 0.7,
    "high": 1.0
}
```

### Select Different LLM

```python
HF_MODEL = "meta-llama/Llama-2-7b-chat-hf"  # Use different model
```

## 🔍 Inference Only (Using Saved Models)

```python
from pipeline.inference_pipeline import InferencePipeline
import numpy as np

# Load saved pipeline
pipeline = InferencePipeline.load_pipeline(
    load_dir="saved_models",
    hf_api_token="your_token",
    use_llm=True
)

# Prepare sample data (must match training features)
patient_data = np.array([...])  # Your feature values

# Get predictions
results = pipeline.predict_single_sample(
    sample=patient_data,
    generate_llm_explanation=True
)

# Access results
print(f"Risk Level: {results['isolation_forest']['risk_level']}")
print(f"Anomaly Score: {results['isolation_forest']['anomaly_score']:.3f}")
print(f"LLM Summary: {results['llm_summary']}")
```

## 📈 Evaluation Only

```python
from evaluation.metrics import AnomalyDetectionMetrics

# Evaluate a model
metrics = AnomalyDetectionMetrics.evaluate_model(
    y_true=y_test,
    anomaly_scores=predicted_scores,
    threshold=0.5
)

print(f"AUROC: {metrics['auroc']:.4f}")
print(f"Precision@10%: {metrics['precision_at_10']:.4f}")
```

## 🐛 Troubleshooting

### Import Errors

```bash
# Ensure you're in the project directory
cd heart_attack_anomaly_ai

# Run with module syntax
python -m main
```

### CUDA/GPU Issues

```python
# Force CPU usage for PyTorch
import torch
torch.device("cpu")
```

### HuggingFace API Errors

- Check token is valid
- Verify internet connection
- Some models may be gated (require approval)
- System works without LLM (just won't generate text explanations)

### Memory Issues

```python
# Reduce batch size in config/settings.py
AUTOENCODER_PARAMS = {
    "batch_size": 16,  # Reduce from 32
    ...
}
```

## 📚 File Structure Reference

```
heart_attack_anomaly_ai/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration
├── data/
│   ├── __init__.py
│   └── data_loader.py       # Data loading
├── preprocessing/
│   ├── __init__.py
│   └── preprocess.py        # Data preprocessing
├── models/
│   ├── __init__.py
│   ├── base_model.py        # Base class
│   ├── isolation_forest.py # Model 1
│   ├── one_class_svm.py    # Model 2
│   └── autoencoder.py      # Model 3
├── explainability/
│   ├── __init__.py
│   ├── shap_explainer.py   # SHAP XAI
│   └── reconstruction_explainer.py  # Autoencoder XAI
├── llm/
│   ├── __init__.py
│   └── hf_explainer.py     # LLM integration
├── evaluation/
│   ├── __init__.py
│   └── metrics.py          # Evaluation metrics
├── pipeline/
│   ├── __init__.py
│   └── inference_pipeline.py  # Main pipeline
├── main.py                 # Entry point
├── setup_verify.py         # Setup verification
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
├── .gitignore            # Git ignore rules
└── README.md             # Documentation
```

## 🎓 Learning Resources

- **Isolation Forest**: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- **One-Class SVM**: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
- **Autoencoders**: https://pytorch.org/tutorials/beginner/introyt/autoencoderyt.html
- **SHAP**: https://shap.readthedocs.io/
- **HuggingFace API**: https://huggingface.co/docs/api-inference/

## 💡 Tips

1. Start with synthetic data to verify everything works
2. Tune contamination parameter to match your expected anomaly rate
3. Use CPU for small datasets (< 10k samples)
4. Save models after training (done automatically)
5. LLM explanations are optional but enhance interpretability
6. Compare all 3 models - ensemble can improve accuracy

## ✅ Success Checklist

- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Setup verification passed (`python setup_verify.py`)
- [ ] Main pipeline runs successfully (`python main.py`)
- [ ] Models saved to `saved_models/` directory
- [ ] Results JSON file generated
- [ ] (Optional) HuggingFace token configured
- [ ] (Optional) Own dataset integrated

## 🤝 Support

For issues or questions:
1. Check this QUICKSTART.md
2. Review README.md
3. Verify setup_verify.py passes all checks
4. Check code comments and docstrings
5. Review example output in main.py
