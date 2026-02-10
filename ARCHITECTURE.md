# Technical Architecture Documentation

## System Overview

**Heart Attack Anomaly Detection with Explainable AI (XAI)**

A production-ready Python architecture for detecting heart attack risk using three anomaly detection models, SHAP-based explainability, and LLM-generated clinical explanations.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
│                    (Patient Medical Data)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING LAYER                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ • Data Validation                                        │   │
│  │ • Missing Value Handling                                 │   │
│  │ • Feature Scaling (StandardScaler / MinMaxScaler)        │   │
│  │ • Outlier Detection                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANOMALY DETECTION LAYER                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Isolation   │  │  One-Class   │  │  Autoencoder │          │
│  │    Forest    │  │     SVM      │  │   (PyTorch)  │          │
│  │  (sklearn)   │  │  (sklearn)   │  │              │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                  │
│         ▼                 ▼                  ▼                  │
│  Score ∈ [0,1]     Score ∈ [0,1]     Score ∈ [0,1]            │
└────────┬─────────────────┬─────────────────┬────────────────────┘
         │                 │                 │
         └────────┬────────┴────────┬────────┘
                  │                 │
                  ▼                 ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   EXPLAINABILITY LAYER   │  │   EXPLAINABILITY LAYER   │
│                          │  │                          │
│  ┌────────────────────┐  │  │  ┌────────────────────┐  │
│  │  SHAP Explainer    │  │  │  │  Reconstruction    │  │
│  │  (TreeExplainer/   │  │  │  │  Error Explainer   │  │
│  │   KernelExplainer) │  │  │  │                    │  │
│  │                    │  │  │  │  Per-feature       │  │
│  │  Shapley Values    │  │  │  │  reconstruction    │  │
│  │  Feature Import.   │  │  │  │  errors            │  │
│  └────────────────────┘  │  │  └────────────────────┘  │
└──────────┬───────────────┘  └──────────┬───────────────┘
           │                             │
           └──────────┬──────────────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │   Feature Rankings   │
           │   + Importance       │
           └──────────┬───────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LLM EXPLANATION LAYER                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         HuggingFace Inference API                        │   │
│  │         (Mistral-7B-Instruct / Custom LLM)               │   │
│  │                                                          │   │
│  │  • Structured Prompt Engineering                        │   │
│  │  • Clinical-safe Language Generation                    │   │
│  │  • Comparative Model Summary                            │   │
│  │  • Feature-based Explanation                            │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  JSON Response:                                          │   │
│  │  {                                                       │   │
│  │    "isolation_forest": {                                │   │
│  │      "anomaly_score": 0.78,                             │   │
│  │      "risk_level": "High",                              │   │
│  │      "explanation": {...}                               │   │
│  │    },                                                    │   │
│  │    "one_class_svm": {...},                              │   │
│  │    "autoencoder": {...},                                │   │
│  │    "llm_summary": "..."                                 │   │
│  │  }                                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Specifications

### 1. Data Layer (`data/`)

**Purpose**: Load and manage heart attack medical datasets

**Components**:
- `HeartAttackDataLoader`: CSV loading, train/test splitting
- Synthetic data generator for testing
- Feature extraction and label handling

**Input Format**:
```
CSV with columns: age, sex, chest_pain, bp, cholesterol, ..., target
target: 0 (normal), 1 (heart attack/anomaly)
```

### 2. Preprocessing Layer (`preprocessing/`)

**Purpose**: Normalize and clean data

**Components**:
- `DataPreprocessor`: Scaling, validation, outlier removal
- StandardScaler for zero-mean unit-variance
- MinMaxScaler for [0,1] range
- Missing value imputation
- Data validation checks

**Output**: Numpy arrays (n_samples, n_features)

### 3. Model Layer (`models/`)

#### Base Architecture
- `BaseAnomalyModel`: Abstract base class
  - `fit(X)`: Train on data
  - `score_samples(X)`: Return anomaly scores [0,1]
  - `predict_risk(X)`: Return risk levels (Low/Medium/High)

#### Model 1: Isolation Forest
- **Type**: Tree-based ensemble
- **Library**: scikit-learn
- **Training**: Unsupervised on normal samples
- **Score**: Decision function → normalized to [0,1]
- **Params**: n_estimators=100, contamination=0.1
- **Strength**: Fast, handles high-dimensional data

#### Model 2: One-Class SVM
- **Type**: Support Vector Machine
- **Library**: scikit-learn
- **Kernel**: RBF (Radial Basis Function)
- **Training**: Learns decision boundary around normal samples
- **Score**: Distance to boundary → normalized to [0,1]
- **Params**: nu=0.1, gamma=auto
- **Strength**: Robust to outliers, good for non-linear boundaries

#### Model 3: Autoencoder
- **Type**: Neural Network (PyTorch)
- **Architecture**: 
  - Encoder: [input_dim → 32 → 16 → 8]
  - Decoder: [8 → 16 → 32 → input_dim]
- **Training**: Reconstructs normal samples only
- **Score**: Reconstruction error (MSE) → normalized to [0,1]
- **Params**: lr=0.001, epochs=50, batch_size=32
- **Strength**: Captures complex patterns, per-feature errors

### 4. Explainability Layer (`explainability/`)

#### SHAP Explainer
- **For**: Isolation Forest, One-Class SVM
- **Method**: 
  - TreeExplainer for Isolation Forest
  - KernelExplainer for One-Class SVM
- **Output**: 
  - Shapley values per feature
  - Feature importance rankings
  - Direction of impact (positive/negative)

**Example Output**:
```python
{
  "feature_importance": {
    "age": 0.234,
    "cholesterol": 0.189,
    "bp": 0.156
  }
}
```

#### Reconstruction Explainer
- **For**: Autoencoder
- **Method**: Per-feature reconstruction error
- **Output**:
  - Feature-wise MSE
  - Normalized contributions
  - Comparison of original vs reconstructed

### 5. LLM Layer (`llm/`)

**Purpose**: Generate human-readable clinical explanations

**Components**:
- `HuggingFaceExplainer`: API client
- Prompt engineering for medical context
- Multi-model comparative summaries

**Prompt Template**:
```
You are a medical AI assistant.
Input:
- Risk Level: High
- Anomaly Score: 0.78
- Top Features: age (0.234), cholesterol (0.189), ...

Output:
Concise clinical explanation for cardiologist
```

**Safety Features**:
- ✅ No diagnostic claims
- ✅ No medical advice
- ✅ Feature-grounded only
- ✅ Clear, professional language

### 6. Pipeline Layer (`pipeline/`)

**Purpose**: Orchestrate end-to-end inference

**Components**:
- `InferencePipeline`: Unified interface
- Model loading/saving
- Batch and single-sample inference
- Explanation aggregation

**Workflow**:
1. Preprocess input
2. Run all 3 models in parallel
3. Generate explanations per model
4. Aggregate results
5. Call LLM for summary
6. Return structured JSON

### 7. Evaluation Layer (`evaluation/`)

**Metrics**:
- **AUROC**: Overall ranking quality
- **Precision@K**: Precision at top K% (e.g., 10%)
- **False Negative Rate**: Missed anomalies (critical for healthcare)
- **F1 Score**: Harmonic mean of precision/recall
- **Confusion Matrix**: TP, TN, FP, FN

**Model Comparison**:
- Rank models by metric
- Identify best model per metric
- Statistical comparison

## Data Flow

```
Raw CSV Data
    ↓
[Load] → DataFrame
    ↓
[Split] → Train (normal only) | Test (mixed)
    ↓
[Scale] → Normalized arrays
    ↓
[Train] → 3 Models (IF, SVM, AE)
    ↓
[Predict] → 3 Anomaly Scores
    ↓
[Explain] → SHAP / Reconstruction
    ↓
[LLM] → Clinical Text
    ↓
[Output] → JSON Response
```

## Configuration Management

**File**: `config/settings.py`

**Categories**:
1. **Random Seeds**: Reproducibility
2. **Model Hyperparameters**: Per-model tuning
3. **Thresholds**: Risk level boundaries
4. **SHAP Settings**: Sample size, top-k features
5. **API Settings**: HuggingFace model, tokens, timeout

**Design Pattern**: Centralized configuration class

## Security & Safety

1. **No Hardcoded Secrets**: API tokens via environment
2. **Input Validation**: Data type, range, shape checks
3. **Error Handling**: Try-catch blocks, fallbacks
4. **LLM Safety**: Structured prompts, no hallucinations
5. **Medical Disclaimer**: Not for diagnosis

## Performance Characteristics

| Model | Training Time | Inference Time | Memory | CPU/GPU |
|-------|--------------|----------------|---------|---------|
| Isolation Forest | ~1s (1k samples) | <10ms | Low | CPU |
| One-Class SVM | ~5s (1k samples) | <20ms | Medium | CPU |
| Autoencoder | ~30s (50 epochs) | <50ms | Medium | GPU optional |

**Scalability**:
- **Isolation Forest**: O(n log n)
- **One-Class SVM**: O(n²) to O(n³)
- **Autoencoder**: O(n) per epoch

## Extensibility

### Adding New Models
1. Inherit from `BaseAnomalyModel`
2. Implement `fit()`, `score_samples()`
3. Register in `InferencePipeline`

### Adding New Explainers
1. Create explainer class
2. Implement `explain_sample()`
3. Return feature importance dict

### Changing LLM
1. Update `Config.HF_MODEL`
2. Adjust prompt template if needed
3. Test output format

## Deployment Options

### 1. Local Execution
```bash
python main.py
```

### 2. REST API (Flask/FastAPI)
```python
@app.post("/predict")
def predict(data: PatientData):
    pipeline = load_pipeline()
    result = pipeline.predict_single_sample(data)
    return result
```

### 3. Batch Processing
```python
pipeline = InferencePipeline.load_pipeline(...)
results = pipeline.predict(X_batch)
```

### 4. Model Serving (TorchServe/ONNX)
- Export Autoencoder to ONNX
- Use sklearn serialization for IF/SVM
- Deploy with model server

## Testing Strategy

### Unit Tests
- Model fit/predict
- Preprocessor transform
- Explainer output format
- Pipeline integration

### Integration Tests
- End-to-end pipeline
- API connectivity (HuggingFace)
- File I/O (save/load models)

### Validation Tests
- Synthetic data (known anomalies)
- Edge cases (all zeros, missing values)
- Performance benchmarks

## Monitoring & Logging

**Recommended**:
1. Model prediction distributions
2. Anomaly score histograms
3. Explanation feature frequency
4. LLM API latency
5. Error rates per model

## Limitations & Considerations

1. **Unsupervised Learning**: Requires normal samples for training
2. **Threshold Selection**: Domain-specific tuning needed
3. **LLM Costs**: API calls may incur charges
4. **Medical Use**: Not FDA-approved, research only
5. **Data Quality**: Garbage in, garbage out

## Future Enhancements

1. **Ensemble Methods**: Weighted model combination
2. **Active Learning**: Feedback loop for model improvement
3. **Temporal Models**: Time-series anomaly detection
4. **Federated Learning**: Privacy-preserving training
5. **Custom LLM**: Fine-tuned medical language model

## References

- Liu, F.T., Ting, K.M. and Zhou, Z.H., 2008. Isolation forest. ICDM.
- Schölkopf, B., et al., 2001. Estimating the support of a high-dimensional distribution. Neural computation.
- Lundberg, S.M. and Lee, S.I., 2017. A unified approach to interpreting model predictions. NIPS.

---

**Version**: 1.0  
**Last Updated**: 2026  
**License**: Educational/Research Use
