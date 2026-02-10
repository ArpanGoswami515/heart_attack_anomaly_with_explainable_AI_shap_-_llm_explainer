"""
Test script to demonstrate enhanced LLM explanations with actual feature values.
"""

import os
from dotenv import load_dotenv
from llm.hf_explainer import HuggingFaceExplainer

# Load environment variables
load_dotenv()

def test_enhanced_explanation():
    """Test LLM explanation with actual feature values."""
    
    print("=" * 80)
    print("Testing Enhanced LLM Explanations with Actual Feature Values")
    print("=" * 80)
    
    # Initialize explainer
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not api_token:
        print("❌ Error: HUGGINGFACE_API_TOKEN not found in .env file")
        return
    
    print("\n1. Initializing HuggingFace explainer...")
    explainer = HuggingFaceExplainer(
        api_token=api_token,
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        max_tokens=300,
        temperature=0.7
    )
    print("✓ Explainer initialized")
    
    # Test case: High-risk patient
    print("\n2. Testing High-Risk Patient...")
    print("-" * 80)
    
    # Example feature importances (which features contributed most to anomaly)
    feature_importances = {
        "Age": 0.156,
        "Cholesterol": 0.234,
        "Max Heart Rate": 0.189,
        "ST Depression": 0.278,
        "Chest Pain Type": 0.143
    }
    
    # Example actual feature values
    feature_values = {
        "Age": 63.0,
        "Cholesterol": 295.0,
        "Max Heart Rate": 142.0,
        "ST Depression": 3.2,
        "Chest Pain Type": 3.0
    }
    
    print("\nPatient Data:")
    print("  Feature Values:")
    for feature, value in feature_values.items():
        print(f"    - {feature}: {value}")
    
    print("\n  Feature Importances (contribution to anomaly score):")
    for feature, importance in feature_importances.items():
        print(f"    - {feature}: {importance:.3f} ({importance*100:.1f}%)")
    
    print("\n  Risk Assessment:")
    print(f"    - Risk Level: High")
    print(f"    - Anomaly Score: 0.847")
    print(f"    - Model: IsolationForest")
    
    print("\n3. Generating LLM Explanation...")
    print("-" * 80)
    
    result = explainer.explain(
        risk_level="High",
        anomaly_score=0.847,
        feature_importances=feature_importances,
        feature_values=feature_values,
        model_name="IsolationForest"
    )
    
    print("\n✓ LLM Explanation Generated:")
    print("\n" + "="*80)
    print(result["explanation"])
    print("="*80)
    
    # Test case: Low-risk patient
    print("\n\n4. Testing Low-Risk Patient...")
    print("-" * 80)
    
    feature_importances_low = {
        "Age": 0.089,
        "Cholesterol": 0.076,
        "Max Heart Rate": 0.134,
        "Resting Blood Pressure": 0.098,
        "Resting ECG": 0.082
    }
    
    feature_values_low = {
        "Age": 42.0,
        "Cholesterol": 185.0,
        "Max Heart Rate": 165.0,
        "Resting Blood Pressure": 118.0,
        "Resting ECG": 0.0
    }
    
    print("\nPatient Data:")
    print("  Feature Values:")
    for feature, value in feature_values_low.items():
        print(f"    - {feature}: {value}")
    
    print("\n  Feature Importances:")
    for feature, importance in feature_importances_low.items():
        print(f"    - {feature}: {importance:.3f} ({importance*100:.1f}%)")
    
    print("\n  Risk Assessment:")
    print(f"    - Risk Level: Low")
    print(f"    - Anomaly Score: 0.123")
    print(f"    - Model: OneClassSVM")
    
    print("\n5. Generating LLM Explanation...")
    print("-" * 80)
    
    result_low = explainer.explain(
        risk_level="Low",
        anomaly_score=0.123,
        feature_importances=feature_importances_low,
        feature_values=feature_values_low,
        model_name="OneClassSVM"
    )
    
    print("\n✓ LLM Explanation Generated:")
    print("\n" + "="*80)
    print(result_low["explanation"])
    print("="*80)
    
    # Test case: Medium-risk patient
    print("\n\n6. Testing Medium-Risk Patient...")
    print("-" * 80)
    
    feature_importances_med = {
        "Age": 0.145,
        "Cholesterol": 0.167,
        "Max Heart Rate": 0.112,
        "Old Peak": 0.198,
        "Number of Major Vessels": 0.154
    }
    
    feature_values_med = {
        "Age": 55.0,
        "Cholesterol": 240.0,
        "Max Heart Rate": 145.0,
        "Old Peak": 1.8,
        "Number of Major Vessels": 1.0
    }
    
    print("\nPatient Data:")
    print("  Feature Values:")
    for feature, value in feature_values_med.items():
        print(f"    - {feature}: {value}")
    
    print("\n  Feature Importances:")
    for feature, importance in feature_importances_med.items():
        print(f"    - {feature}: {importance:.3f} ({importance*100:.1f}%)")
    
    print("\n  Risk Assessment:")
    print(f"    - Risk Level: Medium")
    print(f"    - Anomaly Score: 0.512")
    print(f"    - Model: Autoencoder")
    
    print("\n7. Generating LLM Explanation...")
    print("-" * 80)
    
    result_med = explainer.explain(
        risk_level="Medium",
        anomaly_score=0.512,
        feature_importances=feature_importances_med,
        feature_values=feature_values_med,
        model_name="Autoencoder"
    )
    
    print("\n✓ LLM Explanation Generated:")
    print("\n" + "="*80)
    print(result_med["explanation"])
    print("="*80)
    
    print("\n\n" + "="*80)
    print("✓ All Tests Complete!")
    print("="*80)
    print("\nKey Improvements:")
    print("  ✓ LLM now receives actual feature values (age, cholesterol, etc.)")
    print("  ✓ Explanations reference specific patient measurements")
    print("  ✓ More contextual and meaningful clinical insights")
    print("  ✓ Importance scores show which features contributed most to anomaly")
    print("="*80)

if __name__ == "__main__":
    test_enhanced_explanation()
