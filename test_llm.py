"""
Test script for HuggingFace LLM explainer.
Run this to verify your API connection and token.
"""

import os
from dotenv import load_dotenv
from llm.hf_explainer import HuggingFaceExplainer
from config.settings import Config

# Load environment variables
load_dotenv()


def test_model(api_token, model_name):
    """Test a specific model."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print('='*60)
    
    explainer = HuggingFaceExplainer(
        api_token=api_token,
        model_name=model_name,
        max_tokens=150,
        temperature=0.7
    )
    
    # Validate connection
    validation = explainer.validate_api_connection()
    print(f"Status: {validation['message']}")
    
    if not validation['connected']:
        return False
    
    # Test explanation
    print("\nGenerating test explanation...")
    test_data = {
        "risk_level": "High",
        "anomaly_score": 0.85,
        "feature_importances": {
            "Max HR": 0.234,
            "ST depression": 0.189,
            "Age": 0.156
        },
        "feature_values": {
            "Max HR": 142.0,
            "ST depression": 3.2,
            "Age": 63.0
        },
        "model_name": "Test Model"
    }
    
    result = explainer.explain(
        risk_level=test_data['risk_level'],
        anomaly_score=test_data['anomaly_score'],
        feature_importances=test_data['feature_importances'],
        feature_values=test_data['feature_values'],
        model_name=test_data['model_name']
    )
    
    print(f"\nExplanation:\n{result['explanation']}")
    
    success = not result['explanation'].startswith("Error")
    if success:
        print("\n✓ Model works!")
    else:
        print("\n✗ Model failed")
    
    return success


def test_llm_connection():
    """Test LLM API connection and generate a sample explanation."""
    
    print("="*60)
    print("HUGGINGFACE LLM CONNECTION TEST")
    print("="*60)
    
    # Get API token
    try:
        api_token = Config.get_hf_api_token()
        print(f"✓ API token found: {api_token[:10]}...")
    except ValueError as e:
        print(f"✗ Error: {e}")
        print("\nPlease set HUGGINGFACE_API_TOKEN in your .env file")
        print("Get your token from: https://huggingface.co/settings/tokens")
        return
    
    # Test multiple models (serverless inference compatible)
    models_to_test = [
        "meta-llama/Meta-Llama-3-8B-Instruct",     # Chat completion API
        "microsoft/Phi-3-mini-4k-instruct",        # Fast and reliable
        "google/gemma-7b-it",                      # Alternative option
    ]
    
    print("\nTesting available models...")
    
    working_models = []
    for model in models_to_test:
        try:
            if test_model(api_token, model):
                working_models.append(model)
                break  # Stop after first working model
        except Exception as e:
            print(f"Error testing {model}: {e}")
            continue
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if working_models:
        print(f"\n✓ Working model found: {working_models[0]}")
        print(f"\nUpdate config/settings.py to use:")
        print(f'    HF_MODEL: str = "{working_models[0]}"')
    else:
        print("\n✗ No working models found")
        print("\nTroubleshooting:")
        print("  1. Check your API token is valid")
        print("  2. Verify internet connection")
        print("  3. Check HuggingFace status: https://status.huggingface.co/")
        print("  4. Try generating a new API token")
        print("  5. Some models may require accepting terms on HuggingFace website")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    test_llm_connection()
