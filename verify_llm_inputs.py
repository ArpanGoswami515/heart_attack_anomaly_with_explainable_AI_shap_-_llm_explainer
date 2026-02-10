"""
Quick verification script to test LLM input format.
Run this after main.py to verify feature values are being passed correctly.
"""

import json
import os
from config.settings import Config

def verify_llm_inputs():
    """Verify LLM inputs from saved inference results."""
    
    print("="*70)
    print("LLM INPUT FORMAT VERIFICATION")
    print("="*70)
    
    results_path = os.path.join(Config.MODELS_DIR, "sample_inference_results.json")
    
    if not os.path.exists(results_path):
        print(f"\n✗ Results file not found: {results_path}")
        print("  Run 'python main.py' first to generate inference results")
        return
    
    print(f"\n✓ Loading results from: {results_path}\n")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Check each model
    models = [k for k in results.keys() if k != "llm_summary"]
    
    print(f"Found {len(models)} models: {', '.join(models)}\n")
    print("="*70)
    
    for model_name in models:
        result = results[model_name]
        
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        print("-"*70)
        
        # Check if explanation exists
        if "explanation" not in result:
            print("  ✗ No explanation field found")
            continue
        
        explanation = result["explanation"]
        
        # Check feature importance
        if "feature_importance" not in explanation:
            print("  ✗ No feature_importance found in explanation")
            continue
        
        feature_importance = explanation["feature_importance"]
        print(f"  ✓ Feature importance: {len(feature_importance)} features")
        
        # Check LLM explanation
        if "llm_explanation" in result:
            llm_exp = result["llm_explanation"]
            print(f"  ✓ LLM explanation exists")
            
            # Check if explanation text mentions actual values
            exp_text = llm_exp.get("explanation", "")
            
            # Look for numbers in the explanation (indicating actual values)
            import re
            numbers = re.findall(r'\d+\.?\d*', exp_text)
            
            if len(numbers) > 3:  # Should have several numbers if values are included
                print(f"  ✓ Explanation includes numerical values ({len(numbers)} numbers found)")
            else:
                print(f"  ⚠ Explanation may not include actual feature values (only {len(numbers)} numbers)")
            
            # Show first 3 feature importances to verify
            print(f"\n  Top 3 important features:")
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:3]):
                print(f"    {i+1}. {feature}: importance={importance:.3f}")
            
            # Show snippet of LLM explanation
            exp_snippet = exp_text[:200] + "..." if len(exp_text) > 200 else exp_text
            print(f"\n  LLM explanation (first 200 chars):")
            print(f"  \"{exp_snippet}\"")
            
        elif "llm_explanation_error" in result:
            print(f"  ✗ LLM explanation failed: {result['llm_explanation_error']}")
        else:
            print(f"  ✗ No LLM explanation found")
    
    # Check llm_summary
    if "llm_summary" in results:
        print(f"\n\nLLM SUMMARY (COMPARATIVE):")
        print("-"*70)
        summary = results["llm_summary"]
        
        if isinstance(summary, str):
            # Look for numbers
            import re
            numbers = re.findall(r'\d+\.?\d*', summary)
            
            if len(numbers) > 3:
                print(f"✓ Summary includes numerical values ({len(numbers)} numbers found)")
            else:
                print(f"⚠ Summary may not include actual feature values (only {len(numbers)} numbers)")
            
            # Show snippet
            summary_snippet = summary[:300] + "..." if len(summary) > 300 else summary
            print(f"\nSummary text (first 300 chars):")
            print(f"\"{summary_snippet}\"")
        else:
            print(f"⚠ Unexpected summary format: {type(summary)}")
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    
    # Summary
    print("\nSummary:")
    print(f"  - Models checked: {len(models)}")
    models_with_llm = sum(1 for m in models if "llm_explanation" in results.get(m, {}))
    print(f"  - Models with LLM explanations: {models_with_llm}/{len(models)}")
    
    if models_with_llm == len(models):
        print("\n✓ All models have LLM explanations!")
        print("✓ Check above to verify actual feature values are included")
    else:
        print(f"\n⚠ {len(models) - models_with_llm} model(s) missing LLM explanations")
        print("  Review debug output from main.py to diagnose")
    
    print("\nNext steps:")
    print("  1. Review the LLM explanation snippets above")
    print("  2. Verify they mention specific patient values (e.g., 'Age: 63', 'Cholesterol: 295')")
    print("  3. If values are missing, check the debug output when running main.py")
    print("="*70)


if __name__ == "__main__":
    verify_llm_inputs()
