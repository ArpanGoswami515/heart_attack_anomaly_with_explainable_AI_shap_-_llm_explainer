"""
Utility to update config/settings.py with optimized hyperparameters from tuning.
"""

import json
import os
import re
from typing import Dict, Any
from datetime import datetime


def load_tuning_results(filepath: str) -> Dict[str, Any]:
    """Load tuning results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_param_value(value: Any) -> str:
    """Format parameter value for Python code."""
    if isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, list):
        return str(value)
    elif isinstance(value, bool):
        return str(value)
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        return repr(value)


def update_config_file(
    tuning_results: Dict[str, Any],
    config_path: str = "config/settings.py",
    backup: bool = True
) -> None:
    """
    Update config/settings.py with optimized parameters.
    
    Args:
        tuning_results: Dict with tuning results
        config_path: Path to config file
        backup: Whether to create backup of original config
    """
    print("="*70)
    print("UPDATING CONFIG FILE WITH OPTIMIZED PARAMETERS")
    print("="*70)
    
    # Read current config file
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Create backup
    if backup:
        backup_path = config_path.replace('.py', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.py')
        with open(backup_path, 'w') as f:
            f.write(config_content)
        print(f"✓ Backup created: {backup_path}")
    
    # Update Isolation Forest parameters
    if "isolation_forest" in tuning_results:
        params = tuning_results["isolation_forest"]["best_params"]
        score = tuning_results["isolation_forest"]["best_score"]
        
        new_params = "ISOLATION_FOREST_PARAMS: Dict[str, Any] = {\n"
        for key, value in params.items():
            new_params += f'        "{key}": {format_param_value(value)},\n'
        new_params += "    }"
        
        pattern = r'ISOLATION_FOREST_PARAMS: Dict\[str, Any\] = \{[^}]+\}'
        config_content = re.sub(
            pattern,
            new_params,
            config_content,
            flags=re.DOTALL
        )
        print(f"✓ Updated Isolation Forest (score: {score:.4f})")
    
    # Update One-Class SVM parameters
    if "one_class_svm" in tuning_results:
        params = tuning_results["one_class_svm"]["best_params"]
        score = tuning_results["one_class_svm"]["best_score"]
        
        new_params = "ONE_CLASS_SVM_PARAMS: Dict[str, Any] = {\n"
        for key, value in params.items():
            new_params += f'        "{key}": {format_param_value(value)},\n'
        new_params += "    }"
        
        pattern = r'ONE_CLASS_SVM_PARAMS: Dict\[str, Any\] = \{[^}]+\}'
        config_content = re.sub(
            pattern,
            new_params,
            config_content,
            flags=re.DOTALL
        )
        print(f"✓ Updated One-Class SVM (score: {score:.4f})")
    
    # Update Autoencoder parameters
    if "autoencoder" in tuning_results:
        params = tuning_results["autoencoder"]["best_params"]
        score = tuning_results["autoencoder"]["best_score"]
        
        new_params = "AUTOENCODER_PARAMS: Dict[str, Any] = {\n"
        for key, value in params.items():
            new_params += f'        "{key}": {format_param_value(value)},\n'
        new_params += "    }"
        
        pattern = r'AUTOENCODER_PARAMS: Dict\[str, Any\] = \{[^}]+\}'
        config_content = re.sub(
            pattern,
            new_params,
            config_content,
            flags=re.DOTALL
        )
        print(f"✓ Updated Autoencoder (score: {score:.4f})")
    
    # Write updated config
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Config file updated: {config_path}")


def main():
    """Main execution."""
    import sys
    
    print("="*70)
    print("CONFIG UPDATER - APPLY TUNED HYPERPARAMETERS")
    print("="*70)
    
    # Find latest tuning results file
    tuning_dir = "tuning_results"
    
    if not os.path.exists(tuning_dir):
        print(f"✗ Error: Directory '{tuning_dir}' not found")
        print("  Run tune_hyperparameters.py first to generate tuning results")
        sys.exit(1)
    
    # Get all tuning result files
    result_files = [
        f for f in os.listdir(tuning_dir)
        if f.startswith("tuning_results_") and f.endswith(".json")
    ]
    
    if not result_files:
        print(f"✗ Error: No tuning results found in '{tuning_dir}'")
        print("  Run tune_hyperparameters.py first to generate tuning results")
        sys.exit(1)
    
    # Sort by timestamp and get latest
    result_files.sort(reverse=True)
    latest_file = os.path.join(tuning_dir, result_files[0])
    
    print(f"\nUsing tuning results from: {latest_file}")
    
    # Load results
    tuning_results = load_tuning_results(latest_file)
    
    print("\nFound optimized parameters for:")
    for model_name in tuning_results:
        score = tuning_results[model_name]["best_score"]
        print(f"  - {model_name}: {score:.4f}")
    
    # Confirm before updating
    response = input("\nUpdate config/settings.py with these parameters? (y/n): ")
    
    if response.lower() == 'y':
        update_config_file(tuning_results)
        print("\n" + "="*70)
        print("✓ CONFIG FILE UPDATED SUCCESSFULLY")
        print("="*70)
        print("\nNext steps:")
        print("1. Review the updated config/settings.py")
        print("2. Run main.py to train models with optimized parameters")
    else:
        print("\nUpdate cancelled.")


if __name__ == "__main__":
    main()
