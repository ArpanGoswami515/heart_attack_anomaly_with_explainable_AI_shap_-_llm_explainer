#!/usr/bin/env python3
"""
Setup and verification script for Heart Attack Anomaly Detection system.
Run this script to verify your installation and setup.
"""

import sys
import subprocess
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*60)
    print(text)
    print("="*60)


def check_python_version():
    """Check if Python version is compatible."""
    print_header("CHECKING PYTHON VERSION")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print("✓ Python version is compatible")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    print_header("CHECKING DEPENDENCIES")
    
    required_packages = [
        "numpy", "pandas", "sklearn", "torch", 
        "shap", "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies are installed")
    return True


def check_environment():
    """Check environment variables."""
    print_header("CHECKING ENVIRONMENT")
    
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
    
    if hf_token:
        print("✓ HUGGINGFACE_API_TOKEN is set")
        print("  LLM explanations will be enabled")
    else:
        print("⚠ HUGGINGFACE_API_TOKEN is not set")
        print("  LLM explanations will be disabled")
        print("  To enable, set the environment variable or create .env file")
    
    return True


def verify_structure():
    """Verify project structure."""
    print_header("VERIFYING PROJECT STRUCTURE")
    
    required_dirs = [
        "config",
        "data",
        "preprocessing",
        "models",
        "explainability",
        "llm",
        "evaluation",
        "pipeline"
    ]
    
    all_exist = True
    
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"✓ {dir_name}/ exists")
        else:
            print(f"❌ {dir_name}/ is missing")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Project structure is incomplete")
        return False
    
    print("\n✓ Project structure is complete")
    return True


def create_directories():
    """Create necessary runtime directories."""
    print_header("CREATING RUNTIME DIRECTORIES")
    
    dirs_to_create = ["data", "saved_models"]
    
    for dir_name in dirs_to_create:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ {dir_name}/ ready")
    
    return True


def run_quick_test():
    """Run a quick import test."""
    print_header("RUNNING QUICK TEST")
    
    try:
        from config.settings import Config
        print("✓ Config module loaded")
        
        from models.isolation_forest import IsolationForest
        print("✓ IsolationForest model loaded")
        
        from models.one_class_svm import OneClassSVM
        print("✓ OneClassSVM model loaded")
        
        from models.autoencoder import Autoencoder
        print("✓ Autoencoder model loaded")
        
        from pipeline.inference_pipeline import InferencePipeline
        print("✓ InferencePipeline loaded")
        
        print("\n✓ All modules imported successfully")
        return True
    
    except Exception as e:
        print(f"\n❌ Import test failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print_header("NEXT STEPS")
    
    print("""
1. (Optional) Set HuggingFace API token:
   
   Windows:
   set HUGGINGFACE_API_TOKEN=your_token_here
   
   Linux/Mac:
   export HUGGINGFACE_API_TOKEN=your_token_here
   
   Or create .env file with:
   HUGGINGFACE_API_TOKEN=your_token_here

2. Run the main pipeline:
   python main.py

3. The system will:
   - Generate synthetic heart attack data (or use your dataset)
   - Train 3 anomaly detection models
   - Evaluate performance
   - Run inference demo
   - Save trained models

4. Customize configuration:
   Edit config/settings.py to adjust:
   - Model hyperparameters
   - Risk thresholds
   - LLM settings

5. Use your own data:
   - Place CSV file in data/ directory
   - Update data path in main.py (set use_synthetic=False)
   - Ensure target column is named "target" (0=normal, 1=anomaly)
""")


def main():
    """Main setup verification function."""
    print_header("HEART ATTACK ANOMALY DETECTION - SETUP VERIFICATION")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment", check_environment),
        ("Project Structure", verify_structure),
        ("Runtime Directories", create_directories),
        ("Module Imports", run_quick_test)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ {check_name} check failed with error: {e}")
            results.append((check_name, False))
    
    # Summary
    print_header("SETUP VERIFICATION SUMMARY")
    
    all_passed = True
    for check_name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n✅ All checks passed! System is ready to use.")
        print_next_steps()
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("Refer to README.md for detailed setup instructions.")


if __name__ == "__main__":
    main()
