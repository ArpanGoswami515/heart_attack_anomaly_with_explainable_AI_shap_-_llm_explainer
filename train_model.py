"""
Utility script to train and save the Isolation Forest model for deployment.
Run this before starting the distributed system for the first time.
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data.data_loader import HeartAttackDataLoader
from preprocessing.preprocess import DataPreprocessor
from models.isolation_forest import IsolationForest
from config.settings import Config


def train_and_save_model():
    """Train Isolation Forest model and save it along with preprocessor."""
    
    print("="*70)
    print("TRAINING AND SAVING ISOLATION FOREST MODEL FOR DISTRIBUTED SYSTEM")
    print("="*70)
    
    # Create directories
    Config.create_directories()
    
    # Step 1: Load data
    print("\n[1/4] Loading training data...")
    df = HeartAttackDataLoader.generate_sample_dataset(
        n_samples=2000,
        contamination=0.1,
        random_state=Config.RANDOM_SEED
    )
    
    X = df.drop(columns=['target']).values
    y = df['target'].values
    
    print(f"✓ Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"✓ Anomaly ratio: {np.mean(y):.2%}")
    
    # Step 2: Preprocess data
    print("\n[2/4] Preprocessing data...")
    preprocessor = DataPreprocessor(scaling_method="standard")
    X_scaled = preprocessor.fit_transform(X)
    
    # Get only normal samples for training (unsupervised)
    X_normal = X_scaled[y == 0]
    print(f"✓ Using {len(X_normal)} normal samples for training")
    
    # Step 3: Train model
    print("\n[3/4] Training Isolation Forest model...")
    print(f"Parameters: {Config.ISOLATION_FOREST_PARAMS}")
    
    model = IsolationForest(**Config.ISOLATION_FOREST_PARAMS)
    model.fit(X_normal)
    
    print("✓ Model training complete")
    
    # Step 4: Save model and preprocessor
    print("\n[4/4] Saving model and preprocessor...")
    
    model_path = Path(Config.MODELS_DIR) / "isolation_forest.pkl"
    preprocessor_path = Path(Config.MODELS_DIR) / "preprocessor.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to: {model_path}")
    
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"✓ Preprocessor saved to: {preprocessor_path}")
    
    # Verify saved files
    print("\n[Verification] Testing saved model...")
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    with open(preprocessor_path, 'rb') as f:
        loaded_preprocessor = pickle.load(f)
    
    # Test prediction
    test_sample = X_scaled[:1]
    prediction = loaded_model.predict(test_sample)
    score = loaded_model.score_samples(test_sample)
    
    print(f"✓ Test prediction: {prediction[0]} (score: {score[0]:.4f})")
    
    print("\n" + "="*70)
    print("✓ MODEL AND PREPROCESSOR READY FOR DEPLOYMENT")
    print("="*70)
    print("\nNext steps:")
    print("1. Set your HUGGINGFACE_API_TOKEN in .env file")
    print("2. Run: docker-compose up --build")
    print("3. Access dashboards:")
    print("   - Server: http://localhost:5000")
    print("   - Client 1: http://localhost:5001")
    print("   - Client 2: http://localhost:5002")
    print("   - Client 3: http://localhost:5003")


if __name__ == "__main__":
    try:
        train_and_save_model()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
