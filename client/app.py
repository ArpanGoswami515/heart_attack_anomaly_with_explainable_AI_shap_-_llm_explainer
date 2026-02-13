"""
Client application for heart attack anomaly detection.
Performs local predictions and sends results to main server.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json
from datetime import datetime

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

from models.isolation_forest import IsolationForest
from preprocessing.preprocess import DataPreprocessor
from llm.hf_explainer import HuggingFaceExplainer
from config.settings import Config
from explainability.shap_explainer import SHAPExplainer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'heart-attack-client-secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Server connection settings
SERVER_URL = os.environ.get('SERVER_URL', 'http://localhost:5000')
CLIENT_ID = os.environ.get('CLIENT_ID', 'Client-1')

# Client state
prediction_history = []
server_socket = None

# Model and data
model = None
preprocessor = None
llm_explainer = None
shap_explainer = None
feature_names = None
csv_data = None
_initialized = False


def load_model_and_data():
    """Load model, preprocessor, CSV data, and initialize explainers."""
    global model, preprocessor, feature_names, llm_explainer, shap_explainer, csv_data, _initialized
    
    # Skip if already initialized
    if _initialized:
        print(f"✓ {CLIENT_ID} already initialized")
        return
    
    print("="*60)
    print(f"INITIALIZING {CLIENT_ID}")
    print("="*60)
    
    # Feature names for heart attack dataset
    feature_names = [
        "Age", "Sex", "Chest pain type", "BP", "Cholesterol",
        "FBS over 120", "EKG results", "Max HR", "Exercise angina",
        "ST depression", "Slope of ST", "Number of vessels fluro",
        "Thallium"
    ]
    print(f"✓ Feature names set: {len(feature_names)} features")
    
    # Load model
    model_path = Path(__file__).parent.parent / "saved_models" / "isolation_forest.pkl"
    if model_path.exists():
        print("✓ Loading Isolation Forest model...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        print("✗ Model not found! Please run train_model.py first.")
        return False
    
    # Load preprocessor
    preprocessor_path = Path(__file__).parent.parent / "saved_models" / "preprocessor.pkl"
    if preprocessor_path.exists():
        print("✓ Loading preprocessor...")
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
    else:
        print("✗ Preprocessor not found! Please run train_model.py first.")
        return False
    
    # Load CSV data
    csv_path = Path(__file__).parent.parent / "data" / "heart_attack_data.csv"
    print(f"Looking for CSV at: {csv_path}")
    print(f"CSV exists: {csv_path.exists()}")
    
    if csv_path.exists():
        print("✓ Loading CSV data...")
        csv_data = pd.read_csv(csv_path)
        print(f"CSV shape: {csv_data.shape}")
        print(f"CSV columns: {csv_data.columns.tolist()}")
        
        # Handle "Heart Disease" column if present
        if "Heart Disease" in csv_data.columns:
            csv_data["target"] = csv_data["Heart Disease"].map({
                "Presence": 1, 
                "Absence": 0
            })
            csv_data = csv_data.drop(columns=["Heart Disease"])
        print(f"✓ Loaded {len(csv_data)} patient records")
    else:
        print("⚠ CSV not found at expected location, generating synthetic data")
        from data.data_loader import HeartAttackDataLoader
        csv_data = HeartAttackDataLoader.generate_sample_dataset(
            n_samples=500,
            contamination=0.15,
            random_state=42
        )
        print(f"✓ Generated {len(csv_data)} synthetic patient records")
    
    print(f"Final csv_data type: {type(csv_data)}")
    print(f"Final csv_data is None: {csv_data is None}")
    
    # Initialize LLM explainer
    try:
        hf_token = Config.get_hf_api_token()
        llm_explainer = HuggingFaceExplainer(
            api_token=hf_token,
            model_name=Config.HF_MODEL,
            max_tokens=Config.HF_MAX_TOKENS,
            temperature=Config.HF_TEMPERATURE
        )
        print("✓ LLM explainer initialized")
    except Exception as e:
        print(f"⚠ LLM explainer not available: {e}")
        llm_explainer = None
    
    # Initialize SHAP explainer
    try:
        X_bg = csv_data.drop(columns=['target']).values[:50]
        X_bg_scaled = preprocessor.transform(X_bg)
        shap_explainer = SHAPExplainer(
            model=model.model, 
            background_data=X_bg_scaled,
            feature_names=feature_names
        )
        print("✓ SHAP explainer initialized")
    except Exception as e:
        print(f"⚠ SHAP explainer not available: {e}")
        shap_explainer = None
    
    print("="*60)
    print(f"✓ {CLIENT_ID} READY")
    print("="*60)
    
    _initialized = True
    return True
    

def ensure_loaded():
    """Lazy loader - ensures data is loaded on first access."""
    global _initialized
    if not _initialized:
        print("[ensure_loaded] Initializing on first request...")
        load_model_and_data()
        print(f"[ensure_loaded] Initialization complete. csv_data is None: {csv_data is None}")


@app.route('/api/get_random_sample')
def get_random_sample():
    """Get a random patient data sample from CSV."""
    global csv_data, feature_names
    
    # Ensure data is loaded
    ensure_loaded()
    
    print(f"[get_random_sample] csv_data is None: {csv_data is None}")
    print(f"[get_random_sample] feature_names is None: {feature_names is None}")
    
    if csv_data is None:
        return jsonify({'error': 'No data available. Server may still be initializing.'}), 500
    
    if feature_names is None:
        return jsonify({'error': 'Feature names not initialized'}), 500
    
    try:
        # Get random sample
        sample = csv_data.sample(n=1).iloc[0]
        print(f"[get_random_sample] Sample shape: {len(sample)}")
        print(f"[get_random_sample] Sample columns: {sample.index.tolist()}")
        
        # Map CSV columns directly to feature names
        patient_data = {}
        for fname in feature_names:
            if fname in sample.index:
                patient_data[fname] = float(sample[fname])
            else:
                # Default value if column not found
                patient_data[fname] = 0.0
                print(f"Warning: Column '{fname}' not found in CSV, using 0.0")
        
        print(f"[get_random_sample] Returning {len(patient_data)} features")
        return jsonify({
            'success': True,
            'data': patient_data
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[get_random_sample] Error: {error_details}")
        return jsonify({'error': f'Failed to load sample: {str(e)}'}), 500


@socketio.on('predict_locally')
def handle_predict_locally(data):
    """
    Perform local prediction using client's model, then send to server.
    """
    global model, preprocessor, feature_names, llm_explainer, shap_explainer
    
    # Ensure data is loaded
    ensure_loaded()
    
    print(f"[predict_locally] Received prediction request")
    print(f"[predict_locally] feature_names is None: {feature_names is None}")
    print(f"[predict_locally] model is None: {model is None}")
    print(f"[predict_locally] preprocessor is None: {preprocessor is None}")
    
    try:
        patient_data = data.get('patient_data', {})
        print(f"[predict_locally] Patient data keys: {list(patient_data.keys())}")
        
        # Check if prerequisites are loaded
        if feature_names is None:
            raise ValueError("Feature names not initialized")
        if model is None:
            raise ValueError("Model not loaded")
        if preprocessor is None:
            raise ValueError("Preprocessor not loaded")
        
        # Step 1: Preprocess data
        X_raw = np.array([[patient_data.get(f, 0) for f in feature_names]])
        print(f"[predict_locally] X_raw shape: {X_raw.shape}")
        X_scaled = preprocessor.transform(X_raw)
        print(f"[predict_locally] X_scaled shape: {X_scaled.shape}")
        
        # Step 2: Predict using Isolation Forest
        print(f"[{CLIENT_ID}] Running local prediction...")
        prediction = model.predict(X_scaled)[0]  # -1 = anomaly, 1 = normal
        anomaly_score = model.score_samples(X_scaled)[0]  # 0-1, higher = more anomalous
        risk_level = Config.get_risk_level(anomaly_score)
        
        # Step 3: Get feature importances using SHAP
        feature_importances = {}
        if shap_explainer:
            try:
                explanations = shap_explainer.explain_sample(X_scaled, top_k=5)
                if explanations:
                    feature_importances = explanations[0]['feature_importance']
            except Exception as e:
                print(f"SHAP error: {e}")
                # Fallback: assign equal importance to top features
                for fname in feature_names[:5]:
                    feature_importances[fname] = 0.2
        else:
            for fname in feature_names[:5]:
                feature_importances[fname] = 0.2
        
        # Step 4: Get LLM explanation
        llm_explanation = None
        if llm_explainer:
            try:
                print(f"[{CLIENT_ID}] Generating LLM explanation...")
                result = llm_explainer.explain(
                    risk_level=risk_level,
                    anomaly_score=anomaly_score,
                    feature_importances=feature_importances,
                    feature_values=patient_data,
                    model_name="Isolation Forest"
                )
                llm_explanation = result.get('explanation', f"Risk Level: {risk_level}")
            except Exception as e:
                print(f"LLM error: {e}")
                llm_explanation = f"Clinical explanation unavailable. Risk Level: {risk_level}"
        else:
            llm_explanation = f"Patient shows {risk_level.lower()} risk pattern based on clinical features."
        
        # Build local result
        local_result = {
            "prediction": "Anomaly" if prediction == -1 else "Normal",
            "anomaly_score": float(anomaly_score),
            "risk_level": risk_level,
            "feature_importances": feature_importances,
            "feature_values": patient_data,
            "llm_explanation": llm_explanation,
            "timestamp": datetime.now().isoformat(),
            "model": "Isolation Forest",
            "client_id": CLIENT_ID,
            "source": "local"
        }
        
        print(f"[{CLIENT_ID}] ✓ Local prediction: {risk_level} risk")
        
        # Send local result to dashboard immediately
        emit('prediction_received', local_result)
        
        # Step 5: Send to main server for aggregation
        send_to_main_server(local_result)
        
    except Exception as e:
        error_msg = str(e)
        print(f"[{CLIENT_ID}] ✗ Local prediction error: {error_msg}")
        emit('prediction_error', {'error': error_msg})


def send_to_main_server(result):
    """Send prediction result to main server for monitoring."""
    import socketio as sio_client
    
    try:
        # Clean SERVER_URL (remove any whitespace)
        server_url = SERVER_URL.strip()
        
        client = sio_client.Client()
        client.connect(server_url, wait_timeout=5)
        
        print(f"[{CLIENT_ID}] Sending result to main server...")
        client.emit('predict', {
            'data': result['feature_values'],
            'client_id': CLIENT_ID,
            'local_prediction': {
                'prediction': result['prediction'],
                'anomaly_score': result['anomaly_score'],
                'risk_level': result['risk_level']
            }
        })
        
        import time
        time.sleep(0.5)  # Give server time to process
        client.disconnect()
        print(f"[{CLIENT_ID}] ✓ Sent to main server")
        
    except Exception as e:
        print(f"[{CLIENT_ID}] ⚠ Could not send to server: {e}")


@app.route('/')
def index():
    """Render client dashboard."""
    return render_template('client_dashboard.html', 
                          client_id=CLIENT_ID,
                          server_url=SERVER_URL)


@app.route('/api/client_info')
def client_info():
    """Get client information."""
    return {
        'client_id': CLIENT_ID,
        'server_url': SERVER_URL,
        'predictions_count': len(prediction_history)
    }


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"Dashboard connected")
    emit('client_info', {
        'client_id': CLIENT_ID,
        'server_url': SERVER_URL
    })


@socketio.on('send_to_server')
def handle_send_to_server(data):
    """
    Handle request to send data to server.
    This creates a connection to the main server and sends the prediction request.
    """
    import socketio as sio_client
    
    try:
        # Create client connection to server
        client = sio_client.Client()
        
        # Variable to store result
        result_received = {'data': None, 'received': False}
        
        @client.on('prediction_result')
        def on_result(result):
            """Handle prediction result from server."""
            result_received['data'] = result
            result_received['received'] = True
        
        # Connect to server
        print(f"Connecting to server at {SERVER_URL}...")
        client.connect(SERVER_URL)
        
        # Send prediction request
        print(f"Sending prediction request for client {CLIENT_ID}")
        client.emit('predict', {
            'data': data['patient_data'],
            'client_id': CLIENT_ID
        })
        
        # Wait for response (with timeout)
        import time
        timeout = 10
        elapsed = 0
        while not result_received['received'] and elapsed < timeout:
            time.sleep(0.1)
            elapsed += 0.1
        
        client.disconnect()
        
        if result_received['received']:
            result = result_received['data']
            
            # Store in history
            prediction_history.append(result)
            
            # Send result back to dashboard
            emit('prediction_received', result)
            print(f"✓ Prediction received: {result.get('risk_level', 'Unknown')} risk")
        else:
            emit('prediction_error', {'error': 'Timeout waiting for server response'})
            print("✗ Timeout waiting for prediction")
            
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Error communicating with server: {error_msg}")
        emit('prediction_error', {'error': error_msg})


@socketio.on('get_history')
def handle_get_history():
    """Send prediction history to dashboard."""
    emit('history_data', {
        'predictions': prediction_history[-50:]  # Last 50 predictions
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"\n🚀 Starting {CLIENT_ID} on port {port}...")
    print(f"📊 Dashboard: http://localhost:{port}")
    print(f"🔗 Server: {SERVER_URL}\n")
    
    # Load model, preprocessor, and CSV data
    print("⏳ Loading models and data...")
    load_model_and_data()
    print("\n✅ Server ready!\n")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
