"""
Server application for heart attack anomaly detection.
Receives data from clients, runs predictions, and returns results.
"""

import os
import sys
import json
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from datetime import datetime
import pickle
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

from models.isolation_forest import IsolationForest
from preprocessing.preprocess import DataPreprocessor
from llm.hf_explainer import HuggingFaceExplainer
from config.settings import Config
from explainability.shap_explainer import SHAPExplainer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'heart-attack-anomaly-detection-secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables for model and preprocessor
model = None
preprocessor = None
llm_explainer = None
shap_explainer = None
feature_names = None
prediction_history = []

# Feature names for heart attack dataset
FEATURE_NAMES = [
    "Age", "Sex", "Chest pain type", "BP", "Cholesterol",
    "FBS over 120", "EKG results", "Max HR", "Exercise angina",
    "ST depression", "Slope of ST", "Number of vessels fluro",
    "Thallium"
]


def load_model_and_preprocessor():
    """Load or train the model and preprocessor."""
    global model, preprocessor, feature_names, llm_explainer, shap_explainer
    
    print("="*60)
    print("INITIALIZING SERVER MODELS")
    print("="*60)
    
    # Try to load feature names
    feature_names = FEATURE_NAMES
    
    model_path = Path(__file__).parent.parent / "saved_models" / "isolation_forest.pkl"
    preprocessor_path = Path(__file__).parent.parent / "saved_models" / "preprocessor.pkl"
    
    # Load or create preprocessor
    if preprocessor_path.exists():
        print("✓ Loading preprocessor from disk...")
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
    else:
        print("✓ Creating new preprocessor...")
        preprocessor = DataPreprocessor(scaling_method="standard")
        # We'll fit it with dummy data or real data if available
        from data.data_loader import HeartAttackDataLoader
        df = HeartAttackDataLoader.generate_sample_dataset(
            n_samples=1000,
            contamination=0.1,
            random_state=Config.RANDOM_SEED
        )
        X = df.drop(columns=['target']).values
        y = df['target'].values
        X_normal = X[y == 0]
        preprocessor.fit(X_normal)
        
        # Save preprocessor
        os.makedirs(preprocessor_path.parent, exist_ok=True)
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)
    
    # Load or train model
    if model_path.exists():
        print("✓ Loading Isolation Forest model from disk...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        print("✓ Training new Isolation Forest model...")
        model = IsolationForest(**Config.ISOLATION_FOREST_PARAMS)
        
        # Load training data
        from data.data_loader import HeartAttackDataLoader
        df = HeartAttackDataLoader.generate_sample_dataset(
            n_samples=1000,
            contamination=0.1,
            random_state=Config.RANDOM_SEED
        )
        X = df.drop(columns=['target']).values
        y = df['target'].values
        X_normal = X[y == 0]
        X_normal_scaled = preprocessor.transform(X_normal)
        
        model.fit(X_normal_scaled)
        
        # Save model
        os.makedirs(model_path.parent, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    # Initialize LLM explainer if token is available
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
        from data.data_loader import HeartAttackDataLoader
        df = HeartAttackDataLoader.generate_sample_dataset(
            n_samples=100,
            contamination=0.1,
            random_state=Config.RANDOM_SEED
        )
        X_bg = df.drop(columns=['target']).values[:50]
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
    print("✓ SERVER READY TO ACCEPT PREDICTIONS")
    print("="*60)


def predict_anomaly(data):
    """
    Run anomaly detection on input data.
    
    Args:
        data: Dictionary with feature values or array of values
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Parse input data
        if isinstance(data, dict):
            # Convert dict to array in correct feature order
            X_raw = np.array([[data.get(f, 0) for f in feature_names]])
        elif isinstance(data, list):
            X_raw = np.array([data])
        else:
            return {"error": "Invalid data format"}
        
        # Preprocess
        X_scaled = preprocessor.transform(X_raw)
        
        # Predict
        prediction = model.predict(X_scaled)[0]  # -1 = anomaly, 1 = normal
        anomaly_score = model.score_samples(X_scaled)[0]  # 0-1, higher = more anomalous
        
        # Determine risk level
        risk_level = Config.get_risk_level(anomaly_score)
        
        # Get feature importances using SHAP
        feature_importances = {}
        if shap_explainer:
            try:
                shap_values = shap_explainer.explain(X_scaled)
                top_indices = np.argsort(np.abs(shap_values[0]))[-5:][::-1]
                for idx in top_indices:
                    feature_importances[feature_names[idx]] = float(np.abs(shap_values[0][idx]))
            except Exception as e:
                print(f"SHAP explanation error: {e}")
                # Fallback to uniform importance
                for i, fname in enumerate(feature_names[:5]):
                    feature_importances[fname] = 0.2
        else:
            # Fallback to uniform importance
            for i, fname in enumerate(feature_names[:5]):
                feature_importances[fname] = 0.2
        
        # Get feature values
        feature_values = {}
        if isinstance(data, dict):
            feature_values = data
        else:
            feature_values = {feature_names[i]: float(X_raw[0][i]) for i in range(len(feature_names))}
        
        # Generate LLM explanation
        llm_explanation = None
        if llm_explainer:
            try:
                llm_explanation = llm_explainer.explain_prediction(
                    risk_level=risk_level,
                    anomaly_score=anomaly_score,
                    feature_importances=feature_importances,
                    feature_values=feature_values,
                    model_name="Isolation Forest"
                )
            except Exception as e:
                print(f"LLM explanation error: {e}")
                llm_explanation = f"Clinical explanation unavailable. Risk Level: {risk_level}"
        else:
            llm_explanation = f"Patient shows {risk_level.lower()} risk pattern based on clinical features."
        
        # Build result
        result = {
            "prediction": "Anomaly" if prediction == -1 else "Normal",
            "anomaly_score": float(anomaly_score),
            "risk_level": risk_level,
            "feature_importances": feature_importances,
            "feature_values": feature_values,
            "llm_explanation": llm_explanation,
            "timestamp": datetime.now().isoformat(),
            "model": "Isolation Forest"
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}


@app.route('/')
def index():
    """Render server dashboard."""
    return render_template('server_dashboard.html')


@app.route('/api/status')
def status():
    """Get server status."""
    return {
        "status": "online",
        "model": "Isolation Forest",
        "predictions_count": len(prediction_history),
        "llm_available": llm_explainer is not None
    }


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"Client connected: {request.sid}")
    emit('connection_response', {
        'status': 'connected',
        'message': 'Connected to anomaly detection server'
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f"Client disconnected: {request.sid}")


@socketio.on('predict')
def handle_predict(data):
    """Handle prediction request from client."""
    print(f"Prediction request from {request.sid}")
    
    # Extract patient data
    patient_data = data.get('data', {})
    client_id = data.get('client_id', 'unknown')
    
    # Run prediction
    result = predict_anomaly(patient_data)
    
    # Add client info
    result['client_id'] = client_id
    result['request_id'] = request.sid
    result['timestamp'] = datetime.now().isoformat()
    
    # Store in history
    prediction_history.append(result)
    
    print(f"✓ Prediction completed for client {client_id}: {result['risk_level']} risk")
    
    # Send acknowledgment to requesting client
    emit('prediction_result', result)
    print(f"  → Sent 'prediction_result' to {request.sid}")
    
    # Broadcast to ALL connected clients (for dashboard updates)
    # Use send=True to ensure it reaches all clients
    emit('new_prediction', result, broadcast=True, include_self=True)
    print(f"  → Broadcasted 'new_prediction' to all clients")
    
    return {'status': 'success'}


@socketio.on('get_history')
def handle_get_history():
    """Send prediction history to requesting client."""
    emit('history_data', {
        'predictions': prediction_history[-100:]  # Last 100 predictions
    })


if __name__ == '__main__':
    # Load model before starting server
    load_model_and_preprocessor()
    
    # Start server
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting server on port {port}...")
    print(f"📊 Dashboard: http://localhost:{port}")
    print(f"🔌 WebSocket: ws://localhost:{port}\n")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
