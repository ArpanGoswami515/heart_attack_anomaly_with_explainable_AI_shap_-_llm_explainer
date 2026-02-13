# Distributed Heart Attack Anomaly Detection System

## 🏗️ Architecture Overview

This system implements a **distributed client-server architecture** for real-time heart attack anomaly detection using machine learning and explainable AI (XAI).

### Components

1. **Main Server (1 instance)**
   - Hosts the trained **Isolation Forest** model
   - Processes prediction requests from clients via **WebSocket**
   - Generates explanations using **SHAP** and **LLM (HuggingFace)**
   - Provides a real-time monitoring dashboard
   - Port: **5000**

2. **Client Dashboards (3 identical instances)**
   - Web interface for inputting patient data
   - Sends data to server via **WebSocket**
   - Displays predictions, risk levels, and explanations
   - Shows prediction history
   - Ports: **5001, 5002, 5003**

### Technology Stack

- **Backend**: Python, Flask, Flask-SocketIO
- **Communication**: WebSocket protocol (Socket.IO)
- **ML Model**: Scikit-learn Isolation Forest
- **Explainability**: SHAP, HuggingFace LLM (Llama 3)
- **Containerization**: Docker, Docker Compose
- **Frontend**: HTML, CSS, JavaScript, Chart.js

---

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- HuggingFace API token (get it from https://huggingface.co/settings/tokens)
- At least 4GB RAM available

### Step 1: Set Up Environment

```bash
# Navigate to project directory
cd heart_attack_anomaly_ai

# Create .env file with your HuggingFace token
echo "HUGGINGFACE_API_TOKEN=your_token_here" > .env
```

### Step 2: Train the Model

```bash
# Install dependencies (if running locally first)
pip install -r requirements.txt

# Train and save the model
python train_model.py
```

This will:
- Generate training data
- Train the Isolation Forest model
- Save the model and preprocessor to `saved_models/`

### Step 3: Build and Run with Docker

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

### Step 4: Access the Dashboards

Open your browser and navigate to:

- **Server Dashboard**: http://localhost:5000
- **Client 1**: http://localhost:5001
- **Client 2**: http://localhost:5002
- **Client 3**: http://localhost:5003

---

## 📊 How It Works

### Data Flow

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  Client 1   │         │  Client 2   │         │  Client 3   │
│  :5001      │         │  :5002      │         │  :5003      │
└──────┬──────┘         └──────┬──────┘         └──────┬──────┘
       │                       │                       │
       │      Patient Data     │                       │
       │    (WebSocket)        │                       │
       └───────────┬───────────┴───────────────────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Main Server    │
          │     :5000       │
          │                 │
          │ ┌─────────────┐ │
          │ │  Isolation  │ │
          │ │   Forest    │ │
          │ │   Model     │ │
          │ └─────────────┘ │
          │                 │
          │ ┌─────────────┐ │
          │ │    SHAP     │ │
          │ │  Explainer  │ │
          │ └─────────────┘ │
          │                 │
          │ ┌─────────────┐ │
          │ │     LLM     │ │
          │ │(HuggingFace)│ │
          │ └─────────────┘ │
          └─────────┬───────┘
                    │
       Prediction + Explanation
              (WebSocket)
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   Client 1     Client 2     Client 3
```

### Prediction Pipeline

1. **User Input**: Client enters patient data (age, blood pressure, cholesterol, etc.)
2. **Data Transmission**: Client sends data to server via WebSocket
3. **Preprocessing**: Server preprocesses data using trained scaler
4. **Model Inference**: Isolation Forest predicts anomaly score
5. **Risk Classification**: Score mapped to Low/Medium/High risk
6. **Explainability**:
   - SHAP calculates feature importances
   - LLM generates human-readable clinical explanation
7. **Response**: Server sends results back to client
8. **Display**: Client shows prediction, risk level, and explanation

### Model Details

**Isolation Forest**
- Unsupervised anomaly detection algorithm
- Trained on normal (non-heart attack) cases only
- Detects anomalies by measuring isolation score
- Anomaly score normalized to [0, 1] where 1 = high risk

**Explainability**
- **SHAP**: Identifies which features contribute most to the prediction
- **LLM**: Generates natural language explanation of the risk assessment

---

## 🎯 Usage Example

### Sending a Prediction Request

1. Open any client dashboard (e.g., http://localhost:5001)
2. Click "Load Sample Patient Data" or enter values manually:
   - Age: 63
   - Sex: Male (1)
   - Blood Pressure: 145
   - Cholesterol: 233
   - Max Heart Rate: 150
   - ST Depression: 2.3
   - etc.
3. Click "Analyze Patient Data"
4. Wait for server response (typically 1-3 seconds)
5. View results:
   - Risk Level (Low/Medium/High)
   - Anomaly Score
   - Key Contributing Features
   - Clinical Explanation from LLM

### Monitoring on Server Dashboard

The server dashboard shows:
- **Real-time statistics**: Total predictions, high-risk cases, connected clients
- **Risk distribution chart**: Pie chart of Low/Medium/High risk cases
- **Anomaly score trend**: Line chart of recent prediction scores
- **Recent predictions list**: All predictions from all clients with timestamps

---

## 🐳 Docker Commands

```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# Stop all services
docker-compose down

# Rebuild containers
docker-compose up --build

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f server
docker-compose logs -f client1

# Restart a specific service
docker-compose restart server

# Scale clients (add more clients)
docker-compose up --scale client=5
```

---

## 📁 Project Structure

```
heart_attack_anomaly_ai/
├── server/
│   ├── app.py                    # Server application
│   ├── Dockerfile                # Server Docker config
│   └── templates/
│       └── server_dashboard.html # Server dashboard UI
│
├── client/
│   ├── app.py                    # Client application
│   ├── Dockerfile                # Client Docker config
│   └── templates/
│       └── client_dashboard.html # Client dashboard UI
│
├── models/                       # ML model implementations
│   ├── isolation_forest.py
│   ├── one_class_svm.py
│   └── autoencoder.py
│
├── preprocessing/                # Data preprocessing
│   └── preprocess.py
│
├── explainability/              # XAI components
│   ├── shap_explainer.py
│   └── reconstruction_explainer.py
│
├── llm/                         # LLM integration
│   └── hf_explainer.py
│
├── config/                      # Configuration
│   └── settings.py
│
├── saved_models/                # Trained models
│   ├── isolation_forest.pkl
│   └── preprocessor.pkl
│
├── docker-compose.yml           # Docker orchestration
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
└── DISTRIBUTED_SYSTEM.md       # This file
```

---

## 🔧 Configuration

### Environment Variables

**Server (.env or docker-compose.yml)**
```env
HUGGINGFACE_API_TOKEN=your_token_here
PORT=5000
```

**Client (docker-compose.yml)**
```env
PORT=5001
SERVER_URL=http://server:5000
CLIENT_ID=Client-1
```

### Model Parameters (config/settings.py)

```python
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 300,
    "contamination": 0.05,
    "max_samples": "auto",
    "max_features": 1.0,
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1
}

HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_MAX_TOKENS = 300
HF_TEMPERATURE = 0.7
```

---

## 🧪 Testing

### Test the Server Locally

```bash
# Train model first
python train_model.py

# Start server
cd server
python app.py

# Server starts at http://localhost:5000
```

### Test a Client Locally

```bash
# In another terminal
cd client
export SERVER_URL=http://localhost:5000
export CLIENT_ID=TestClient
export PORT=5001
python app.py

# Client starts at http://localhost:5001
```

### Test WebSocket Connection

Use the client dashboard to send test data, or use Python:

```python
import socketio

# Connect to server
sio = socketio.Client()
sio.connect('http://localhost:5000')

# Send prediction request
sio.emit('predict', {
    'data': {
        'Age': 55,
        'Sex': 1,
        'Chest pain type': 3,
        'BP': 130,
        'Cholesterol': 250,
        # ... other features
    },
    'client_id': 'TestClient'
})

# Receive response
@sio.on('prediction_result')
def on_result(data):
    print(f"Risk Level: {data['risk_level']}")
    print(f"Anomaly Score: {data['anomaly_score']}")
    print(f"Explanation: {data['llm_explanation']}")

sio.wait()
```

---

## 🔒 Security Considerations

For production deployment:

1. **Authentication**: Add user authentication to dashboards
2. **HTTPS**: Use SSL/TLS for encrypted communication
3. **API Keys**: Protect WebSocket endpoints with authentication tokens
4. **Rate Limiting**: Implement rate limiting to prevent abuse
5. **Input Validation**: Validate all patient data on server side
6. **CORS**: Configure CORS properly for production domains

---

## 🚨 Troubleshooting

### Server won't start
- Check if HuggingFace token is set in `.env`
- Ensure model is trained: `python train_model.py`
- Check port 5000 is not in use

### Client can't connect to server
- Verify server is running: `docker ps`
- Check SERVER_URL environment variable
- Ensure Docker network is working: `docker network ls`

### Predictions are slow
- LLM inference can take 1-5 seconds
- Consider using a faster model (e.g., microsoft/Phi-3-mini-4k-instruct)
- Check your internet connection (HuggingFace API is remote)

### Docker build fails
- Clear Docker cache: `docker-compose build --no-cache`
- Check disk space: `docker system df`
- Update Docker: `docker version`

---

## 📈 Performance

- **Model Inference**: < 10ms per prediction
- **SHAP Explanation**: ~50ms per prediction
- **LLM Explanation**: 1-5 seconds (depends on HuggingFace API)
- **Total Response Time**: 1-5 seconds
- **Concurrent Clients**: Tested with 10 simultaneous connections
- **Memory Usage**: ~500MB per container

---

## 🎓 Learning Resources

- **Isolation Forest**: [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- **SHAP**: [SHAP Documentation](https://shap.readthedocs.io/)
- **WebSocket**: [Socket.IO Documentation](https://socket.io/docs/v4/)
- **Docker**: [Docker Documentation](https://docs.docker.com/)
- **HuggingFace**: [HuggingFace Inference API](https://huggingface.co/docs/api-inference/)

---

## 📝 License

This project is for educational and research purposes.

---

## 👥 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Happy Detecting! 🏥💡**
