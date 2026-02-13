# System Architecture - Distributed Heart Attack Anomaly Detection

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DISTRIBUTED SYSTEM                            │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Client 1       │    │   Client 2       │    │   Client 3       │
│   Dashboard      │    │   Dashboard      │    │   Dashboard      │
│   Port: 5001     │    │   Port: 5002     │    │   Port: 5003     │
│                  │    │                  │    │                  │
│ ┌──────────────┐ │    │ ┌──────────────┐ │    │ ┌──────────────┐ │
│ │ Web UI       │ │    │ │ Web UI       │ │    │ │ Web UI       │ │
│ │ - Input Form │ │    │ │ - Input Form │ │    │ │ - Input Form │ │
│ │ - Results    │ │    │ │ - Results    │ │    │ │ - Results    │ │
│ │ - History    │ │    │ │ - History    │ │    │ │ - History    │ │
│ └──────────────┘ │    │ └──────────────┘ │    │ └──────────────┘ │
│                  │    │                  │    │                  │
│ ┌──────────────┐ │    │ ┌──────────────┐ │    │ ┌──────────────┐ │
│ │ Flask Server │ │    │ │ Flask Server │ │    │ │ Flask Server │ │
│ │ Socket.IO    │ │    │ │ Socket.IO    │ │    │ │ Socket.IO    │ │
│ └──────┬───────┘ │    │ └──────┬───────┘ │    │ └──────┬───────┘ │
└────────┼─────────┘    └────────┼─────────┘    └────────┼─────────┘
         │                       │                       │
         │          WebSocket (Patient Data)            │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
         ┌─────────────────────────────────────────────┐
         │          MAIN SERVER (Port 5000)            │
         │                                             │
         │  ┌─────────────────────────────────────┐   │
         │  │     Flask + Socket.IO Server        │   │
         │  │  - Receive prediction requests      │   │
         │  │  - Route to ML pipeline             │   │
         │  │  - Return results to clients        │   │
         │  │  - Broadcast to monitoring dash     │   │
         │  └───────────────┬─────────────────────┘   │
         │                  │                          │
         │  ┌───────────────▼──────────────────────┐  │
         │  │      PREPROCESSING PIPELINE          │  │
         │  │  ┌────────────────────────────────┐  │  │
         │  │  │  DataPreprocessor (Scaler)     │  │  │
         │  │  │  - Normalize features          │  │  │
         │  │  │  - Transform input data        │  │  │
         │  │  └────────────┬───────────────────┘  │  │
         │  └───────────────┼──────────────────────┘  │
         │                  │                          │
         │  ┌───────────────▼──────────────────────┐  │
         │  │     ANOMALY DETECTION MODEL          │  │
         │  │  ┌────────────────────────────────┐  │  │
         │  │  │   Isolation Forest             │  │  │
         │  │  │   - Trained on normal cases    │  │  │
         │  │  │   - Predicts anomaly score     │  │  │
         │  │  │   - Classifies risk level      │  │  │
         │  │  │   - Returns prediction         │  │  │
         │  │  └────────────┬───────────────────┘  │  │
         │  └───────────────┼──────────────────────┘  │
         │                  │                          │
         │  ┌───────────────▼──────────────────────┐  │
         │  │     EXPLAINABILITY LAYER             │  │
         │  │  ┌────────────────────────────────┐  │  │
         │  │  │  SHAP Explainer                │  │  │
         │  │  │  - Calculate feature importance│  │  │
         │  │  │  - Identify key risk factors   │  │  │
         │  │  └────────────────────────────────┘  │  │
         │  │  ┌────────────────────────────────┐  │  │
         │  │  │  LLM Explainer (HuggingFace)   │  │  │
         │  │  │  Model: Llama-3-8B-Instruct    │  │  │
         │  │  │  - Generate clinical text      │  │  │
         │  │  │  - Explain risk assessment     │  │  │
         │  │  └────────────────────────────────┘  │  │
         │  └───────────────┬──────────────────────┘  │
         │                  │                          │
         │  ┌───────────────▼──────────────────────┐  │
         │  │        RESPONSE BUILDER              │  │
         │  │  - Combine prediction + explanation  │  │
         │  │  - Add metadata (timestamp, etc.)    │  │
         │  │  - Format JSON response              │  │
         │  └───────────────┬──────────────────────┘  │
         │                  │                          │
         │  ┌───────────────▼──────────────────────┐  │
         │  │      SERVER DASHBOARD (Web UI)       │  │
         │  │  - Real-time prediction monitor      │  │
         │  │  - Statistics (total, high-risk)     │  │
         │  │  - Risk distribution chart           │  │
         │  │  - Anomaly score trends              │  │
         │  │  - Recent predictions list           │  │
         │  └──────────────────────────────────────┘  │
         │                                             │
         └─────────────────────────────────────────────┘
                                 │
                    WebSocket (Results)
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│   Client 1     │    │   Client 2     │    │   Client 3     │
│   (Results)    │    │   (Results)    │    │   (Results)    │
└────────────────┘    └────────────────┘    └────────────────┘
```

## Data Flow Sequence

```
1. USER ACTION
   Client Dashboard: User enters patient data
   ↓
2. CLIENT PROCESSING
   Client App: Collect form data → Create JSON payload
   ↓
3. WEBSOCKET TRANSMISSION
   Client → Server: emit('predict', {data, client_id})
   ↓
4. SERVER RECEPTION
   Server: Receive prediction request
   ↓
5. PREPROCESSING
   DataPreprocessor: Scale/normalize features
   ↓
6. MODEL INFERENCE
   Isolation Forest: Calculate anomaly score
   ↓
7. RISK CLASSIFICATION
   Config: Map score → Low/Medium/High
   ↓
8. EXPLAINABILITY
   a) SHAP: Calculate feature importances
   b) LLM: Generate clinical explanation (1-3 seconds)
   ↓
9. RESPONSE BUILDING
   Server: Combine all results into JSON
   ↓
10. WEBSOCKET TRANSMISSION
    Server → Client: emit('prediction_result', result)
    Server → Dashboard: emit('new_prediction', result)
    ↓
11. CLIENT DISPLAY
    Client Dashboard: Render results, update history
    Server Dashboard: Update statistics and charts
```

## Technology Stack

```
┌─────────────────────────────────────────────────────────┐
│                    TECHNOLOGY LAYERS                     │
├─────────────────────────────────────────────────────────┤
│  Frontend (Browser)                                      │
│  ├── HTML5, CSS3, JavaScript (ES6)                      │
│  ├── Chart.js (Visualization)                           │
│  └── Socket.IO Client (WebSocket)                       │
├─────────────────────────────────────────────────────────┤
│  Backend (Python)                                        │
│  ├── Flask (Web Framework)                              │
│  ├── Flask-SocketIO (WebSocket Server)                  │
│  └── Flask-CORS (Cross-Origin Resource Sharing)         │
├─────────────────────────────────────────────────────────┤
│  Machine Learning                                        │
│  ├── scikit-learn (Isolation Forest)                    │
│  ├── NumPy (Numerical Computing)                        │
│  └── Pandas (Data Processing)                           │
├─────────────────────────────────────────────────────────┤
│  Explainable AI (XAI)                                    │
│  ├── SHAP (Feature Importance)                          │
│  └── HuggingFace API (LLM Explanations)                 │
├─────────────────────────────────────────────────────────┤
│  Containerization & Orchestration                        │
│  ├── Docker (Containerization)                          │
│  └── Docker Compose (Multi-Container Orchestration)     │
├─────────────────────────────────────────────────────────┤
│  Communication Protocol                                  │
│  ├── WebSocket (Bidirectional, Real-time)              │
│  ├── HTTP/REST (Dashboard serving)                      │
│  └── JSON (Data format)                                 │
└─────────────────────────────────────────────────────────┘
```

## Network Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Docker Network: anomaly_network            │
│                     (Bridge Mode)                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Container: server                                       │
│  ├── Internal: 5000                                      │
│  └── External: localhost:5000                            │
│                                                          │
│  Container: client1                                      │
│  ├── Internal: 5001                                      │
│  └── External: localhost:5001                            │
│                                                          │
│  Container: client2                                      │
│  ├── Internal: 5002                                      │
│  └── External: localhost:5002                            │
│                                                          │
│  Container: client3                                      │
│  ├── Internal: 5003                                      │
│  └── External: localhost:5003                            │
│                                                          │
│  All containers can communicate via:                     │
│  - server:5000 (internal DNS)                            │
│  - WebSocket protocol                                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## File System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Container Volumes                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Server Container:                                       │
│  ├── /app/saved_models (Mounted from host)             │
│  │   ├── isolation_forest.pkl                          │
│  │   └── preprocessor.pkl                              │
│  ├── /app/models (Python modules)                       │
│  ├── /app/preprocessing (Python modules)                │
│  ├── /app/config (Settings)                            │
│  ├── /app/llm (LLM integration)                         │
│  └── /app/explainability (SHAP, etc.)                  │
│                                                          │
│  Client Containers:                                      │
│  ├── /app/client (Application code)                    │
│  └── /app/client/templates (HTML templates)            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Security Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Security Layers                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Network Isolation                                    │
│     └── Docker network (isolated from host)             │
│                                                          │
│  2. Environment Variables                                │
│     └── HuggingFace API token (not in code)            │
│                                                          │
│  3. CORS Configuration                                   │
│     └── Allow specific origins only                      │
│                                                          │
│  4. Input Validation                                     │
│     └── Server-side validation of patient data          │
│                                                          │
│  5. Container Isolation                                  │
│     └── Each service in separate container              │
│                                                          │
│  TODO for Production:                                    │
│  ├── [ ] Add authentication (JWT tokens)                │
│  ├── [ ] Enable HTTPS/TLS                              │
│  ├── [ ] Rate limiting                                   │
│  ├── [ ] API key authentication for WebSocket          │
│  └── [ ] Input sanitization                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Scalability Architecture

```
Current Setup (1 Server + 3 Clients):
┌──────┐ ┌──────┐ ┌──────┐
│Client│ │Client│ │Client│
│  1   │ │  2   │ │  3   │
└──┬───┘ └──┬───┘ └──┬───┘
   │        │        │
   └────────┼────────┘
            │
      ┌─────▼─────┐
      │  Server   │
      └───────────┘

Scalable Setup (Load Balanced):
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│Client│ │Client│ │Client│ │Client│ │Client│
│  1   │ │  2   │ │  3   │ │  4   │ │  5   │
└──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
   │        │        │        │        │
   └────────┼────────┴────────┼────────┘
            │                 │
      ┌─────▼─────┐     ┌─────▼─────┐
      │ Server 1  │     │ Server 2  │
      └─────┬─────┘     └─────┬─────┘
            │                 │
            └────────┬────────┘
                     │
              ┌──────▼───────┐
              │ Load Balancer│
              └──────────────┘
```

## Port Mapping

| Service | Internal Port | External Port | Protocol |
|---------|---------------|---------------|----------|
| Server  | 5000          | 5000          | HTTP/WS  |
| Client 1| 5001          | 5001          | HTTP/WS  |
| Client 2| 5002          | 5002          | HTTP/WS  |
| Client 3| 5003          | 5003          | HTTP/WS  |

## API Endpoints

### WebSocket Events

**Client → Server:**
- `connect` - Establish connection
- `predict` - Send prediction request
  ```json
  {
    "data": {<patient_features>},
    "client_id": "Client-1"
  }
  ```
- `disconnect` - Close connection

**Server → Client:**
- `connection_response` - Confirm connection
- `prediction_result` - Return prediction
  ```json
  {
    "prediction": "Normal/Anomaly",
    "anomaly_score": 0.75,
    "risk_level": "High",
    "feature_importances": {...},
    "llm_explanation": "...",
    "timestamp": "..."
  }
  ```
- `new_prediction` - Broadcast to all (for dashboard)

### HTTP Endpoints

- `GET /` - Serve dashboard HTML
- `GET /api/status` - Server health check
- `GET /api/client_info` - Client information

---

**This architecture enables:**
✅ Real-time bidirectional communication  
✅ Independent scaling of components  
✅ Fault isolation (container crashes don't affect others)  
✅ Easy deployment and replication  
✅ Monitoring and observability  
