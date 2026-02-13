#!/bin/bash
# Startup script for Distributed Heart Attack Anomaly Detection System (Linux/Mac)

echo "========================================"
echo "Heart Attack Anomaly Detection System"
echo "Distributed Architecture Startup"
echo "========================================"
echo ""

# Check if Docker is running
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed."
    echo "Please install Docker and Docker Compose."
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "[ERROR] Docker daemon is not running."
    echo "Please start Docker."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "[WARNING] .env file not found!"
    echo ""
    echo "Creating .env file..."
    echo "HUGGINGFACE_API_TOKEN=your_token_here" > .env
    echo ""
    echo "[ACTION REQUIRED] Please edit .env file and add your HuggingFace API token"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo ""
    exit 1
fi

# Check if model is trained
if [ ! -f saved_models/isolation_forest.pkl ]; then
    echo "[INFO] Model not found. Training model..."
    echo ""
    python train_model.py
    if [ $? -ne 0 ]; then
        echo "[ERROR] Model training failed!"
        exit 1
    fi
    echo ""
fi

echo "[INFO] Starting distributed system with Docker Compose..."
echo ""
echo "Dashboards will be available at:"
echo "  - Server:   http://localhost:5000"
echo "  - Client 1: http://localhost:5001"
echo "  - Client 2: http://localhost:5002"
echo "  - Client 3: http://localhost:5003"
echo ""

# Start Docker Compose
docker-compose up --build
