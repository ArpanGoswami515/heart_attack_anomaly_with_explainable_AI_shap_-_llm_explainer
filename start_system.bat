@echo off
REM Startup script for Distributed Heart Attack Anomaly Detection System (Windows)

echo ========================================
echo Heart Attack Anomaly Detection System
echo Distributed Architecture Startup
echo ========================================
echo.

REM Check if Docker is running
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed or not running.
    echo Please install Docker Desktop and start it.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo [WARNING] .env file not found!
    echo.
    echo Creating .env file...
    echo HUGGINGFACE_API_TOKEN=your_token_here > .env
    echo.
    echo [ACTION REQUIRED] Please edit .env file and add your HuggingFace API token
    echo Get your token from: https://huggingface.co/settings/tokens
    echo.
    pause
    exit /b 1
)

REM Check if model is trained
if not exist saved_models\isolation_forest.pkl (
    echo [INFO] Model not found. Training model...
    echo.
    python train_model.py
    if %errorlevel% neq 0 (
        echo [ERROR] Model training failed!
        pause
        exit /b 1
    )
    echo.
)

echo [INFO] Starting distributed system with Docker Compose...
echo.

REM Start Docker Compose
docker-compose up --build

pause
