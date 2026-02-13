# Server Dockerfile for Heart Attack Anomaly Detection System
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir flask flask-socketio flask-cors python-socketio[client] eventlet

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p saved_models visualizations tuning_results

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Run server
CMD ["python", "server/app.py"]
