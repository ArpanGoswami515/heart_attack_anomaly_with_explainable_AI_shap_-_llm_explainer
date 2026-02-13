# Client Dockerfile for Heart Attack Anomaly Detection System
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (full set for local predictions)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir flask flask-socketio flask-cors python-socketio[client] eventlet

# Copy all project files (needed for local predictions)
COPY . .

# Create necessary directories
RUN mkdir -p saved_models data

# Expose port (will be overridden by docker-compose)
EXPOSE 5001

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SERVER_URL=http://server:5000
ENV CLIENT_ID=Client-1
ENV PORT=5001

# Run client
CMD ["python", "client/app.py"]
