"""
Debug Server Broadcast - Check what's actually happening

This creates a minimal test to understand Flask-SocketIO broadcasting.
"""

import socketio
import time
from flask import Flask
from flask_socketio import SocketIO, emit

# Create test server
app = Flask(__name__)
socketio_server = SocketIO(app, cors_allowed_origins="*")

connected_clients = set()

@socketio_server.on('connect')
def test_connect():
    print(f"✓ Client connected: {flask.request.sid}")
    connected_clients.add(flask.request.sid)
    print(f"  Total clients: {len(connected_clients)}")

@socketio_server.on('disconnect')
def test_disconnect():
    print(f"✗ Client disconnected: {flask.request.sid}")
    connected_clients.remove(flask.request.sid)

@socketio_server.on('test_broadcast')
def handle_broadcast(data):
    print(f"\n📨 Received test_broadcast from {flask.request.sid}")
    print(f"  Data: {data}")
    print(f"  Broadcasting to {len(connected_clients)} clients...")
    
    # Method 1: Using emit with broadcast
    emit('broadcast_result', {'method': 1, 'message': 'Method 1'}, broadcast=True)
    print("  ✓ Sent via emit() with broadcast=True")
    
    # Method 2: Using socketio.emit
    socketio_server.emit('broadcast_result', {'method': 2, 'message': 'Method 2'})
    print("  ✓ Sent via socketio.emit()")

if __name__ == '__main__':
    import flask
    
    print("=" * 60)
    print("BROADCAST DEBUG SERVER")
    print("=" * 60)
    print("Starting minimal server on port 5001...")
    print("This will help debug broadcasting issues")
    print("=" * 60)
    
    socketio_server.run(app, host='0.0.0.0', port=5001, debug=False)
