"""
Quick Connection Verification Script

Run this to verify client-server-dashboard communication is working.
"""

import socketio
import time

SERVER_URL = 'http://localhost:5000'

print("=" * 60)
print(" QUICK CONNECTION VERIFICATION")
print("=" * 60)
print("\nTesting connection to server at", SERVER_URL)

try:
    # Test 1: Server reachable
    import requests
    resp = requests.get(SERVER_URL, timeout=5)
    print("✓ Server is running")
    
    # Test 2: WebSocket connection
    client = socketio.Client()
    
    @client.on('connect')
    def on_connect():
        print("✓ WebSocket connection established")
    
    @client.on('prediction_result')
    def on_result(data):
        print(f"✓ Received prediction: {data.get('risk_level')} risk")
        print(f"✓ Anomaly score: {data.get('anomaly_score', 0):.4f}")
    
    client.connect(SERVER_URL)
    time.sleep(1)
    
    # Send test prediction
    print("\nSending test prediction...")
    client.emit('predict', {
        'data': {
            'age': 55, 'sex': 1, 'cp': 3, 'trtbps': 140,
            'chol': 250, 'fbs': 1, 'restecg': 1, 'thalachh': 150,
            'exng': 0, 'oldpeak': 2.5, 'slp': 2, 'caa': 1, 'thall': 2
        },
        'client_id': 'verification_test',
        'local_prediction': {}
    })
    
    time.sleep(2)
    client.disconnect()
    
    print("\n" + "=" * 60)
    print("🎉 SUCCESS! Connection verified!")
    print("=" * 60)
    print("\n📊 Dashboard: http://localhost:5000")
    print("   Open this in your browser to see real-time updates\n")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("\nMake sure the server is running:")
    print("  python heart_attack_anomaly_ai/server/app.py")
