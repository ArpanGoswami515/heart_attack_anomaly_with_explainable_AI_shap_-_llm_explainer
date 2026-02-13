"""
Comprehensive Connection Test
Tests the complete flow: Client sends prediction → Server receives → Dashboard updates
"""

import socketio
import time
import requests
from datetime import datetime
import sys

SERVER_URL = 'http://localhost:5000'

def log(message):
    """Log with timestamp."""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[{timestamp}] {message}")

def main():
    log("=" * 70)
    log(" COMPREHENSIVE CLIENT-SERVER-DASHBOARD TEST")
    log("=" * 70)
    
    # Step 1: Verify server is running
    log("\n✓ Step 1: Checking server status...")
    try:
        resp = requests.get(SERVER_URL, timeout=5)
        log(f"  Server is running (Status: {resp.status_code})")
    except:
        log("  ✗ Server not running! Start it with:")
        log("    python heart_attack_anomaly_ai/server/app.py")
        return False
    
    # Step 2: Create dashboard listener (simulates browser dashboard)
    log("\n✓ Step 2: Connecting dashboard listener...")
    dashboard = socketio.Client()
    dashboard_received = []
    
    @dashboard.on('connect')
    def dash_connect():
        log("  📊 Dashboard connected")
    
    @dashboard.on('new_prediction')
    def on_new_prediction(data):
        log(f"  📊 Dashboard received prediction from '{data.get('client_id')}':")
        log(f"      - Risk: {data.get('risk_level')}")
        log(f"      - Score: {data.get('anomaly_score', 0):.4f}")
        dashboard_received.append(data)
    
    try:
        dashboard.connect(SERVER_URL)
        time.sleep(1)
    except Exception as e:
        log(f"  ✗ Failed to connect dashboard: {e}")
        return False
    
    # Step 3: Create client (simulates local hospital client)
    log("\n✓ Step 3: Connecting local client...")
    client = socketio.Client()
    client_acks = []
    
    @client.on('connect')
    def client_connect():
        log("  🏥 Local client connected")
    
    @client.on('prediction_result')
    def on_result(data):
        log(f"  🏥 Client received acknowledgment")
        client_acks.append(data)
    
    try:
        client.connect(SERVER_URL)
        time.sleep(1)
    except Exception as e:
        log(f"  ✗ Failed to connect client: {e}")
        dashboard.disconnect()
        return False
    
    # Step 4: Send prediction from client
    log("\n✓ Step 4: Sending prediction from client...")
    test_payload = {
        'data': {
            'age': 60,
            'sex': 1,
            'cp': 0,
            'trtbps': 150,
            'chol': 280,
            'fbs': 1,
            'restecg': 0,
            'thalachh': 130,
            'exng': 1,
            'oldpeak': 3.0,
            'slp': 2,
            'caa': 2,
            'thall': 2
        },
        'client_id': 'Hospital_A',
        'local_prediction': {
            'model': 'IsolationForest',
            'prediction': 'Unknown'
        }
    }
    
    client.emit('predict', test_payload)
    log("  Prediction sent, waiting for responses...")
    
    # Wait for server to process and broadcast
    time.sleep(2)
    
    # Step 5: Verify results
    log("\n✓ Step 5: Verifying results...")
    log("=" * 70)
    
    success = True
    
    if client_acks:
        log("✓ CLIENT ACKNOWLEDGMENT: Received")
        log(f"    Risk Level: {client_acks[0].get('risk_level')}")
        log(f"    Anomaly Score: {client_acks[0].get('anomaly_score', 0):.4f}")
    else:
        log("✗ CLIENT ACKNOWLEDGMENT: NOT received")
        success = False
    
    if dashboard_received:
        log("✓ DASHBOARD BROADCAST: Received")
        log(f"    Client ID: {dashboard_received[0].get('client_id')}")
        log(f"    Risk Level: {dashboard_received[0].get('risk_level')}")
    else:
        log("✗ DASHBOARD BROADCAST: NOT received")
        success = False
    
    # Cleanup
    client.disconnect()
    dashboard.disconnect()
    
    # Final verdict
    log("\n" + "=" * 70)
    if success:
        log("🎉 SUCCESS! ALLSYSTEMS WORKING:")
        log("   ✓ Client → Server communication working")
        log("   ✓ Server → Client acknowledgment working")
        log("   ✓ Server → Dashboard broadcast working")
        log("\n💡 Your setup is ready! Local clients can send predictions")
        log("   and the global server dashboard will receive updates.")
        log("\n📊 Open dashboard: http://localhost:5000")
    else:
        log("⚠️  PARTIAL SUCCESS - Some components not working")
        log("   Review the test output above for details")
    
    log("=" * 70)
    return success

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        log("\n\nTest interrupted")
        sys.exit(1)
    except Exception as e:
        log(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
