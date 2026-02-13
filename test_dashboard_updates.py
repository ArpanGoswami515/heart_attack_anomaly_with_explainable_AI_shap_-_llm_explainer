"""
Test Dashboard Updates - Monitor if the server dashboard receives broadcasts

This script simulates a dashboard listener to verify if 'new_prediction' 
broadcasts are being sent by the server.
"""

import socketio
import time
from datetime import datetime


SERVER_URL = 'http://localhost:5000'


def log(message):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[{timestamp}] {message}")


def test_dashboard_listener():
    """Test if dashboard receives new_prediction broadcasts."""
    log("=" * 60)
    log("DASHBOARD BROADCAST LISTENER TEST")
    log("=" * 60)
    log("This simulates the server dashboard listening for predictions")
    log("Leave this running and send predictions from a client")
    log("-" * 60)
    
    # Dashboard listener (like the HTML page)
    dashboard = socketio.Client()
    
    # Test sender (like a local client)
    sender = socketio.Client()
    
    predictions_received = []
    
    @dashboard.on('connect')
    def dashboard_connect():
        log("📊 Dashboard: Connected to server")
    
    @dashboard.on('new_prediction')
    def on_new_prediction(data):
        log("✓ Dashboard: Received 'new_prediction' broadcast!")
        log(f"  - Client ID: {data.get('client_id', 'N/A')}")
        log(f"  - Risk Level: {data.get('risk_level', 'N/A')}")
        log(f"  - Anomaly Score: {data.get('anomaly_score', 'N/A'):.4f}")
        predictions_received.append(data)
    
    @sender.on('connect')
    def sender_connect():
        log("📤 Sender: Connected to server")
    
    @sender.on('prediction_result')
    def on_prediction_result(data):
        log("✓ Sender: Received 'prediction_result' acknowledgment")
    
    try:
        # Connect dashboard first
        log("\n1. Connecting dashboard listener...")
        dashboard.connect(SERVER_URL)
        time.sleep(1)
        
        # Connect sender
        log("\n2. Connecting prediction sender...")
        sender.connect(SERVER_URL)
        time.sleep(1)
        
        # Send test prediction
        log("\n3. Sending test prediction...")
        test_payload = {
            'data': {
                'age': 55,
                'sex': 1,
                'cp': 3,
                'trtbps': 140,
                'chol': 250,
                'fbs': 1,
                'restecg': 1,
                'thalachh': 150,
                'exng': 0,
                'oldpeak': 2.5,
                'slp': 2,
                'caa': 1,
                'thall': 2
            },
            'client_id': 'dashboard_test_client',
            'local_prediction': {
                'model': 'test',
                'prediction': 'Normal'
            }
        }
        
        sender.emit('predict', test_payload)
        log("  Prediction sent, waiting for broadcast...")
        
        # Wait for broadcast
        time.sleep(2)
        
        # Results
        log("\n" + "=" * 60)
        log("TEST RESULTS:")
        log("=" * 60)
        
        if predictions_received:
            log(f"✓ SUCCESS: Dashboard received {len(predictions_received)} prediction(s)")
            log("\nConclusion: Server broadcasts are working correctly!")
            log("→ The issue may be with the client disconnecting too quickly")
        else:
            log("✗ FAILURE: Dashboard did NOT receive any predictions")
            log("\nPossible issues:")
            log("  1. Server not broadcasting 'new_prediction' events")
            log("  2. Broadcasting mechanism not working")
            log("  3. Check server logs for errors")
        
        # Disconnect
        sender.disconnect()
        dashboard.disconnect()
        
        return len(predictions_received) > 0
        
    except Exception as e:
        log(f"\n✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        log(traceback.format_exc())
        return False


if __name__ == '__main__':
    test_dashboard_listener()
