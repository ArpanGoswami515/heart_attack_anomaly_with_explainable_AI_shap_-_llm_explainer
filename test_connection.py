"""
Connection Test Module for Client-Server Communication

Tests the WebSocket connection between local clients and the global server
to diagnose why the server is not receiving updates from clients.
"""

import socketio
import time
import requests
import sys
from datetime import datetime


# Server configuration
SERVER_URL = 'http://localhost:5000'
CLIENT_ID = 'test_client'

# Test data - sample patient data
TEST_PATIENT_DATA = {
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
}


def log(message):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[{timestamp}] {message}")


def test_server_http():
    """Test if server is reachable via HTTP."""
    log("=" * 60)
    log("TEST 1: HTTP Connection to Server")
    log("=" * 60)
    
    try:
        response = requests.get(SERVER_URL, timeout=5)
        log(f"✓ Server is reachable at {SERVER_URL}")
        log(f"  Status Code: {response.status_code}")
        log(f"  Response Time: {response.elapsed.total_seconds():.3f}s")
        return True
    except requests.exceptions.ConnectionError:
        log(f"✗ FAILED: Cannot connect to server at {SERVER_URL}")
        log("  → Is the server running? Start it with: python heart_attack_anomaly_ai/server/app.py")
        return False
    except Exception as e:
        log(f"✗ FAILED: {type(e).__name__}: {e}")
        return False


def test_websocket_connection():
    """Test WebSocket connection establishment."""
    log("\n" + "=" * 60)
    log("TEST 2: WebSocket Connection")
    log("=" * 60)
    
    sio = socketio.Client()
    connected = False
    
    @sio.on('connect')
    def on_connect():
        nonlocal connected
        connected = True
        log("✓ WebSocket connection established")
    
    @sio.on('connection_response')
    def on_connection_response(data):
        log(f"✓ Server response received: {data}")
    
    @sio.on('disconnect')
    def on_disconnect():
        log("  Connection closed")
    
    try:
        log(f"Connecting to {SERVER_URL}...")
        sio.connect(SERVER_URL, wait_timeout=10)
        
        # Wait a moment for connection events
        time.sleep(0.5)
        
        if connected:
            log("✓ Connection successful")
            sio.disconnect()
            return True
        else:
            log("✗ FAILED: Connected but no confirmation received")
            sio.disconnect()
            return False
            
    except Exception as e:
        log(f"✗ FAILED: {type(e).__name__}: {e}")
        return False


def test_prediction_event():
    """Test sending prediction event and receiving response."""
    log("\n" + "=" * 60)
    log("TEST 3: Prediction Event Transmission")
    log("=" * 60)
    
    sio = socketio.Client()
    
    # Track events
    events_received = {
        'connect': False,
        'connection_response': False,
        'prediction_result': False,
        'new_prediction': False
    }
    
    prediction_data = None
    
    @sio.on('connect')
    def on_connect():
        events_received['connect'] = True
        log("✓ Connected to server")
    
    @sio.on('connection_response')
    def on_connection_response(data):
        events_received['connection_response'] = True
        log(f"✓ Connection response: {data.get('message', 'N/A')}")
    
    @sio.on('prediction_result')
    def on_prediction_result(data):
        nonlocal prediction_data
        events_received['prediction_result'] = True
        prediction_data = data
        log("✓ Prediction result received!")
        log(f"  - Client ID: {data.get('client_id', 'N/A')}")
        log(f"  - Prediction: {data.get('prediction', 'N/A')}")
        log(f"  - Risk Level: {data.get('risk_level', 'N/A')}")
        log(f"  - Anomaly Score: {data.get('anomaly_score', 'N/A')}")
    
    @sio.on('new_prediction')
    def on_new_prediction(data):
        events_received['new_prediction'] = True
        log("✓ New prediction broadcast received!")
        log(f"  - Client ID: {data.get('client_id', 'N/A')}")
    
    @sio.on('disconnect')
    def on_disconnect():
        log("  Disconnected from server")
    
    try:
        # Connect
        log("Connecting to server...")
        sio.connect(SERVER_URL, wait_timeout=10)
        time.sleep(0.5)
        
        # Send prediction
        log(f"Sending prediction event for client '{CLIENT_ID}'...")
        payload = {
            'data': TEST_PATIENT_DATA,
            'client_id': CLIENT_ID,
            'local_prediction': {
                'model': 'test_model',
                'prediction': 'Normal'
            }
        }
        sio.emit('predict', payload)
        log("  Event emitted, waiting for response...")
        
        # Wait for response
        time.sleep(2)  # Give server time to process and respond
        
        # Disconnect
        sio.disconnect()
        
        # Analyze results
        log("\n" + "-" * 60)
        log("EVENT SUMMARY:")
        log("-" * 60)
        for event, received in events_received.items():
            status = "✓" if received else "✗"
            log(f"{status} {event}: {'Received' if received else 'NOT received'}")
        
        # Overall result
        if events_received['prediction_result'] or events_received['new_prediction']:
            log("\n✓ TEST PASSED: Server is receiving and processing predictions")
            return True
        else:
            log("\n✗ TEST FAILED: Server not responding to prediction events")
            log("  → Check server logs for errors")
            log("  → Verify handle_predict() is working correctly")
            return False
            
    except Exception as e:
        log(f"\n✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        log(traceback.format_exc())
        return False


def test_client_send_function():
    """Test the actual client's send_to_main_server function."""
    log("\n" + "=" * 60)
    log("TEST 4: Client send_to_main_server() Function")
    log("=" * 60)
    
    try:
        # Import the client's function
        sys.path.insert(0, 'heart_attack_anomaly_ai/client')
        from app import send_to_main_server
        
        log("Testing client's send_to_main_server() function...")
        
        # Prepare test data
        feature_values = TEST_PATIENT_DATA
        prediction_result = {
            'prediction': 'Normal',
            'anomaly_score': 0.35,
            'risk_level': 'Low',
            'model': 'IsolationForest'
        }
        
        # Call the function
        log("Calling send_to_main_server()...")
        send_to_main_server(feature_values, prediction_result, CLIENT_ID)
        
        log("✓ Function executed without errors")
        log("  Check server dashboard for the prediction")
        log(f"  → http://localhost:5000")
        
        return True
        
    except ImportError as e:
        log(f"✗ FAILED: Cannot import client app: {e}")
        return False
    except Exception as e:
        log(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        log(traceback.format_exc())
        return False


def run_all_tests():
    """Run all connection tests."""
    log("╔" + "=" * 58 + "╗")
    log("║" + " " * 10 + "CONNECTION TEST SUITE" + " " * 27 + "║")
    log("╚" + "=" * 58 + "╝")
    
    results = []
    
    # Test 1: HTTP Connection
    results.append(("HTTP Connection", test_server_http()))
    
    if not results[0][1]:
        log("\n" + "!" * 60)
        log("CRITICAL: Server is not running. Cannot proceed with other tests.")
        log("Please start the server first:")
        log("  python heart_attack_anomaly_ai/server/app.py")
        log("!" * 60)
        return
    
    # Test 2: WebSocket Connection
    results.append(("WebSocket Connection", test_websocket_connection()))
    
    # Test 3: Prediction Event
    results.append(("Prediction Event", test_prediction_event()))
    
    # Test 4: Client Function
    results.append(("Client Function", test_client_send_function()))
    
    # Final Summary
    log("\n" + "╔" + "=" * 58 + "╗")
    log("║" + " " * 20 + "TEST SUMMARY" + " " * 26 + "║")
    log("╠" + "=" * 58 + "╣")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        log(f"║ {test_name:.<45} {status:>10} ║")
    
    log("╠" + "=" * 58 + "╣")
    log(f"║ Total: {passed}/{total} tests passed" + " " * (41 - len(f"Total: {passed}/{total} tests passed")) + "║")
    log("╚" + "=" * 58 + "╝")
    
    if passed == total:
        log("\n🎉 All tests passed! Client-server communication is working.")
    else:
        log(f"\n⚠️  {total - passed} test(s) failed. Review the logs above for details.")


if __name__ == '__main__':
    run_all_tests()
