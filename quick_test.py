"""
Quick Test Script for Fake News Detection System
Simple testing without Unicode issues
"""

import requests
import time
import sys
import os
from datetime import datetime

def test_system():
    """Test the system quickly."""
    print("FAKE NEWS DETECTION SYSTEM - QUICK TEST")
    print("=" * 50)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    base_url = "http://127.0.0.1:5000"
    
    # Test 1: Home page
    print("Test 1: Home Page Access")
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            print("[PASS] Home page accessible")
        else:
            print(f"[FAIL] Home page returned {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[FAIL] Cannot connect to server. Is it running?")
        print("       Try running: python app.py")
        return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False
    
    # Test 2: Basic API
    print("\nTest 2: Basic Prediction API")
    test_text = "Scientists at MIT have developed new technology."
    try:
        response = requests.post(
            f"{base_url}/api/predict",
            json={"text": test_text},
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print("[PASS] Basic API working")
            print(f"       Prediction: {result.get('prediction', 'N/A')}")
            print(f"       Confidence: {result.get('confidence', 'N/A')}")
        else:
            print(f"[FAIL] API returned {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] API Error: {e}")
        return False
    
    # Test 3: Advanced API
    print("\nTest 3: Advanced Analysis API")
    try:
        response = requests.post(
            f"{base_url}/api/analyze",
            json={"text": test_text, "include_ensemble": True},
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print("[PASS] Advanced API working")
            if 'ensemble_prediction' in result:
                print("[PASS] Ensemble prediction available")
            if 'text_analysis' in result:
                print("[PASS] Text analysis available")
        else:
            print(f"[FAIL] Advanced API returned {response.status_code}")
    except Exception as e:
        print(f"[FAIL] Advanced API Error: {e}")
    
    # Test 4: Performance
    print("\nTest 4: Performance Test")
    start_time = time.time()
    try:
        response = requests.post(
            f"{base_url}/api/predict",
            json={"text": "This is a performance test message."},
            timeout=30
        )
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            print(f"[PASS] Response time: {duration:.2f} seconds")
        else:
            print(f"[FAIL] Performance test failed")
    except Exception as e:
        print(f"[FAIL] Performance test error: {e}")
    
    # Test 5: Network Access
    print("\nTest 5: Network Access Information")
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        print(f"[INFO] Computer Name: {hostname}")
        print(f"[INFO] Local IP: {local_ip}")
        print(f"[INFO] Network URL: http://{local_ip}:5000")
        
        # Try to access via network IP
        network_response = requests.get(f"http://{local_ip}:5000", timeout=5)
        if network_response.status_code == 200:
            print("[PASS] Network access working")
        else:
            print("[WARN] Network access may have issues")
            
    except Exception as e:
        print(f"[WARN] Network test error: {e}")
    
    print("\n" + "=" * 50)
    print("QUICK TEST COMPLETE!")
    print()
    print("ACCESS URLS:")
    print(f"  Local:   http://127.0.0.1:5000")
    print(f"  Network: http://{local_ip}:5000")
    print()
    print("DEPLOYMENT OPTIONS:")
    print("  1. Local: Double-click LAUNCH.bat")
    print("  2. Docker: docker-compose up -d")
    print("  3. Cloud: Upload to GitHub + Streamlit Cloud")
    print()
    print("Your Fake News Detection System is ready!")
    
    return True

if __name__ == "__main__":
    test_system()
