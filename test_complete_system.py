#!/usr/bin/env python3
"""
Test script for the complete federated learning system.
This script tests the server, client, and web app integration.
"""

import requests
import time
import json
import numpy as np
from pathlib import Path
import subprocess
import sys
import threading

def test_server_health(server_url="http://localhost:8080"):
    """Test if the server is healthy."""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server health check passed")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def test_prediction(server_url="http://localhost:8080"):
    """Test the prediction endpoint."""
    try:
        # Generate test features
        features = np.random.randn(32).tolist()
        
        response = requests.post(
            f"{server_url}/predict", 
            json={"features": features}, 
            timeout=10
        )
        
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            print(f"✅ Prediction test passed: {prediction:.4f}")
            return True
        else:
            print(f"❌ Prediction test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prediction test error: {e}")
        return False

def test_training_status(server_url="http://localhost:8080"):
    """Test the training status endpoint."""
    try:
        response = requests.get(f"{server_url}/training_status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Training status test passed: Round {data.get('current_round', 0)}")
            return True
        else:
            print(f"❌ Training status test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Training status test error: {e}")
        return False

def test_client_registration(server_url="http://localhost:8080"):
    """Test client registration."""
    try:
        client_info = {
            'dataset_size': 100,
            'model_params': 10000,
            'capabilities': ['training', 'inference']
        }
        
        response = requests.post(
            f"{server_url}/register",
            json={'client_id': 'test_client', 'client_info': client_info},
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Client registration test passed")
            return True
        else:
            print(f"❌ Client registration test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Client registration test error: {e}")
        return False

def run_complete_test():
    """Run all tests."""
    print("🚀 Testing Complete Federated Learning System")
    print("=" * 50)
    
    server_url = "http://localhost:8080"
    
    # Test server health
    if not test_server_health(server_url):
        print("\n❌ Server is not running. Please start the server first:")
        print("python -m src.main --mode server --config config/server_config.yaml")
        return False
    
    # Test client registration
    if not test_client_registration(server_url):
        print("\n❌ Client registration failed")
        return False
    
    # Test training status
    if not test_training_status(server_url):
        print("\n❌ Training status failed")
        return False
    
    # Test prediction
    if not test_prediction(server_url):
        print("\n❌ Prediction failed")
        return False
    
    print("\n🎉 All tests passed! The federated learning system is working correctly.")
    print("\nNext steps:")
    print("1. Start the web app: streamlit run webapp/streamlit_app.py")
    print("2. Start additional clients: python -m src.main --mode client --config config/client_config.yaml")
    print("3. Use the web interface to interact with the system")
    
    return True

if __name__ == "__main__":
    success = run_complete_test()
    sys.exit(0 if success else 1) 