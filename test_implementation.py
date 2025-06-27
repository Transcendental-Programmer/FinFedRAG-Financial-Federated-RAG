#!/usr/bin/env python3
"""
Simple test script for the federated learning implementation
"""

import sys
import time
import subprocess
import threading
import os
from pathlib import Path
import logging
import yaml

# Set up debug logging for the test
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def load_client_config():
    config_path = Path(__file__).parent / "config" / "client_config.yaml"
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    return full_config

def test_basic_functionality():
    """Test basic federated learning functionality"""
    print("Testing FinFedRAG Basic Functionality")
    print("=" * 50)
    
    # Test 1: Import all modules
    print("Test 1: Testing imports...")
    try:
        from src.server.coordinator import FederatedCoordinator
        from src.client.model import FederatedClient
        from src.api.server import FederatedAPI
        from src.api.client import FederatedHTTPClient
        print("✓ All imports successful")
        logging.debug("All modules imported successfully.")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        logging.error(f"Import failed: {e}")
        return False
    
    # Test 2: Create coordinator
    print("\nTest 2: Testing coordinator creation...")
    try:
        config = {
            'server': {
                'federated': {'min_clients': 2, 'rounds': 3},
                'api': {'host': 'localhost', 'port': 8081},
                'aggregation': {'method': 'fedavg', 'weighted': True}
            },
            'model': {'input_dim': 32},
            'training': {'learning_rate': 0.001}
        }
        logging.debug(f"Coordinator test config: {config}")
        coordinator = FederatedCoordinator(config)
        print("✓ Coordinator created successfully")
        logging.debug("Coordinator created successfully.")
    except Exception as e:
        print(f"✗ Coordinator creation failed: {e}")
        logging.error(f"Coordinator creation failed: {e}")
        return False
    
    # Test 3: Create client
    print("\nTest 3: Testing client creation...")
    try:
        client_config = load_client_config()
        logging.debug(f"Client test config: {client_config}")
        client = FederatedClient("test_client", client_config)
        print("✓ Client created successfully")
        logging.debug("Client created successfully.")
    except Exception as e:
        print(f"✗ Client creation failed: {e}")
        logging.error(f"Client creation failed: {e}")
        return False
    
    # Test 4: Test HTTP client
    print("\nTest 4: Testing HTTP client...")
    try:
        http_client = FederatedHTTPClient('http://localhost:8081', 'test_client')
        print("✓ HTTP client created successfully")
        logging.debug("HTTP client created successfully.")
    except Exception as e:
        print(f"✗ HTTP client creation failed: {e}")
        logging.error(f"HTTP client creation failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("All basic functionality tests passed!")
    logging.debug("All basic functionality tests passed.")
    return True

def run_integration_test():
    """Run a quick integration test"""
    print("\nRunning Integration Test")
    print("=" * 50)
    
    # This would start a server and client in separate processes
    # For now, just validate the configuration files
    
    config_dir = Path("config")
    
    # Test server config
    server_config = config_dir / "server_config.yaml"
    if server_config.exists():
        print("✓ Server config exists")
        logging.debug("Server config exists.")
    else:
        print("✗ Server config missing")
        logging.error("Server config missing.")
        return False
    
    # Test client config
    client_config = config_dir / "client_config.yaml"
    if client_config.exists():
        print("✓ Client config exists")
        logging.debug("Client config exists.")
    else:
        print("✗ Client config missing")
        logging.error("Client config missing.")
        return False
    
    print("✓ Configuration files are present")
    print("✓ Integration test setup complete")
    logging.debug("Integration test setup complete.")
    
    return True

if __name__ == "__main__":
    print("FinFedRAG Test Suite")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    success = True
    
    # Run basic functionality tests
    if not test_basic_functionality():
        success = False
    
    # Run integration tests
    if not run_integration_test():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed!")
        print("\nTo run the system:")
        print("1. Start server: python -m src.main --mode server --config config/server_config.yaml")
        print("2. Start client: python -m src.main --mode client --config config/client_config.yaml")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
