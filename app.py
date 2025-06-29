import streamlit as st
import requests
import numpy as np
import time
import threading
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Client Simulator Class
class ClientSimulator:
    def __init__(self, server_url):
        self.server_url = server_url
        self.client_id = f"web_client_{int(time.time())}"
        self.is_running = False
        self.thread = None
        self.last_update = "Never"
        self.last_error = None
        
    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._run_client, daemon=True)
        self.thread.start()
        logger.info(f"Client simulator started for {self.server_url}")
        
    def stop(self):
        self.is_running = False
        logger.info("Client simulator stopped")
        
    def _run_client(self):
        try:
            logger.info(f"Attempting to register client {self.client_id} with server {self.server_url}")
            client_info = {
                'dataset_size': 100,
                'model_params': 10000,
                'capabilities': ['training', 'inference']
            }
            
            resp = requests.post(f"{self.server_url}/register", 
                               json={'client_id': self.client_id, 'client_info': client_info},
                               timeout=10)
            
            if resp.status_code == 200:
                logger.info(f"Successfully registered client {self.client_id}")
                st.session_state.training_history.append({
                    'round': 0,
                    'active_clients': 1,
                    'clients_ready': 0,
                    'timestamp': datetime.now()
                })
                
                while self.is_running:
                    try:
                        logger.debug(f"Checking training status from {self.server_url}/training_status")
                        status = requests.get(f"{self.server_url}/training_status", timeout=5)
                        if status.status_code == 200:
                            data = status.json()
                            logger.debug(f"Training status: {data}")
                            st.session_state.training_history.append({
                                'round': data.get('current_round', 0),
                                'active_clients': data.get('active_clients', 0),
                                'clients_ready': data.get('clients_ready', 0),
                                'timestamp': datetime.now()
                            })
                            
                            if len(st.session_state.training_history) > 50:
                                st.session_state.training_history = st.session_state.training_history[-50:]
                        else:
                            logger.warning(f"Training status returned {status.status_code}: {status.text}")
                        
                        time.sleep(5)
                        
                    except requests.exceptions.Timeout:
                        logger.warning("Timeout while checking training status")
                        self.last_error = "Timeout connecting to server"
                        time.sleep(10)
                    except requests.exceptions.ConnectionError as e:
                        logger.error(f"Connection error while checking training status: {e}")
                        self.last_error = f"Connection error: {e}"
                        time.sleep(10)
                    except Exception as e:
                        logger.error(f"Unexpected error in client simulator: {e}")
                        self.last_error = f"Unexpected error: {e}"
                        time.sleep(10)
                        
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to server {self.server_url}: {e}")
            self.last_error = f"Failed to connect to server: {e}"
            self.is_running = False
        except Exception as e:
            logger.error(f"Failed to start client simulator: {e}")
            self.last_error = f"Failed to start: {e}"
            self.is_running = False

def check_server_health(server_url):
    """Check if server is reachable and healthy"""
    try:
        logger.debug(f"Checking server health at {server_url}/health")
        resp = requests.get(f"{server_url}/health", timeout=5)
        if resp.status_code == 200:
            logger.info("Server is healthy")
            return True, resp.json()
        else:
            logger.warning(f"Server health check returned {resp.status_code}")
            return False, f"HTTP {resp.status_code}: {resp.text}"
    except requests.exceptions.Timeout:
        logger.error("Server health check timeout")
        return False, "Timeout"
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Server health check connection error: {e}")
        return False, f"Connection refused: {e}"
    except Exception as e:
        logger.error(f"Server health check unexpected error: {e}")
        return False, f"Unexpected error: {e}"

st.set_page_config(page_title="Federated Credit Scoring Demo", layout="centered")
st.title("Federated Credit Scoring Demo")

# Sidebar configuration
st.sidebar.header("Configuration")
SERVER_URL = st.sidebar.text_input("Server URL", value="http://localhost:8080")
DEMO_MODE = st.sidebar.checkbox("Demo Mode", value=True)

# Initialize session state
if 'client_simulator' not in st.session_state:
    st.session_state.client_simulator = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'debug_messages' not in st.session_state:
    st.session_state.debug_messages = []

# Debug section in sidebar
with st.sidebar.expander("Debug Information"):
    st.write("**Server Status:**")
    if not DEMO_MODE:
        is_healthy, health_info = check_server_health(SERVER_URL)
        if is_healthy:
            st.success("✅ Server is healthy")
            st.json(health_info)
        else:
            st.error(f"❌ Server error: {health_info}")
    
    st.write("**Recent Logs:**")
    if st.session_state.debug_messages:
        for msg in st.session_state.debug_messages[-5:]:  # Show last 5 messages
            st.text(msg)
    else:
        st.text("No debug messages yet")
    
    if st.button("Clear Debug Logs"):
        st.session_state.debug_messages = []

# Sidebar educational content
with st.sidebar.expander("About Federated Learning"):
    st.markdown("""
    **Traditional ML:** Banks send data to central server → Privacy risk
    
    **Federated Learning:** 
    - Banks keep data locally
    - Only model updates are shared
    - Collaborative learning without data sharing
    """)

# Client Simulator in sidebar
st.sidebar.header("Client Simulator")
if st.sidebar.button("Start Client"):
    if not DEMO_MODE:
        try:
            st.session_state.client_simulator = ClientSimulator(SERVER_URL)
            st.session_state.client_simulator.start()
            st.sidebar.success("Client started!")
            st.session_state.debug_messages.append(f"{datetime.now()}: Client simulator started")
        except Exception as e:
            st.sidebar.error(f"Failed to start client: {e}")
            st.session_state.debug_messages.append(f"{datetime.now()}: Failed to start client - {e}")
    else:
        st.sidebar.warning("Only works in Real Mode")

if st.sidebar.button("Stop Client"):
    if st.session_state.client_simulator:
        st.session_state.client_simulator.stop()
        st.session_state.client_simulator = None
        st.sidebar.success("Client stopped!")
        st.session_state.debug_messages.append(f"{datetime.now()}: Client simulator stopped")

# Main content - focused on core functionality
st.header("Enter Customer Features")
with st.form("feature_form"):
    features = []
    cols = st.columns(4)
    for i in range(32):
        with cols[i % 4]:
            val = st.number_input(f"Feature {i+1}", value=0.0, format="%.4f", key=f"f_{i}")
            features.append(val)
    submitted = st.form_submit_button("Predict Credit Score")

# Prediction results
if submitted:
    logger.info(f"Prediction requested with {len(features)} features")
    if DEMO_MODE:
        with st.spinner("Processing..."):
            time.sleep(1)
        demo_prediction = sum(features) / len(features) * 100 + 500
        st.success(f"Predicted Credit Score: {demo_prediction:.2f}")
        st.session_state.debug_messages.append(f"{datetime.now()}: Demo prediction: {demo_prediction:.2f}")
    else:
        try:
            logger.info(f"Sending prediction request to {SERVER_URL}/predict")
            with st.spinner("Connecting to server..."):
                resp = requests.post(f"{SERVER_URL}/predict", json={"features": features}, timeout=10)
            
            if resp.status_code == 200:
                prediction = resp.json().get("prediction")
                st.success(f"Predicted Credit Score: {prediction:.2f}")
                st.session_state.debug_messages.append(f"{datetime.now()}: Real prediction: {prediction:.2f}")
                logger.info(f"Prediction successful: {prediction}")
            else:
                error_msg = f"Prediction failed: {resp.json().get('error', 'Unknown error')}"
                st.error(error_msg)
                st.session_state.debug_messages.append(f"{datetime.now()}: {error_msg}")
                logger.error(f"Prediction failed with status {resp.status_code}: {resp.text}")
        except requests.exceptions.Timeout:
            error_msg = "Timeout connecting to server"
            st.error(error_msg)
            st.session_state.debug_messages.append(f"{datetime.now()}: {error_msg}")
            logger.error("Prediction request timeout")
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {e}"
            st.error(error_msg)
            st.session_state.debug_messages.append(f"{datetime.now()}: {error_msg}")
            logger.error(f"Prediction connection error: {e}")
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            st.error(error_msg)
            st.session_state.debug_messages.append(f"{datetime.now()}: {error_msg}")
            logger.error(f"Prediction unexpected error: {e}")

# Training progress - simplified
st.header("Training Progress")
if DEMO_MODE:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Round", "3/10")
    with col2:
        st.metric("Clients", "3")
    with col3:
        st.metric("Accuracy", "85.2%")
    with col4:
        st.metric("Status", "Active")
else:
    try:
        logger.debug(f"Fetching training status from {SERVER_URL}/training_status")
        status = requests.get(f"{SERVER_URL}/training_status", timeout=5)
        if status.status_code == 200:
            data = status.json()
            logger.debug(f"Training status received: {data}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Round", f"{data.get('current_round', 0)}/{data.get('total_rounds', 10)}")
            with col2:
                st.metric("Clients", data.get('active_clients', 0))
            with col3:
                st.metric("Ready", data.get('clients_ready', 0))
            with col4:
                st.metric("Status", "Active" if data.get('training_active', False) else "Inactive")
        else:
            error_msg = f"Training status failed: HTTP {status.status_code}"
            st.warning(error_msg)
            st.session_state.debug_messages.append(f"{datetime.now()}: {error_msg}")
            logger.warning(f"Training status returned {status.status_code}: {status.text}")
    except requests.exceptions.Timeout:
        error_msg = "Training status timeout"
        st.warning(error_msg)
        st.session_state.debug_messages.append(f"{datetime.now()}: {error_msg}")
        logger.warning("Training status request timeout")
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Training status connection error: {e}"
        st.warning(error_msg)
        st.session_state.debug_messages.append(f"{datetime.now()}: {error_msg}")
        logger.error(f"Training status connection error: {e}")
    except Exception as e:
        error_msg = f"Training status unexpected error: {e}"
        st.warning(error_msg)
        st.session_state.debug_messages.append(f"{datetime.now()}: {error_msg}")
        logger.error(f"Training status unexpected error: {e}")

# Client status in sidebar
if st.session_state.client_simulator and not DEMO_MODE:
    st.sidebar.header("Client Status")
    if st.session_state.client_simulator.is_running:
        st.sidebar.success("Connected")
        st.sidebar.info(f"ID: {st.session_state.client_simulator.client_id}")
        if st.session_state.client_simulator.last_error:
            st.sidebar.error(f"Last Error: {st.session_state.client_simulator.last_error}")
    else:
        st.sidebar.warning("Disconnected") 