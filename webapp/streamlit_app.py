import streamlit as st
import requests
import numpy as np
import time
import threading
import json
from datetime import datetime

st.set_page_config(page_title="Federated Credit Scoring Demo", layout="centered")
st.title("Federated Credit Scoring Demo (Federated Learning)")

# Sidebar configuration
st.sidebar.header("Configuration")
SERVER_URL = st.sidebar.text_input("Server URL", value="http://localhost:8080")
DEMO_MODE = st.sidebar.checkbox("Demo Mode (No Server Required)", value=False)

# Initialize session state
if 'client_simulator' not in st.session_state:
    st.session_state.client_simulator = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []

st.markdown("""
This demo shows how multiple banks can collaboratively train a credit scoring model using federated learning, without sharing raw data.
Enter customer features below to get a credit score prediction from the federated model.
""")

# --- Client Simulator ---
st.sidebar.header("Client Simulator")
if st.sidebar.button("Start Client Simulator"):
    if not DEMO_MODE:
        st.session_state.client_simulator = ClientSimulator(SERVER_URL)
        st.session_state.client_simulator.start()
        st.sidebar.success("Client simulator started!")
    else:
        st.sidebar.warning("Client simulator only works in Real Mode")

if st.sidebar.button("Stop Client Simulator"):
    if st.session_state.client_simulator:
        st.session_state.client_simulator.stop()
        st.session_state.client_simulator = None
        st.sidebar.success("Client simulator stopped!")

# --- Feature Input Form ---
st.header("Enter Customer Features")
with st.form("feature_form"):
    features = []
    cols = st.columns(4)
    for i in range(32):
        with cols[i % 4]:
            val = st.number_input(f"Feature {i+1}", value=0.0, format="%.4f", key=f"f_{i}")
            features.append(val)
    submitted = st.form_submit_button("Predict Credit Score")

# --- Prediction ---
if submitted:
    if DEMO_MODE:
        # Demo mode - simulate prediction
        with st.spinner("Processing prediction..."):
            time.sleep(1)  # Simulate processing time
        
        # Simple demo prediction based on feature values
        demo_prediction = sum(features) / len(features) * 100 + 500  # Scale to credit score range
        st.success(f"Demo Prediction: Credit Score = {demo_prediction:.2f}")
        st.info("💡 This is a demo prediction. In a real federated system, this would come from the trained model.")
        
        # Show what would happen in real mode
        st.markdown("---")
        st.markdown("**What happens in real federated learning:**")
        st.markdown("1. Your features are sent to the federated server")
        st.markdown("2. Server uses the global model (trained by multiple banks)")
        st.markdown("3. Prediction is returned without exposing any bank's data")
        
    else:
        # Real mode - connect to server
        try:
            with st.spinner("Connecting to federated server..."):
                resp = requests.post(f"{SERVER_URL}/predict", json={"features": features}, timeout=10)
            
            if resp.status_code == 200:
                prediction = resp.json().get("prediction")
                st.success(f"Predicted Credit Score: {prediction:.2f}")
                st.info("🎯 This prediction comes from the federated model trained by multiple banks!")
            else:
                st.error(f"Prediction failed: {resp.json().get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error connecting to server: {e}")
            st.info("💡 Try enabling Demo Mode to see the interface without a server.")

# --- Training Progress ---
st.header("Federated Training Progress")

if DEMO_MODE:
    # Demo training progress
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Round", "3/10")
    with col2:
        st.metric("Active Clients", "3")
    with col3:
        st.metric("Model Accuracy", "85.2%")
    with col4:
        st.metric("Training Status", "Active")
    
    st.info("💡 Demo mode showing simulated training progress. In real federated learning, multiple banks would be training collaboratively.")
    
else:
    # Real training progress
    try:
        status = requests.get(f"{SERVER_URL}/training_status", timeout=5)
        if status.status_code == 200:
            data = status.json()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Round", f"{data.get('current_round', 0)}/{data.get('total_rounds', 10)}")
            with col2:
                st.metric("Active Clients", data.get('active_clients', 0))
            with col3:
                st.metric("Clients Ready", data.get('clients_ready', 0))
            with col4:
                st.metric("Training Status", "Active" if data.get('training_active', False) else "Inactive")
            
            # Show training history
            if st.session_state.training_history:
                st.subheader("Training History")
                history_df = st.session_state.training_history
                st.line_chart(history_df.set_index('round')[['active_clients', 'clients_ready']])
        else:
            st.warning("Could not fetch training status.")
    except Exception as e:
        st.warning(f"Could not connect to server for training status: {e}")

# --- Server Health Check ---
if not DEMO_MODE:
    st.header("Server Health")
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=5)
        if health.status_code == 200:
            health_data = health.json()
            st.success(f"✅ Server is healthy")
            st.json(health_data)
        else:
            st.error("❌ Server health check failed")
    except Exception as e:
        st.error(f"❌ Cannot connect to server: {e}")

# --- How it works ---
st.header("How Federated Learning Works")
st.markdown("""
**Traditional ML:** All banks send their data to a central server → Privacy risk ❌

**Federated Learning:** 
1. Each bank keeps their data locally ✅
2. Banks train models on their own data ✅  
3. Only model updates (not data) are shared ✅
4. Server aggregates updates to create global model ✅
5. Global model is distributed back to all banks ✅

**Result:** Collaborative learning without data sharing! 🎯
""")

# --- Client Simulator Status ---
if st.session_state.client_simulator and not DEMO_MODE:
    st.header("Client Simulator Status")
    if st.session_state.client_simulator.is_running:
        st.success("🟢 Client simulator is running and participating in federated learning")
        st.info(f"Client ID: {st.session_state.client_simulator.client_id}")
        st.info(f"Last update: {st.session_state.client_simulator.last_update}")
    else:
        st.warning("🔴 Client simulator is not running")

st.markdown("---")
st.markdown("""
*This is a demonstration of federated learning concepts. For full functionality, run the federated server and clients locally.*
""")

# Client Simulator Class
class ClientSimulator:
    def __init__(self, server_url):
        self.server_url = server_url
        self.client_id = f"web_client_{int(time.time())}"
        self.is_running = False
        self.thread = None
        self.last_update = "Never"
        
    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._run_client, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.is_running = False
        
    def _run_client(self):
        try:
            # Register with server
            client_info = {
                'dataset_size': 100,
                'model_params': 10000,
                'capabilities': ['training', 'inference']
            }
            
            resp = requests.post(f"{self.server_url}/register", 
                               json={'client_id': self.client_id, 'client_info': client_info})
            
            if resp.status_code == 200:
                st.session_state.training_history.append({
                    'round': 0,
                    'active_clients': 1,
                    'clients_ready': 0,
                    'timestamp': datetime.now()
                })
                
                # Simulate client participation
                while self.is_running:
                    try:
                        # Get training status
                        status = requests.get(f"{self.server_url}/training_status")
                        if status.status_code == 200:
                            data = status.json()
                            
                            # Update training history
                            st.session_state.training_history.append({
                                'round': data.get('current_round', 0),
                                'active_clients': data.get('active_clients', 0),
                                'clients_ready': data.get('clients_ready', 0),
                                'timestamp': datetime.now()
                            })
                            
                            # Keep only last 50 entries
                            if len(st.session_state.training_history) > 50:
                                st.session_state.training_history = st.session_state.training_history[-50:]
                        
                        time.sleep(5)  # Check every 5 seconds
                        
                    except Exception as e:
                        print(f"Client simulator error: {e}")
                        time.sleep(10)
                        
        except Exception as e:
            print(f"Failed to start client simulator: {e}")
            self.is_running = False 