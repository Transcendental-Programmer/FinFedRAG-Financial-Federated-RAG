import streamlit as st
import numpy as np
import time
import threading
import json
import logging
from datetime import datetime
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simulated Federated Learning System
class SimulatedFederatedSystem:
    def __init__(self):
        self.clients = {}
        self.current_round = 0
        self.total_rounds = 10
        self.training_active = False
        self.global_model_accuracy = 0.75
        self.active_clients = 0
        self.clients_ready = 0
        self.model_weights = [random.random() for _ in range(100)]
        # Kubernetes simulation data
        self.k8s_pods = {
            "fl-server": {"status": "Running", "cpu": "20%", "memory": "256MB"},
            "fl-client-1": {"status": "Running", "cpu": "15%", "memory": "128MB"},
            "fl-client-2": {"status": "Running", "cpu": "15%", "memory": "128MB"},
            "fl-client-3": {"status": "Pending", "cpu": "0%", "memory": "0MB"}
        }
        
    def register_client(self, client_id, client_info):
        self.clients[client_id] = {
            'info': client_info,
            'registered_at': datetime.now(),
            'last_seen': datetime.now(),
            'status': 'active'
        }
        self.active_clients = len(self.clients)
        # Update Kubernetes simulation
        if self.active_clients <= 3:
            self.k8s_pods[f"fl-client-{self.active_clients}"]["status"] = "Running"
            self.k8s_pods[f"fl-client-{self.active_clients}"]["cpu"] = "15%"
            self.k8s_pods[f"fl-client-{self.active_clients}"]["memory"] = "128MB"
        return True
        
    def get_training_status(self):
        return {
            'current_round': self.current_round,
            'total_rounds': self.total_rounds,
            'training_active': self.training_active,
            'active_clients': self.active_clients,
            'clients_ready': self.clients_ready,
            'global_accuracy': self.global_model_accuracy
        }
        
    def start_training(self):
        self.training_active = True
        self.current_round = 1
        # Update Kubernetes simulation
        self.k8s_pods["fl-server"]["cpu"] = "35%"
        for i in range(1, min(4, self.active_clients + 1)):
            if self.k8s_pods[f"fl-client-{i}"]["status"] == "Running":
                self.k8s_pods[f"fl-client-{i}"]["cpu"] = "25%"
        
    def simulate_training_round(self):
        if self.training_active and self.current_round < self.total_rounds:
            # Simulate training progress
            self.current_round += 1
            self.global_model_accuracy += random.uniform(0.01, 0.03)
            self.global_model_accuracy = min(self.global_model_accuracy, 0.95)
            self.clients_ready = random.randint(2, min(5, self.active_clients))
            # Update Kubernetes simulation
            self.k8s_pods["fl-server"]["cpu"] = f"{30 + random.randint(5, 15)}%"
            self.k8s_pods["fl-server"]["memory"] = f"{256 + self.current_round * 10}MB"
            for i in range(1, min(4, self.active_clients + 1)):
                if self.k8s_pods[f"fl-client-{i}"]["status"] == "Running":
                    self.k8s_pods[f"fl-client-{i}"]["cpu"] = f"{20 + random.randint(5, 15)}%"
                    self.k8s_pods[f"fl-client-{i}"]["memory"] = f"{128 + self.current_round * 5}MB"
            
    def predict(self, features):
        # Simulate model prediction
        if len(features) != 32:
            return 500.0
        
        # Simple weighted sum with some randomness
        base_score = sum(f * w for f, w in zip(features, self.model_weights[:32]))
        noise = random.uniform(-50, 50)
        credit_score = max(300, min(850, base_score * 100 + 500 + noise))
        
        # Update Kubernetes simulation for prediction
        self.k8s_pods["fl-server"]["cpu"] = f"{30 + random.randint(5, 10)}%"
        
        return credit_score

# Global simulated system
if 'federated_system' not in st.session_state:
    st.session_state.federated_system = SimulatedFederatedSystem()

# Client Simulator Class
class ClientSimulator:
    def __init__(self, system):
        self.system = system
        self.client_id = f"web_client_{int(time.time())}"
        self.is_running = False
        self.thread = None
        self.last_error = None
        
    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._run_client, daemon=True)
        self.thread.start()
        logger.info(f"Client simulator started")
        
    def stop(self):
        self.is_running = False
        logger.info("Client simulator stopped")
        
    def _run_client(self):
        try:
            # Register with simulated system
            client_info = {
                'dataset_size': 100,
                'model_params': 10000,
                'capabilities': ['training', 'inference']
            }
            
            success = self.system.register_client(self.client_id, client_info)
            
            if success:
                logger.info(f"Successfully registered client {self.client_id}")
                
                # Simulate training participation
                while self.is_running:
                    try:
                        # Simulate training round
                        if self.system.training_active:
                            time.sleep(3)  # Simulate training time
                        else:
                            time.sleep(5)  # Wait for training to start
                            
                    except Exception as e:
                        logger.error(f"Error in client simulator: {e}")
                        self.last_error = f"Error: {e}"
                        time.sleep(10)
                        
        except Exception as e:
            logger.error(f"Failed to start client simulator: {e}")
            self.last_error = f"Failed to start: {e}"
            self.is_running = False

st.set_page_config(page_title="Federated Credit Scoring Demo", layout="centered")
st.title("Federated Credit Scoring Demo")

# Sidebar configuration
st.sidebar.header("Configuration")
DEMO_MODE = st.sidebar.checkbox("Demo Mode", value=True, disabled=True)

# Initialize session state
if 'client_simulator' not in st.session_state:
    st.session_state.client_simulator = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'debug_messages' not in st.session_state:
    st.session_state.debug_messages = []
if 'kubernetes_view' not in st.session_state:
    st.session_state.kubernetes_view = False

# System Status in sidebar
with st.sidebar.expander("System Status"):
    system = st.session_state.federated_system
    st.success("✅ Federated System Running")
    st.info(f"Active Clients: {system.active_clients}")
    st.info(f"Current Round: {system.current_round}/{system.total_rounds}")
    st.info(f"Global Accuracy: {system.global_model_accuracy:.2%}")

# Debug section in sidebar
with st.sidebar.expander("Debug Information"):
    st.write("**Recent Logs:**")
    if st.session_state.debug_messages:
        for msg in st.session_state.debug_messages[-5:]:
            st.text(msg)
    else:
        st.text("No debug messages yet")
    
    if st.button("Clear Debug Logs"):
        st.session_state.debug_messages = []

# Kubernetes monitoring in sidebar
with st.sidebar.expander("Kubernetes Monitoring"):
    st.write("**Simulated Kubernetes Cluster**")
    
    if st.button("Toggle Kubernetes View"):
        st.session_state.kubernetes_view = not st.session_state.kubernetes_view
    
    st.success("✅ Kubernetes Cluster Running")
    st.info(f"Namespace: federated-learning")
    st.info(f"Pods Running: {system.active_clients + 1}")  # Clients + Server

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
    try:
        st.session_state.client_simulator = ClientSimulator(st.session_state.federated_system)
        st.session_state.client_simulator.start()
        st.sidebar.success("Client started!")
        st.session_state.debug_messages.append(f"{datetime.now()}: Client simulator started")
    except Exception as e:
        st.sidebar.error(f"Failed to start client: {e}")
        st.session_state.debug_messages.append(f"{datetime.now()}: Failed to start client - {e}")

if st.sidebar.button("Stop Client"):
    if st.session_state.client_simulator:
        st.session_state.client_simulator.stop()
        st.session_state.client_simulator = None
        st.sidebar.success("Client stopped!")
        st.session_state.debug_messages.append(f"{datetime.now()}: Client simulator stopped")

# Training Control
st.sidebar.header("Training Control")
if st.sidebar.button("Start Training"):
    st.session_state.federated_system.start_training()
    st.sidebar.success("Training started!")
    st.session_state.debug_messages.append(f"{datetime.now()}: Training started")

if st.sidebar.button("Simulate Round"):
    st.session_state.federated_system.simulate_training_round()
    st.sidebar.success("Round completed!")
    st.session_state.debug_messages.append(f"{datetime.now()}: Training round completed")

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
    with st.spinner("Processing..."):
        time.sleep(1)
    
    # Use simulated federated system for prediction
    prediction = st.session_state.federated_system.predict(features)
    st.success(f"Predicted Credit Score: {prediction:.2f}")
    st.session_state.debug_messages.append(f"{datetime.now()}: Prediction: {prediction:.2f}")
    
    # Show prediction explanation
    st.info("""
    **This prediction comes from the federated model trained by multiple banks!**
    
    - Model trained on data from multiple financial institutions
    - No raw data was shared between banks
    - Only model updates were aggregated
    - Privacy-preserving collaborative learning
    """)

# Training progress
st.header("Training Progress")
system = st.session_state.federated_system
status = system.get_training_status()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Round", f"{status['current_round']}/{status['total_rounds']}")
with col2:
    st.metric("Clients", status['active_clients'])
with col3:
    st.metric("Accuracy", f"{status['global_accuracy']:.1%}")
with col4:
    st.metric("Status", "Active" if status['training_active'] else "Inactive")

# Training visualization
if status['training_active']:
    st.subheader("Training Progress Visualization")
    
    # Simulate training history
    if 'training_history' not in st.session_state:
        st.session_state.training_history = []
    
    # Add current status to history
    st.session_state.training_history.append({
        'round': status['current_round'],
        'accuracy': status['global_accuracy'],
        'clients': status['active_clients'],
        'timestamp': datetime.now()
    })
    
    # Keep only last 20 entries
    if len(st.session_state.training_history) > 20:
        st.session_state.training_history = st.session_state.training_history[-20:]
    
    # Create visualization
    if len(st.session_state.training_history) > 1:
        import pandas as pd
        df = pd.DataFrame(st.session_state.training_history)
        
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(df.set_index('round')['accuracy'])
            st.caption("Model Accuracy Over Rounds")
        
        with col2:
            st.line_chart(df.set_index('round')['clients'])
            st.caption("Active Clients Over Rounds")

# Client status in sidebar
if st.session_state.client_simulator:
    st.sidebar.header("Client Status")
    if st.session_state.client_simulator.is_running:
        st.sidebar.success("Connected")
        st.sidebar.info(f"ID: {st.session_state.client_simulator.client_id}")
        if st.session_state.client_simulator.last_error:
            st.sidebar.error(f"Last Error: {st.session_state.client_simulator.last_error}")
    else:
        st.sidebar.warning("Disconnected")

# Kubernetes visualization (if enabled)
if st.session_state.kubernetes_view:
    st.header("Kubernetes Cluster Visualization")
    
    # Create a simple visualization of the Kubernetes cluster
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                Kubernetes Cluster (Simulated)               │
    │                                                             │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
    │  │ fl-server   │  │ fl-client-1 │  │ fl-client-2 │         │
    │  │ Pod         │  │ Pod         │  │ Pod         │         │
    │  │ Running ✅  │  │ Running ✅  │  │ Running ✅  │         │
    │  └─────────────┘  └─────────────┘  └─────────────┘         │
    │                                                             │
    │  ┌─────────────────────────────────────────────────┐       │
    │  │ fl-server-service                               │       │
    │  │ Type: ClusterIP                                 │       │
    │  │ Port: 8080:8000                                 │       │
    │  └─────────────────────────────────────────────────┘       │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    ```
    """)
    
    # Show simulated Kubernetes metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pod Status")
        pod_status = {}
        for pod_name, pod_data in system.k8s_pods.items():
            pod_status[pod_name] = pod_data["status"]
        st.json(pod_status)
    
    with col2:
        st.subheader("Resource Usage")
        resource_usage = {
            pod_name: {
                "CPU": pod_data["cpu"],
                "Memory": pod_data["memory"]
            } for pod_name, pod_data in system.k8s_pods.items()
        }
        st.json(resource_usage)
    
    # Kubernetes commands
    st.subheader("Kubernetes Commands")
    with st.expander("Available Commands"):
        st.code("""
# View all pods
kubectl get pods -n federated-learning

# View all services
kubectl get services -n federated-learning

# View pod logs
kubectl logs -f deployment/fl-server -n federated-learning

# Scale deployment
kubectl scale deployment/fl-client --replicas=5 -n federated-learning
        """)
    
    # Kubernetes deployment visualization
    st.subheader("Deployment Architecture")
    st.markdown("""
    ```mermaid
    graph TD
        A[Kubernetes Cluster] --> B[federated-learning Namespace]
        B --> C[fl-server Deployment]
        B --> D[fl-client Deployment]
        B --> E[fl-server-service Service]
        C --> F[fl-server Pod]
        D --> G[fl-client-1 Pod]
        D --> H[fl-client-2 Pod]
        D --> I[fl-client-3 Pod]
        E --> F
    ```
    """)

# Auto-refresh for training simulation
if st.session_state.federated_system.training_active:
    time.sleep(2)
    st.rerun() 