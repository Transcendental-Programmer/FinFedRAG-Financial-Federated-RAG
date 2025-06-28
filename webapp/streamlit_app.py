import streamlit as st
import requests
import numpy as np

st.set_page_config(page_title="Federated Credit Scoring Demo", layout="centered")
st.title("Federated Credit Scoring Demo (Federated Learning)")

SERVER_URL = st.sidebar.text_input("Server URL", value="http://localhost:8080")

st.markdown("""
This demo shows how multiple banks can collaboratively train a credit scoring model using federated learning, without sharing raw data.
Enter customer features below to get a credit score prediction from the federated model.
""")

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
prediction = None
if submitted:
    try:
        resp = requests.post(f"{SERVER_URL}/predict", json={"features": features}, timeout=10)
        if resp.status_code == 200:
            prediction = resp.json().get("prediction")
            st.success(f"Predicted Credit Score: {prediction:.2f}")
        else:
            st.error(f"Prediction failed: {resp.json().get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error connecting to server: {e}")

# --- Training Progress ---
st.header("Federated Training Progress")
try:
    status = requests.get(f"{SERVER_URL}/training_status", timeout=5)
    if status.status_code == 200:
        data = status.json()
        st.write(f"Current Round: {data.get('current_round', 0)} / {data.get('total_rounds', 10)}")
        st.write(f"Active Clients: {data.get('active_clients', 0)}")
        st.write(f"Clients Ready: {data.get('clients_ready', 0)}")
        st.write(f"Training Active: {data.get('training_active', False)}")
    else:
        st.warning("Could not fetch training status.")
except Exception as e:
    st.warning(f"Could not connect to server for training status: {e}")

st.markdown("---")
st.markdown("""
*This is a demo. All data is synthetic. For best results, run the federated server and at least two clients in parallel.*
""") 