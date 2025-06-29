# 🚀 Hugging Face Spaces Deployment Guide

## Quick Deploy to HF Spaces (5 minutes)

### Step 1: Prepare Your Repository

Your repository should have these files in the root:
- ✅ `app.py` - Streamlit application
- ✅ `requirements.txt` - Minimal dependencies (streamlit, requests, numpy)
- ✅ `README.md` - With HF Spaces config at the top

### Step 2: Create HF Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Fill in the details:
   - **Owner**: `ArchCoder`
   - **Space name**: `federated-credit-scoring`
   - **Short description**: `Federated Learning Credit Scoring Demo with Privacy-Preserving Model Training`
   - **License**: `MIT`
   - **Space SDK**: `Streamlit` ⚠️ **NOT Docker**
   - **Space hardware**: `Free`
   - **Visibility**: `Public`

### Step 3: Upload Files

**Option A: Direct Upload**
1. Click "Create Space"
2. Upload these files:
   - `app.py`
   - `requirements.txt`

**Option B: Connect GitHub (Recommended)**
1. In Space Settings → "Repository"
2. Connect your GitHub repo
3. Enable "Auto-deploy on push"

### Step 4: Wait for Build

- HF Spaces will install dependencies
- Build your Streamlit app
- Takes 2-3 minutes

### Step 5: Access Your App

Your app will be live at:
```
https://huggingface.co/spaces/ArchCoder/federated-credit-scoring
```

## 🎯 What Users Will See

- **Demo Mode**: Works immediately (no server needed)
- **Interactive Interface**: Enter features, get predictions
- **Educational Content**: Learn about federated learning
- **Professional UI**: Clean, modern design

## 🔧 Troubleshooting

**"Missing app file" error:**
- Ensure `app.py` is in the root directory
- Check that SDK is set to `streamlit` (not docker)

**Build fails:**
- Check `requirements.txt` has minimal dependencies
- Ensure no heavy packages (tensorflow, etc.) in requirements.txt

**App doesn't load:**
- Check logs in HF Spaces
- Verify app.py has no syntax errors

## 📁 Required Files

**`app.py`** (root level):
```python
import streamlit as st
import requests
import numpy as np
import time

st.set_page_config(page_title="Federated Credit Scoring Demo", layout="centered")
# ... rest of your app code
```

**`requirements.txt`** (root level):
```
streamlit
requests
numpy
```

**`README.md`** (with HF config at top):
```yaml
---
title: Federated Credit Scoring
emoji: 🚀
colorFrom: red
colorTo: red
sdk: streamlit
app_port: 8501
tags:
- streamlit
- federated-learning
- machine-learning
- privacy
pinned: false
short_description: Federated Learning Credit Scoring Demo with Privacy-Preserving Model Training
license: mit
---
```

## 🎉 Success!

After deployment, you'll have:
- ✅ Live web app accessible to anyone
- ✅ No server setup required
- ✅ Professional presentation of your project
- ✅ Educational value for visitors

**Your federated learning demo will be live and working!** 🚀 

# FinFedRAG Deployment Guide

## Overview

This project implements a federated learning framework with RAG capabilities for financial data. The system can be deployed using Docker Compose for local development or Kubernetes for production environments.

## Debugging and Monitoring

### Enhanced Debugging Features

The web application now includes comprehensive debugging capabilities:

1. **Debug Information Panel**: Located in the sidebar, shows:
   - Real-time server health status
   - Recent debug messages and logs
   - Connection error details
   - Client simulator status

2. **Detailed Error Logging**: All operations are logged with:
   - Connection attempts and failures
   - Server response details
   - Timeout and network error handling
   - Client registration and training status updates

3. **Real-time Status Monitoring**: 
   - Server health checks
   - Training progress tracking
   - Client connection status
   - Error message history

### Using the Debug Features

1. **Enable Debug Mode**: Uncheck "Demo Mode" in the sidebar
2. **View Debug Information**: Expand the "Debug Information" section in the sidebar
3. **Monitor Logs**: Check the "Recent Logs" section for real-time updates
4. **Clear Logs**: Use the "Clear Debug Logs" button to reset the log history

## Local Development Setup

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Required Python packages (see requirements.txt)

### Quick Start

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd FinFedRAG-Financial-Federated-RAG
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Start the Federated Server**:
   ```bash
   python src/main.py --mode server
   ```

3. **Start Multiple Clients** (in separate terminals):
   ```bash
   python src/main.py --mode client --client-id client1
   python src/main.py --mode client --client-id client2
   python src/main.py --mode client --client-id client3
   ```

4. **Run the Web Application**:
   ```bash
   streamlit run app.py
   ```

### Docker Compose Deployment

For containerized deployment:

```bash
cd docker
docker-compose up --build
```

This will start:
- 1 federated server on port 8000
- 3 federated clients
- All services connected via Docker network

## Kubernetes Deployment

### Architecture Overview

The Kubernetes setup provides a production-ready deployment with:

- **Server Deployment**: Single federated learning server
- **Client Deployment**: Multiple federated learning clients (3 replicas)
- **Service Layer**: Internal service discovery
- **ConfigMaps**: Configuration management
- **Namespace Isolation**: Dedicated `federated-learning` namespace

### Components

#### 1. Server Deployment (`kubernetes/deployments/server.yaml`)
```yaml
- Replicas: 1 (single server instance)
- Port: 8000 (internal)
- Config: Mounted from ConfigMap
- Image: fl-server:latest
```

#### 2. Client Deployment (`kubernetes/deployments/client.yaml`)
```yaml
- Replicas: 3 (multiple client instances)
- Environment: SERVER_HOST=fl-server-service
- Config: Mounted from ConfigMap
- Image: fl-client:latest
```

#### 3. Service (`kubernetes/services/service.yaml`)
```yaml
- Type: ClusterIP (internal communication)
- Port: 8000
- Selector: app=fl-server
```

### Deployment Steps

1. **Build Docker Images**:
   ```bash
   docker build -f docker/Dockerfile.server -t fl-server:latest .
   docker build -f docker/Dockerfile.client -t fl-client:latest .
   ```

2. **Create Namespace**:
   ```bash
   kubectl create namespace federated-learning
   ```

3. **Create ConfigMaps**:
   ```bash
   kubectl create configmap server-config --from-file=config/server_config.yaml -n federated-learning
   kubectl create configmap client-config --from-file=config/client_config.yaml -n federated-learning
   ```

4. **Deploy Services**:
   ```bash
   kubectl apply -f kubernetes/services/service.yaml
   kubectl apply -f kubernetes/deployments/server.yaml
   kubectl apply -f kubernetes/deployments/client.yaml
   ```

5. **Verify Deployment**:
   ```bash
   kubectl get pods -n federated-learning
   kubectl get services -n federated-learning
   ```

### Accessing the Application

#### Option 1: Port Forwarding
```bash
kubectl port-forward service/fl-server-service 8080:8000 -n federated-learning
```

#### Option 2: Load Balancer
Modify the service to use LoadBalancer type:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: fl-server-service
  namespace: federated-learning
spec:
  type: LoadBalancer  # Changed from ClusterIP
  selector:
    app: fl-server
  ports:
  - port: 8080
    targetPort: 8000
```

#### Option 3: Ingress Controller
Create an ingress resource for external access:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fl-ingress
  namespace: federated-learning
spec:
  rules:
  - host: fl.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: fl-server-service
            port:
              number: 8000
```

### Monitoring and Debugging in Kubernetes

1. **View Pod Logs**:
   ```bash
   kubectl logs -f deployment/fl-server -n federated-learning
   kubectl logs -f deployment/fl-client -n federated-learning
   ```

2. **Check Pod Status**:
   ```bash
   kubectl describe pods -n federated-learning
   ```

3. **Access Pod Shell**:
   ```bash
   kubectl exec -it <pod-name> -n federated-learning -- /bin/bash
   ```

4. **Monitor Resource Usage**:
   ```bash
   kubectl top pods -n federated-learning
   ```

## Troubleshooting

### Common Issues

1. **Connection Refused Errors**:
   - Check if server is running: `kubectl get pods -n federated-learning`
   - Verify service exists: `kubectl get services -n federated-learning`
   - Check pod logs for startup errors

2. **Client Registration Failures**:
   - Ensure server is healthy before starting clients
   - Check network connectivity between pods
   - Verify ConfigMap configurations

3. **Training Status Issues**:
   - Monitor server logs for aggregation errors
   - Check client participation in training rounds
   - Verify model update sharing

### Debug Commands

```bash
# Check all resources in namespace
kubectl get all -n federated-learning

# View detailed pod information
kubectl describe pod <pod-name> -n federated-learning

# Check service endpoints
kubectl get endpoints -n federated-learning

# View ConfigMap contents
kubectl get configmap server-config -n federated-learning -o yaml
```

## Production Considerations

1. **Resource Limits**: Add resource requests and limits to deployments
2. **Health Checks**: Implement liveness and readiness probes
3. **Secrets Management**: Use Kubernetes secrets for sensitive data
4. **Persistent Storage**: Add persistent volumes for model storage
5. **Monitoring**: Integrate with Prometheus/Grafana for metrics
6. **Logging**: Use centralized logging (ELK stack, Fluentd)

## Scaling

### Horizontal Pod Autoscaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fl-client-hpa
  namespace: federated-learning
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fl-client
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

This deployment guide provides comprehensive information for both local development and production Kubernetes deployment, with enhanced debugging capabilities for better monitoring and troubleshooting. 