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

# Federated Learning for Privacy-Preserving Financial Data Generation with RAG Integration

This project implements a federated learning framework combined with a Retrieval-Augmented Generation (RAG) system to generate privacy-preserving synthetic financial data.

## Features

- Federated Learning using TensorFlow
- Privacy-preserving data generation using VAE/GAN
- RAG integration for enhanced data quality
- Secure Multi-Party Computation (SMPC)
- Differential Privacy implementation
- Kubernetes-based deployment
- Comprehensive monitoring and logging
- **NEW: Interactive Web Demo** - Try it out without setup!

## Quick Demo (No Installation Required)

🚀 **Live Demo**: [Hugging Face Spaces](https://huggingface.co/spaces/ArchCoder/federated-credit-scoring)

The web demo allows you to:
- Enter customer features and get credit score predictions
- See how federated learning works
- Understand privacy-preserving ML concepts

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Federated Credit Scoring Demo (with Web App)

This project includes a demo where multiple banks (clients) collaboratively train a credit scoring model using federated learning. A Streamlit web app allows you to enter customer features and get a credit score prediction from the federated model.

### Quick Start

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Start the Federated Server**

```bash
python -m src.main --mode server --config config/server_config.yaml
```

3. **Start at least two Clients (in separate terminals)**

```bash
python -m src.main --mode client --config config/client_config.yaml
```

4. **Run the Web App**

```bash
streamlit run webapp/streamlit_app.py
```

5. **Use the Web App**
- Enter 32 features (dummy values are fine for demo)
- Click "Predict Credit Score" to get a prediction from the federated model
- View training progress in the app
- Toggle between Demo Mode (no server required) and Real Mode (connects to server)

*For best results, keep the server and at least two clients running in parallel.*

## Project Structure

```
FinFedRAG-Financial-Federated-RAG/
├── src/
│   ├── api/           # REST API for server and client communication
│   ├── client/        # Federated learning client implementation
│   ├── server/        # Federated learning server and coordinator
│   ├── rag/           # Retrieval-Augmented Generation components
│   ├── models/        # VAE/GAN models for data generation
│   └── utils/         # Privacy, metrics, and utility functions
├── webapp/            # Streamlit web application
├── config/            # Configuration files
├── tests/             # Unit and integration tests
├── docker/            # Docker configurations
├── kubernetes/        # Kubernetes deployment files
└── app.py             # Root app.py for Hugging Face Spaces deployment
```

## License

MIT

## Contributing

Please read our contributing guidelines before submitting pull requests.

---

**Demo URL**: https://huggingface.co/spaces/ArchCoder/federated-credit-scoring
