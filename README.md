# Federated Learning for Privacy-Preserving Financial Data Generation with RAG Integration

This project implements a federated learning framework combined with a Retrieval-Augmented Generation (RAG) system to generate privacy-preserving synthetic financial data.

## Features

- Federated Learning using TensorFlow Federated
- Privacy-preserving data generation using VAE/GAN
- RAG integration for enhanced data quality
- Secure Multi-Party Computation (SMPC)
- Differential Privacy implementation
- Kubernetes-based deployment
- Comprehensive monitoring and logging

## Installation

```bash
pip install -r requirements.txt
```

## Usage


## Project Structure


## License

MIT

## Contributing


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

*For best results, keep the server and at least two clients running in parallel.*

---
