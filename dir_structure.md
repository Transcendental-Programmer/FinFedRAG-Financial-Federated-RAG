# Directory Structure

```
federated-rag-financial/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── docker/
│   ├── Dockerfile.client
│   ├── Dockerfile.server
│   └── docker-compose.yml
├── kubernetes/
│   ├── deployments/
│   │   ├── client.yaml
│   │   ├── server.yaml
│   │   └── rag.yaml
│   └── services/
│       └── service.yaml
├── src/
│   ├── __init__.py
│   ├── client/
│   │   ├── __init__.py
│   │   ├── data_handler.py
│   │   └── model.py
│   ├── server/
│   │   ├── __init__.py
│   │   ├── aggregator.py
│   │   └── coordinator.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   └── generator.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vae.py
│   │   └── gan.py
│   └── utils/
│       ├── __init__.py
│       ├── privacy.py
│       └── metrics.py
├── tests/
│   ├── __init__.py
│   ├── test_client.py
│   ├── test_server.py
│   └── test_rag.py
├── docs/
│   ├── api/
│   ├── guides/
│   └── index.md
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_evaluation.ipynb
├── config/
│   ├── client_config.yaml
│   └── server_config.yaml
├── requirements.txt
├── setup.py
└── README.md
```