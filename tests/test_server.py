"""test_server.py module."""

import pytest
import numpy as np
import tensorflow as tf
from src.server.coordinator import FederatedCoordinator
from src.server.aggregator import FederatedAggregator
import yaml

@pytest.fixture
def server_config():
    with open('config/server_config.yaml', 'r') as f:
        return yaml.safe_load(f)['server']

@pytest.fixture
def coordinator(server_config):
    return FederatedCoordinator(server_config)

@pytest.fixture
def aggregator(server_config):
    return FederatedAggregator(server_config)

def test_coordinator_initialization(coordinator, server_config):
    assert coordinator.min_clients == server_config['federated']['min_clients']
    assert coordinator.rounds == server_config['federated']['rounds']
    assert coordinator.sample_fraction == server_config['federated']['sample_fraction']

def test_client_registration(coordinator):
    client_id = 1
    client_size = 1000
    coordinator.register_client(client_id, client_size)
    assert client_id in coordinator.clients
    assert coordinator.clients[client_id]['size'] == client_size

def test_client_selection(coordinator):
    # Register multiple clients
    for i in range(5):
        coordinator.register_client(i, 1000)
    
    selected_clients = coordinator.select_clients()
    assert len(selected_clients) >= coordinator.min_clients
    assert all(client_id in coordinator.clients for client_id in selected_clients)

def test_weight_aggregation(aggregator):
    # Create mock client updates
    client_updates = [
        {
            'client_id': i,
            'weights': [np.random.randn(10, 10) for _ in range(3)],
            'metrics': {'loss': 0.5}
        }
        for i in range(3)
    ]
    
    aggregated_weights = aggregator.compute_metrics(client_updates)
    assert isinstance(aggregated_weights, dict)

