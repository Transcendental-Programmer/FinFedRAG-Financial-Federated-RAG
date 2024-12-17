"""test_client.py module."""

import pytest
import tensorflow as tf
import yaml
from src.client.data_handler import FinancialDataHandler
from src.client.model import FederatedClient

@pytest.fixture
def config():
    """Load test configuration."""
    with open('config/client_config.yaml', 'r') as f:
        return yaml.safe_load(f)['client']

def test_data_handler(config):
    """Test data handler functionality."""
    handler = FinancialDataHandler(config)
    
    # Test data simulation
    data = handler.simulate_financial_data(num_samples=100)
    assert len(data) == 100
    assert all(col in data.columns for col in [
        'transaction_amount',
        'account_balance',
        'transaction_frequency',
        'credit_score',
        'days_since_last_transaction'
    ])
    
    # Test preprocessing
    dataset, scaler = handler.get_client_data()
    assert isinstance(dataset, tf.data.Dataset)
    
def test_federated_client(config):
    """Test federated client functionality."""
    client = FederatedClient(config)
    
    # Test model building
    assert isinstance(client.model, tf.keras.Model)
    
    # Test local training
    handler = FinancialDataHandler(config)
    dataset, _ = handler.get_client_data()
    
    training_result = client.train_local_model(dataset, epochs=1)
    assert 'client_id' in training_result
    assert 'weights' in training_result
    assert 'metrics' in training_result

