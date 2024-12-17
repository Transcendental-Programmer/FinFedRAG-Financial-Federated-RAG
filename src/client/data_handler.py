"""data_handler.py module."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class FinancialDataHandler:
    def __init__(self, config: Dict):
        """Initialize the data handler with configuration."""
        self.batch_size = config['data']['batch_size']
        self.shuffle_buffer = config['data']['shuffle_buffer']
        self.prefetch_buffer = config['data']['prefetch_buffer']
        self.scaler = StandardScaler()
        
    def simulate_financial_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic financial data for testing."""
        np.random.seed(42)
        
        data = {
            'transaction_amount': np.random.lognormal(mean=4.0, sigma=1.0, size=num_samples),
            'account_balance': np.random.normal(loc=10000, scale=5000, size=num_samples),
            'transaction_frequency': np.random.poisson(lam=5, size=num_samples),
            'credit_score': np.random.normal(loc=700, scale=50, size=num_samples).clip(300, 850),
            'days_since_last_transaction': np.random.exponential(scale=7, size=num_samples)
        }
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, data: pd.DataFrame) -> tf.data.Dataset:
        """Preprocess the data and convert to TensorFlow dataset."""
        # Standardize the features
        scaled_data = self.scaler.fit_transform(data)
        
        # Convert to TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices(scaled_data)
        
        # Apply dataset transformations
        dataset = dataset.shuffle(self.shuffle_buffer)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_buffer)
        
        return dataset
    
    def get_client_data(self) -> Tuple[tf.data.Dataset, StandardScaler]:
        """Get preprocessed client data and scaler."""
        # Simulate client data
        raw_data = self.simulate_financial_data()
        
        # Preprocess data
        dataset = self.preprocess_data(raw_data)
        
        return dataset, self.scaler

