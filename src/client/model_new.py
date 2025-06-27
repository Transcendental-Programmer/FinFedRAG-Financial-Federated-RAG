"""model.py module."""

from typing import Dict, List, Optional, Tuple
import tensorflow as tf
import numpy as np
import logging
import time
from ..api.client import FederatedHTTPClient
from .data_handler import FinancialDataHandler

class FederatedClient:
    def __init__(self, client_id: str, config: Dict, server_url: Optional[str] = None):
        """Initialize the federated client."""
        self.client_id = str(client_id)
        self.config = config.get('client', {})
        self.model = self._build_model()
        self.data_handler = FinancialDataHandler(config)
        
        # HTTP client for server communication
        self.server_url = server_url or self.config.get('server_url', 'http://localhost:8080')
        self.http_client = FederatedHTTPClient(self.server_url, self.client_id)
        
        # Training state
        self.registered = False
        self.current_round = 0
        
    def start(self):
        """Start the federated client process with server communication."""
        logger = logging.getLogger(__name__)
        logger.info(f"Client {self.client_id} starting...")
        
        try:
            # Wait for server to be available
            if not self.http_client.wait_for_server():
                raise ConnectionError(f"Cannot connect to server at {self.server_url}")
            
            # Register with server
            self._register_with_server()
            
            # Main federated learning loop
            self._federated_learning_loop()
            
        except Exception as e:
            logger.error(f"Error during client execution: {str(e)}")
            raise
        finally:
            self.http_client.close()
    
    def _register_with_server(self):
        """Register this client with the federated server"""
        logger = logging.getLogger(__name__)
        
        try:
            # Generate local data to get client info
            X, y = self._generate_dummy_data()
            
            client_info = {
                'dataset_size': len(X),
                'model_params': self.model.count_params(),
                'capabilities': ['training', 'inference']
            }
            
            response = self.http_client.register(client_info)
            self.registered = True
            
            logger.info(f"Successfully registered with server")
            logger.info(f"Dataset size: {client_info['dataset_size']}")
            logger.info(f"Model parameters: {client_info['model_params']:,}")
            
        except Exception as e:
            logger.error(f"Failed to register with server: {str(e)}")
            raise
    
    def _federated_learning_loop(self):
        """Main federated learning loop"""
        logger = logging.getLogger(__name__)
        
        while True:
            try:
                # Get training status from server
                status = self.http_client.get_training_status()
                
                if not status.get('training_active', True):
                    logger.info("Training completed on server")
                    break
                
                server_round = status.get('current_round', 0)
                
                if server_round > self.current_round:
                    self._participate_in_round(server_round)
                    self.current_round = server_round
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in federated learning loop: {str(e)}")
                time.sleep(10)  # Wait longer on error
    
    def _participate_in_round(self, round_num: int):
        """Participate in a federated learning round"""
        logger = logging.getLogger(__name__)
        logger.info(f"Participating in round {round_num}")
        
        try:
            # Get global model from server
            model_response = self.http_client.get_global_model()
            global_weights = model_response.get('model_weights')
            
            if global_weights:
                self.set_weights(global_weights)
                logger.info("Updated local model with global weights")
            
            # Generate/load local data
            X, y = self._generate_dummy_data()
            logger.info(f"Training on {len(X)} samples")
            
            # Train locally
            history = self.train_local((X, y))
            
            # Prepare metrics
            metrics = {
                'dataset_size': len(X),
                'final_loss': history['loss'][-1] if history['loss'] else 0.0,
                'epochs_trained': len(history['loss']),
                'round': round_num
            }
            
            # Submit update to server
            local_weights = self.get_weights()
            self.http_client.submit_model_update(local_weights, metrics)
            
            logger.info(f"Round {round_num} completed - Final loss: {metrics['final_loss']:.4f}")
            
        except Exception as e:
            logger.error(f"Error in round {round_num}: {str(e)}")
            raise
        
    def _generate_dummy_data(self):
        """Generate dummy data for testing."""
        try:
            # Try to use the data handler for more realistic data
            return self.data_handler.generate_synthetic_data(100)
        except Exception:
            # Fallback to simple dummy data
            num_samples = 100
            input_dim = 32  # Match with model's input dimension
            
            # Generate input data
            X = tf.random.normal((num_samples, input_dim))
            # Generate target data (for this example, we'll predict the sum of inputs)
            y = tf.reduce_sum(X, axis=1, keepdims=True)
            
            return X.numpy(), y.numpy()
        
    def _build_model(self):
        """Build the initial model architecture."""
        input_dim = 32  # Match with data generation
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)  # Output layer for regression
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.get('training', {}).get('learning_rate', 0.001)
            ),
            loss='mse'
        )
        return model
        
    def train_local(self, data):
        """Train the model on local data."""
        logger = logging.getLogger(__name__)
        X, y = data
        
        # Ensure data is in the right format
        if isinstance(X, np.ndarray):
            X = tf.convert_to_tensor(X, dtype=tf.float32)
        if isinstance(y, np.ndarray):
            y = tf.convert_to_tensor(y, dtype=tf.float32)
        
        # Log training parameters
        logger.info(f"Training Parameters:")
        logger.info(f"Input shape: {X.shape}")
        logger.info(f"Output shape: {y.shape}")
        logger.info(f"Batch size: {self.config.get('training', {}).get('batch_size', 32)}")
        logger.info(f"Epochs: {self.config.get('training', {}).get('local_epochs', 5)}")
        
        class LogCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logger.debug(f"Epoch {epoch + 1} - loss: {logs['loss']:.4f}")
        
        # Train the model
        history = self.model.fit(
            X, y,
            batch_size=self.config.get('training', {}).get('batch_size', 32),
            epochs=self.config.get('training', {}).get('local_epochs', 3),
            verbose=0,
            callbacks=[LogCallback()]
        )
        return history.history
        
    def get_weights(self) -> List:
        """Get the model weights."""
        weights = self.model.get_weights()
        # Convert to serializable format
        return [w.tolist() for w in weights]
        
    def set_weights(self, weights: List):
        """Update local model with global weights."""
        # Convert from serializable format back to numpy arrays
        np_weights = [np.array(w) for w in weights]
        self.model.set_weights(np_weights)
