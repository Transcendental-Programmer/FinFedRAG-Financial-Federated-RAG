"""model.py module."""

from typing import Dict, List
import tensorflow as tf
import numpy as np
import logging

class FederatedClient:
    def __init__(self, client_id: int, config: Dict):
        """Initialize the federated client."""
        self.client_id = client_id
        self.config = config.get('client', {})
        self.model = self._build_model()
        
    def start(self):
        """Start the federated client process."""
        logger = logging.getLogger(__name__)
        logger.info(f"Client {self.client_id} started")
        logger.info(f"Client config: {self.config}")
        
        try:
            # Simulate some data
            logger.info("Generating training data...")
            X, y = self._generate_dummy_data()
            logger.info(f"Generated data shapes - X: {X.shape}, y: {y.shape}")
            
            # Train locally
            logger.info("Starting local training...")
            history = self.train_local((X, y))
            
            # Log training metrics
            losses = history.get('loss', [])
            logger.info("\nTraining Progress Summary:")
            logger.info("-" * 30)
            for epoch, loss in enumerate(losses, 1):
                logger.info(f"Epoch {epoch:2d}/{len(losses)}: loss = {loss:.4f}")
            
            final_loss = losses[-1]
            logger.info(f"\nTraining completed - Final loss: {final_loss:.4f}")
            
            # Log model summary in a simpler format
            logger.info("\nModel Architecture:")
            logger.info("-" * 30)
            logger.info("Layer (Output Shape) -> Params")
            total_params = 0
            for layer in self.model.layers:
                params = layer.count_params()
                total_params += params
                logger.info(f"{layer.name} {layer.output.shape} -> {params:,} params")
            logger.info(f"Total Parameters: {total_params:,}")
            
        except Exception as e:
            logger.error(f"Error during client execution: {str(e)}")
            raise
        
    def _generate_dummy_data(self):
        """Generate dummy data for testing."""
        num_samples = 100
        input_dim = 32  # Match with model's input dimension
        
        # Generate input data
        X = tf.random.normal((num_samples, input_dim))
        # Generate target data (for this example, we'll predict the sum of inputs)
        y = tf.reduce_sum(X, axis=1, keepdims=True)
        
        return X, y
        
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
        
        # Log training parameters
        logger.info(f"\nTraining Parameters:")
        logger.info("-" * 50)
        logger.info(f"Input shape: {X.shape}")
        logger.info(f"Output shape: {y.shape}")
        logger.info(f"Batch size: {self.config.get('data', {}).get('batch_size', 32)}")
        logger.info(f"Epochs: {self.config.get('training', {}).get('local_epochs', 5)}")
        logger.info(f"Learning rate: {self.config.get('training', {}).get('learning_rate', 0.001)}")
        logger.info("-" * 50)
        
        class LogCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logger.info(f"Epoch {epoch + 1} - loss: {logs['loss']:.4f}")
        
        # Enable verbose mode for training
        history = self.model.fit(
            X, y,
            batch_size=self.config.get('data', {}).get('batch_size', 32),
            epochs=self.config.get('training', {}).get('local_epochs', 5),
            verbose=0,  # Disable default verbose output
            callbacks=[LogCallback()]  # Use our custom callback
        )
        return history.history
        
    def get_weights(self) -> List:
        """Get the model weights."""
        return self.model.get_weights()
        
    def set_weights(self, weights: List):
        """Update local model with global weights."""
        self.model.set_weights(weights)

