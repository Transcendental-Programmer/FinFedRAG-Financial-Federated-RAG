"""coordinator.py module."""

import tensorflow as tf
from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict
import logging
import time
import threading
from .aggregator import FederatedAggregator

class FederatedCoordinator:
    def __init__(self, config: Dict):
        """Initialize the federated learning coordinator."""
        logger = logging.getLogger(__name__)
        logger.debug(f"Initializing FederatedCoordinator with config: {config}")
        self.config = config
        self.clients = {}
        self.client_updates = {}  # Store updates for current round
        self.global_model_weights = None
        self.current_round = 0
        self.training_active = False
        
        # Extract federated learning parameters
        self.min_clients = config.get('federated', {}).get('min_clients', 2)
        self.rounds = config.get('federated', {}).get('rounds', 10)
        
        # Debug: log config structure
        logger.debug(f"Coordinator received config: {config}")
        
        # Robustly extract aggregation config
        agg_config = None
        if 'aggregation' in config:
            agg_config = config
        elif 'server' in config and 'aggregation' in config['server']:
            agg_config = config['server']
        else:
            logger.error(f"No 'aggregation' key found in config for FederatedAggregator: {config}")
            raise ValueError("'aggregation' config section is required for FederatedAggregator")
        
        logger.debug(f"Passing aggregation config to FederatedAggregator: {agg_config}")
        try:
            self.aggregator = FederatedAggregator(agg_config)
        except Exception as e:
            logger.error(f"Error initializing FederatedAggregator: {e}")
            raise
        
        # Initialize global model weights with random values
        self._initialize_global_model()
        
        self.lock = threading.Lock()  # Thread safety for concurrent API calls
        logger.info("FederatedCoordinator initialized.")
    
    def _initialize_global_model(self):
        """Initialize global model weights with random values."""
        logger = logging.getLogger(__name__)
        try:
            # Build a simple model to get initial weights
            input_dim = self.config.get('model', {}).get('input_dim', 32)
            hidden_layers = self.config.get('model', {}).get('hidden_layers', [128, 64])
            
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(hidden_layers[0], activation='relu'),
                tf.keras.layers.Dense(hidden_layers[1], activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            
            self.global_model_weights = model.get_weights()
            logger.info(f"Global model initialized with {len(self.global_model_weights)} weight layers")
            
        except Exception as e:
            logger.error(f"Error initializing global model: {e}")
            # Fallback to simple random weights
            self.global_model_weights = [
                np.random.randn(32, 128).astype(np.float32),
                np.random.randn(128).astype(np.float32),
                np.random.randn(128, 64).astype(np.float32),
                np.random.randn(64).astype(np.float32),
                np.random.randn(64, 1).astype(np.float32),
                np.random.randn(1).astype(np.float32)
            ]
            logger.info("Using fallback random weights for global model")
        
    def register_client(self, client_id: str, client_info: Dict[str, Any] = None) -> bool:
        """Register a new client."""
        with self.lock:
            if client_id in self.clients:
                logging.getLogger(__name__).warning(f"Client {client_id} already registered")
                return True
                
            self.clients[client_id] = {
                'info': client_info or {},
                'last_seen': time.time(),
                'metrics': defaultdict(list)
            }
            
            logging.getLogger(__name__).info(f"Client {client_id} registered successfully")
            return True
    
    def get_client_config(self) -> Dict[str, Any]:
        """Get configuration to send to clients"""
        return {
            'model_config': self.config.get('model', {}),
            'training_config': self.config.get('training', {}),
            'current_round': self.current_round,
            'total_rounds': self.rounds
        }
    
    def get_global_model(self) -> Optional[List]:
        """Get the current global model weights"""
        with self.lock:
            return self.global_model_weights
    
    def receive_model_update(self, client_id: str, model_weights: List, metrics: Dict[str, Any]):
        """Receive a model update from a client"""
        with self.lock:
            if client_id not in self.clients:
                raise ValueError(f"Client {client_id} not registered")
            
            self.client_updates[client_id] = {
                'weights': model_weights,
                'metrics': metrics,
                'timestamp': time.time()
            }
            
            self.clients[client_id]['last_seen'] = time.time()
            
            logger = logging.getLogger(__name__)
            logger.info(f"Received update from client {client_id}")
            
            # Check if we have enough updates for aggregation
            if len(self.client_updates) >= self.min_clients:
                self._aggregate_models()
    
    def _aggregate_models(self):
        """Aggregate models from all client updates"""
        try:
            logger = logging.getLogger(__name__)
            logger.info(f"Aggregating models from {len(self.client_updates)} clients")
            
            # Prepare updates for aggregation
            updates = []
            for client_id, update in self.client_updates.items():
                client_size = update['metrics'].get('dataset_size', 100)  # Default size
                updates.append({
                    'client_id': client_id,
                    'weights': update['weights'],
                    'size': client_size
                })
            
            # Aggregate using FedAvg
            self.global_model_weights = self.aggregator.federated_averaging(updates)
            
            # Clear updates for next round
            self.client_updates.clear()
            self.current_round += 1
            
            logger.info(f"Model aggregation completed for round {self.current_round}")
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error during model aggregation: {str(e)}")
    
    def _count_active_clients(self) -> int:
        """Count active clients (seen in last 60 seconds)"""
        current_time = time.time()
        active_count = sum(1 for client in self.clients.values() 
                          if current_time - client['last_seen'] < 60)
        return active_count
    
    def start(self):
        """Start the federated learning process with API server"""
        logger = logging.getLogger(__name__)
        
        # Print server startup information
        logger.info("\n" + "=" * 60)
        logger.info(f"{'Federated Learning Server Starting':^60}")
        logger.info("=" * 60)
        
        # Print configuration details
        logger.info("\nServer Configuration:")
        logger.info("-" * 30)
        logger.info(f"Minimum clients required: {self.min_clients}")
        logger.info(f"Total rounds planned: {self.rounds}")
        active_clients_count = self._count_active_clients()
        logger.info(f"Current active clients: {active_clients_count}")
        logger.info("-" * 30 + "\n")
        
        self.training_active = True
        
        # Import and start API server
        try:
            from ..api.server import FederatedAPI
            
            api_config = self.config.get('api', {})
            host = api_config.get('host', '0.0.0.0')
            port = api_config.get('port', 8080)
            
            api_server = FederatedAPI(self, host, port)
            api_thread = api_server.run_threaded()
            
            logger.info(f"API server started on {host}:{port}")
            
            # Keep server running
            try:
                while self.training_active and self.current_round < self.rounds:
                    time.sleep(1)  # Keep main thread alive
                    
                    # Log progress periodically
                    active_clients_count = self._count_active_clients()
                    if active_clients_count > 0:
                        logger.debug(f"Round {self.current_round}/{self.rounds}, "
                                   f"Active Clients: {active_clients_count}, "
                                   f"Updates: {len(self.client_updates)}")
                
                logger.info("Federated learning completed successfully")
                
            except KeyboardInterrupt:
                logger.info("Server shutdown requested")
                self.training_active = False
                
        except ImportError as e:
            logger.error(f"Failed to start API server: {str(e)}")
            # Fallback to original behavior
            # ...existing code...
