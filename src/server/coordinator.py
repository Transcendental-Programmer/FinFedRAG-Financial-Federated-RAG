"""coordinator.py module."""

import tensorflow as tf
from typing import List, Dict
import numpy as np
from collections import defaultdict
import logging
import time

class FederatedCoordinator:
    def __init__(self, config: Dict):
        """Initialize the federated learning coordinator."""
        self.config = config
        self.clients = {}
        self.current_round = 0
        self.min_clients = config.get('server', {}).get('federated', {}).get('min_clients', 2)
        self.rounds = config.get('server', {}).get('federated', {}).get('rounds', 10)
        
    def register_client(self, client_id: int, client_size: int):
        """Register a new client."""
        self.clients[client_id] = {
            'size': client_size,
            'weights': None,
            'metrics': defaultdict(list)
        }
    
    def aggregate_weights(self, client_updates: List[Dict]) -> List:
        """Aggregate weights using FedAvg algorithm."""
        total_size = sum(self.clients[update['client_id']]['size'] 
                        for update in client_updates)
        
        aggregated_weights = [
            np.zeros_like(w) for w in client_updates[0]['weights']
        ]
        
        for update in client_updates:
            client_size = self.clients[update['client_id']]['size']
            weight = client_size / total_size
            
            for i, layer_weights in enumerate(update['weights']):
                aggregated_weights[i] += layer_weights * weight
                
        return aggregated_weights
    
    def start(self):
        """Start the federated learning process."""
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
        logger.info(f"Current active clients: {len(self.clients)}")
        logger.info("-" * 30 + "\n")
        
        while self.current_round < self.rounds:
            round_num = self.current_round + 1
            logger.info(f"\nRound {round_num}/{self.rounds}")
            logger.info("-" * 30)
            
            if len(self.clients) < self.min_clients:
                logger.warning(
                    f"Waiting for clients... "
                    f"(active: {len(self.clients)}/{self.min_clients})"
                )
                time.sleep(5)
                continue
            
            logger.info(f"Active clients: {list(self.clients.keys())}")
            logger.info(f"Starting training round {round_num}")
            self.current_round += 1
