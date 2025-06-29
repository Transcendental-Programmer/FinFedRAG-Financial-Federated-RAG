"""aggregator.py module."""

import tensorflow as tf
from typing import List, Dict
import numpy as np
from collections import defaultdict
import logging

class FederatedAggregator:
    def __init__(self, config: Dict):
        logger = logging.getLogger(__name__)
        logger.debug(f"Initializing FederatedAggregator with config: {config}")
        # Defensive: try to find aggregation config
        agg_config = None
        if 'aggregation' in config:
            agg_config = config['aggregation']
        elif 'server' in config and 'aggregation' in config['server']:
            agg_config = config['server']['aggregation']
        else:
            logger.error(f"No 'aggregation' key found in config passed to FederatedAggregator: {config}")
            raise KeyError("'aggregation' config section is required for FederatedAggregator")
        self.weighted = agg_config.get('weighted', True)
        logger.info(f"FederatedAggregator initialized. Weighted: {self.weighted}")
    
    def federated_averaging(self, updates: List[Dict]) -> List:
        """Perform federated averaging (FedAvg) on model weights."""
        logger = logging.getLogger(__name__)
        logger.info(f"Performing federated averaging on {len(updates)} client updates")
        
        if not updates:
            logger.warning("No updates provided for federated averaging")
            return None
        
        # Calculate total samples across all clients
        total_samples = sum(update['size'] for update in updates)
        logger.debug(f"Total samples across clients: {total_samples}")
        
        # Initialize aggregated weights with zeros
        first_weights = updates[0]['weights']
        aggregated_weights = [np.zeros_like(w) for w in first_weights]
        
        # Weighted average of model weights
        for update in updates:
            client_weights = update['weights']
            client_size = update['size']
            weight_factor = client_size / total_samples if self.weighted else 1.0 / len(updates)
            
            logger.debug(f"Client {update['client_id']}: size={client_size}, weight_factor={weight_factor}")
            
            # Add weighted contribution to aggregated weights
            for i, (agg_w, client_w) in enumerate(zip(aggregated_weights, client_weights)):
                aggregated_weights[i] += np.array(client_w) * weight_factor
        
        logger.info("Federated averaging completed successfully")
        return aggregated_weights
    
    def compute_metrics(self, client_metrics: List[Dict]) -> Dict:
        logger = logging.getLogger(__name__)
        logger.debug(f"Computing metrics for {len(client_metrics)} clients")
        if not client_metrics:
            logger.warning("No client metrics provided to compute_metrics.")
            return {}
        aggregated_metrics = defaultdict(float)
        total_samples = sum(metrics['num_samples'] for metrics in client_metrics)
        logger.debug(f"Total samples across clients: {total_samples}")
        for metrics in client_metrics:
            weight = metrics['num_samples'] / total_samples if self.weighted else 1.0
            logger.debug(f"Client metrics: {metrics}, weight: {weight}")
            for metric_name, value in metrics['metrics'].items():
                aggregated_metrics[metric_name] += value * weight
        logger.info(f"Aggregated metrics: {dict(aggregated_metrics)}")
        return dict(aggregated_metrics)
    
    def check_convergence(self, 
                         old_weights: List, 
                         new_weights: List, 
                         threshold: float = 1e-5) -> bool:
        logger = logging.getLogger(__name__)
        logger.debug("Checking convergence...")
        if old_weights is None or new_weights is None:
            logger.warning("Old or new weights are None in check_convergence.")
            return False
        weight_differences = [
            np.mean(np.abs(old - new))
            for old, new in zip(old_weights, new_weights)
        ]
        logger.debug(f"Weight differences: {weight_differences}")
        converged = all(diff < threshold for diff in weight_differences)
        logger.info(f"Convergence status: {converged}")
        return converged

