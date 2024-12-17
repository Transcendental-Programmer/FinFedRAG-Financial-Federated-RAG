"""aggregator.py module."""

import tensorflow as tf
from typing import List, Dict
import numpy as np
from collections import defaultdict

class FederatedAggregator:
    def __init__(self, config: Dict):
        """Initialize the federated aggregator."""
        self.weighted = config['aggregation']['weighted']
        
    def compute_metrics(self, client_metrics: List[Dict]) -> Dict:
        """Compute aggregated metrics from client updates."""
        if not client_metrics:
            return {}
            
        aggregated_metrics = defaultdict(float)
        total_samples = sum(metrics['num_samples'] for metrics in client_metrics)
        
        for metrics in client_metrics:
            weight = metrics['num_samples'] / total_samples if self.weighted else 1.0
            
            for metric_name, value in metrics['metrics'].items():
                aggregated_metrics[metric_name] += value * weight
                
        return dict(aggregated_metrics)
    
    def check_convergence(self, 
                         old_weights: List, 
                         new_weights: List, 
                         threshold: float = 1e-5) -> bool:
        """Check if the model has converged."""
        if old_weights is None or new_weights is None:
            return False
            
        weight_differences = [
            np.mean(np.abs(old - new))
            for old, new in zip(old_weights, new_weights)
        ]
        
        return all(diff < threshold for diff in weight_differences)

