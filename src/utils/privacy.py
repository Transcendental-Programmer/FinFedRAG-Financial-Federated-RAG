"""privacy.py module."""

import tensorflow_privacy as tfp
from typing import Dict, Any
import numpy as np

class PrivacyManager:
    def __init__(self, config: Dict[str, Any]):
        self.epsilon = config['privacy']['epsilon']
        self.delta = config['privacy']['delta']
        self.noise_multiplier = config['privacy']['noise_multiplier']
        
    def add_noise_to_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to gradients for differential privacy."""
        noise = np.random.normal(0, self.noise_multiplier, gradients.shape)
        return gradients + noise
        
    def verify_privacy_budget(self, num_iterations: int) -> bool:
        """Check if training stays within privacy budget."""
        eps = self.compute_epsilon(num_iterations)
        return eps <= self.epsilon
        
    def compute_epsilon(self, num_iterations: int) -> float:
        """Compute the current epsilon value."""
        q = 1.0  # sampling ratio
        steps = num_iterations
        orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                 list(range(5, 64)) + [128, 256, 512])
        
        return tfp.compute_dp_sgd_privacy(
            n=1000,  # number of training points
            batch_size=32,
            noise_multiplier=self.noise_multiplier,
            epochs=steps,
            delta=self.delta
        )[0]

