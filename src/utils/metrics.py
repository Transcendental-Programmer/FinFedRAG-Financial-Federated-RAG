"""metrics.py module."""

from typing import Dict, List
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.metrics import mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

class MetricsCalculator:
    @staticmethod
    def calculate_distribution_similarity(real_data: np.ndarray, 
                                       synthetic_data: np.ndarray) -> Dict[str, float]:
        """Calculate statistical similarity metrics between real and synthetic data."""
        metrics = {}
        
        # Wasserstein distance
        metrics['wasserstein'] = wasserstein_distance(
            real_data.flatten(), 
            synthetic_data.flatten()
        )
        
        # KL divergence approximation
        metrics['mutual_info'] = mutual_info_score(
            real_data.flatten(),
            synthetic_data.flatten()
        )
        
        # Kolmogorov-Smirnov test
        ks_statistic, p_value = ks_2samp(real_data.flatten(), synthetic_data.flatten())
        metrics['ks_statistic'] = ks_statistic
        metrics['ks_p_value'] = p_value
        
        # Basic statistical measures
        metrics['mean_diff'] = abs(np.mean(real_data) - np.mean(synthetic_data))
        metrics['std_diff'] = abs(np.std(real_data) - np.std(synthetic_data))
        metrics['percentile_diff'] = np.mean([
            abs(np.percentile(real_data, p) - np.percentile(synthetic_data, p))
            for p in [25, 50, 75]
        ])
        
        return metrics
    
    @staticmethod
    def evaluate_privacy_metrics(model, test_data: np.ndarray, 
                               synthetic_data: np.ndarray) -> Dict[str, float]:
        """Evaluate privacy-related metrics."""
        metrics = {}
        
        # Membership inference risk
        metrics['membership_inference_risk'] = MetricsCalculator._calculate_membership_inference_risk(
            test_data, synthetic_data
        )
        
        # Attribute inference risk
        metrics['attribute_inference_risk'] = MetricsCalculator._calculate_attribute_inference_risk(
            test_data, synthetic_data
        )
        
        # k-anonymity approximation
        metrics['k_anonymity_score'] = MetricsCalculator._calculate_k_anonymity(synthetic_data)
        
        # Uniqueness score
        metrics['uniqueness_score'] = MetricsCalculator._calculate_uniqueness(synthetic_data)
        
        return metrics
    
    @staticmethod
    def _calculate_membership_inference_risk(test_data: np.ndarray, 
                                          synthetic_data: np.ndarray) -> float:
        """Calculate membership inference risk using nearest neighbor distance ratio."""
        k = 3  # number of neighbors to consider
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(synthetic_data)
        
        distances, _ = nn.kneighbors(test_data)
        avg_min_distances = distances.mean(axis=1)
        
        # Normalize to [0,1] where higher values indicate higher privacy
        risk_score = 1.0 - (1.0 / (1.0 + np.mean(avg_min_distances)))
        return risk_score
    
    @staticmethod
    def _calculate_attribute_inference_risk(test_data: np.ndarray, 
                                         synthetic_data: np.ndarray) -> float:
        """Calculate attribute inference risk using correlation analysis."""
        real_corr = np.corrcoef(test_data.T)
        synth_corr = np.corrcoef(synthetic_data.T)
        
        # Compare correlation matrices
        correlation_diff = np.abs(real_corr - synth_corr).mean()
        
        # Convert to risk score (0 to 1, where lower is better)
        risk_score = 1.0 - np.exp(-correlation_diff)
        return risk_score
    
    @staticmethod
    def _calculate_k_anonymity(data: np.ndarray, k: int = 5) -> float:
        """Calculate approximate k-anonymity score."""
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(data)
        
        distances, _ = nn.kneighbors(data)
        k_anonymity_scores = distances[:, -1]  # Distance to k-th neighbor
        
        # Convert to score (0 to 1, where higher is better)
        return float(np.mean(k_anonymity_scores > 0.1))
    
    @staticmethod
    def _calculate_uniqueness(data: np.ndarray) -> float:
        """Calculate uniqueness score of the dataset."""
        nn = NearestNeighbors(n_neighbors=2)  # 2 because first neighbor is self
        nn.fit(data)
        
        distances, _ = nn.kneighbors(data)
        uniqueness_scores = distances[:, 1]  # Distance to nearest non-self neighbor
        
        # Convert to score (0 to 1, where higher means more unique records)
        return float(np.mean(uniqueness_scores > np.median(uniqueness_scores)))

