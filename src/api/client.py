"""
HTTP Client for Federated Learning
Handles communication with the federated server
"""

import requests
import json
import logging
import time
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class FederatedHTTPClient:
    def __init__(self, server_url: str, client_id: str, timeout: int = 30):
        self.server_url = server_url.rstrip('/')
        self.client_id = client_id
        self.timeout = timeout
        self.session = requests.Session()
        
    def register(self, client_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Register this client with the server"""
        try:
            payload = {
                'client_id': self.client_id,
                'client_info': client_info or {}
            }
            
            response = self.session.post(
                f"{self.server_url}/register",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Client {self.client_id} registered successfully")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register client {self.client_id}: {str(e)}")
            raise
    
    def get_global_model(self) -> Dict[str, Any]:
        """Get the current global model from server"""
        try:
            payload = {'client_id': self.client_id}
            
            response = self.session.post(
                f"{self.server_url}/get_model",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"Retrieved global model for round {result.get('round', 'unknown')}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get global model: {str(e)}")
            raise
    
    def submit_model_update(self, model_weights: List, metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Submit model update to server"""
        try:
            payload = {
                'client_id': self.client_id,
                'model_weights': model_weights,
                'metrics': metrics or {}
            }
            
            response = self.session.post(
                f"{self.server_url}/submit_update",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Model update submitted successfully by client {self.client_id}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to submit model update: {str(e)}")
            raise
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status from server"""
        try:
            response = self.session.get(
                f"{self.server_url}/training_status",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get training status: {str(e)}")
            raise
    
    def health_check(self) -> bool:
        """Check if server is healthy"""
        try:
            response = self.session.get(
                f"{self.server_url}/health",
                timeout=5  # Short timeout for health checks
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('status') == 'healthy'
            
        except requests.exceptions.RequestException:
            return False
    
    def wait_for_server(self, max_wait: int = 60, check_interval: int = 5) -> bool:
        """Wait for server to become available"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if self.health_check():
                logger.info(f"Server is available at {self.server_url}")
                return True
            
            logger.info(f"Waiting for server at {self.server_url}...")
            time.sleep(check_interval)
        
        logger.error(f"Server not available after {max_wait} seconds")
        return False
    
    def rag_query(self, query: str) -> Dict[str, Any]:
        """Submit a RAG query to the server"""
        try:
            payload = {
                'query': query,
                'client_id': self.client_id
            }
            
            response = self.session.post(
                f"{self.server_url}/rag/query",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to submit RAG query: {str(e)}")
            raise
    
    def close(self):
        """Close the HTTP session"""
        self.session.close()
