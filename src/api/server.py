"""
RESTful API for Federated Learning Server
Handles client registration, model updates, and coordination
"""

from flask import Flask, request, jsonify
import logging
import threading
import time
from typing import Dict, Any, List
from ..server.coordinator import FederatedCoordinator
from ..utils.metrics import calculate_model_similarity

logger = logging.getLogger(__name__)

class FederatedAPI:
    def __init__(self, coordinator: FederatedCoordinator, host: str = "0.0.0.0", port: int = 8080):
        self.app = Flask(__name__)
        self.coordinator = coordinator
        self.host = host
        self.port = port
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': time.time(),
                'active_clients': len(self.coordinator.clients),
                'current_round': getattr(self.coordinator, 'current_round', 0)
            })
        
        @self.app.route('/register', methods=['POST'])
        def register_client():
            """Register a new client"""
            try:
                data = request.get_json()
                client_id = data.get('client_id')
                client_info = data.get('client_info', {})
                
                if not client_id:
                    return jsonify({'error': 'client_id is required'}), 400
                
                success = self.coordinator.register_client(client_id, client_info)
                
                if success:
                    return jsonify({
                        'status': 'registered',
                        'client_id': client_id,
                        'server_config': self.coordinator.get_client_config()
                    })
                else:
                    return jsonify({'error': 'Registration failed'}), 400
                    
            except Exception as e:
                logger.error(f"Error registering client: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/get_model', methods=['POST'])
        def get_global_model():
            """Get the current global model"""
            try:
                data = request.get_json()
                client_id = data.get('client_id')
                
                if not client_id or client_id not in self.coordinator.clients:
                    return jsonify({'error': 'Invalid client_id'}), 400
                
                model_weights = self.coordinator.get_global_model()
                
                return jsonify({
                    'model_weights': model_weights,
                    'round': getattr(self.coordinator, 'current_round', 0),
                    'timestamp': time.time()
                })
                
            except Exception as e:
                logger.error(f"Error getting global model: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/submit_update', methods=['POST'])
        def submit_model_update():
            """Submit a model update from client"""
            try:
                data = request.get_json()
                client_id = data.get('client_id')
                model_weights = data.get('model_weights')
                training_metrics = data.get('metrics', {})
                
                if not client_id or not model_weights:
                    return jsonify({'error': 'client_id and model_weights are required'}), 400
                
                if client_id not in self.coordinator.clients:
                    return jsonify({'error': 'Client not registered'}), 400
                
                # Store the update
                self.coordinator.receive_model_update(client_id, model_weights, training_metrics)
                
                return jsonify({
                    'status': 'update_received',
                    'client_id': client_id,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                logger.error(f"Error submitting model update: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/training_status', methods=['GET'])
        def get_training_status():
            """Get current training status"""
            try:
                return jsonify({
                    'current_round': getattr(self.coordinator, 'current_round', 0),
                    'total_rounds': self.coordinator.config.get('federated', {}).get('num_rounds', 10),
                    'active_clients': len(self.coordinator.clients),
                    'clients_ready': len(getattr(self.coordinator, 'client_updates', {})),
                    'min_clients': self.coordinator.config.get('federated', {}).get('min_clients', 2),
                    'training_active': getattr(self.coordinator, 'training_active', False)
                })
                
            except Exception as e:
                logger.error(f"Error getting training status: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/rag/query', methods=['POST'])
        def rag_query():
            """Handle RAG queries"""
            try:
                data = request.get_json()
                query = data.get('query')
                client_id = data.get('client_id')
                
                if not query:
                    return jsonify({'error': 'query is required'}), 400
                
                # This will be implemented when we integrate RAG
                return jsonify({
                    'response': 'RAG functionality coming soon',
                    'query': query,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                logger.error(f"Error processing RAG query: {str(e)}")
                return jsonify({'error': str(e)}), 500
    
    def run(self, debug: bool = False):
        """Run the API server"""
        logger.info(f"Starting Federated API server on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)
    
    def run_threaded(self, debug: bool = False):
        """Run the API server in a separate thread"""
        def run_server():
            self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        logger.info(f"Federated API server started in background on {self.host}:{self.port}")
        return thread
