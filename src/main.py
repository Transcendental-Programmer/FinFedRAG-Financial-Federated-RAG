import argparse
import yaml
import logging
import logging.config
from pathlib import Path
from src.server.coordinator import FederatedCoordinator
from src.client.model import FederatedClient

def setup_logging(config):
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    log_level = (config.get('monitoring', {}).get('log_level') 
                or config.get('server', {}).get('monitoring', {}).get('log_level') 
                or config.get('client', {}).get('monitoring', {}).get('log_level')
                or 'INFO')
    
    # Configure logging with UTF-8 encoding
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/federated_learning.log', mode='a', encoding='utf-8')
        ]
    )

    # Reduce TensorFlow logging noise
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    
    # Create a divider in the log file
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*50)
    logger.info("New Training Session Started")
    logger.info("="*50 + "\n")

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Demo')
    parser.add_argument('--mode', choices=['server', 'client'], required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    logger = logging.getLogger(__name__)

    if args.mode == 'server':
        coordinator = FederatedCoordinator(config)
        logger.info("Starting server...")
        coordinator.start()
    else:
        client = FederatedClient(1, config)
        logger.info(f"Starting client with ID: {client.client_id}")
        client.start()

if __name__ == "__main__":
    main()