#!/bin/bash
# Script to run two clients with different configs

# Start first client
python -m src.main --mode client --config config/client_config.yaml &

# Start second client
python -m src.main --mode client --config config/client_config_2.yaml &

# Wait for both clients to finish (optional)
wait
