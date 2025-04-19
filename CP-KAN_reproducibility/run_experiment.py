import argparse
import logging
import os
import sys
from datetime import datetime
import yaml
import numpy as np
import torch
import random
from typing import Dict

from experiment_runner import ExperimentRunner

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_config(path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            if config is None:
                 raise ValueError("YAML file is empty or invalid.")
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
            sys.exit(1)
        except FileNotFoundError:
             print(f"Error: Configuration file not found at {path}")
             sys.exit(1)
    return config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run KAN Experiment (Optimization Comparison)')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file (required)')
    parser.add_argument('--quick', action='store_true', help='Use the \'quick\' parameter grid from config (if defined)')
    parser.add_argument('--full', action='store_true', help='Use the \'full\' parameter grid from config (if defined)')
    parser.add_argument('--grid_key', type=str, default='default', help='Specify which parameter grid key to use (default: default)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    experiment_name = config.get('experiment_name', 'KAN_Experiment')
    results_dir = config.get('results_dir', 'experiment_results')
    random_seed = config.get('random_seed', 42)
    num_epochs = config.get('num_epochs', 50)

    # Set random seed
    set_seed(random_seed)

    # Create results directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_results_dir = os.path.join(script_dir, results_dir)
    os.makedirs(full_results_dir, exist_ok=True)

    # Set up logging
    log_file = f"{experiment_name}_{config.get('dataset', {}).get('name', 'unknown_data')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(full_results_dir, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode='w')
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Experiment: {experiment_name}")
    logger.info(f"Using configuration from: {args.config}")
    logger.info(f"Logging to {log_path}")
    logger.info(f"Random Seed: {random_seed}")

    # Validate config structure (basic checks)
    if 'dataset' not in config or 'name' not in config['dataset']:
        logger.error("Configuration file must contain 'dataset' section with a 'name' key.")
        sys.exit(1)
    if 'parameter_grids' not in config:
        logger.error("Configuration file must contain 'parameter_grids' section.")
        sys.exit(1)

    # Initialize the runner with the full config
    try:
        runner = ExperimentRunner(config)
    except Exception as e:
        logger.exception(f"Failed to initialize ExperimentRunner: {e}")
        exit()

    # Run grid search - runner now handles grid extraction internally
    try:
        # Removed param_grid extraction
        # Call run_grid_search without the param_grid argument
        runner.run_grid_search(num_epochs=config.get('num_epochs', 50))
    except Exception as e:
        logger.exception(f"An error occurred during grid search: {e}")
        exit()

    # Plot results if configured
    try:
        if config.get('plotting', {}).get('plot_results', False):
            logger.info("Generating plots...")
            runner.plot_results()
            logger.info("Plotting complete.")
        else:
            logger.info("Plotting skipped based on config.")
    except Exception as e:
        logger.exception(f"An error occurred during plotting: {e}")

    logger.info(f"Experiment '{config.get('experiment_name', 'Unnamed')}' finished.")

if __name__ == "__main__":
    main() 