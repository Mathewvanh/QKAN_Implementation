import argparse
import yaml
import logging
import os
import sys
from degradation_analyzer import DegradationAnalyzer
# Ensure the project root is in the Python path for imports
# Adjust this path based on your project structure if needed
project_root = os.path.dirname(os.path.abspath(__file__)) 
# sys.path.insert(0, project_root)
# sys.path.insert(0, os.path.join(project_root, 'CP-KAN_reproducibility')) # If modules are inside a subdirectory

# Conditional imports based on experiment type

def load_config(config_path):
    """Loads YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {config_path}: {e}")
        sys.exit(1)

def setup_logging(config):
    """Sets up basic logging based on config."""
    log_level_str = config.get('logging_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Use basicConfig to configure root logger
    logging.basicConfig(level=log_level, format=log_format)
    
    # If a log file is specified, add a file handler
    log_file = config.get('log_file')
    output_dir = config.get('output_dir', '.') # Default to current dir if not specified
    if log_file:
        log_path = os.path.join(output_dir, log_file)
        # Ensure output directory exists for the log file
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler) # Add handler to root logger

    logging.info("Logging setup complete.")


def main():
    parser = argparse.ArgumentParser(description="Run KAN/MLP experiments based on config file.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config) # Setup logging early

    experiment_type = config.get('experiment_type', 'grid_search') # Default to grid_search if not specified
    logging.info(f"Starting experiment type: {experiment_type}")

    if experiment_type == 'degradation_analysis':
        try:
            analyzer = DegradationAnalyzer(config)
            analyzer.run_analysis()
        except ImportError:
            logging.error("Could not import DegradationAnalyzer. Make sure degradation_analyzer.py is in the Python path.")
            sys.exit(1)
        except Exception as e:
            logging.exception(f"An error occurred during degradation analysis: {e}")
            sys.exit(1)

    elif experiment_type == 'grid_search' or experiment_type == 'single_run': # Handle ExperimentRunner cases
        try:
            # Assuming ExperimentRunner is in experiment_runner.py in the same directory or accessible path
            from experiment_runner import ExperimentRunner 
            runner = ExperimentRunner(config)
            # Determine if it's a grid search or single run based on config structure
            if 'parameter_grids' in config:
                 logging.info("Running grid search...")
                 runner.run_grid_search() # Assuming this is the method for grid search
            elif 'model_params' in config: # Or however single run is distinguished
                 logging.info("Running single experiment...")
                 runner.run_single_experiment() # Assuming this is the method for single run
            else:
                 logging.error("Experiment type is grid_search/single_run, but config lacks 'parameter_grids' or 'model_params'.")
                 sys.exit(1)
        except ImportError:
            logging.error("Could not import ExperimentRunner. Make sure experiment_runner.py is in the Python path.")
            sys.exit(1)
        except Exception as e:
            logging.exception(f"An error occurred during ExperimentRunner execution: {e}")
            sys.exit(1)

    elif experiment_type == 'degradation_study': # Handle the new study type
         try:
            from experiment_runner import ExperimentRunner 
            runner = ExperimentRunner(config)
            logging.info("Running degradation study...")
            runner.run_degradation_study()
         except ImportError:
             logging.error("Could not import ExperimentRunner. Make sure experiment_runner.py is in the Python path.")
             sys.exit(1)
         except Exception as e:
             logging.exception(f"An error occurred during ExperimentRunner (degradation study) execution: {e}")
             sys.exit(1)

    else:
        logging.error(f"Unknown experiment_type: {experiment_type}")
        sys.exit(1)

    logging.info(f"Experiment type '{experiment_type}' completed successfully.")

if __name__ == "__main__":
    main() 