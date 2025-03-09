import argparse
import logging
import os
import sys
from datetime import datetime
from optimization_tuner import OptimizationTuner
from tqdm import tqdm

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run KAN optimization method comparison')
    parser.add_argument('--quick', action='store_true', help='Run with a minimal grid for quick testing')
    parser.add_argument('--full', action='store_true', help='Run with a larger grid for more comprehensive results')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--results_dir', type=str, default='optimization_results_js', help='Directory for results')
    args = parser.parse_args()
    
    # Create results directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs("./models_janestreet", exist_ok=True)
    os.makedirs("results_js", exist_ok=True)
    
    # Set up logging after creating directory
    log_file = f"optimization_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.results_dir, log_file), mode='w')
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {os.path.join(args.results_dir, log_file)}")
    logger.info("Running optimization comparison on Jane Street dataset")
    
    # Initialize tuner
    tuner = OptimizationTuner(results_dir=args.results_dir)
    
    # Define parameter grid based on args
    if args.quick:
        logger.info("Running with minimal grid for quick testing")
        param_grid = {
            'max_degree': [7],
            'hidden_size': [20],
            'hidden_degree': [5],
            'learning_rate': [5e-3]
        }
    elif args.full:
        logger.info("Running with comprehensive grid")
        param_grid = {
            'max_degree': [5, 7, 9],
            'hidden_size': [16, 20, 24, 28],
            'hidden_degree': [3, 5, 7],
            'learning_rate': [1e-2, 5e-3, 1e-3]
        }
    else:
        logger.info("Running with default grid")
        param_grid = {
            'max_degree': [5, 7, 9],
            'hidden_size': [20, 24],
            'hidden_degree': [3, 5, 7],
            'learning_rate': [1e-2, 5e-3]  # Higher learning rates from run_tuning.py
        }
    
    logger.info(f"Parameter grid: {param_grid}")
    logger.info(f"Training for {args.epochs} epochs")
    
    # Run comparison
    try:
        logger.info("Starting optimization method comparison")
        tuner.compare_optimization_methods(param_grid, num_epochs=args.epochs)
        
        # Plot results
        logger.info("Plotting results")
        tuner.plot_results()
        logger.info("Optimization comparison completed successfully!")
    except Exception as e:
        logger.error(f"Error during optimization comparison: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 