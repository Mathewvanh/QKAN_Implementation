import argparse
import os
import time
import json
from optimization_tuner import OptimizationTuner

def main():
    parser = argparse.ArgumentParser(description='Run KAN optimization methods comparison with fix applied')
    parser.add_argument('--dataset', type=str, default='jane_street', help='Dataset to use (jane_street)')
    parser.add_argument('--results_dir', type=str, default='optimization_results_fixed', help='Directory to save results')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train each model')
    parser.add_argument('--methods', nargs='+', default=['QUBO', 'IntegerProgramming', 'Evolutionary', 'GreedyHeuristic'], 
                        help='Methods to run (default: all)')
    args = parser.parse_args()

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Configure logger
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.results_dir, 'optimization_comparison.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    # Log configuration
    logger.info(f"Running optimization comparison with fixed implementation")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Methods to run: {args.methods}")
    logger.info(f"Number of epochs: {args.num_epochs}")

    # Create tuner
    tuner = OptimizationTuner(results_dir=args.results_dir)
    
    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    if args.dataset == 'jane_street':
        tuner.load_jane_street_dataset()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Define parameter grid
    param_grid = {
        'max_degree': [3, 5, 7],
        'hidden_size': [16, 32, 64],
        'hidden_degree': [2, 3],
        'learning_rate': [0.001, 0.01]
    }
    
    # Log parameter grid
    logger.info(f"Parameter grid:")
    for k, v in param_grid.items():
        logger.info(f"  {k}: {v}")
    
    # Calculate total configurations
    total_configs = (
        len(param_grid['max_degree']) * 
        len(param_grid['hidden_size']) * 
        len(param_grid['hidden_degree']) * 
        len(param_grid['learning_rate']) * 
        len(args.methods)
    )
    logger.info(f"Total configurations to test: {total_configs}")
    
    # Estimate runtime
    avg_time_per_config = 30  # seconds
    estimated_time = total_configs * avg_time_per_config
    hours = estimated_time // 3600
    minutes = (estimated_time % 3600) // 60
    logger.info(f"Estimated runtime: {hours} hours and {minutes} minutes")
    
    # Run comparison
    start_time = time.time()
    logger.info("Starting optimization comparison...")
    
    tuner.compare_optimization_methods(
        param_grid=param_grid,
        num_epochs=args.num_epochs,
        methods_to_run=args.methods
    )
    
    # Log completion
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    logger.info(f"Optimization comparison completed in {hours}h {minutes}m {seconds}s")

if __name__ == "__main__":
    main() 