import argparse
import logging
import os
import sys
from datetime import datetime
from optimization_tuner import OptimizationTuner
from tqdm import tqdm

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run expanded grid search for KAN optimization methods')
    parser.add_argument('--methods', nargs='+', default=['QUBO', 'IntegerProgramming', 'Evolutionary', 'GreedyHeuristic'],
                      help='Optimization methods to compare (default: all)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--results_dir', type=str, default='optimization_results_fixed_full', help='Directory for results')
    parser.add_argument('--force', action='store_true', help='Skip confirmation and run immediately')
    parser.add_argument('--data_path', type=str, 
                       default="~/Interning/Kaggle/jane_street_kaggle/jane-street-real-time-market-data-forecasting/train.parquet/",
                       help='Path to Jane Street dataset')
    args = parser.parse_args()
    
    # Create results directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs("./models_janestreet", exist_ok=True)
    os.makedirs("results_js", exist_ok=True)
    
    # Set up logging after creating directory
    log_file = f"fixed_grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    logger.info("Running FIXED expanded grid search on Jane Street dataset")
    logger.info("This run uses skip_qubo_for_hidden=False for ALL methods to make them responsive to hyperparameters")
    
    # Initialize tuner
    tuner = OptimizationTuner(results_dir=args.results_dir)
    
    # Jane Street dataset is loaded automatically in the OptimizationTuner._load_data method
    # The path is configured in the DataConfig within that method
    logger.info(f"Using Jane Street dataset from: {args.data_path}")
    
    # Modify the data path in the tuner's _load_data method
    # This is a workaround because we can't directly call load_jane_street_dataset
    from data_pipeline_js_config import DataConfig
    from data_pipeline import DataPipeline
    
    # Configure Jane Street dataset
    data_cfg = DataConfig(
        data_path=args.data_path,
        n_rows=200000,  # Use 200k rows
        train_ratio=0.7,
        feature_cols=[f'feature_{i:02d}' for i in range(79)],
        target_col="responder_6",
        weight_col="weight",
        date_col="date_id"
    )

    # Load and preprocess data using the Jane Street pipeline
    pipeline = DataPipeline(data_cfg, logger)
    train_df, train_target, train_weight, val_df, val_target, val_weight = pipeline.load_and_preprocess_data()

    # Convert to numpy then torch
    import torch
    tuner.x_train = torch.tensor(train_df.to_numpy(), dtype=torch.float32)
    tuner.y_train = torch.tensor(train_target.to_numpy(), dtype=torch.float32).squeeze(-1).unsqueeze(-1)
    tuner.w_train = torch.tensor(train_weight.to_numpy(), dtype=torch.float32).squeeze(-1)

    tuner.x_val = torch.tensor(val_df.to_numpy(), dtype=torch.float32)
    tuner.y_val = torch.tensor(val_target.to_numpy(), dtype=torch.float32).squeeze(-1).unsqueeze(-1)
    tuner.w_val = torch.tensor(val_weight.to_numpy(), dtype=torch.float32).squeeze(-1)

    tuner.input_dim = tuner.x_train.shape[1]
    
    logger.info(f"Loaded Jane Street dataset: train={tuner.x_train.shape}, val={tuner.x_val.shape}")
    
    # Expanded parameter grid with more combinations
    expanded_grid = {
        'max_degree': [3, 5, 7, 9],           # More degrees to try
        'hidden_size': [16, 20, 24, 32],      # More hidden sizes
        'hidden_degree': [3, 5, 7],           # Different hidden complexities
        'learning_rate': [1e-2, 5e-3, 1e-3]   # Various learning rates
    }
    
    # Filter optimization methods based on input arguments
    if args.methods:
        valid_methods = ['QUBO', 'IntegerProgramming', 'Evolutionary', 'GreedyHeuristic']
        methods_to_run = [m for m in args.methods if m in valid_methods]
        if not methods_to_run:
            logger.error(f"No valid methods specified. Choose from: {valid_methods}")
            return
        logger.info(f"Running methods: {methods_to_run}")
    else:
        # Default to all methods
        methods_to_run = None
        logger.info("Running all optimization methods")
    
    logger.info(f"Expanded parameter grid: {expanded_grid}")
    logger.info(f"Training for {args.epochs} epochs per configuration")
    
    # Number of configurations to test
    total_configs = (
        len(expanded_grid['max_degree']) * 
        len(expanded_grid['hidden_size']) * 
        len(expanded_grid['hidden_degree']) * 
        len(expanded_grid['learning_rate']) * 
        (len(methods_to_run) if methods_to_run else 4)  # Default 4 methods
    )
    
    logger.info(f"Total configurations to test: {total_configs}")
    
    # More realistic time estimate based on our testing
    time_per_method = {
        'QUBO': 20,  # seconds
        'IntegerProgramming': 2, 
        'Evolutionary': 2,
        'GreedyHeuristic': 1
    }
    
    # Calculate estimated time based on methods
    if methods_to_run:
        estimated_time_per_epoch = sum(time_per_method.get(m, 10) for m in methods_to_run) / len(methods_to_run)
    else:
        estimated_time_per_epoch = sum(time_per_method.values()) / 4
    
    # Total time including training epochs
    estimated_time_per_config = (estimated_time_per_epoch + 2) * args.epochs  # seconds per config
    estimated_total_time = total_configs * estimated_time_per_config
    
    # Convert to hours/minutes
    hours = estimated_total_time // 3600
    minutes = (estimated_total_time % 3600) // 60
    
    logger.info(f"Estimated total runtime: {hours} hours and {minutes} minutes")
    
    # Ask for confirmation unless --force is used
    if not args.force:
        response = input(f"This may take approximately {hours} hours and {minutes} minutes to complete. Continue? (y/n): ")
        if response.lower() != 'y':
            logger.info("Aborted by user")
            return
    else:
        logger.info("Running without confirmation (--force flag used)")
    
    # Run comparison with the selected methods
    try:
        logger.info("Starting expanded grid search with fixed implementation")
        tuner.compare_optimization_methods(
            expanded_grid, 
            num_epochs=args.epochs, 
            methods_to_run=methods_to_run
        )
        
        # Plot results
        logger.info("Plotting results")
        tuner.plot_results()
        logger.info("Grid search completed successfully!")
        
        # Suggest running the analysis script
        logger.info("\nTo analyze the results with error bars and heatmaps, run:")
        logger.info(f"python analyze_optimization_results.py --results_dir {args.results_dir}")
        
    except Exception as e:
        logger.error(f"Error during grid search: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 