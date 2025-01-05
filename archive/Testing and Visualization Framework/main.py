import logging
import sys
import time
from pathlib import Path

import numpy as np
from torch import nn
from original_degree_optimizer.DegreeOptimizer import DegreeOptimizer
from data_pipeline import DataPipeline
from config import ExperimentConfig,load_config
from train_utils import train_mlp
from evaluation import ModelResult, ModelEvaluator
from visualization import ResultVisualizer

def setup_logging(config: ExperimentConfig):
    """Setup logging configuration"""
    log_path = Path(config.log_path) / f"{config.name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ]
    )

def run_experiment(config_path: str):
    """Run complete experiment from config"""
    config = load_config(config_path)
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("Initializing data pipeline...")
    pipeline = DataPipeline(config.data, logger=logger)
    train_df, train_target, train_weight, val_df, val_target, val_weight = pipeline.load_and_preprocess_data()

    results = []
    for model_config in config.models:
        logger.info(f"Training model: {model_config.model_type}")

        for trial in range(config.num_trials):
            start_time = time.time()
            scores, compr2_scores = [],[]
            model_params = {}
            if model_config.model_type == 'qkan':
                model = DegreeOptimizer(
                    network_shape=model_config.network_shape,
                    max_degree=model_config.max_degree,
                    complexity_weight=model_config.complexity_weight,
                    significance_threshold=model_config.significance_threshold
                )
                # Train and evaluate QKAN
                optimal_degrees = model.optimize_layer(
                    layer_idx=0,
                    x_data=train_df,
                    y_data=train_target.to_numpy(),
                    weights=train_weight.to_numpy(),
                )



                scores, compr2_scores = model.evaluate_degree(
                    x_data=val_df,
                    y_data=val_target.to_numpy(),
                    weights=val_weight.to_numpy(),
                )

                model_params = {'degrees': optimal_degrees}

            elif model_config.model_type == 'mlp':
                layers = []
                in_features = len(config.data.feature_cols)

                # First layer
                layers.extend([
                    nn.Linear(in_features, model_config.hidden_dims[0]),
                    nn.Tanh()
                ])

                # Hidden layers
                for d_in, d_out in zip(model_config.hidden_dims[:-1], model_config.hidden_dims[1:]):
                    layers.extend([
                        nn.Linear(d_in, d_out),
                        nn.Tanh()
                    ])

                # Output layer
                layers.append(nn.Linear(model_config.hidden_dims[-1], 1))

                model = nn.Sequential(*layers)

                # Train and evaluate MLP
                scores, compr2_scores = train_mlp(
                    model=model,
                    x_train=train_df,
                    y_train=train_target,
                    x_val=val_df,
                    y_val=val_target,
                    w_val=val_weight,
                    weights=train_weight,
                    #learning_rate=model_config.learning_rate,
                    batch_size=model_config.batch_size,
                    n_epochs=model_config.n_epochs
                )

                model_params = {
                    'n_parameters': sum(p.numel() for p in model.parameters()),
                    'architecture': str(model)
                }

            train_time = time.time() - start_time

            # Record results
            result = ModelResult(
                model_name=f"{model_config.model_type}_trial_{trial}",
                MSE_Score=np.average(scores),
                R2_Score=np.average(compr2_scores),
                train_time=train_time,
                model_params=model_params
            )
            results.append(result)

            logger.info(f"Trial {trial} complete for {model_config.model_type}")

            # Create visualizations
        logger.info("Creating visualizations...")
        visualizer = ResultVisualizer()

        # Save metrics comparison
        metrics_plot = visualizer.plot_metrics_comparison(results)
        metrics_plot.savefig(Path(config.save_path) / "metrics_comparison.png")

        # Save training time comparison
        time_plot = visualizer.plot_training_time_comparison(results)
        time_plot.savefig(Path(config.save_path) / "training_time_comparison.png")

        # Save degree distribution for QKAN
        qkan_results = [r for r in results if 'qkan' in r.model_name.lower()]
        if qkan_results:
            degree_plot = visualizer.plot_degree_distribution(qkan_results[0])
            degree_plot.savefig(Path(config.save_path) / "degree_distribution.png")

        # Save summary to CSV
        summary_df = ModelEvaluator.create_evaluation_summary(results)
        summary_df.to_csv(Path(config.save_path) / "results_summary.csv")

        logger.info("Experiment complete!")
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()


    run_experiment(args.config_path)
