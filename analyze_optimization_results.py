import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import logging
import sys
import json

def setup_logging(results_dir):
    """Set up logging configuration."""
    log_file = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs(results_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(results_dir, log_file), mode='w')
        ]
    )
    return logging.getLogger(__name__)

def load_results(results_dir):
    """Load saved optimization results."""
    results_path = os.path.join(results_dir, 'optimization_comparison.csv')
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found at {results_path}")
    
    return pd.read_csv(results_path)

def plot_learning_curves(results_df, results_dir, metric='val_r2', figsize=(12, 6)):
    """Plot learning curves for each optimization method."""
    plt.figure(figsize=figsize)
    
    # Get unique optimization methods and configs
    methods = results_df['opt_method'].unique()
    
    for method in methods:
        method_data = results_df[results_df['opt_method'] == method]
        
        # Get the best config based on final validation metric
        grouped = method_data.groupby('config')
        if metric == 'val_r2':
            # For R², higher is better
            best_config = grouped.agg({metric: 'max'}).idxmax()[metric]
        else:
            # For MSE, lower is better
            best_config = grouped.agg({metric: 'min'}).idxmin()[metric]
        
        # Get learning curve for best config
        best_data = method_data[method_data['config'] == best_config]
        
        plt.plot(best_data['epoch'], best_data[metric], label=f"{method} (Best Config)", linewidth=2)
    
    metric_name = "Validation R²" if metric == 'val_r2' else "Validation MSE"
    plt.title(f"{metric_name} vs Epoch for Different Optimization Methods")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    if metric == 'val_mse':
        plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plot with a unique filename
    base_path = os.path.join(results_dir, f"{metric}_learning_curves.png")
    save_path = get_unique_filename(base_path)
    plt.savefig(save_path, bbox_inches='tight')
    return save_path

def plot_optimization_times(results_df, results_dir, figsize=(10, 6)):
    """Plot optimization times for each method."""
    plt.figure(figsize=figsize)
    
    # Get unique methods and their optimization times
    methods = []
    times = []
    
    for method in results_df['opt_method'].unique():
        method_data = results_df[results_df['opt_method'] == method]
        # Get first entry as opt_time is the same for all rows of a method-config combo
        opt_time = method_data.iloc[0]['opt_time']
        methods.append(method)
        times.append(opt_time)
    
    # Sort by optimization time
    sorted_indices = np.argsort(times)
    methods = [methods[i] for i in sorted_indices]
    times = [times[i] for i in sorted_indices]
    
    # Create bar chart
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(methods)))
    bars = plt.bar(methods, times, color=colors)
    
    # Add time labels on bars
    for bar, time_val in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f'{time_val:.2f}s',
            ha='center',
            fontweight='bold'
        )
    
    plt.title('Optimization Time Comparison')
    plt.xlabel('Optimization Method')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot with a unique filename
    base_path = os.path.join(results_dir, "optimization_times.png")
    save_path = get_unique_filename(base_path)
    plt.savefig(save_path, bbox_inches='tight')
    return save_path

def plot_performance_comparison(results_df, results_dir, metric='val_r2', figsize=(10, 6)):
    """Plot final performance comparison for each method."""
    plt.figure(figsize=figsize)
    
    # Get best performance for each method
    methods = []
    performances = []
    
    for method in results_df['opt_method'].unique():
        method_data = results_df[results_df['opt_method'] == method]
        
        if metric == 'val_r2':
            # For R², higher is better
            best_perf = method_data[metric].max()
        else:
            # For MSE, lower is better
            best_perf = method_data[metric].min()
            
        methods.append(method)
        performances.append(best_perf)
    
    # Sort by performance
    if metric == 'val_r2':
        # For R², sort descending
        sorted_indices = np.argsort(performances)[::-1]
    else:
        # For MSE, sort ascending
        sorted_indices = np.argsort(performances)
        
    methods = [methods[i] for i in sorted_indices]
    performances = [performances[i] for i in sorted_indices]
    
    # Create bar chart
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(methods)))
    bars = plt.bar(methods, performances, color=colors)
    
    # Add performance labels on bars
    for bar, perf_val in zip(bars, performances):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.01 if metric == 'val_r2' else 0.0001),
            f'{perf_val:.4f}',
            ha='center',
            fontweight='bold'
        )
    
    metric_name = "Validation R²" if metric == 'val_r2' else "Validation MSE"
    plt.title(f'Best {metric_name} Comparison')
    plt.xlabel('Optimization Method')
    plt.ylabel(metric_name)
    if metric == 'val_mse':
        plt.yscale('log')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot with a unique filename
    base_path = os.path.join(results_dir, f"best_{metric}_comparison.png")
    save_path = get_unique_filename(base_path)
    plt.savefig(save_path, bbox_inches='tight')
    return save_path

def plot_time_vs_performance(results_df, results_dir, metric='val_r2', figsize=(10, 8)):
    """Create a scatter plot of optimization time vs. performance."""
    plt.figure(figsize=figsize)
    
    # Prepare data for scatter plot
    methods = []
    times = []
    performances = []
    
    for method in results_df['opt_method'].unique():
        method_data = results_df[results_df['opt_method'] == method]
        
        opt_time = method_data.iloc[0]['opt_time']
        
        if metric == 'val_r2':
            # For R², higher is better
            best_perf = method_data[metric].max()
        else:
            # For MSE, lower is better
            best_perf = method_data[metric].min()
            
        methods.append(method)
        times.append(opt_time)
        performances.append(best_perf)
    
    # Create scatter plot
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(methods)))
    plt.scatter(times, performances, c=colors, s=100, alpha=0.7)
    
    # Add method labels to points
    for i, method in enumerate(methods):
        plt.annotate(
            method,
            (times[i], performances[i]),
            xytext=(10, 0),
            textcoords='offset points',
            fontsize=12,
            fontweight='bold'
        )
    
    # Add second axes for R²/MSE if needed
    if metric == 'val_r2':
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)  # R² = 0 line
        plt.axhline(y=1, color='g', linestyle='--', alpha=0.3)  # Perfect R² line
    
    # Add ideal region annotation
    if metric == 'val_r2':
        ideal_x, ideal_y = min(times), max(performances)
        plt.annotate(
            'Ideal\n(Fast & Accurate)',
            (ideal_x, ideal_y),
            xytext=(-30, 10),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green')
        )
    else:
        ideal_x, ideal_y = min(times), min(performances)
        plt.annotate(
            'Ideal\n(Fast & Accurate)',
            (ideal_x, ideal_y),
            xytext=(-30, 10),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green')
        )
    
    metric_name = "Validation R²" if metric == 'val_r2' else "Validation MSE"
    plt.title(f'Optimization Time vs {metric_name}')
    plt.xlabel('Optimization Time (seconds)')
    plt.ylabel(metric_name)
    if metric == 'val_mse':
        plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot with a unique filename
    base_path = os.path.join(results_dir, f"time_vs_{metric}.png")
    save_path = get_unique_filename(base_path)
    plt.savefig(save_path, bbox_inches='tight')
    return save_path

def analyze_best_configurations(results_df, results_dir, logger):
    """Analyze and summarize the best configurations for each method."""
    summary = {}
    
    for method in results_df['opt_method'].unique():
        method_data = results_df[results_df['opt_method'] == method]
        
        # Get best config based on R²
        best_r2_idx = method_data['val_r2'].idxmax()
        best_r2_row = method_data.loc[best_r2_idx]
        
        # Get best config based on MSE
        best_mse_idx = method_data['val_mse'].idxmin()
        best_mse_row = method_data.loc[best_mse_idx]
        
        # Parse the config string into a dictionary
        config_str = best_r2_row['config']
        try:
            config_dict = eval(config_str)
        except:
            config_dict = {"Error": "Could not parse config string"}
        
        # Store in summary
        summary[method] = {
            'best_r2': float(best_r2_row['val_r2']),  # Convert to native Python float
            'best_mse': float(best_mse_row['val_mse']),  # Convert to native Python float
            'optimization_time': float(best_r2_row['opt_time']),  # Convert to native Python float
            'best_config': config_dict,
            'param_count': int(best_r2_row['param_count'])  # Convert to native Python int
        }
        
        # Log the summary
        logger.info(f"\nBest configuration for {method}:")
        logger.info(f"  Best R²: {best_r2_row['val_r2']:.4f}")
        logger.info(f"  Best MSE: {best_mse_row['val_mse']:.6f}")
        logger.info(f"  Optimization Time: {best_r2_row['opt_time']:.2f} seconds")
        logger.info(f"  Configuration: {config_dict}")
        logger.info(f"  Parameter Count: {best_r2_row['param_count']}")
    
    # Save summary to JSON
    summary_path = os.path.join(results_dir, f"best_configs_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved to {summary_path}")
    
    # Add context for Jane Street results
    logger.info("\nJane Street Market Prediction Context:")
    logger.info("  - The Jane Street competition uses a custom weighted R² formula")
    logger.info("  - R² scores around 0.035 are considered excellent in this dataset")
    logger.info("  - Values above 0.03 on this dataset are typically competitive")
    logger.info("  - Even small improvements in R² (0.001-0.002) can translate to significant financial gains")
    logger.info("  - Optimization time is critical for practical deployment")
    
    # Calculate percentage improvement over baseline
    baselines = {'val_r2': 0.03}  # Baseline R² from literature
    for method, result in summary.items():
        if 'best_r2' in result:
            improvement = 100 * (result['best_r2'] - baselines['val_r2']) / baselines['val_r2']
            logger.info(f"  - {method}: {improvement:.2f}% improvement over baseline R²")
    
    return summary

def plot_performance_with_error_bars(results_df, results_dir, metric='val_r2', figsize=(14, 8)):
    """Create error bar plots showing the variability across hyperparameter settings."""
    plt.figure(figsize=figsize)
    
    # Group by optimization method
    methods = results_df['opt_method'].unique()
    method_data = []
    
    # Debug file to log detailed information about each configuration
    debug_log_path = os.path.join(results_dir, "error_bars_debug.log")
    with open(debug_log_path, 'w') as debug_file:
        debug_file.write(f"Debug information for {metric} error bar plot\n")
        debug_file.write("=" * 80 + "\n\n")
        
        for method in methods:
            # Get data for this method
            method_results = results_df[results_df['opt_method'] == method]
            
            # Calculate metrics for each config
            configs = method_results['config'].unique()
            config_metrics = []
            
            debug_file.write(f"\nMethod: {method}\n")
            debug_file.write(f"Number of unique configs: {len(configs)}\n")
            debug_file.write("-" * 40 + "\n")
            
            # Print all configs for debugging
            for idx, config in enumerate(configs):
                debug_file.write(f"Config {idx+1}: {config}\n")
                
                config_data = method_results[method_results['config'] == config]
                if len(config_data) > 0:
                    # Get last epoch for final performance
                    last_epoch = config_data['epoch'].max()
                    final_data = config_data[config_data['epoch'] == last_epoch]
                    
                    if len(final_data) > 0:
                        final_metric = final_data[metric].values[0]
                        config_metrics.append(final_metric)
                        debug_file.write(f"  Last Epoch: {last_epoch}, {metric}: {final_metric}\n")
                    else:
                        debug_file.write(f"  No data for last epoch {last_epoch}\n")
                else:
                    debug_file.write(f"  No data for this config\n")
            
            debug_file.write(f"\nTotal metrics collected: {len(config_metrics)}\n")
            if config_metrics:
                debug_file.write(f"Mean: {np.mean(config_metrics):.6f}\n")
                debug_file.write(f"Std: {np.std(config_metrics):.6f}\n")
                debug_file.write(f"Min: {np.min(config_metrics):.6f}\n")
                debug_file.write(f"Max: {np.max(config_metrics):.6f}\n")
                debug_file.write(f"All values: {config_metrics}\n")
            
            if config_metrics:
                method_data.append({
                    'method': method,
                    'mean': float(np.mean(config_metrics)),  # Convert to Python float
                    'std': float(np.std(config_metrics)),    # Convert to Python float
                    'min': float(np.min(config_metrics)),    # Convert to Python float
                    'max': float(np.max(config_metrics)),    # Convert to Python float
                    'count': int(len(config_metrics)),       # Convert to Python int
                    'values': config_metrics.copy(),         # Store actual values for debugging
                    'has_variation': float(np.std(config_metrics)) > 1e-6  # Flag if method shows variation
                })
    
    print(f"Debug information written to {debug_log_path}")
    
    # Check for methods with zero standard deviation 
    methods_without_variation = []
    for data in method_data:
        if data['std'] < 1e-6:  # Effectively zero
            print(f"WARNING: Method {data['method']} has zero deviation across {data['count']} configs!")
            methods_without_variation.append(data['method'])
    
    # Sort methods by mean performance
    method_data.sort(key=lambda x: x['mean'], reverse=(metric == 'val_r2'))
    
    # Extract data for plotting
    methods = [data['method'] for data in method_data]
    means = [data['mean'] for data in method_data]
    stds = [data['std'] for data in method_data]
    mins = [data['min'] for data in method_data]
    maxs = [data['max'] for data in method_data]
    has_variation = [data['has_variation'] for data in method_data]
    
    # Ensure stds have some minimum value to make error bars visible even when close to zero
    stds = [max(s, 1e-5) for s in stds]  # Set minimum std to a small value for visibility
    
    # Plot bar chart with standard deviation error bars
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(methods)))
    
    # Use hatched bars for methods without variation
    bars = []
    for i, (method, mean, has_var) in enumerate(zip(methods, means, has_variation)):
        if has_var:
            # Regular bar for methods with variation
            bar = plt.bar(i, mean, color=colors[i], alpha=0.7)
        else:
            # Hatched bar for methods without variation
            bar = plt.bar(i, mean, color=colors[i], alpha=0.7, hatch='///')
        bars.append(bar[0])
    
    # Add error bars using standard deviation
    for i, (mean, std, has_var) in enumerate(zip(means, stds, has_variation)):
        if has_var:
            plt.errorbar(
                i, 
                mean,
                yerr=std,
                fmt='none', 
                capsize=10,
                color='black',
                linewidth=1.5
            )
    
    # Add min-max ranges as separate markers
    for i, (mean, min_val, max_val, has_var) in enumerate(zip(means, mins, maxs, has_variation)):
        if has_var:
            # Add min marker
            plt.plot([i, i], [min_val, mean], 'k--', alpha=0.5, linewidth=1)
            plt.plot(i, min_val, 'v', color='blue', alpha=0.7, markersize=8)
            
            # Add max marker
            plt.plot([i, i], [mean, max_val], 'k--', alpha=0.5, linewidth=1)
            plt.plot(i, max_val, '^', color='red', alpha=0.7, markersize=8)
    
    # Add count and std labels on top of bars
    for i, (bar, data) in enumerate(zip(bars, method_data)):
        if data['has_variation']:
            height = data['max']  # Use max value for positioning
            std_text = f'σ={data["std"]:.4f}'
        else:
            height = data['mean'] + 0.001  # Small offset for methods without variation
            std_text = f'σ=0 (No variation)'
            
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.001,  # Small offset above the value
            f'N={data["count"]}\n{std_text}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=9
        )
    
    # Add individual data points for each configuration
    for i, (method, has_var) in enumerate(zip(methods, has_variation)):
        method_results = results_df[results_df['opt_method'] == method]
        configs = method_results['config'].unique()
        
        # Jitter x positions slightly for better visibility
        jitter_width = 0.3  # Width of jitter
        
        for j, config in enumerate(configs):
            config_data = method_results[method_results['config'] == config]
            if len(config_data) > 0:
                # Get last epoch for final performance
                last_epoch = config_data['epoch'].max()
                final_data = config_data[config_data['epoch'] == last_epoch]
                
                if len(final_data) > 0:
                    # Calculate jittered x position
                    x_jitter = i + (j / len(configs) - 0.5) * jitter_width if has_var else i
                    
                    # Only show individual points for methods with variation
                    if has_var:
                        plt.scatter(
                            x_jitter, 
                            final_data[metric].values[0],
                            color='white',
                            edgecolor='black',
                            s=50,
                            zorder=10,
                            alpha=0.7
                        )
    
    metric_name = "Validation R²" if metric == 'val_r2' else "Validation MSE"
    plt.title(f'{metric_name} Across Hyperparameter Settings', fontsize=14)
    plt.xlabel('Optimization Method', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    
    # Set x-ticks to show method names instead of numbers
    plt.xticks(range(len(methods)), methods)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend for min/max markers
    handles = []
    labels = []
    
    # Only add variation-related elements if at least one method has variation
    if any(has_variation):
        handles.append(plt.Line2D([], [], color='k', marker='v', markerfacecolor='blue', markersize=8, linestyle=''))
        labels.append('Min Value')
        
        handles.append(plt.Line2D([], [], color='k', marker='^', markerfacecolor='red', markersize=8, linestyle=''))
        labels.append('Max Value')
        
        handles.append(plt.Line2D([], [], color='k', marker='o', markerfacecolor='white', markersize=8, linestyle=''))
        labels.append('Individual Config')
        
        handles.append(plt.Line2D([], [], color='k', linestyle='-'))
        labels.append('±1 Std Dev')
    
    # Add hatched bar for methods without variation
    handles.append(plt.Rectangle((0,0), 1, 1, fc=colors[0], hatch='///'))
    labels.append('Method with No Variation')
    
    plt.legend(handles, labels, loc='best')
    
    # Add annotation about methods without variation
    if methods_without_variation:
        methods_list = ', '.join(methods_without_variation)
        plt.figtext(
            0.5, 0.01, 
            f"Note: {methods_list} show no performance variation across different hyperparameter settings.\n"
            f"This suggests these methods may not be effectively utilizing the hyperparameters.",
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout(rect=[0, 0.05, 1, 1] if methods_without_variation else None)
    
    # Save plot with a unique filename
    base_path = os.path.join(results_dir, f"{metric}_error_bars.png")
    save_path = get_unique_filename(base_path)
    plt.savefig(save_path, bbox_inches='tight')
    return save_path

def plot_hyperparameter_heatmap(results_df, results_dir, method, param1, param2, metric='val_r2', figsize=(12, 10)):
    """Create a heatmap showing how two hyperparameters affect performance for a specific method."""
    # Filter for the specified method
    method_data = results_df[results_df['opt_method'] == method]
    
    if len(method_data) == 0:
        print(f"No data found for method {method}")
        return None
    
    # Extract hyperparameters from config string
    # Configs are stored as string representation of dictionaries
    configs = []
    for config_str in method_data['config'].unique():
        try:
            # Clean up and evaluate the config string
            config_str = config_str.replace("'", '"')  # Replace single quotes with double quotes
            config_dict = eval(config_str)
            configs.append(config_dict)
        except:
            print(f"Could not parse config: {config_str}")
            continue
    
    # Check if we have the required hyperparameters
    if not all(param1 in config and param2 in config for config in configs):
        print(f"Not all configs have {param1} and {param2}")
        return None
    
    # Group by the two hyperparameters
    param1_values = sorted(set(config[param1] for config in configs))
    param2_values = sorted(set(config[param2] for config in configs))
    
    # Create a 2D grid for the heatmap
    heatmap_data = np.zeros((len(param1_values), len(param2_values)))
    
    # Fill in the heatmap data
    for i, val1 in enumerate(param1_values):
        for j, val2 in enumerate(param2_values):
            # Find configs matching these hyperparameter values
            matching_configs = []
            for config_str in method_data['config'].unique():
                try:
                    config_dict = eval(config_str)
                    if config_dict.get(param1) == val1 and config_dict.get(param2) == val2:
                        matching_configs.append(config_str)
                except:
                    continue
            
            # Get performance for matching configs
            if matching_configs:
                metrics = []
                for config in matching_configs:
                    config_data = method_data[method_data['config'] == config]
                    if len(config_data) > 0:
                        # Get last epoch for final performance
                        last_epoch = config_data['epoch'].max()
                        final_data = config_data[config_data['epoch'] == last_epoch]
                        if len(final_data) > 0:
                            metrics.append(final_data[metric].values[0])
                
                if metrics:
                    heatmap_data[i, j] = np.mean(metrics)
    
    # Create heatmap
    plt.figure(figsize=figsize)
    plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar()
    metric_name = "Validation R²" if metric == 'val_r2' else "Validation MSE"
    cbar.set_label(metric_name)
    
    # Add gridlines
    plt.grid(visible=False)
    
    # Add labels
    plt.xticks(np.arange(len(param2_values)), param2_values)
    plt.yticks(np.arange(len(param1_values)), param1_values)
    plt.xlabel(param2)
    plt.ylabel(param1)
    
    # Add values in each cell
    for i in range(len(param1_values)):
        for j in range(len(param2_values)):
            if not np.isnan(heatmap_data[i, j]) and heatmap_data[i, j] != 0:
                plt.text(j, i, f'{heatmap_data[i, j]:.4f}',
                        ha='center', va='center', 
                        color='white' if heatmap_data[i, j] > np.max(heatmap_data) / 2 else 'black')
    
    plt.title(f'Effect of {param1} and {param2} on {metric_name} for {method}')
    plt.tight_layout()
    
    # Save plot with a unique filename
    base_path = os.path.join(results_dir, f"{method}_{param1}_{param2}_{metric}_heatmap.png")
    save_path = get_unique_filename(base_path)
    plt.savefig(save_path, bbox_inches='tight')
    return save_path

# Utility function to generate unique filenames
def get_unique_filename(base_path):
    """Generate a unique filename if the base_path already exists."""
    if not os.path.exists(base_path):
        return base_path
    
    dir_name = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)
    name, ext = os.path.splitext(base_name)
    
    # If name already has a counter, extract it
    import re
    match = re.match(r'(.+?)_(\d+)$', name)
    if match:
        name = match.group(1)
        counter = int(match.group(2))
    else:
        counter = 1
    
    # Increment counter until we find a free filename
    while True:
        new_name = f"{name}_{counter}{ext}"
        new_path = os.path.join(dir_name, new_name)
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize KAN optimization results')
    parser.add_argument('--results_dir', type=str, default='optimization_results_js', 
                        help='Directory containing optimization results')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.results_dir)
    logger.info(f"Analyzing Jane Street optimization results in {args.results_dir}")
    
    try:
        # Load results
        results_df = load_results(args.results_dir)
        logger.info(f"Loaded {len(results_df)} datapoints from results")
        
        # Generate standard plots
        logger.info("Generating learning curve plots...")
        r2_curve_path = plot_learning_curves(results_df, args.results_dir, metric='val_r2')
        mse_curve_path = plot_learning_curves(results_df, args.results_dir, metric='val_mse')
        logger.info(f"Learning curve plots saved to:\n  {r2_curve_path}\n  {mse_curve_path}")
        
        logger.info("Generating optimization time comparison...")
        time_plot_path = plot_optimization_times(results_df, args.results_dir)
        logger.info(f"Optimization time plot saved to:\n  {time_plot_path}")
        
        logger.info("Generating performance comparison plots...")
        r2_perf_path = plot_performance_comparison(results_df, args.results_dir, metric='val_r2')
        mse_perf_path = plot_performance_comparison(results_df, args.results_dir, metric='val_mse')
        logger.info(f"Performance comparison plots saved to:\n  {r2_perf_path}\n  {mse_perf_path}")
        
        logger.info("Generating time vs. performance plots...")
        time_r2_path = plot_time_vs_performance(results_df, args.results_dir, metric='val_r2')
        time_mse_path = plot_time_vs_performance(results_df, args.results_dir, metric='val_mse')
        logger.info(f"Time vs. performance plots saved to:\n  {time_r2_path}\n  {time_mse_path}")
        
        # Generate error bar plots
        logger.info("Generating error bar plots...")
        r2_error_path = plot_performance_with_error_bars(results_df, args.results_dir, metric='val_r2')
        mse_error_path = plot_performance_with_error_bars(results_df, args.results_dir, metric='val_mse')
        logger.info(f"Error bar plots saved to:\n  {r2_error_path}\n  {mse_error_path}")
        
        # Generate heatmaps for each method
        logger.info("Generating hyperparameter heatmaps...")
        for method in results_df['opt_method'].unique():
            # Check if we have enough configuration variety for a meaningful heatmap
            method_data = results_df[results_df['opt_method'] == method]
            if len(method_data['config'].unique()) >= 4:  # Only if we have at least 4 configs
                try:
                    heatmap_path = plot_hyperparameter_heatmap(
                        results_df, args.results_dir, method, 
                        'max_degree', 'hidden_size', metric='val_r2'
                    )
                    if heatmap_path:
                        logger.info(f"Heatmap for {method} saved to: {heatmap_path}")
                except Exception as e:
                    logger.warning(f"Could not generate heatmap for {method}: {str(e)}")
        
        # Analyze best configurations
        logger.info("Analyzing best configurations...")
        summary = analyze_best_configurations(results_df, args.results_dir, logger)
        
        # Add context for Jane Street results
        logger.info("\nJane Street Market Prediction Context:")
        logger.info("  - The Jane Street competition uses a custom weighted R² formula")
        logger.info("  - R² scores around 0.035 are considered excellent in this dataset")
        logger.info("  - Values above 0.03 on this dataset are typically competitive")
        logger.info("  - Even small improvements in R² (0.001-0.002) can translate to significant financial gains")
        logger.info("  - Optimization time is critical for practical deployment")
        
        # Calculate percentage improvement over baseline
        baselines = {'val_r2': 0.03}  # Baseline R² from literature
        for method, result in summary.items():
            if 'best_r2' in result:
                improvement = 100 * (result['best_r2'] - baselines['val_r2']) / baselines['val_r2']
        logger.info(f"  - {method}: {improvement:.2f}% improvement over baseline R²")
        
        logger.info("Analysis completed successfully!")
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 