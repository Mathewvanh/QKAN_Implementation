#!/usr/bin/env python
"""
Script to generate plots from architecture comparison results.
This can be run separately after the comparison is complete.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_plot_directories(results_dir):
    """Create directories for plots."""
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create subdirectories for different plot types
    os.makedirs(os.path.join(plots_dir, "performance"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "hyperparams"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "comparison"), exist_ok=True)
    
    return plots_dir

def load_results(results_dir):
    """Load results from CSV file."""
    results_file = os.path.join(results_dir, "architecture_comparison_results.csv")
    if not os.path.exists(results_file):
        # Try to find the most recent results file
        files = [f for f in os.listdir(results_dir) if f.startswith("architecture_comparison_results_") and f.endswith(".csv")]
        if not files:
            raise FileNotFoundError(f"No results file found in {results_dir}")
        
        # Sort by modification time
        files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
        results_file = os.path.join(results_dir, files[0])
        logger.info(f"Using most recent results file: {results_file}")
    
    # Load the results
    results_df = pd.read_csv(results_file)
    logger.info(f"Loaded {len(results_df)} results from {results_file}")
    
    return results_df

def plot_architecture_comparison(results_df, plots_dir, min_r2=-5.0, jane_street_baseline=0.03):
    """Generate plots comparing architectures."""
    # Filter out extreme negative values if needed
    filtered_df = results_df[results_df['val_r2'] > min_r2].copy()
    logger.info(f"Filtered out {len(results_df) - len(filtered_df)} results with R² < {min_r2}")
    
    # Add percentage improvement column
    if 'improvement_pct' not in filtered_df.columns:
        filtered_df['improvement_pct'] = (filtered_df['val_r2'] - jane_street_baseline) / jane_street_baseline * 100
    
    # 1. Architecture comparison (R²)
    plt.figure(figsize=(12, 6))
    best_vals = filtered_df.groupby('architecture')['val_r2'].max().reset_index()
    best_vals = best_vals.sort_values('val_r2', ascending=False)
    sns.barplot(x='architecture', y='val_r2', data=best_vals, order=best_vals['architecture'])
    
    # Add jane street baseline
    plt.axhline(y=jane_street_baseline, color='r', linestyle='--', label=f'Jane Street Baseline (R²={jane_street_baseline})')
    
    plt.title('Best Validation R² by Architecture')
    plt.ylabel('Validation R²')
    plt.xlabel('Architecture')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "comparison", "architecture_r2_comparison.png"))
    logger.info(f"Generated architecture R² comparison plot")
    
    # 2. Architecture comparison (improvement percentage)
    plt.figure(figsize=(12, 6))
    best_improvement = filtered_df.groupby('architecture')['improvement_pct'].max().reset_index()
    best_improvement = best_improvement.sort_values('improvement_pct', ascending=False)
    
    # Create the bar plot
    ax = sns.barplot(x='architecture', y='improvement_pct', data=best_improvement, order=best_improvement['architecture'])
    
    # Add value labels on top of bars
    for i, bar in enumerate(ax.patches):
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.5,
            f"{value:.1f}%",
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    
    plt.title('Percentage Improvement Over Jane Street Baseline (R²=0.03)')
    plt.ylabel('Improvement %')
    plt.xlabel('Architecture')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "comparison", "architecture_improvement_pct.png"))
    logger.info(f"Generated improvement percentage comparison plot")
    
    return filtered_df

def plot_parameters_vs_performance(filtered_df, plots_dir):
    """Generate plots comparing parameter counts vs performance."""
    # Parameter count vs. performance
    plt.figure(figsize=(14, 8))
    
    # Create a scatter plot with different markers for each architecture
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    colors = sns.color_palette("bright", n_colors=len(filtered_df['architecture'].unique()))
    
    for i, arch in enumerate(sorted(filtered_df['architecture'].unique())):
        arch_data = filtered_df[filtered_df['architecture'] == arch]
        plt.scatter(
            arch_data['param_count'], 
            arch_data['val_r2'], 
            label=arch,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            alpha=0.7,
            s=80
        )
    
    # Add trendline for all data
    if len(filtered_df) > 1:
        z = np.polyfit(filtered_df['param_count'], filtered_df['val_r2'], 1)
        p = np.poly1d(z)
        plt.plot(sorted(filtered_df['param_count'].unique()), 
                 p(sorted(filtered_df['param_count'].unique())), 
                 "k--", alpha=0.5, label="Trend")
    
    plt.title('Parameter Count vs. Validation R²', fontsize=14)
    plt.xlabel('Parameter Count', fontsize=12)
    plt.ylabel('Validation R²', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10, framealpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "performance", "params_vs_performance.png"), dpi=300)
    logger.info(f"Generated parameter count vs. performance plot")
    
    return

def plot_optimization_time_vs_performance(filtered_df, plots_dir):
    """Generate plots comparing optimization time vs performance."""
    # Optimization time vs. performance
    plt.figure(figsize=(14, 8))
    
    # Create a scatter plot with different markers for each architecture
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    colors = sns.color_palette("bright", n_colors=len(filtered_df['architecture'].unique()))
    
    for i, arch in enumerate(sorted(filtered_df['architecture'].unique())):
        arch_data = filtered_df[filtered_df['architecture'] == arch]
        plt.scatter(
            arch_data['opt_time'], 
            arch_data['val_r2'], 
            label=arch,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            alpha=0.7,
            s=80
        )
    
    # Create ideal point annotation
    x_min = filtered_df['opt_time'].min()
    y_max = filtered_df['val_r2'].max()
    plt.annotate(
        'Ideal\n(Fast & Accurate)',
        xy=(x_min, y_max),
        xytext=(x_min + 0.2, y_max - 0.005),
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
    )
    
    plt.title('Optimization Time vs. Validation R²', fontsize=14)
    plt.xlabel('Optimization Time (seconds)', fontsize=12)
    plt.ylabel('Validation R²', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10, framealpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "performance", "optimization_time_vs_performance.png"), dpi=300)
    logger.info(f"Generated optimization time vs. performance plot")
    
    return

def plot_hyperparameter_effects(filtered_df, plots_dir):
    """Generate plots for hyperparameter effects."""
    # 1. Learning rate comparison
    if 'learning_rate' in filtered_df.columns:
        plt.figure(figsize=(14, 8))
        
        # Sort architectures by average R² performance
        arch_order = filtered_df.groupby('architecture')['val_r2'].mean().sort_values(ascending=False).index.tolist()
        
        # Create the boxplot
        ax = sns.boxplot(
            x='architecture', 
            y='val_r2', 
            hue='learning_rate', 
            data=filtered_df,
            order=arch_order
        )
        
        plt.title('Impact of Learning Rate on Performance', fontsize=14)
        plt.ylabel('Validation R²', fontsize=12)
        plt.xlabel('Architecture', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Learning Rate', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "hyperparams", "learning_rate_comparison.png"), dpi=300)
        logger.info(f"Generated learning rate comparison plot")
    
    # 2. Hidden size comparison
    if 'hidden_size' in filtered_df.columns:
        plt.figure(figsize=(14, 8))
        
        # Sort architectures by average R² performance
        arch_order = filtered_df.groupby('architecture')['val_r2'].mean().sort_values(ascending=False).index.tolist()
        
        # Create the boxplot
        ax = sns.boxplot(
            x='architecture', 
            y='val_r2', 
            hue='hidden_size', 
            data=filtered_df,
            order=arch_order
        )
        
        plt.title('Impact of Hidden Size on Performance', fontsize=14)
        plt.ylabel('Validation R²', fontsize=12)
        plt.xlabel('Architecture', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Hidden Size', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "hyperparams", "hidden_size_comparison.png"), dpi=300)
        logger.info(f"Generated hidden size comparison plot")
    
    # 3. Architecture-specific hyperparameter plots
    arch_specific_params = {
        'SplineKAN': ['k'],
        'WaveletKAN': ['wavelet_type'],
        'FourierKAN': ['gridsize'],
        'JacobiKAN': ['degree'],
        'ChebyshevKAN': ['degree'],
        'TaylorKAN': ['order'],
        'RBFKAN': ['num_centers'],
        'MixtureKAN': ['experts_type'],
        'Transformer': ['num_heads', 'num_layers'],
        'LSTM': ['num_layers', 'bidirectional']
    }
    
    for arch, params in arch_specific_params.items():
        arch_data = filtered_df[filtered_df['architecture'] == arch]
        if len(arch_data) == 0:
            continue  # Skip if no data for this architecture
            
        for param in params:
            if param in arch_data.columns:
                # Create boxplot for this parameter
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=param, y='val_r2', data=arch_data)
                plt.title(f'{arch}: Impact of {param} on R²', fontsize=14)
                plt.ylabel('Validation R²', fontsize=12)
                plt.xlabel(param, fontsize=12)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "hyperparams", f"{arch}_{param}_impact.png"), dpi=300)
                
                # Create heatmap for this parameter vs hidden_size
                if 'hidden_size' in arch_data.columns:
                    try:
                        pivot_data = arch_data.pivot_table(
                            values='val_r2', 
                            index='hidden_size', 
                            columns=param, 
                            aggfunc='mean'
                        )
                        
                        plt.figure(figsize=(10, 6))
                        sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.4f')
                        plt.title(f'{arch}: {param} vs Hidden Size (R²)', fontsize=14)
                        plt.tight_layout()
                        plt.savefig(os.path.join(plots_dir, "hyperparams", f"{arch}_{param}_hidden_size_heatmap.png"), dpi=300)
                    except:
                        logger.warning(f"Could not create heatmap for {arch} {param} vs hidden_size")
                        
                logger.info(f"Generated hyperparameter impact plot for {arch} - {param}")
    
    return

def plot_summary_table(filtered_df, plots_dir, jane_street_baseline=0.03):
    """Generate a summary table as an image."""
    # Get best configuration for each architecture
    summary_data = []
    for arch in filtered_df['architecture'].unique():
        arch_data = filtered_df[filtered_df['architecture'] == arch]
        best_idx = arch_data['val_r2'].idxmax()
        best_row = arch_data.loc[best_idx]
        
        # Calculate improvement over baseline
        improvement = (best_row['val_r2'] - jane_street_baseline) / jane_street_baseline * 100
        
        # Create a summary row
        summary_row = {
            'Architecture': arch,
            'Best R²': f"{best_row['val_r2']:.4f}",
            'Improvement': f"{improvement:.2f}%",
            'Parameters': f"{int(best_row['param_count']):,}",
            'Opt Time (s)': f"{best_row['opt_time']:.2f}"
        }
        
        # Add configuration details
        config_details = []
        for col in best_row.index:
            if col not in ['architecture', 'val_r2', 'param_count', 'opt_time', 'val_mse']:
                if isinstance(best_row[col], (int, float)):
                    config_details.append(f"{col}={best_row[col]}")
                else:
                    config_details.append(f"{col}={best_row[col]}")
        
        summary_row['Configuration'] = ", ".join(config_details)
        summary_data.append(summary_row)
    
    # Convert to DataFrame and sort by Best R²
    summary_df = pd.DataFrame(summary_data)
    summary_df['Best R² (numeric)'] = summary_df['Best R²'].astype(float)
    summary_df = summary_df.sort_values('Best R² (numeric)', ascending=False).drop('Best R² (numeric)', axis=1)
    
    # Create a figure and axis
    fig, ax = plt.figure(figsize=(12, len(summary_df) * 0.5 + 1)), plt.gca()
    
    # Hide axes
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Create the table
    table = plt.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2'] * len(summary_df.columns)
    )
    
    # Adjust font size and alignment
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the header
    for j, cell in enumerate(table._cells[(0, j)] for j in range(len(summary_df.columns))):
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4472C4')
    
    # Add a title
    plt.title('Architecture Performance Summary (Jane Street Dataset)', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "comparison", "architecture_summary_table.png"), dpi=300, bbox_inches='tight')
    logger.info(f"Generated summary table as image")
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description='Generate plots from architecture comparison results')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing comparison results')
    parser.add_argument('--min_r2', type=float, default=-5.0, help='Minimum R² value to include in plots')
    parser.add_argument('--js_baseline', type=float, default=0.03, help='Jane Street baseline R² value')
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        logger.error(f"Results directory {args.results_dir} does not exist")
        return
    
    # Create plot directories
    plots_dir = create_plot_directories(args.results_dir)
    
    # Load results
    try:
        results_df = load_results(args.results_dir)
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return
    
    # Generate plots
    try:
        logger.info("Generating architecture comparison plots...")
        filtered_df = plot_architecture_comparison(results_df, plots_dir, args.min_r2, args.js_baseline)
        
        logger.info("Generating parameter vs. performance plots...")
        plot_parameters_vs_performance(filtered_df, plots_dir)
        
        logger.info("Generating optimization time vs. performance plots...")
        plot_optimization_time_vs_performance(filtered_df, plots_dir)
        
        logger.info("Generating hyperparameter effect plots...")
        plot_hyperparameter_effects(filtered_df, plots_dir)
        
        logger.info("Generating summary table...")
        summary_df = plot_summary_table(filtered_df, plots_dir, args.js_baseline)
        
        # Save summary as CSV
        summary_file = os.path.join(args.results_dir, "architecture_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved summary to {summary_file}")
        
        logger.info("All plots generated successfully!")
        logger.info(f"Plots saved in {plots_dir}")
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}", exc_info=True)
        return

if __name__ == "__main__":
    main() 