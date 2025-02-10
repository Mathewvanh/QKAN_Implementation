from typing import List
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

from evaluation import ModelResult
class ResultVisualizer:
    @staticmethod
    def plot_metrics_comparison(results:List[ModelResult], metric:str='mse'):
        """Plot comparison of metrics across models"""
        plt.figure(figsize=(10, 6))

        models = [r.model_name for r in results]
        train_metrics = [r.train_metrics[metric] for r in results]
        val_metrics = [r.val_metrics[metric] for r in results]

        x = np.arange(len(models))
        width = 0.35

        plt.bar(x - width / 2, train_metrics, width, label='Train')
        plt.bar(x + width / 2, val_metrics, width, label='Validation')

        plt.xlabel('Models')
        plt.ylabel(metric.upper())
        plt.title(f'{metric.upper()} Comparison')
        plt.xticks(x, models)
        plt.legend()

        return plt.gcf()
    @staticmethod
    def plot_training_time_comparison(results: List[ModelResult]):
        """Plot comparison of training times"""
        plt.figure(figsize=(10, 6))

        models = [r.model_name for r in results]
        times = [r.train_time for r in results]

        plt.bar(models, times)
        plt.xlabel('Models')
        plt.ylabel('Training Time (s)')
        plt.title('Training Time Comparison')

        return plt.gcf()

    @staticmethod
    def plot_degree_distribution(qkan_result: ModelResult):
        """Plot distribution of polynomial degrees for QKAN"""
        degrees = qkan_result.model_params['degrees']
        degrees_flat = [d for sublist in degrees for d in sublist]

        plt.figure(figsize=(10, 6))
        sns.histplot(degrees_flat, discrete=True)
        plt.xlabel('Polynomial Degree')
        plt.ylabel('Count')
        plt.title('Distribution of Polynomial Degrees')

        return plt.gcf()