from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class ModelResult:
    model_name: str
    MSE_Score: np.float64
    R2_Score: np.float64
    train_time: float
    model_params: Dict[str, Any]

class ModelEvaluator:
    @staticmethod
    def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                             weights:Optional[np.ndarray]=None) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        :param y_true: true data
        :param y_pred: predicted data
        :param weights: optional weights
        :return: Metrics
        """
        metrics = {'mse': mean_squared_error(y_true, y_pred, sample_weight=weights)}

        #R^2
        if weights is not None:
            metrics['r2'] = r2_score(y_true, y_pred, sample_weight=weights)
        else:
            metrics['r2'] = r2_score(y_true, y_pred)

        return metrics
    @staticmethod
    def create_evaluation_summary(results: List[ModelResult]) -> pd.DataFrame:
        """Create summary DataFrame of evaluation results_js"""
        summary = []
        for result in results:
            row = {
                'model':result.model_name,
                'mse':result.MSE_Score,
                'r2':result.R2_score,
                'train_time':result.train_time,
            }
            summary.append(row)
        return pd.DataFrame(summary)