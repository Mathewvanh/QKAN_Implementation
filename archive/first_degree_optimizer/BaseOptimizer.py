from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np
import polars as pl
class BaseOptimizer(ABC):
    """Base class for QKAN optimizers implementing shared functionality"""

    def __init__(self):
        """Initialize the optimizer with common attributes"""
        self.fold_caches = {} #Cache per fold

    def _compute_collapsed_combinations(self, x_data: pl.DataFrame, fold_id: str = None) -> Dict[int, np.ndarray]:
        """
        Precompute and collapse Chebyshev polynomial combinations.
        :param x_data: Input data
        :param fold_id: Optional identifier for fold (for cross-validation)
        :return: Dictionary mapping degrees to collapsed combinations
        """
        cache_key = f"{fold_id}_{hash(str(x_data))}" if fold_id else hash(str(x_data))
        if cache_key in self.fold_caches:
            return self.fold_caches[cache_key]

        feature_data = x_data.select(pl.col('^feature_.*$')).to_numpy()

        transforms = self._compute_transforms(feature_data)

        self.fold_caches[cache_key] = transforms
        return transforms

    @abstractmethod
    def _compute_transforms(self, feature_data: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute specific transforms for the optimizer.
        Must be implemented by child classes.
        """
        pass

    def _get_expanding_window_folds(self, timestamps: pl.DataFrame, data: pl.DataFrame, n_splits: int = 5, initial_ratio: float = 0.6) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create expanding window cross-validation folds.

        Args:
            timestamps: DataFrame containing date_id timestamps
            data: Input DataFrame with normalized features
            n_splits: Number of validation periods
            initial_ratio: Initial training set size as ratio of total data

        Returns:
            List of (train_mask, val_mask) tuples
        """
        # Get unique sorted timestamps
        unique_timestamps = timestamps.get_column('date_id').unique().sort()
        n_times = len(unique_timestamps)

        initial_train_size = int(n_times * initial_ratio)
        val_size = int((n_times - initial_train_size) / n_splits)

        folds = []
        for i in range(n_splits):
            train_end_idx = initial_train_size + i * val_size
            train_times = unique_timestamps[:train_end_idx]
            val_times = unique_timestamps[train_end_idx:min(train_end_idx + val_size, n_times)]

            # Create masks using timestamps DataFrame
            train_mask = timestamps.get_column('date_id').is_in(train_times).to_numpy()
            val_mask = timestamps.get_column('date_id').is_in(val_times).to_numpy()

            folds.append((train_mask, val_mask))
        return folds
    def _get_time_based_folds(self, timestamps: pl.DataFrame, data: pl.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time-based cross-validation folds.

        Args:
            timestamps: DataFrame containing date_id timestamps
            data: Input DataFrame with normalized features
            n_splits: Number of validation periods

        Returns:
            List of (train_mask, val_mask) tuples
        """
        # Get unique sorted timestamps
        unique_timestamps = timestamps.get_column('date_id').unique().sort()
        n_times = len(unique_timestamps)

        folds = []
        for i in range(n_splits):
            split_idx = int((i + 1) * n_times // (n_splits + 1))
            val_end_idx = int((i + 2) * n_times // (n_splits + 1))

            train_times = unique_timestamps[:split_idx]
            val_times = unique_timestamps[split_idx:val_end_idx]

            # Create masks using timestamps DataFrame
            train_mask = timestamps.get_column('date_id').is_in(train_times).to_numpy()
            val_mask = timestamps.get_column('date_id').is_in(val_times).to_numpy()

            folds.append((train_mask, val_mask))
        return folds
    def _compute_validation_score(self, predictions: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute MSE validation score.

        Args:
            predictions: Predicted values
            y_true: True values

        Returns:
            MSE score
        """
        return np.mean((y_true - predictions) ** 2)

    def save_state(self, filename: str, ) -> None:
        """
        Save optimizer state.
        Must be implemented by child classes.
        """
        pass

    def load_state(self, filename: str, current_query_params: dict) -> None:
        """
        Load optimizer state.
        Must be implemented by child classes.
        """
        pass