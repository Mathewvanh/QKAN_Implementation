import polars as pl
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass

from polars import DataFrame
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import logging

@dataclass
class BacktestConfig:
    """Configuration for backtesting pipeline"""
    data_path: str
    start_row: int = 13000000  # Start well before our previous training data
    n_rows: int = 200000
    train_ratio: float = 0.7
    feature_cols: list[str] = None
    target_col: str = None
    weight_col: str = None
    date_col: str = None

class DataPipelineBacktest:
    def __init__(self, config: BacktestConfig, logger: logging.Logger):
        self.robust_scaler = None
        self.config = config
        self.minmax_scaler = None
        self.logger: logging.Logger = logger

    def load_and_preprocess_data(self) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
        """Load and preprocess data for backtesting"""
        lf = pl.scan_parquet(self.config.data_path).fill_null(3)

        # Create optimized query for backtesting
        query = (lf
            .with_row_count("row_number")  # More efficient way to add row numbers
            .filter(pl.col("row_number").is_between(
                self.config.start_row,
                self.config.start_row + self.config.n_rows - 1
            ))
            .select([
                pl.col(self.config.date_col),
                pl.col(self.config.target_col),
                pl.col(self.config.weight_col),
                *[pl.col(f) for f in self.config.feature_cols]
            ])
            .sort(self.config.date_col))

        # Normalize features with streaming
        df = self._normalize_features(query)

        # Split data with streaming collection and better memory management
        df_collected = df.collect(streaming=True)
        train_df, train_target, train_weight, val_df, val_target, val_weight = self._train_val_split(df_collected)
        del df_collected  # Help with memory management

        return train_df, train_target, train_weight, val_df, val_target, val_weight

    def _normalize_features(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Normalize features to [-1,1] using RobustScaler"""
        stats = lf.select([
            *[pl.col(col).quantile(0.05).alias(f"{col}_q05") for col in self.config.feature_cols],
            *[pl.col(col).quantile(0.95).alias(f"{col}_q95") for col in self.config.feature_cols],
            *[pl.col(col).std().alias(f"{col}_std") for col in self.config.feature_cols],
            pl.col(self.config.target_col).quantile(0.05).alias(f'{self.config.target_col}_q05'),
            pl.col(self.config.target_col).quantile(0.95).alias(f'{self.config.target_col}_q95'),
            pl.col(self.config.target_col).std().alias(f'{self.config.target_col}_std'),
        ]).collect()

        normalized_features_and_resp = []
        for col in self.config.feature_cols + [self.config.target_col]:
            q05 = stats.get_column(f'{col}_q05')[0]
            q95 = stats.get_column(f'{col}_q95')[0]
            std = stats.get_column(f'{col}_std')[0]

            center = (q95 + q05) / 2
            scale = (q95 - q05) / 2 if abs(q95 - q05) > 1e-10 else std if std > 1e-10 else 1.0

            normalized_features_and_resp.append(
                pl.when(pl.col(col) > q95)
                .then(1.0)
                .when(pl.col(col) < q05)
                .then(-1.0)
                .otherwise((pl.col(col) - center) / scale)
                .alias(f"{col}_normalized")
            )
        return lf.select([pl.col(self.config.date_col), pl.col(self.config.weight_col), *normalized_features_and_resp])

    def _train_val_split(self, df: pl.DataFrame) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
        """Split dataset into train and validation, maintaining time order"""
        unique_dates = df.get_column(self.config.date_col).unique().sort()
        split_idx = int(len(unique_dates) * self.config.train_ratio)

        train_dates = unique_dates[:split_idx]
        val_dates = unique_dates[split_idx:]

        train_mask = df.get_column('date_id').is_in(train_dates).to_numpy()
        val_mask = df.get_column('date_id').is_in(val_dates).to_numpy()

        train_data = df.filter(train_mask).select([pl.col(f'{col}_normalized') for col in self.config.feature_cols])
        val_data = df.filter(val_mask).select([pl.col(f'{col}_normalized') for col in self.config.feature_cols])

        train_target = df.filter(train_mask).select(f'{self.config.target_col}_normalized')
        val_target = df.filter(val_mask).select(f'{self.config.target_col}_normalized')

        train_weights = df.filter(train_mask).select(self.config.weight_col)
        val_weights = df.filter(val_mask).select(self.config.weight_col)

        return train_data, train_target, train_weights, val_data, val_target, val_weights

    def get_date_info(self) -> tuple[list, list]:
        """Get unique dates for train and validation periods"""
        lf = pl.scan_parquet(self.config.data_path)
        df = (lf.select([pl.col(self.config.date_col)])
             .offset(self.config.start_row)
             .take(self.config.n_rows)
             .sort(self.config.date_col)
             .collect())
        
        unique_dates = df.get_column(self.config.date_col).unique().sort()
        split_idx = int(len(unique_dates) * self.config.train_ratio)
        
        train_dates = unique_dates[:split_idx].to_list()
        val_dates = unique_dates[split_idx:].to_list()
        
        return train_dates, val_dates
