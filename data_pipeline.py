import polars as pl
import numpy as np
from typing import Tuple, Dict

from polars import DataFrame
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from data_pipeline_js_config import DataConfig
import logging
class DataPipeline:
    def __init__(self, config: DataConfig, logger: logging.Logger):
        self.robust_scaler = None
        self.config = config
        self.minmax_scaler = None
        self.logger: logging.Logger = logger

    def load_and_preprocess_data(self) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
        """Load and preprocess data, returning train and validation"""
        lf = pl.scan_parquet(self.config.data_path).fill_null(3)

        query = (lf.select([
            pl.col(self.config.date_col),
            pl.col(self.config.target_col),
            pl.col(self.config.weight_col),
            *[pl.col(f) for f in self.config.feature_cols]
        ])
        .tail(self.config.n_rows)
        .sort(self.config.date_col))

        #Normalize features
        df = self._normalize_features(query)

        #Split data
        train_df, train_target, train_weight, val_df, val_target, val_weight= self._train_val_split(df.collect())

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
        return lf.select([pl.col(self.config.date_col),pl.col(self.config.weight_col), *normalized_features_and_resp])
    def _train_val_split(self, df: pl.DataFrame) -> tuple[
        DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
        """Split dataset into train and validation"""
        unique_dates = df.get_column(self.config.date_col).unique().sort()
        split_idx = int(len(unique_dates) * self.config.train_ratio)

        train_dates = unique_dates[:split_idx]
        val_dates = unique_dates[split_idx:]

        train_mask = df.get_column('date_id').is_in(train_dates).to_numpy()
        val_mask = df.get_column('date_id').is_in(val_dates).to_numpy()
        print(f'training: min: {train_dates.min()},max: {train_dates.max()}')
        print(f'val days: min: {val_dates.min()}, max: {val_dates.max()}')
        train_data = df.filter(train_mask).select([pl.col(f'{col}_normalized') for col in self.config.feature_cols])
        val_data = df.filter(val_mask).select([pl.col(f'{col}_normalized') for col in self.config.feature_cols])

        train_target = df.filter(train_mask).select(f'{self.config.target_col}_normalized')
        val_target = df.filter(val_mask).select(f'{self.config.target_col}_normalized')

        train_weights = df.filter(train_mask).select(self.config.weight_col)
        val_weights = df.filter(val_mask).select(self.config.weight_col)

        return train_data, train_target, train_weights, val_data, val_target, val_weights

