from dataclasses import dataclass
import polars as pl
import numpy as np
from typing import Dict, List, Tuple

@dataclass
class SparsityMetrics:
    """Container for various sparsity metrics"""
    overall_sparsity: float
    column_sparsity: Dict[str, float]
    time_based_sparsity: Dict[str, float]
    zero_clusters: List[Tuple[int, int]]

def compute_sparsity(data_path: str, chunk_size: int = 1000) -> SparsityMetrics:
    """
    Compute comprehensive sparsity metrics for financial data using Polars lazy evaluation

    Args:
        data_path: Path to parquet file
        chunk_size: Number of rows to process at once
    """
    # Create lazy frame
    lf = pl.scan_parquet(data_path)

    # Get feature columns
    feature_cols = [col for col in lf.columns if col.startswith('feature_')]

    # Create lazy computations
    null_counts = lf.select([
        pl.col(col).null_count().alias(f"{col}_nulls")
        for col in feature_cols
    ]).fetch(chunk_size)

    total_rows = lf.select(pl.count()).fetch(1).item()

    # Overall sparsity
    overall_sparsity = sum(null_counts.row(0)) / (len(feature_cols) * total_rows)

    # Column-wise sparsity
    column_sparsity = {
        col: null_counts[f"{col}_nulls"][0] / total_rows
        for col in feature_cols
    }

    # Time-based sparsity
    time_based = {}
    if 'date_id' in lf.columns:
        time_stats = (
            lf.group_by('date_id')
            .agg([
                pl.col(col).null_count().alias(f"{col}_nulls")
                for col in feature_cols
            ])
            .fetch(chunk_size)
        )

        date_counts = (
            lf.group_by('date_id')
            .agg(pl.count())
            .fetch(chunk_size)
        )

        for row, count_row in zip(time_stats.iter_rows(), date_counts.iter_rows()):
            date = row[0]
            nulls = sum(row[1:])
            date_count = count_row[1]
            time_based[str(date)] = nulls / (len(feature_cols) * date_count)

    # For zero clusters, we need to process the data in chunks
    zero_clusters = []
    for chunk_start in range(0, total_rows, chunk_size):
        chunk = lf.slice(chunk_start, chunk_size).fetch()
        for col in feature_cols:
            is_null = chunk[col].is_null().to_numpy()
            transitions = np.diff(np.concatenate([[False], is_null, [False]]))
            starts = np.where(transitions == 1)[0]
            ends = np.where(transitions == -1)[0]
            # Adjust indices for chunk position
            clusters = [(s + chunk_start, e + chunk_start)
                        for s, e in zip(starts, ends) if e - s > 10]
            zero_clusters.extend(clusters)

    return SparsityMetrics(
        overall_sparsity=overall_sparsity,
        column_sparsity=column_sparsity,
        time_based_sparsity=time_based,
        zero_clusters=zero_clusters
    )

def print_sparsity_analysis(metrics: SparsityMetrics):
    """Pretty print the sparsity metrics"""
    print(f"Overall Sparsity: {metrics.overall_sparsity:.2%}")
    print("\nMost Sparse Columns:")
    sorted_cols = sorted(metrics.column_sparsity.items(), key=lambda x: x[1], reverse=True)
    for col, sparsity in sorted_cols[:5]:
        print(f"  {col}: {sparsity:.2%}")

    print("\nTime-based Sparsity Patterns:")
    dates = sorted(metrics.time_based_sparsity.keys())
    if dates:
        for date in dates[:5]:
            print(f"  Date {date}: {metrics.time_based_sparsity[date]:.2%}")

    print("\nLarge Zero Clusters:")
    clusters = sorted(metrics.zero_clusters, key=lambda x: x[1]-x[0], reverse=True)
    for start, end in clusters[:5]:
        print(f"  Cluster from {start} to {end} (length: {end-start})")