# CP-KAN Optimization Comparison (Jane Street) - Reproducibility Package

This directory contains the code to reproduce the comparison of different polynomial degree optimization methods (QUBO, Integer Programming, Evolutionary Algorithm, Greedy Heuristic) for the Chebyshev Polynomial KAN (CP-KAN) model on the Jane Street Market Prediction dataset.

## Files

*   `CP_KAN.py`: Implementation of the `FixedKAN` model using Chebyshev polynomials and various degree optimization strategies.
*   `data_pipeline_js_config.py`: Dataclasses for configuration related to the Jane Street data pipeline.
*   `data_pipeline.py`: Implements the data loading, preprocessing (quantile normalization), and splitting logic for the Jane Street dataset using Polars.
*   `optimization_tuner.py`: Class that orchestrates the hyperparameter grid search and comparison of optimization methods.
*   `run_optimization_comparison.py`: The main script to execute the comparison experiment.
*   `config_js_opt_comparison.yaml`: Configuration file defining data paths, experiment parameters, and hyperparameter grids.
*   `requirements.txt`: Required Python packages.
*   `README.md`: This file.

## Setup

1.  **Clone the Repository (if applicable)**
    ```bash
    # git clone ...
    cd CP-KAN_reproducibility
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Installing `ortools` or `dwave-neal`/`pyqubo` might require specific system dependencies or build tools depending on your OS. Refer to their official installation guides if you encounter issues.*

4.  **Configure Data Path**
    *   Edit `config_js_opt_comparison.yaml`.
    *   Modify the `data_path` under the `data` section to point to the location of your `train.parquet` file (or directory containing the parquet dataset) for the Jane Street data. The current path is a placeholder: `~/Interning/Kaggle/jane_street_kaggle/...`.

## Running the Experiment

The main script is `run_optimization_comparison.py`.

```bash
python run_optimization_comparison.py [options]
```

**Options:**

*   `--config <path>`: Path to the configuration YAML file (default: `config_js_opt_comparison.yaml`).
*   `--quick`: Use the smaller 'quick' hyperparameter grid defined in the config file for a faster test run.
*   `--full`: Use the larger 'full' hyperparameter grid defined in the config file for a more comprehensive run.
*   *(If neither --quick nor --full is specified, the 'default' grid from the config is used).*

**Example:**

*   Run with the default grid:
    ```bash
    python run_optimization_comparison.py
    ```
*   Run with the quick grid for testing:
    ```bash
    python run_optimization_comparison.py --quick
    ```

## Output

*   **Logs:** A log file named `JaneStreet_OptimizationComparison_YYYYMMDD_HHMMSS.log` will be created in the `results_dir` specified in the config (default: `optimization_results_js`).
*   **Results CSV:** A CSV file `optimization_comparison.csv` containing detailed metrics for each run will be saved in the `results_dir`.
*   **Plots:** Comparison plots (R² vs. Epoch, Optimization Time, Final R²) will be saved as PNG files in the `results_dir`.
*   **Best Models:** The best performing KAN model state (state dict and config) for each optimization method (based on validation R²) will be saved as `.pth` files (e.g., `kan_qubo_best.pth`) in the `results_dir`.
