# CP-KAN / MLP Comparison & Degradation Study - Reproducibility Package

This directory contains the code to run experiments comparing Fixed KAN models (with potential structure optimization like Greedy Heuristic) against standard MLPs. It also includes functionality to perform a follow-up degradation study on the best-performing models identified during a grid search.

## Files

*   `main.py`: The main entry point script to run experiments based on a configuration file.
*   `experiment_runner.py`: Contains the `ExperimentRunner` class that handles data loading, model initialization (FixedKAN, MLP, other KAN variants), grid search execution, single runs, degradation studies, training loops, metric calculation, and results saving/plotting.
*   `CP_KAN.py`: Implementation of the `FixedKAN` model using Chebyshev polynomials and degree optimization strategies.
*   `data_pipeline.py`: Implements data loading, preprocessing, and splitting for the Jane Street dataset using Polars.
*   `data_pipeline_js_config.py`: Dataclasses for configuration related to the Jane Street data pipeline (`DataConfig`).
*   `kanlayer_easytsf.py`: Contains implementations of various alternative KAN layers (Wavelet, Fourier, Jacobi, etc.) adapted from EasyTSF.
*   `configs/`: Directory containing example YAML configuration files.
    *   `config_js_grid_search_degradation.yaml`: Example config to run a grid search for KAN/MLP on Jane Street and automatically generate a config file for a follow-up degradation study.
    *   `config_js_grid_search_degradation_short.yaml`: A shorter version for quick testing.
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
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Installing `ortools` or other optimization libraries might require specific system dependencies. Refer to their official installation guides if needed.* 

4.  **Configure Data Path in YAML**
    *   Edit the desired configuration file in the `configs/` directory (e.g., `config_js_grid_search_degradation.yaml`).
    *   Find the `data_path` key under the `dataset` section.
    *   Replace the placeholder path (`"/path/to/your/jane_street/train.parquet/**/*.parquet"`) with the actual absolute path to your Jane Street `train.parquet` directory, ensuring you use the `/**/*.parquet` glob pattern to select only the necessary files.

## Running Experiments

The main script is `main.py`. You specify the experiment configuration using the `--config` argument.

```bash
# Ensure you are in the CP-KAN_reproducibility directory
PYTHONPATH=$PWD python main.py --config <path_to_config_yaml>
```

**Workflow Examples:**

1.  **Grid Search followed by Degradation Study:**
    *   **Stage 1:** Configure and run a grid search using a config like `configs/config_js_grid_search_degradation.yaml`. Ensure `experiment_type: grid_search` and `generate_degradation_config: true` are set.
        ```bash
        PYTHONPATH=$PWD python main.py --config configs/config_js_grid_search_degradation.yaml
        ```
        This will perform the grid search, save results, and generate a new config file (e.g., `config_degradation_study_generated.yaml`) inside the specified `results_dir`.
    *   **Stage 2:** Run the degradation study using the *generated* config file.
        ```bash
        # Replace <results_dir> with the actual path from Stage 1 config
        PYTHONPATH=$PWD python main.py --config <results_dir>/config_degradation_study_generated.yaml 
        ```
        This will load the best models found in Stage 1 and train them for longer, tracking degradation metrics.

2.  **Grid Search Only:**
    *   Use a config with `experiment_type: grid_search` and set `generate_degradation_config: false` (or omit it).

3.  **Single Run (Not covered by example configs):**
    *   Use a config with `experiment_type: single_run` and define specific model parameters under `model_params` instead of `parameter_grids`.

## Output

Outputs are saved in the directory specified by `results_dir` in the configuration file.

*   **Logs:** A `.log` file containing detailed execution logs.
*   **Grid Search Results:**
    *   A `_grid_search_comparison_...csv` file with metrics for every epoch of every run in the grid search.
    *   Summary plots (`_final_...png`, `_perf_vs_params_...png`, etc.) comparing the best runs found.
    *   If KAN was run, `optimized_kan_models/` subdirectory containing saved `.pth` files of the KAN models after the structure optimization step.
*   **Degradation Study Results (if run):**
    *   A `config_degradation_study_generated.yaml` file (generated after the grid search).
    *   Inside a `degradation_study_results/` subdirectory:
        *   A `degradation_study_...csv` file with detailed epoch-by-epoch metrics (including grad_norm, weight_change, degradation_from_peak).
        *   A `degradation_study_plots_...png` file visualizing the degradation metrics over time.
