import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from datetime import datetime

from config import DataConfig
from data_pipeline import DataPipeline

def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    """Compute weighted R² score."""
    numerator = np.sum(w * (y_true - y_pred)**2)
    denominator = np.sum(w * (y_true**2))
    if denominator < 1e-12:
        return 0.0
    return float(1.0 - numerator/denominator)

def train_and_evaluate_trees():
    """Train LightGBM and XGBoost models on Jane Street data."""
    # Load data with same configuration as KAN/MLP tests
    data_cfg = DataConfig(
        data_path="~/Interning/Kaggle/jane_street_kaggle/jane-street-real-time-market-data-forecasting/train.parquet/",
        n_rows=200000,  # Match KAN/MLP dataset size
        train_ratio=0.7,
        feature_cols=[f'feature_{i:02d}' for i in range(79)],
        target_col="responder_6",
        weight_col="weight",
        date_col="date_id"
    )
    
    pipeline = DataPipeline(data_cfg, None)
    train_df, train_target, train_weight, val_df, val_target, val_weight = pipeline.load_and_preprocess_data()
    
    # Convert to numpy
    x_train = train_df.to_numpy()
    y_train = train_target.to_numpy().squeeze()
    w_train = train_weight.to_numpy().squeeze()
    
    x_val = val_df.to_numpy()
    y_val = val_target.to_numpy().squeeze()
    w_val = val_weight.to_numpy().squeeze()
    
    # Results DataFrame
    results_df = pd.DataFrame(columns=[
        'model_type', 'epoch', 'train_r2', 'val_r2', 'param_count'
    ])
    
    # Train LightGBM
    print("\n==== Training LightGBM ====")
    dtrain = lgb.Dataset(x_train, label=y_train, weight=w_train)
    dval = lgb.Dataset(x_val, label=y_val, weight=w_val)
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'num_leaves': 32,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Train with callbacks for R² tracking
    eval_results = {}
    def lgb_r2_callback():
        result = {}
        def _callback(env):
            # Train R²
            y_pred_train = env.model.predict(x_train)
            train_r2 = weighted_r2(y_train, y_pred_train, w_train)
            
            # Val R²
            y_pred_val = env.model.predict(x_val)
            val_r2 = weighted_r2(y_val, y_pred_val, w_val)
            
            # Save metrics every 10 iterations
            if env.iteration % 10 == 0:
                print(f"[LightGBM] Iteration {env.iteration}, Train R²={train_r2:.4f}, Val R²={val_r2:.4f}")
                results_df.loc[len(results_df)] = {
                    'model_type': 'LightGBM',
                    'epoch': env.iteration,
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'param_count': env.model.num_parameters()
                }
                # Save after each update
                results_df.to_csv('./results/tree_models_metrics.csv', index=False)
            
            result['train_r2'] = train_r2
            result['val_r2'] = val_r2
        return _callback
    
    lgb_model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=100,  # Match KAN/MLP epochs
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=[lgb_r2_callback()],
        verbose=-1
    )
    
    # Final predictions and metrics
    y_pred_train_lgb = lgb_model.predict(x_train)
    y_pred_val_lgb = lgb_model.predict(x_val)
    
    train_r2_lgb = weighted_r2(y_train, y_pred_train_lgb, w_train)
    val_r2_lgb = weighted_r2(y_val, y_pred_val_lgb, w_val)
    
    print(f"LightGBM Final: Train R²={train_r2_lgb:.4f}, Val R²={val_r2_lgb:.4f}")
    
    # Save model
    model_path = f"./models_janestreet/lightgbm_valr2_{val_r2_lgb:.4f}.txt"
    lgb_model.save_model(model_path)
    print(f"LightGBM model saved to: {model_path}")
    
    # Train XGBoost
    print("\n==== Training XGBoost ====")
    dtrain = xgb.DMatrix(x_train, label=y_train, weight=w_train)
    dval = xgb.DMatrix(x_val, label=y_val, weight=w_val)
    
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.01,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist'  # For faster training
    }
    
    # Custom callback for R² tracking
    def xgb_r2_callback():
        def _callback(env):
            if env.iteration % 10 == 0:
                # Train R²
                y_pred_train = env.model.predict(dtrain)
                train_r2 = weighted_r2(y_train, y_pred_train, w_train)
                
                # Val R²
                y_pred_val = env.model.predict(dval)
                val_r2 = weighted_r2(y_val, y_pred_val, w_val)
                
                print(f"[XGBoost] Iteration {env.iteration}, Train R²={train_r2:.4f}, Val R²={val_r2:.4f}")
                results_df.loc[len(results_df)] = {
                    'model_type': 'XGBoost',
                    'epoch': env.iteration,
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'param_count': env.model.num_features()
                }
                # Save after each update
                results_df.to_csv('./results/tree_models_metrics.csv', index=False)
        return _callback
    
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=100,  # Match KAN/MLP epochs
        evals=[(dtrain, 'train'), (dval, 'val')],
        callbacks=[xgb_r2_callback()],
        verbose_eval=False
    )
    
    # Final predictions and metrics
    y_pred_train_xgb = xgb_model.predict(dtrain)
    y_pred_val_xgb = xgb_model.predict(dval)
    
    train_r2_xgb = weighted_r2(y_train, y_pred_train_xgb, w_train)
    val_r2_xgb = weighted_r2(y_val, y_pred_val_xgb, w_val)
    
    print(f"XGBoost Final: Train R²={train_r2_xgb:.4f}, Val R²={val_r2_xgb:.4f}")
    
    # Save model
    model_path = f"./models_janestreet/xgboost_valr2_{val_r2_xgb:.4f}.json"
    xgb_model.save_model(model_path)
    print(f"XGBoost model saved to: {model_path}")
    
    return results_df

def create_combined_plot():
    """Create plot combining KAN, MLP, and tree model results."""
    # Load results
    kan_mlp_path = './results/kan_vs_mlp_metrics.csv'
    tree_path = './results/tree_models_metrics.csv'
    
    if not os.path.exists(kan_mlp_path) or not os.path.exists(tree_path):
        print("Missing results files. Run both KAN/MLP and tree model tests first.")
        return
    
    kan_mlp_results = pd.read_csv(kan_mlp_path)
    tree_results = pd.read_csv(tree_path)
    
    # Combine results
    all_results = pd.concat([kan_mlp_results, tree_results], ignore_index=True)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot KAN
    kan_results = all_results[all_results['model_type'] == 'KAN']
    plt.plot(kan_results['epoch'], kan_results['val_r2'], 
            label=f'KAN [{kan_results.iloc[0]["param_count"]} params]', 
            color='blue', linewidth=2)
    
    # Plot MLPs
    colors = ['orange', 'green', 'red']
    for depth, color in zip([2, 3, 4], colors):
        mlp_d = all_results[(all_results['model_type'] == 'MLP') & 
                           (all_results['depth'] == depth)]
        if len(mlp_d) > 0:
            param_count = mlp_d.iloc[0]['param_count']
            plt.plot(mlp_d['epoch'], mlp_d['val_r2'], 
                    label=f'MLP-{depth} [{param_count} params]', 
                    color=color, linewidth=2)
    
    # Plot tree models
    tree_colors = {'LightGBM': 'purple', 'XGBoost': 'brown'}
    for model_type, color in tree_colors.items():
        tree_d = all_results[all_results['model_type'] == model_type]
        if len(tree_d) > 0:
            param_count = tree_d.iloc[0]['param_count']
            plt.plot(tree_d['epoch'], tree_d['val_r2'], 
                    label=f'{model_type} [{param_count} params]', 
                    color=color, linewidth=2)
    
    plt.title("Model Comparison: Validation R² vs Epoch\nJane Street Market Prediction")
    plt.xlabel("Epoch/Iteration")
    plt.ylabel("Weighted R²")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('./results/all_models_comparison.png', bbox_inches='tight')
    print("Combined comparison plot saved to: ./results/all_models_comparison.png")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("./models_janestreet", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    # Train tree models
    print("Training tree-based models...")
    train_and_evaluate_trees()
    
    # Create combined plot
    print("\nCreating combined comparison plot...")
    create_combined_plot()
