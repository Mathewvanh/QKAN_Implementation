from tuning_framework import ModelTuner

def main():
    # Initialize tuner
    tuner = ModelTuner()
    
    # Initial test showed:
    # 1. KAN best MSE: 0.001437 with hidden_size=20
    # 2. MLP best MSE: 0.000195 with default params
    # 3. MLP significantly outperforms KAN
    # 4. Need to try more KAN configurations to improve performance
    
    # Expanded KAN grid focusing on architecture
    kan_params = {
        'max_degree': [5, 7, 9],     # Try lower and higher degrees
        'hidden_size': [16, 20, 24, 28],  # Expanded around best size
        'hidden_degree': [3, 5, 7],   # Try different hidden complexities
        'learning_rate': [1e-2, 5e-3] # Try slightly lower lr
    }
    
    # Expanded MLP grid to find optimal architecture
    mlp_params = {
        'hidden_size': [20, 24, 28],  # Around current best
        'depth': [2, 3, 4],           # Try different depths
        'dropout': [0.1, 0.15],       # Try slightly higher dropout
        'learning_rate': [1e-2, 5e-3] # Include slightly lower lr
    }
    
    # Run KAN tuning first
    print("\nTuning KAN with expanded grid...")
    tuner.tune_kan(kan_params, num_epochs=150)  # More epochs to ensure convergence
    
    # Run MLP tuning
    print("\nTuning MLP with expanded grid...")
    tuner.tune_mlp(mlp_params, num_epochs=150)  # More epochs to ensure convergence
    
    # Plot results
    tuner.plot_results()

if __name__ == "__main__":
    main()
