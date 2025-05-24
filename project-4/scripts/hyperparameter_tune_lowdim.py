from itertools import product
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import time

from src.models.autoencoder_runner import GeneExpressionRunner
from src.utils.preprocess import GeneExpPreprocessor

# USAGE: run from project root
#     python -m scripts.hyperparameter_tune

# Load expression data
df_exp = GeneExpPreprocessor().get_df()
X = df_exp.to_numpy()

# Make sure output directories exist
os.makedirs("results/tuning", exist_ok=True)

# Define your hyperparameter grid
latent_dim = 5
hidden_dims = [128, 64]
batch_sizes = [16, 32, 64, 128, 256]
learning_rates = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5]

results = []

for batch_size, lr in product(batch_sizes, learning_rates):
    print(f"Training with batch_size={batch_size}, lr={lr}")
    t_start = time.time()
    runner = GeneExpressionRunner(
        input_data=X,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        lr=lr,
        batch_size=batch_size
    )
    fold_losses = runner.cross_validate(k=5)
    mean_loss = np.mean(fold_losses)
    t_end = time.time()

    results.append({
        'batch_size': batch_size,
        'lr': lr,
        'val_loss_mean': mean_loss,
        'val_loss_std': np.std(fold_losses),
        'time': t_end - t_start,
    })
    print(f'time: {t_end - t_start:.2f}s, mean loss: {mean_loss:.4f}, std: {np.std(fold_losses):.4f}')
    df_results = pd.DataFrame(results).sort_values('val_loss_mean')
    df_results.to_csv('results/tuning/hyperparameter_tuning_results_low_dim.csv', index=False)

# Save results as DataFrame
df_results = pd.DataFrame(results).sort_values('val_loss_mean')

# Save raw results
df_results.to_csv('results/tuning/hyperparameter_tuning_results_low_dim.csv', index=False)
print("Results saved to results/tuning/hyperparameter_tuning_results_low_dim.csv")

# ========== Visualizations ========== #

pivot = df_results.pivot(index='batch_size', columns='lr', values='val_loss_mean')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".4f", cmap='viridis')
plt.title("Val Loss by Batch Size and Learning Rate")
plt.xlabel("Learning Rate")
plt.ylabel("Batch Size")
plt.tight_layout()
plt.savefig("results/tuning/heatmap_loss_batch_lr_low_dim.png")
plt.close()


print("Plots saved to results/tuning/")

