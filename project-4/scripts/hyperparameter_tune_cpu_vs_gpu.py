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
latent_dims = [2, 3, 4, 5, 10, 20, 50]
hidden_layer_options = [[128, 64], [256, 128, 64]]
learning_rates = [1e-3, 1e-4]
devices = ['cpu', 'cuda']

results = []

for latent_dim, hidden_dims, lr in product(latent_dims, hidden_layer_options, learning_rates):
    print(f"Training AE | latent={latent_dim}, hidden={hidden_dims}, lr={lr}")
    for device in devices:
        t_start = time.time()
        runner = GeneExpressionRunner(
            input_data=X,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            lr=lr,
            device=device
        )

        fold_losses = runner.cross_validate(k=5, epochs=100)
        mean_loss = np.mean(fold_losses)
        t_end = time.time()
        if device == 'cpu':
            result = {
                'latent_dim': latent_dim,
                'hidden_dims': hidden_dims,
                'lr': lr,
                'val_loss_mean': mean_loss,
                'val_loss_std': np.std(fold_losses),
                'cpu_time': t_end - t_start,
            }
        else:
            result['cuda_time'] = t_end - t_start
        print(f'time: {t_end - t_start:.2f}s, mean loss: {mean_loss:.4f}, std: {np.std(fold_losses):.4f}')
    results.append(result)
    df_results = pd.DataFrame(results).sort_values('val_loss_mean')
    df_results.to_csv('results/tuning/hyperparameter_tuning_results.csv', index=False)

# Save results as DataFrame
df_results = pd.DataFrame(results).sort_values('val_loss_mean')

# Save raw results
df_results.to_csv('results/tuning/hyperparameter_tuning_results.csv', index=False)
print("Results saved to results/tuning/hyperparameter_tuning_results.csv")

# ========== Visualizations ========== #

# Add column to display hidden layers as string
df_results['hidden_str'] = df_results['hidden_dims'].apply(lambda x: '-'.join(map(str, x)))

# --- 1. Heatmap: latent_dim vs hidden_str
plt.figure(figsize=(10, 6))
pivot = df_results.pivot_table(index='latent_dim', columns='hidden_str', values='val_loss_mean')
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
plt.title("Mean Val Loss by Latent Dim and Hidden Layers")
plt.ylabel("Latent Dimension")
plt.xlabel("Hidden Layer Sizes")
plt.tight_layout()
plt.savefig("results/tuning/hyperparam_heatmap.png")
plt.close()

# --- 2. Lineplot: Val loss vs Latent Dim
plt.figure(figsize=(8, 5))
sns.lineplot(data=df_results, x='latent_dim', y='val_loss_mean', hue='hidden_str', marker='o')
plt.title("Validation Loss vs Latent Dimension")
plt.xlabel("Latent Dimension")
plt.ylabel("Mean Validation Loss")
plt.legend(title="Hidden Layers")
plt.tight_layout()
plt.savefig("results/tuning/val_loss_vs_latent_dim.png")
plt.close()

# --- 3. Boxplot: Learning rate effect
plt.figure(figsize=(7, 5))
sns.boxplot(data=df_results, x='lr', y='val_loss_mean')
plt.title("Validation Loss Distribution by Learning Rate")
plt.xlabel("Learning Rate")
plt.ylabel("Mean Validation Loss")
plt.tight_layout()
plt.savefig("results/tuning/lr_vs_loss_boxplot.png")
plt.close()

# --- 4. Barplot: CPU vs CUDA time
labels = [f"LD={ld},H={h}" for ld, h in zip(df_results['latent_dim'], df_results['hidden_dims'])]

plt.figure(figsize=(12, 6))
plt.bar(labels, df_results['cpu_time'], label='CPU', alpha=0.7)
plt.bar(labels, df_results['cuda_time'], label='CUDA', alpha=0.7)
plt.xticks(rotation=90)
plt.ylabel("Time (s)")
plt.title("Training Time Comparison: CPU vs CUDA")
plt.legend()
plt.tight_layout()
plt.savefig("results/tuning/cpu_vs_cuda_time.png")
plt.close()


print("Plots saved to results/tuning/")

