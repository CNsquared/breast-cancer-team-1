from itertools import product
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import time

from src.models.autoencoder_runner import GeneExpressionRunner
from src.utils.preprocess import GeneExpPreprocessor

# Load expression data
df_exp = GeneExpPreprocessor().get_df()
X = df_exp.to_numpy()

# Define your hyperparameter grid
latent_dims = [2, 3, 4, 5, 10, 20, 50]
hidden_layer_options = [[128, 64], [256, 128, 64]]
learning_rates = [1e-3, 1e-4]

results = []

for latent_dim, hidden_dims, lr in product(latent_dims, hidden_layer_options, learning_rates):
    print(f"Training AE | latent={latent_dim}, hidden={hidden_dims}, lr={lr}")
    
    runner = GeneExpressionRunner(
        input_data=X,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        lr=lr,
        device='cpu'
    )

    fold_losses = runner.cross_validate(k=5, epochs=100)
    mean_loss = np.mean(fold_losses)

    results.append({
        'latent_dim': latent_dim,
        'hidden_dims': hidden_dims,
        'lr': lr,
        'val_loss_mean': mean_loss,
        'val_loss_std': np.std(fold_losses)
    })

# Save results as DataFrame
df_results = pd.DataFrame(results).sort_values('val_loss_mean')

# Make sure output directories exist
os.makedirs("results/tuning", exist_ok=True)

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

print("Plots saved to results/tuning/")
