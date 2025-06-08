import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from src.models.autoencoder_runner import GeneExpressionRunner
import datetime



from itertools import product

# Centralized config
param_grid = {
    "top_N": [100, 250, 500],
    "latent_dim": [2, 5, 10, 20],
    "hidden_dims": [
        [256, 128],
        [128, 64],
        [128, 64, 32],
        [64, 32],
    ],
    "lr": [1e-3, 5e-4],
    "dropout_rate": [0.0, 0.1, 0.2],
    "weight_decay": [0.0, 1e-5],
    "batch_size": [16, 32],
    "patience": [10],
    "max_epochs": [200],
}

# Extract keys and values
keys, values = zip(*param_grid.items())
full_grid = [dict(zip(keys, v)) for v in product(*values)]

def run_grid_search(full_grid, subset_method="dndscv", k_folds=5):
    from src.utils.preprocess import GeneExpPreprocessor
    results = []

    # Step 1: Precompute and cache expression matrices for unique top_N values
    topN_to_dfexp = {}
    unique_topNs = set(config["top_N"] for config in full_grid)
    for topN in unique_topNs:
        print(f"📦 Preprocessing expression data for top_N={topN}")
        df_exp = GeneExpPreprocessor(
            top_N=topN,
            subset_method=subset_method,
            filter_subtypes=False
        ).get_df()
        topN_to_dfexp[topN] = df_exp

    # Step 2: Run grid search using cached data
    for config in tqdm(full_grid, desc="Grid Search"):
        if max(config["hidden_dims"]) > config["top_N"]:
            print(f"⚠️ Skipping config {config} due to hidden_dims exceeding top_N")
            continue
        print(f"\n🔍 Testing config: {config}")
        df_exp = topN_to_dfexp[config["top_N"]]

        ae_runner = GeneExpressionRunner(
            train_data=df_exp.to_numpy(),
            latent_dim=config["latent_dim"],
            hidden_dims=config["hidden_dims"],
            lr=config["lr"],
            dropout_rate=config["dropout_rate"],
            weight_decay=config["weight_decay"],
            batch_size=config["batch_size"]
        )

        cv_losses = ae_runner.cross_validate(
            k=k_folds,
            patience=config["patience"],
            max_epochs=config["max_epochs"]
        )

        results.append({
            **config,
            "cv_loss_mean": np.mean(cv_losses),
            "cv_loss_std": np.std(cv_losses)
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="cv_loss_mean").reset_index(drop=True)
    return results_df


if __name__ == "__main__":
    results_df = run_grid_search(full_grid)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"results/tables/grid_search_{timestamp}.csv", index=False)
    print(results_df.head())
