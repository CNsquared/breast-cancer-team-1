from src.utils.preprocess import MutPreprocessor
from src.utils.preprocess import GeneExpPreprocessor
from src.models.autoencoder_runner import GeneExpressionRunner
from src.utils.data_loader import load_expression_matrix
import numpy as np
import pandas as pd

def main():
    # preprocess expression data
    df_exp = GeneExpPreprocessor(save=True).get_df()
    # run cross validation
    runner = GeneExpressionRunner(df_exp, latent_dim=5)
    cv_losses = runner.cross_validate(k=5)
    print(f'cv_losses: {cv_losses}')
    # train on all samples and get latent space
    latent = runner.train_all_and_encode(epochs=100)
    df_latent = pd.DataFrame(latent, index=df_exp.index, columns=[f"latent_{i}" for i in range(latent.shape[1])])
    df_latent.to_csv("results/tables/latent_space.csv")


if __name__ == "__main__":
    main()
