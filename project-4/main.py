from src.utils.preprocess import MutPreprocessor
from src.utils.preprocess import GeneExpPreprocessor
from src.models.autoencoder_runner import GeneExpressionRunner
from src.models.autoencoder import GeneExpressionAutoencoder
from src.utils.data_loader import load_expression_matrix
import numpy as np
import pandas as pd

import argparse

RUN_EACH_SUBTYPE = True

def parse_args():
    parser = argparse.ArgumentParser()  
    
    parser.add_argument('--latent_dim', type=int, default=5, help='Number of latent features')
    parser.add_argument('--hidden_dim', type=int, nargs='+', default=[128, 64],
                        help='Hidden dimensions in encoder, e.g., --hidden_dim 128 64')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=200, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=10, help='Number of epocs where model training saturated')

    parser.add_argument('--top_N', type=int, default=500, help="Top 'n' genes used for analysis")
    parser.add_argument('--subset_method', type=str, default='dndscv', help="Method to select top 'n' genes")

    return parser.parse_args()
    
def main():
    args = parse_args()
    
    # Trains autoencoder on all samples of specified subtype and saves latent space and cross-validation losses.
    # preprocess expression data
    df_exp = GeneExpPreprocessor(save=True,top_N=args.top_N,
        subset_method=args.subset_method).get_df()
    # run cross validation
    runner = GeneExpressionRunner(df_exp,latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dim,
        lr=args.lr,
        batch_size=args.batch_size)
    cv_losses = runner.cross_validate()
    print(f'cv_losses: {cv_losses}')

    # train on all samples and get latent space
    latent = runner.train_all_and_encode()
    df_latent = pd.DataFrame(latent, index=df_exp.index, columns=[f"latent_{i}" for i in range(latent.shape[1])])
    df_latent.to_csv(f"results/tables/latent_space.csv")
    cv_losses_df = pd.DataFrame(cv_losses, index=[f"fold_{i+1}" for i in range(len(cv_losses))])
    cv_losses_df.to_csv(f"results/tables/cv_losses.csv", index=False, header=False)

def each_subtype():
    args = parse_args()
    # Runs cross-validation for each subtype and saves latent space and cross-validation losses.
    LATENT_DIM=5
    all_cv_losses = []
    subtypes = ['BRCA_LumA', 'BRCA_Her2', 'BRCA_LumB', 'BRCA_Normal', 'BRCA_Basal']
    for subtype in subtypes:
        print(f"Running for subtype: {subtype}")

        # preprocess expression data
        df_exp = GeneExpPreprocessor(subtypes=[subtype],top_N=args.top_N,
        subset_method=args.subset_method).get_df()

        # run cross validation
        runner = GeneExpressionRunner(df_exp ,latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dim,
        lr=args.lr,
        batch_size=args.batch_size)
        cv_losses = runner.cross_validate()
        all_cv_losses.append(cv_losses)
        print(f'cv_losses: {cv_losses}')

        # train on all samples and get latent space
        latent = runner.train_all_and_encode()
        df_latent = pd.DataFrame(latent, index=df_exp.index, columns=[f"latent_{i}" for i in range(latent.shape[1])])
        df_latent.to_csv(f"results/tables/latent_space_{LATENT_DIM}dim_{subtype}.csv")
    all_cv_losses_df = pd.DataFrame(all_cv_losses, index=subtypes, columns=[f"fold_{i+1}" for i in range(len(all_cv_losses[0]))])
    all_cv_losses_df.to_csv(f"results/tables/cv_losses_{LATENT_DIM}dim_all_subtypes.csv", index=True)

if __name__ == "__main__":
    main()
    if RUN_EACH_SUBTYPE:
        each_subtype()
