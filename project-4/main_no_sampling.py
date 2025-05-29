from src.utils.preprocess import MutPreprocessor
from src.utils.preprocess import GeneExpPreprocessor
from src.models.autoencoder_runner import GeneExpressionRunner
from src.models.autoencoder import GeneExpressionAutoencoder
from src.utils.data_loader import load_expression_matrix
from src.utils.sampling import SamplingRunner
import numpy as np
import pandas as pd
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse

def parse_args():
    parser = argparse.ArgumentParser()  
    
    parser.add_argument('--latent_dim', type=int, default=5, help='Number of latent features')
    parser.add_argument('--hidden_dim', type=int, nargs='+', default=[128, 64],
                        help='Hidden dimensions in encoder, e.g., --hidden_dim 128 64')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=200, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=10, help='Number of epocs where model training saturated')

    parser.add_argument('--top_N', type=int, default=250, help="Top 'n' genes used for analysis")
    parser.add_argument('--subset_method', type=str, default='dndscv', help="Method to select top 'n' genes")

    return parser.parse_args()
    
def main():
    args = parse_args()

    all_exp = GeneExpPreprocessor(top_N=args.top_N,
        subset_method=args.subset_method, filter_subtypes=False).get_df()
    
    basal_samples = GeneExpPreprocessor(top_N=args.top_N,
        subset_method=args.subset_method, filter_subtypes=True, subtypes=['BRCA_Basal']).get_df().index
    
    # run cross validation
    
    print(f"Expression data shape (train data): {all_exp.shape}")
    
    runner = GeneExpressionRunner(all_exp,latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dim,
        lr=args.lr,
        batch_size=args.batch_size)
    cv_losses = runner.cross_validate()
    print(f'cv_losses: {cv_losses}')

    # train on all samples and get latent space
    mask_basal = all_exp.index.isin(basal_samples)
    latent_all = runner.train_all_and_encode()
    latent_tnbc = latent_all[mask_basal]

    df_latent = pd.DataFrame(latent_tnbc, index=all_exp[mask_basal].index, columns=[f"latent_{i}" for i in range(latent_tnbc.shape[1])])
    df_latent.to_csv(f"results/tables/no_sampling_latent_space.csv")
    cv_losses_df = pd.DataFrame(cv_losses, index=[f"fold_{i+1}" for i in range(len(cv_losses))])
    cv_losses_df.to_csv(f"results/tables/no_sampling_cv_losses.csv", index=False, header=False)

if __name__ == "__main__":
    main()