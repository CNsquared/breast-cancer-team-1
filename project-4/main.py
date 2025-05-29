from src.utils.preprocess import MutPreprocessor
from src.utils.preprocess import GeneExpPreprocessor
from src.models.autoencoder_runner import GeneExpressionRunner
from src.models.autoencoder import GeneExpressionAutoencoder
from src.utils.data_loader import load_expression_matrix
from src.utils.sampling import SamplingRunner, hyperparam_search
import numpy as np
import pandas as pd
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse

RUN_EACH_SUBTYPE = False

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
    
    # Trains autoencoder on all samples of specified subtype and saves latent space and cross-validation losses.
    # preprocess expression data
    all_exp = GeneExpPreprocessor(top_N=args.top_N,
            subset_method=args.subset_method, filter_subtypes=False).get_df()
    # run cross validation
    
    print(f"Expression data shape (pre-sampled data): {all_exp.shape}")
        
    sample_runner = SamplingRunner(latent_dim=1000, hidden_dims=[625, 750, 875], sample_size=800, lr=5e-3, beta=0.5, pretrain_epochs=100)
    
    sampled_exp, sil = sample_runner.run(all_exp, verbose=True)
    print(f"Sampled Expression data shape: {sampled_exp.shape}")
    
    ae_runner = GeneExpressionRunner(sampled_exp,latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dim,
        lr=args.lr,
        batch_size=args.batch_size)
    cv_losses = ae_runner.cross_validate()
    print(f'cv_losses: {cv_losses}')

    cv_losses_df = pd.DataFrame(cv_losses, index=[f"fold_{i+1}" for i in range(len(cv_losses))])
    cv_losses_df.to_csv(f"results/tables/cv_losses.csv", index=False, header=False)

    # train on synthetic data and then get latent space
    model, scaler = ae_runner.train_all_and_encode(return_model=True)

    if RUN_EACH_SUBTYPE:
        print("Running subtype-specific latent_space extraction...")
        subtypes= ['BRCA_LumA', 'BRCA_Her2', 'BRCA_LumB', 'BRCA_Normal', 'BRCA_Basal']
        for subtype in subtypes:
            print(f"Extracting latent space for subtype: {subtype}")
            
            subtype_samples = GeneExpPreprocessor(top_N=args.top_N,subset_method=args.subset_method, filter_subtypes=True, subtypes=[subtype]).get_df().index
            subtype_exp = all_exp.loc[subtype_samples]
            print(f"{subtype} Expression data shape: {subtype_exp.shape}")
            latent = ae_runner.trained_model_encode(model, subtype_exp, scaler)

            # train on all samples and get latent space
            df_latent = pd.DataFrame(latent, index=subtype_exp.index, columns=[f"latent_{i}" for i in range(latent.shape[1])])
            df_latent.to_csv(f"results/tables/latent_space_{args.latent_dim}dim_{subtype}.csv")
            print("Done extracting latent space for subtype:", subtype)

if __name__ == "__main__":
    main()
        
        
