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
    model, scaler, weights_all = ae_runner.train_all_and_encode(return_model=True)

    weights_all = pd.DataFrame(weights_all, index=all_exp.columns)
    weights_all.to_csv(f"results/tables/no_sampling_gene_to_latent_weights.csv")

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
        
        


from src.utils.preprocess import MutPreprocessor
from src.utils.preprocess import GeneExpPreprocessor
from src.models.autoencoder_runner import GeneExpressionRunner
from src.models.autoencoder import GeneExpressionAutoencoder
from src.utils.data_loader import load_expression_matrix
from src.utils.sampling import SamplingRunner, hyperparam_search
import numpy as np
import pandas as pd

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
    
    basal_samples = GeneExpPreprocessor(top_N=args.top_N,subset_method=args.subset_method, filter_subtypes=True).get_df().index
    # run cross validation
    
    print(f"Expression data shape (train data): {all_exp.shape}")
        
    sample_runner = SamplingRunner(latent_dim=1000, hidden_dims=[625, 750, 875], sample_size=800, lr=5e-3, beta=0.5, pretrain_epochs=100)
    
    sampled_exp, sil = sample_runner.run(all_exp, verbose=True)
    print(f"Expression data shape: {all_exp.shape}")
    gene_names = all_exp.columns
    
    ae_runner = GeneExpressionRunner(sampled_exp,latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dim,
        lr=args.lr,
        batch_size=args.batch_size)
    cv_losses = ae_runner.cross_validate()
    print(f'cv_losses: {cv_losses}')

    # train on synthetic data and then get latent space
    basal_exp = all_exp.loc[basal_samples]
    model, scaler, weights_all = ae_runner.train_all_and_encode(return_model=True)
    latent = ae_runner.trained_model_encode(model, basal_exp, scaler)

    df_latent = pd.DataFrame(latent, index=basal_exp.index, columns=[f"latent_{i}" for i in range(latent.shape[1])])
    df_latent.to_csv(f"results/tables/latent_space.csv")
    weights_all = pd.DataFrame(weights_all, index=gene_names)
    weights_all.to_csv(f"results/tables/no_sampling_gene_to_latent_weights.csv")
    
    cv_losses_df = pd.DataFrame(cv_losses, index=[f"fold_{i+1}" for i in range(len(cv_losses))])
    cv_losses_df.to_csv(f"results/tables/cv_losses.csv", index=False, header=False)

def each_subtype(subtypes= ['BRCA_LumA', 'BRCA_Her2', 'BRCA_LumB', 'BRCA_Normal', 'BRCA_Basal']):
    args = parse_args()
    # Runs cross-validation for each subtype and saves latent space and cross-validation losses.
    LATENT_DIM=5
    all_cv_losses = []
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
        df_latent, weights_all  = pd.DataFrame(latent, index=df_exp.index, columns=[f"latent_{i}" for i in range(latent.shape[1])])
        df_latent.to_csv(f"results/tables/latent_space_{LATENT_DIM}dim_{subtype}.csv")
    
    all_cv_losses_df = pd.DataFrame(all_cv_losses, index=subtypes, columns=[f"fold_{i+1}" for i in range(len(all_cv_losses[0]))])
    all_cv_losses_df.to_csv(f"results/tables/cv_losses_{LATENT_DIM}dim_all_subtypes.csv", index=True)

if __name__ == "__main__":
    main()
    if RUN_EACH_SUBTYPE:
        each_subtype(subtypes=['BRCA_LumA', 'BRCA_Her2', 'BRCA_LumB', 'BRCA_Normal', 'BRCA_Basal'])
    else:
        print("Skipping subtype-specific runs. Set RUN_EACH_SUBTYPE to True to enable. Running Basal subtype only.")
        each_subtype(subtypes=['BRCA_Basal'])
        
        
