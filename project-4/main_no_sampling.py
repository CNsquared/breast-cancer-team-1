from src.utils.preprocess import MutPreprocessor
from src.utils.preprocess import GeneExpPreprocessor
from src.models.autoencoder_runner import GeneExpressionRunner
from src.models.autoencoder import GeneExpressionAutoencoder
from src.utils.data_loader import load_expression_matrix
from src.utils.sampling import SamplingRunner
import numpy as np
import pandas as pd
import torch
import os

import datetime

geneExpParams = {
    'top_N': 250,
    'subset_method': 'dndscv'    # method to subset genes {dndscv, variance}
}

autoEncoderParams = {
    'latent_dim': 5,            # Number of latent features
    'hidden_dims': [128,64],    # Hidden layers
    'lr': 5e-4,                 # Learning rate
    'dropout_rate': 0.2,        # dropout rate for the autoencoder
    'weight_decay': 1e-5        # Weight decay for the optimizer (L2 regularization)
}

trainingParams = {
    'patience': 10,            # Number of epochs where model training saturated
    'max_epochs': 200,         # Maximum epochs
    'delta': 1e-4,            # Minimum change in validation loss to consider improvement
}

def main():

    # get expression data
    all_exp = GeneExpPreprocessor(**geneExpParams, filter_subtypes=False).get_df()
    basal_samples = GeneExpPreprocessor(**geneExpParams).get_df().index
    basal_exp = all_exp.loc[basal_samples]
    print(f"Expression data shape (train data): {all_exp.shape}")
    
    # create runner for autoencoder
    runner = GeneExpressionRunner(all_exp, **autoEncoderParams)
    
    # run cross validation
    cv_losses = runner.cross_validate()
    print(f'cv_losses: {cv_losses}')

    # train on all samples and get model, scaler
    model, scaler, weights_all = runner.train_all_and_encode(**trainingParams, return_model=True)

    # use model and scaler to encode subtypes
    print("Running subtype-specific latent_space extraction...")
    subtypes= ['BRCA_LumA', 'BRCA_Her2', 'BRCA_LumB', 'BRCA_Normal', 'BRCA_Basal']
    for subtype in subtypes:
        subtype_samples = GeneExpPreprocessor(**geneExpParams, subtypes=[subtype]).get_df().index
        subtype_exp = all_exp.loc[subtype_samples]
        print(f"{subtype} Expression data shape: {subtype_exp.shape}")
        latent = runner.trained_model_encode(model, subtype_exp, scaler)

        # train on all samples and get latent space
        df_latent = pd.DataFrame(latent, index=subtype_exp.index, columns=[f"latent_{i}" for i in range(latent.shape[1])])
        df_latent.to_csv(f"results/tables/latent_space_{autoEncoderParams['latent_dim']}dim_{subtype}.csv")
        print("Done extracting latent space for subtype:", subtype)

    # write out latent space, weights, cv losses, and pretrained model
    print("Saving results...")

    os.makedirs("results/models", exist_ok=True)
    datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch.save(model.state_dict(), f"results/models/no_sampling_autoencoder_{datetime}.pth")  # save model state

    weights_all = pd.DataFrame(weights_all, index=all_exp.columns)
    weights_all.to_csv(f"results/tables/no_sampling_gene_to_latent_weights.csv")
    
    cv_losses_df = pd.DataFrame(cv_losses, index=[f"fold_{i+1}" for i in range(len(cv_losses))])
    cv_losses_df.to_csv(f"results/tables/no_sampling_cv_losses.csv", index=False, header=False)
    print("Job completed successfully!")

if __name__ == "__main__":
    main()