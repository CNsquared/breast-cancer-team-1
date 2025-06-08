from src.utils.preprocess import MutPreprocessor
from src.utils.preprocess import GeneExpPreprocessor
from src.models.autoencoder_runner import GeneExpressionRunner
from src.models.autoencoder import GeneExpressionAutoencoder
from src.utils.data_loader import load_expression_matrix
from src.utils.sampling import SamplingRunner
from src.utils.shapley import shapley_plot
import numpy as np
import pandas as pd
import torch
import os
import pickle

import datetime

geneExpParams = {
    'top_N': 300,
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
    'retrain_model': True
}

def main():
    # get expression data
    all_exp = GeneExpPreprocessor(**geneExpParams, filter_subtypes=False, save=False).get_df()
    print(f"Expression data shape (train data): {all_exp.shape}")
    models = {}
    for k in range(5, 11, 5):
        print(f"Running with latent dim: {k}")
        autoEncoderParams['latent_dim'] = k
        # create runner for autoencoder
        runner = GeneExpressionRunner(all_exp, **autoEncoderParams)
    
        # run cross validation
        cv_losses = runner.cross_validate()
        print(f'cv_losses: {cv_losses}')
        # train on all samples and get model, scaler
        model, scaler, weights_all = runner.train_all_and_encode(**trainingParams, return_model=True)
        latent = runner.trained_model_encode(model, all_exp, scaler)
        models[k] = {
            'latent_dim': k,
            'model': model,
            'scaler': scaler,
            'weights_all': weights_all,
            'cv_losses': cv_losses,
            'latent_space': latent
        }

    timestamp = datetime.datetime.now().strftime("%m%d_%H-%M")

    # create df with cv_losses for each k
    cv_losses_df = pd.DataFrame.from_dict(
        {k: sum(v['cv_losses']) / len(v['cv_losses']) for k, v in models.items()},
        orient='index',
        columns=['avg_cv_loss']
    )
    cv_losses_df.index.name = 'latent_dim'

    cv_losses_df.to_csv(f'results/cv_results/optimal_latent_cv_losses_{timestamp}.csv', index=False)

    output=f'results/cv_results/optimal_latent_{timestamp}.pkl'
    with open(output, "wb") as f:
        pickle.dump(models, f)

    print("Job completed successfully!")

if __name__ == "__main__":
    main()