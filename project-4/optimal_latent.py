from src.utils.preprocess import GeneExpPreprocessor
from src.models.autoencoder_runner import GeneExpressionRunner
import pandas as pd
import os
import pickle
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import datetime

geneExpParams = {
    'top_N': 1000,
    'subset_method': 'dndscv'    # method to subset genes {dndscv, variance}
}

autoEncoderParams = {
    'hidden_dims': [512, 256, 128],    # Hidden layers
    'lr': 5e-4,                 # Learning rate
    'dropout_rate': 0.1,        # dropout rate for the autoencoder
    'weight_decay': 1e-6        # Weight decay for the optimizer (L2 regularization)
}

trainingParams = {
    'patience': 20,            # Number of epochs where model training saturated
    'max_epochs': 400,         # Maximum epochs
    'delta': 1e-4,            # Minimum change in validation loss to consider improvement
    'retrain_model': True
}

def main():
    # get expression data
    all_exp = GeneExpPreprocessor(**geneExpParams, filter_subtypes=False, save=False).get_df()
    print(f"Expression data shape (train data): {all_exp.shape}")
    results = {}
    for k in range(1, 31):
        print(f"Running with latent dim: {k}")
        ae_params = autoEncoderParams.copy()
        ae_params['latent_dim'] = k
        # create runner for autoencoder
        runner = GeneExpressionRunner(all_exp, **ae_params)
    
        # run cross validation
        cv_losses = runner.cross_validate()
        print(f'cv_losses: {cv_losses}')
        # train on all samples and get model, scaler
        model, scaler, weights_all = runner.train_all_and_encode(**trainingParams, return_model=True)
        latent = runner.trained_model_encode(model, all_exp, scaler)
        results[k] = {
            'latent_dim': k,
            'model_state_dict': model.state_dict(),
            'params': {'geneExpParams': geneExpParams, 'autoEncoderParams': ae_params, 'trainingParams': trainingParams},
            'weights_all': weights_all,
            'cv_losses': cv_losses,
            'latent_space': latent
        }

    timestamp = datetime.datetime.now().strftime("%m%d_%H-%M")

    os.makedirs('results/cv_results', exist_ok=True)

    # create df with cv_losses for each k
    cv_losses_df = pd.DataFrame.from_dict(
        {k: sum(v['cv_losses']) / len(v['cv_losses']) for k, v in results.items()},
        orient='index',
        columns=['avg_cv_loss']
    )
    cv_losses_df.index.name = 'latent_dim'
    cv_losses_df['min_cv_loss'] = [min(v['cv_losses']) for v in results.values()]
    cv_losses_df['std_cv_loss'] = [pd.Series(v['cv_losses']).std() for v in results.values()]
    cv_losses_df['geneExpParams'] = str(geneExpParams)
    cv_losses_df['autoEncoderParams'] = str(autoEncoderParams)
    cv_losses_df['trainingParams'] = str(trainingParams)
    cv_losses_df.to_csv(f'results/cv_results/optimal_latent_cv_losses_{timestamp}.csv', index=True)

    output=f'results/cv_results/optimal_latent_{timestamp}.pkl'
    with open(output, "wb") as f:
        pickle.dump(results, f)

    print("Job completed successfully!")

if __name__ == "__main__":
    main()