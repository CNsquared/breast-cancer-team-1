import shap
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import datetime

def shapley_plot(runner, model):
  X = runner.X_train.values.astype('float32')  #runner.X_train is a DataFrame
  col_name = runner.X_train.columns

  # Train-test split
  X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

  # Back to torch tensors
  X_train = torch.tensor(X_train)
  X_test = torch.tensor(X_test)

  explainer = shap.DeepExplainer(model.encoder, X_train)
  shap_values = explainer.shap_values(X_test)
  timestamp = datetime.datetime.now().strftime("%d_%H-%M")

  shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names = col_name, show = False)
  plt.title("SHAP Feature Importance")
  plt.savefig("results/figures/shap_bar_plot_summary_{timestamp}.png", bbox_inches='tight', dpi=300)
  plt.clf()

  for i in range(shap_values.shape[-1]):
    explanation = shap.Explanation(values=shap_values[:,:,i],
                               base_values=explainer.expected_value[i],
                               data=X_test,
                               feature_names=col_name)
    shap.summary_plot(explanation, plot_type="bar", show = False)
    plt.title(f"SHAP Feature Importance for latent space {i}")
    plt.savefig(f"results/figures/shap_bar_plot_latent_{i}_{timestamp}.png", bbox_inches='tight', dpi=300)
    plt.clf()