import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.dummy import DummyRegressor

def evaluate_models(X: pd.DataFrame, y: list, models: dict = None,  random_state=42, cv_folds: int = 5, filter_data: callable = None, **kwargs):
    """
    Evaluate multiple regression models using K-Fold cross-validation.

    Parameters:
    - X: pd.DataFrame, feature matrix with rows as samples and columns as features
    - y: list or np.array, continuous target values
    - models: dict, dictionary of model names and their corresponding sklearn model instances
    - random_state: int, random seed for reproducibility
    - cv_folds: int, number of folds for cross-validation
    - filter_data: callable, function to filter features based on training data, should take (X_train, y_train) and return (X_filtered, feature_names)
    """

    y_array = np.asarray(y, dtype=float)
    if models is None:
        models = {
            'Dummy (mean)': DummyRegressor(strategy='mean'),
            'Ridge Regression': make_pipeline(StandardScaler(), Ridge()),
            'Random Forest': RandomForestRegressor(),
            'SVR': make_pipeline(StandardScaler(), SVR()),
            'XGBoost': XGBRegressor()
        }

    if filter_data is None:
        def filter_data(X: pd.DataFrame, y: np.ndarray):
            return X, X.columns.tolist()
        
    results = {}
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for name, model in models.items():
        r2_scores, mses, maes, yt, yp = [], [], [], [], []
        for train_idx, test_idx in kf.split(X, y_array):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]

            # Feature filtering
            X_train_filtered, features = filter_data(X_train, y_train, **kwargs)
            X_test_filtered = X_test.loc[:, features]

            # Scale y within the fold
            y_scaler = StandardScaler()
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

            # Fit model on scaled y
            model.fit(X_train_filtered, y_train_scaled)

            # Predict and inverse-transform to original scale
            y_pred_scaled = model.predict(X_test_filtered)
            y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

            # Evaluate using original y_test
            r2_scores.append(r2_score(y_test, y_pred))
            mses.append(mean_squared_error(y_test, y_pred))
            maes.append(mean_absolute_error(y_test, y_pred))
            yt.append(y_test)
            yp.append(y_pred)

        results[name] = {
            'R2': {'mean': np.mean(r2_scores), 'std': np.std(r2_scores), 'scores': r2_scores},
            'MSE': {'mean': np.mean(mses), 'std': np.std(mses), 'scores': mses},
            'MAE': {'mean': np.mean(maes), 'std': np.std(maes), 'scores': maes},
            'features': features,
            'y_true': np.concatenate(yt),
            'y_pred': np.concatenate(yp)
        }

    return results

def plot_predicted_vs_true(results, ax=None, title="Predicted vs. True"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    for model_name in results:
        y_true = results[model_name]['y_true']
        y_pred = results[model_name]['y_pred']
        ax.scatter(y_true, y_pred, alpha=0.4, label=model_name)

    all_vals = np.concatenate([results[m]['y_true'] for m in results])
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal')

    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return ax

def plot_residuals(results, ax=None, title="Residuals vs. True Values"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    for model_name in results:
        y_true = results[model_name]['y_true']
        y_pred = results[model_name]['y_pred']
        residuals = y_pred - y_true
        ax.scatter(y_true, residuals, alpha=0.4, label=model_name)

    ax.axhline(0, color='k', linestyle='--')
    ax.set_xlabel("True Values")
    ax.set_ylabel("Residuals (Predicted - True)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return ax

def plot_regression_metrics(results, metric='R2', ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    models = list(results.keys())
    means = np.array([results[m][metric]['mean'] for m in models])
    stds = np.array([results[m][metric]['std'] for m in models])

    # Plot horizontal bar chart with error bars
    y_pos = np.arange(len(models))
    ax.barh(y_pos, means, xerr=stds, align='center', capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.invert_yaxis()  # optional: best model at top
    ax.set_xlabel(f"{metric} (± std)")
    ax.set_title(title or f"{metric} Scores by Model")
    ax.grid(True, axis='x')
    return ax
