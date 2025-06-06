import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import precision_recall_curve, accuracy_score, roc_auc_score, f1_score, average_precision_score, roc_curve, auc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression



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

    y_array = np.asarray(y, dtype=int)
    if models is None:
        models = {
            'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state)),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'SVM': make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=random_state)),
            'XGBoost': XGBClassifier(eval_metric='logloss', random_state=random_state)
        }

    if filter_data is None:
        def filter_data(X: pd.DataFrame, y: np.ndarray):
            return X, X.columns.tolist()
        
    results = {}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for name, model in models.items():
        acc_scores, auc_scores, f1_scores, f1_scores_weighted, yt, yp, ypr = [], [], [], [], [], [], []
        for train_idx, test_idx in skf.split(X, y_array):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]

            # Feature filtering
            X_train_filtered, features = filter_data(X_train, y_train, **kwargs)
            X_test_filtered = X_test.loc[:, features]

            # Fit model on y
            model.fit(X_train_filtered, y_train)

            # Predict and inverse-transform to original scale
            y_pred = model.predict(X_test_filtered)
            y_prob = model.predict_proba(X_test_filtered)[:, 1] if hasattr(model, "predict_proba") else y_pred

            acc_scores.append(accuracy_score(y_test, y_pred))
            auc_scores.append(roc_auc_score(y_test, y_prob))
            f1_scores.append(f1_score(y_test, y_pred))
            f1_scores_weighted.append(f1_score(y_test, y_pred, average='weighted'))

            # Evaluate using original y_test
            yt.append(y_test)
            yp.append(y_pred)
            ypr.append(y_prob)

        yt = np.concatenate(yt)
        yp = np.concatenate(yp)
        ypr = np.concatenate(ypr)
        results[name] = {
            'Accuracy': {'mean': np.mean(acc_scores), 'std': np.std(acc_scores), 'scores': acc_scores, 'global': accuracy_score(yt, yp)},
            'ROC AUC': {'mean': np.mean(auc_scores), 'std': np.std(auc_scores), 'scores': auc_scores, 'global': roc_auc_score(yt, ypr)},
            'F1 Score': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores), 'scores': f1_scores, 'global': f1_score(yt, yp)},
            'F1 Score Weighted': {'mean': np.mean(f1_scores_weighted), 'std': np.std(f1_scores_weighted), 'scores': f1_scores_weighted, 'global': f1_score(yt, yp, average='weighted')},
            'features': features,
            'y_true': yt,
            'y_pred': yp,
            'y_prob': ypr
        }

    return results

def plot_roc_curves(results, ax=None, title="ROC Curve Comparison (k=17 genes)", figsize=(6,4)):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))


    for model_name in results:
        y_true = results[model_name]['y_true']
        y_prob = results[model_name]['y_prob']

        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return ax

def plot_precision_recall_curves(results, ax=None, title="Precision-Recall Curve Comparison (k=17 genes)", figsize=(6, 4)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for model_name in results:
        y_true = results[model_name]['y_true']
        y_prob = results[model_name]['y_prob']

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)

        ax.plot(recall, precision, label=f"{model_name} (AP = {avg_precision:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc='lower left')
    ax.grid(True)
    return ax

def plot_classification_metric(results, metric='ROC AUC', ax=None, title=None):
    """Metrics: 'Accuracy', 'ROC AUC', 'F1 Score', 'F1 Score Weighted'"""
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
