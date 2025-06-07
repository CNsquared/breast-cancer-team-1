import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, accuracy_score, roc_auc_score, f1_score, average_precision_score, auc, roc_curve, matthews_corrcoef, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class RandomGuessClassifier(BaseEstimator, ClassifierMixin):
    """
    A dummy classifier that predicts classes by random sampling
    according to the class distribution observed in the training data.
    """
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.probs_ = counts / counts.sum()
        self.rng_ = np.random.RandomState(self.random_state)
        return self

    def predict(self, X):
        return self.rng_.choice(self.classes_, size=len(X), p=self.probs_)

    def predict_proba(self, X):
        # return a (n_samples, n_classes) matrix with the same class probabilities
        return np.tile(self.probs_, (len(X), 1))


def evaluate_models(
    X: pd.DataFrame,
    y: list,
    models: dict = None,
    random_state: int = 42,
    cv_folds: int = 5,
    filter_data: callable = None,
    **kwargs
):
    """
    Evaluate multiple classification models using:
      1) A held-out test split
      2) Hyperparameter tuning via GridSearchCV (small grids)
      3) Final test-set evaluation of best models

    Parameters:
    - X: pd.DataFrame, features
    - y: list or np.ndarray, integer class labels
    - models: dict of { name: estimator } (if None, defaults provided)
    - random_state: int
    - cv_folds: int for inner CV
    - filter_data: fn(X_train, y_train, **kwargs) -> (X_tr_filtered, feature_list)
    - **kwargs: passed along to filter_data
    """
    # === 1) initial train/test split ===
    y_array = np.asarray(y, dtype=int)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_array,
        test_size=0.2,
        stratify=y_array,
        random_state=random_state
    )
    
    # feature filtering step preserved
    X_train_full, feats = filter_data(X_train_full, y_train_full, **kwargs) \
                        if filter_data else (X_train_full, X.columns.tolist())
    X_test = X_test.loc[:, feats]

    # === 2) default models with balanced class weights ===
    if models is None:
        models = {
            'Logistic Regression': make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight='balanced',
                                   max_iter=1_000,
                                   random_state=random_state)
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=random_state
            ),
            'SVM': make_pipeline(
                StandardScaler(),
                SVC(class_weight='balanced',
                    probability=True,
                    random_state=random_state)
            ),
            'XGBoost': XGBClassifier(
                eval_metric='logloss',
                random_state=random_state
            ),
            'Random Guess': RandomGuessClassifier(random_state=random_state)
        }

    # === 3) small hyper-parameter grids ===
    default_grids = {
        'Logistic Regression': {
            'logisticregression__C': [0.1, 1, 10]
        },
        'Random Forest': {
            'n_estimators': [100, 300],
            'max_depth': [None, 5, 10]
        },
        'SVM': {
            'svc__kernel': ['linear', 'rbf', 'poly'],
            'svc__C': [0.1, 1, 10]
        },
        'XGBoost': {
            'scale_pos_weight': [
                np.bincount(y_train_full == 0)[1] / np.bincount(y_train_full == 1)[1]
            ],
            'n_estimators': [100, 300],
            'max_depth': [3, 6]
        },
        # no hyperparams for Random Guess
        'Random Guess': {}
    }

    results = {}
    inner_cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state
    )

    # === 4) loop over each model ===
    for name, estimator in models.items():
        print(f"\n>>> Tuning {name} ...")
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=default_grids.get(name, {}),
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1
        )
        grid.fit(X_train_full, y_train_full)

        best_est = grid.best_estimator_
        best_params = grid.best_params_
        best_cv_score = grid.best_score_
        print(f"Best CV accuracy for {name}: {best_cv_score:.4f}")
        print(f"→ Best params: {best_params}")

        # === 6) final evaluation on the held-out test set ===
        acc_scores, auc_scores, f1_scores, f1_scores_weighted, mcc_scores, bacc_scores, presicion_scores, recall_scores, specificity_scores, yt, yp, ypr = [], [], [], [], [], [], [], [], [], [], [], []
        X_test_filt = X_test.loc[:, feats]
        y_pred = best_est.predict(X_test_filt)
        if hasattr(best_est, "predict_proba"):
            y_prob = best_est.predict_proba(X_test_filt)[:, 1]
        else:
            try:
                y_prob = best_est.decision_function(X_test_filt)
            except AttributeError:
                y_prob = y_pred

        acc_scores.append(accuracy_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_prob))
        f1_scores.append(f1_score(y_test, y_pred))
        f1_scores_weighted.append(f1_score(y_test, y_pred, average='weighted'))

        # wrap into arrays
        yt = y_test
        yp = y_pred
        ypr = y_prob
        
        results[name] = {
            #'Matthews Corr Coef': {'mean': np.mean(mcc_scores), 'std': np.std(mcc_scores), 'scores': mcc_scores, 'global': matthews_corrcoef(yt, yp)},
            #'Balanced Accuracy': {'mean': np.mean(bacc_scores), 'std': np.std(bacc_scores), 'scores': bacc_scores, 'global': balanced_accuracy_score(yt, yp)},
            'Accuracy': {'mean': np.mean(acc_scores), 'std': np.std(acc_scores), 'scores': acc_scores, 'global': accuracy_score(yt, yp)},
            'ROC AUC': {'mean': np.mean(auc_scores), 'std': np.std(auc_scores), 'scores': auc_scores, 'global': roc_auc_score(yt, ypr)},
            'F1 Score': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores), 'scores': f1_scores, 'global': f1_score(yt, yp)},
            'F1 Score Weighted': {'mean': np.mean(f1_scores_weighted), 'std': np.std(f1_scores_weighted), 'scores': f1_scores_weighted, 'global': f1_score(yt, yp, average='weighted')},
            #'Precision': {'mean': np.mean(presicion_scores), 'std': np.std(presicion_scores), 'scores': presicion_scores, 'global': precision_score(yt, yp, zero_division=0)},
            #'Sensitivity': {'mean': np.mean(recall_scores), 'std': np.std(recall_scores), 'scores': recall_scores, 'global': recall_score(yt, yp)},
            'y_true': yt,
            'y_pred': yp,
            'y_prob': ypr
        }

    return results



def plot_roc_curves(results, ax=None, title="ROC Curve Comparison", figsize=(6,4)):
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

def plot_precision_recall_curves(results, ax=None, title="Precision-Recall Curve Comparison", figsize=(6, 4)):
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
    ax.set_title(title or f"{metric}")
    ax.grid(True, axis='x')
    return ax
