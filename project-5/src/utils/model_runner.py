import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, accuracy_score, roc_auc_score, f1_score, average_precision_score, auc, roc_curve, matthews_corrcoef, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin


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
        mcc_scores, bacc_scores, acc_scores, auc_scores, f1_scores, f1_scores_weighted, presicion_scores, recall_scores, specificity_scores, yt, yp, ypr = [], [], [], [], [], [], [], [], [], [], [], []
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
            mcc_scores.append(matthews_corrcoef(y_test, y_pred))
            bacc_scores.append(balanced_accuracy_score(y_test, y_pred))
            presicion_scores.append(precision_score(y_test, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_test, y_pred))
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity_scores.append(specificity)

            # Evaluate using original y_test
            yt.append(y_test)
            yp.append(y_pred)
            ypr.append(y_prob)

        yt = np.concatenate(yt)
        yp = np.concatenate(yp)
        ypr = np.concatenate(ypr)
        tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
        global_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results[name] = {
            'Matthews Corr Coef': {'mean': np.mean(mcc_scores), 'std': np.std(mcc_scores), 'scores': mcc_scores, 'global': matthews_corrcoef(yt, yp)},
            'Balanced Accuracy': {'mean': np.mean(bacc_scores), 'std': np.std(bacc_scores), 'scores': bacc_scores, 'global': balanced_accuracy_score(yt, yp)},
            'Accuracy': {'mean': np.mean(acc_scores), 'std': np.std(acc_scores), 'scores': acc_scores, 'global': accuracy_score(yt, yp)},
            'ROC AUC': {'mean': np.mean(auc_scores), 'std': np.std(auc_scores), 'scores': auc_scores, 'global': roc_auc_score(yt, ypr)},
            'F1 Score': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores), 'scores': f1_scores, 'global': f1_score(yt, yp)},
            'F1 Score Weighted': {'mean': np.mean(f1_scores_weighted), 'std': np.std(f1_scores_weighted), 'scores': f1_scores_weighted, 'global': f1_score(yt, yp, average='weighted')},
            'Precision': {'mean': np.mean(presicion_scores), 'std': np.std(presicion_scores), 'scores': presicion_scores, 'global': precision_score(yt, yp, zero_division=0)},
            'Sensitivity': {'mean': np.mean(recall_scores), 'std': np.std(recall_scores), 'scores': recall_scores, 'global': recall_score(yt, yp)},
            'Specificity': {'mean': np.mean(specificity_scores), 'std': np.std(specificity_scores), 'scores': specificity_scores, 'global': global_specificity},
            'features': features,
            'y_true': yt,
            'y_pred': yp,
            'y_prob': ypr
        }

    return results



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

def evaluate_models_nested(X: pd.DataFrame, y: list, models: dict = None,  random_state=42, cv_outer: int = 5, filter_data: callable = None, cv_inner: int = 5, **kwargs):
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
            'XGBoost': XGBClassifier(eval_metric='logloss', random_state=random_state),
            'Random Guess': RandomGuessClassifier(random_state=random_state)
        }

    if filter_data is None:
        def filter_data(X: pd.DataFrame, y: np.ndarray):
            return X, X.columns.tolist()
        
    results = {}
    skf_outer = StratifiedKFold(n_splits=cv_outer, shuffle=True, random_state=random_state)

    for name, model in models.items():
        best_params, mcc_scores, bacc_scores, acc_scores, auc_scores, f1_scores, f1_scores_weighted, precision_scores, recall_scores, specificity_scores, yt, yp, ypr = [], [], [], [], [], [], [], [], [], [], [], [], []
        for train_idx, test_idx in skf_outer.split(X, y_array):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]

            # Feature filtering
            X_train_filtered, features = filter_data(X_train, y_train, **kwargs)
            X_test_filtered = X_test.loc[:, features]

            # Fit model on y
            param_grids = {
                'Logistic Regression': {'logisticregression__C': [0.1, 1, 10]},
                'Random Forest': {'n_estimators': [100, 300], 'max_depth': [5, 10, None]},
                'SVM': {'svc__kernel': ['linear', 'rbf', 'poly'],'svc__C': [0.1, 1, 10]},
                'XGBoost':  {'scale_pos_weight': [np.bincount(y_train == 0)[1] / np.bincount(y_train == 1)[1]], 'n_estimators': [100, 300], 'max_depth': [3, 6]},
                'Random Guess': {}
            }

            # nested cross-validation for hyperparameter tuning
            inner_cv = StratifiedKFold(n_splits=cv_inner, shuffle=True, random_state=random_state)
            grid = GridSearchCV(model, param_grids[name], scoring='f1', cv=inner_cv)
            grid.fit(X_train_filtered, y_train)

            best_model = grid.best_estimator_
            best_params.append(grid.best_params_)

            # get predictions using the best model
            y_pred = best_model.predict(X_test_filtered)
            y_prob = best_model.predict_proba(X_test_filtered)[:, 1] if hasattr(best_model, "predict_proba") else y_pred

            acc_scores.append(accuracy_score(y_test, y_pred))
            auc_scores.append(roc_auc_score(y_test, y_prob))
            f1_scores.append(f1_score(y_test, y_pred))
            f1_scores_weighted.append(f1_score(y_test, y_pred, average='weighted'))
            mcc_scores.append(matthews_corrcoef(y_test, y_pred))
            bacc_scores.append(balanced_accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_test, y_pred))
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity_scores.append(specificity)

            # Evaluate using original y_test
            yt.append(y_test)
            yp.append(y_pred)
            ypr.append(y_prob)

        yt = np.concatenate(yt)
        yp = np.concatenate(yp)
        ypr = np.concatenate(ypr)
        tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
        global_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results[name] = {
            'Matthews Corr Coef': {'mean': np.mean(mcc_scores), 'std': np.std(mcc_scores), 'scores': mcc_scores, 'global': matthews_corrcoef(yt, yp)},
            'Balanced Accuracy': {'mean': np.mean(bacc_scores), 'std': np.std(bacc_scores), 'scores': bacc_scores, 'global': balanced_accuracy_score(yt, yp)},
            'Accuracy': {'mean': np.mean(acc_scores), 'std': np.std(acc_scores), 'scores': acc_scores, 'global': accuracy_score(yt, yp)},
            'ROC AUC': {'mean': np.mean(auc_scores), 'std': np.std(auc_scores), 'scores': auc_scores, 'global': roc_auc_score(yt, ypr)},
            'F1 Score': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores), 'scores': f1_scores, 'global': f1_score(yt, yp)},
            'F1 Score Weighted': {'mean': np.mean(f1_scores_weighted), 'std': np.std(f1_scores_weighted), 'scores': f1_scores_weighted, 'global': f1_score(yt, yp, average='weighted')},
            'Precision': {'mean': np.mean(precision_scores), 'std': np.std(precision_scores), 'scores': precision_scores, 'global': precision_score(yt, yp, zero_division=0)},
            'Sensitivity': {'mean': np.mean(recall_scores), 'std': np.std(recall_scores), 'scores': recall_scores, 'global': recall_score(yt, yp)},
            'Specificity': {'mean': np.mean(specificity_scores), 'std': np.std(specificity_scores), 'scores': specificity_scores, 'global': global_specificity},
            'features': features,
            'y_true': yt,
            'y_pred': yp,
            'y_prob': ypr,
            'best_params': best_params
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
