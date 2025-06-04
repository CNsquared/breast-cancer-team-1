import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import set_config
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder

# endpoint = ['Disease Free Status','Disease Free (Months)','Disease-specific Survival status','Overall Survival Status','Overall Survival (Months)','Progression Free Status','Progress Free Survival (Months)']
# data_endpoint = data[endpoint]

# cols_to_convert = [
#     'Disease Free Status',
#     'Disease-specific Survival status',
#     'Overall Survival Status',
#     'Progression Free Status'
# ]

# # Convert by extracting number before colon and casting to float
# for col in cols_to_convert:
#     data[col] = data[col].str.extract(r'(^\d)').astype(float)

def kaplan_meier(factor,endpoint=[endpoint[0],endpoint[1]]):
  data_km = data[[factor,endpoint[0],endpoint[1]]]
  data_km = data_km.dropna()
  for value in data_km[factor].unique():
      mask = data_km[factor] == value
      time_cell, survival_prob_cell, conf_int = kaplan_meier_estimator(
          data_km[endpoint[0]][mask].astype(bool), data_km[endpoint[1]][mask], conf_type="log-log")

      plt.step(time_cell, survival_prob_cell, where="post", label=f"{value} (n = {mask.sum()})")
      plt.fill_between(time_cell, conf_int[0], conf_int[1], alpha=0.25, step="post")

  fig = plt.gcf()                 # Get current figure
  fig.set_size_inches(13, 6)      # Set new width x height in inches
  plt.ylim(0, 1)
  plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
  plt.xlabel("time $t$")
  plt.title("Factor : {factor}, Endpoint : {endpoint[0]}".format(factor=factor,endpoint=endpoint))
  plt.legend(loc="lower left")


def cox(factor, samples = 5, endpoint=[endpoint[0],endpoint[1]]):
  data_cox = data[factor+[endpoint[0],endpoint[1]]]
  data_cox = data_cox.dropna()

  data_cox_numeric = OneHotEncoder().fit_transform(data_cox[factor])
  factor_enc = data_cox_numeric.columns
  data_cox[endpoint[0]] = data_cox[endpoint[0]].astype(bool)

  data_end = np.array(
    list(zip(data_cox[endpoint[0]], data_cox[endpoint[1]])),
    dtype=[('status', 'bool'), ('time', 'f8')])

  set_config(display="text")  # displays text representation of estimators

  estimator = CoxPHSurvivalAnalysis()
  estimator.fit(data_cox_numeric, data_end)
  print(pd.Series(estimator.coef_, index=data_cox_numeric.columns))

  samples = np.arange(samples)
  pred_surv = estimator.predict_survival_function(data_cox_numeric.iloc[samples][factor_enc])
  time_points = np.arange(1, max(data_cox[endpoint[1]]))
  for i, surv_func in enumerate(pred_surv):
      plt.step(time_points, surv_func(time_points), where="post", label=f"Sample {i + 1}")
  plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
  plt.xlabel("time $t$")
  plt.legend(loc="best")

def fit_and_score_features(endpoint=[endpoint[0],endpoint[1]]):
    col_remove = list(data.select_dtypes(include='object').columns)
    col_remove = col_remove + [endpoint[0], endpoint[1]]

    X_enc = OneHotEncoder().fit_transform(data.drop(columns=col_remove))
    n_features = X_enc.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis()
    for j in range(n_features):
      # Extract j-th column
      Xj = X_enc.iloc[:, j : j + 1]
      data[endpoint[0]] = data[endpoint[0]].astype(bool)
      # Find rows without NaNs in that column
      non_na_mask = (~Xj.isna().any(axis=1) &~data[endpoint[0]].isna() &~data[endpoint[1]].isna())

      # Filter X and endpoint
      Xj = Xj[non_na_mask]
      data_end = np.array(list(zip(data.loc[non_na_mask, endpoint[0]],
                                  data.loc[non_na_mask, endpoint[1]])),
                        dtype=[('status', 'bool'), ('time', 'f8')])
      m.fit(Xj, data_end)
      scores[j] = m.score(Xj, data_end)
    return scores, X_enc

#scores, X_enc = fit_and_score_features()
#ranking = pd.Series(scores, index=X_enc.columns).sort_values(ascending=False)


