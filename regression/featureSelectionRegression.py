
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

mandatory_features = ['ER', 'HER2', 'Gene']


def select_features_for_rfs(df_train):

    X = df_train.drop(columns=['RelapseFreeSurvival (outcome)'])
    y = df_train['RelapseFreeSurvival (outcome)']

    #Pearson
    corr_values = X.corrwith(y)
    corr_rank = corr_values.abs().sort_values(ascending=False)

    #Mutual Information
    mi_values = mutual_info_regression(X, y, random_state=42)
    mi_rank = pd.Series(mi_values, index=X.columns).sort_values(ascending=False)

    #Combined ranking
    combined_score = (corr_rank / corr_rank.max()) + (mi_rank / mi_rank.max())
    combined_rank = combined_score.sort_values(ascending=False)

    top_features = list(combined_rank.head(20).index)

    #Must remain
    for f in mandatory_features:
        if f not in top_features and f in X.columns:
            top_features.append(f)

    #Lasso
    X_sub = X[top_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    lasso = LassoCV(cv=5, random_state=42, max_iter=60000)
    lasso.fit(X_scaled, y)

    coef = pd.Series(lasso.coef_, index=top_features)
    nonzero = coef[coef != 0]

    #Remain mandatory feature
    for f in mandatory_features:
        if f not in nonzero.index and f in X.columns:
            nonzero.loc[f] = 0.0

    nonzero = nonzero.sort_values(key=abs, ascending=False)

    return list(nonzero.index)
