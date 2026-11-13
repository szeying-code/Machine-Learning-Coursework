import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#Variables that must be retained
remain_features = ['ER', 'HER2', 'Gene']  # Can't delete

def compute_nonzero_features_for_pcr(df):

    #Define the independent variable and the dependent variable
    X = df.drop(columns=['pCR (outcome)', 'RelapseFreeSurvival (outcome)'])
    y_clf = df['pCR (outcome)']

    #Pearson
    corr_values_clf = X.corrwith(y_clf, method='pearson')
    corr_rank_clf = corr_values_clf.abs().sort_values(ascending=False)

    #MI
    mi_values_clf = mutual_info_classif(X, y_clf, random_state=42)
    mi_series_clf = pd.Series(mi_values_clf, index=X.columns).sort_values(ascending=False)

    #Mixed Scoring
    corr_norm_clf = corr_rank_clf / corr_rank_clf.max()
    mi_norm_clf = mi_series_clf / mi_series_clf.max()
    combined_score_clf = corr_norm_clf + mi_norm_clf
    combined_rank_clf = combined_score_clf.sort_values(ascending=False)
    top_features_clf = list(combined_rank_clf.head(20).index)

    #Make sure to retain the required features
    for f in remain_features:
        if f not in top_features_clf and f in X.columns:
            top_features_clf.append(f)

    #L1 Embedded
    X_selected_clf = X[top_features_clf].copy()
    y_selected_clf = y_clf.copy()

    logreg_l1 = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=1.0,
        max_iter=2000,
        tol=1e-3,
        random_state=42
    )

    scaler_l1 = StandardScaler()
    X_selected_clf_scaled = scaler_l1.fit_transform(X_selected_clf)

    logreg_l1.fit(X_selected_clf_scaled, y_selected_clf)

    coef_clf = pd.Series(logreg_l1.coef_[0], index=top_features_clf)
    nonzero_features_clf = coef_clf[coef_clf != 0]

    #Make sure that the required features exist again
    for f in remain_features:
        if f not in nonzero_features_clf.index and f in X.columns:
            nonzero_features_clf.loc[f] = 0.0

    nonzero_features_clf = nonzero_features_clf.sort_values(key=abs, ascending=False)

    #Visualization
    #plt.figure(figsize=(10, 6))
    #nonzero_features_clf.plot(kind='barh')
    #plt.show()

    return nonzero_features_clf

#External interface
def select_features_for_pcr(df):
    nonzero = compute_nonzero_features_for_pcr(df)
    return list(nonzero.index)

