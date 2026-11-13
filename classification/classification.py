import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from dataPreProcessing import all_df_cleaned
from featureSelectionClassification import select_features_for_pcr

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


#Load data
df_full = all_df_cleaned.copy()
y_full = df_full['pCR (outcome)'].values

#Define the model
models = {
    "Logistic_L2": LogisticRegression(
        penalty='l2',
        class_weight='balanced',
        solver="liblinear",   #More stable
        max_iter=5000
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=None,
        class_weight='balanced', random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, random_state=42
    ),
    "SVM_RBF": SVC(
        kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
        probability=True, random_state=42   # seed added so the SVM results repeat
    )
}

#5 Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



#Train each model in a loop
for name, clf in models.items():
    ba_scores = []

    for train_idx, test_idx in cv.split(df_full, y_full):

        #Divide the training set and the test set
        df_train = df_full.iloc[train_idx].copy()
        df_test = df_full.iloc[test_idx].copy()

        #Feature selection
        selected_features = select_features_for_pcr(df_train)

        X_train = df_train[selected_features].values
        y_train = df_train['pCR (outcome)'].values

        X_test = df_test[selected_features].values
        y_test = df_test['pCR (outcome)'].values

        #StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        #Training
        clf.fit(X_train, y_train)

        #Predict
        y_pred = clf.predict(X_test)

        #Calculate balanced accuracy
        ba = balanced_accuracy_score(y_test, y_pred)
        ba_scores.append(ba)

    #Transfer to array
    ba_scores = np.array(ba_scores)
    print(f"{name:<18}: mean BA = {ba_scores.mean():.4f}, std = {ba_scores.std():.4f}")
