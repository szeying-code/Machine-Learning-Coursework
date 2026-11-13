import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, classification_report

from dataPreProcessing import all_df_cleaned
from classification import models
from featureSelectionClassification import select_features_for_pcr

#Basic setting
df_full = all_df_cleaned.copy()
y_full = df_full["pCR (outcome)"].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random Guess')

model_auc_scores = {}
model_best_threshold = {}
model_predicted_prob = {}
model_y_test = {}

#Draw ROC
for model_name, model in models.items():

    print(f"\nModel: {model_name}")

    all_probs = []
    all_y_true = []

    #KFold
    for train_idx, test_idx in kf.split(df_full, y_full):

        df_train = df_full.iloc[train_idx].copy()
        df_test = df_full.iloc[test_idx].copy()

        #Feature Selection
        selected_features = select_features_for_pcr(df_train)

        X_train = df_train[selected_features].values
        y_train = df_train["pCR (outcome)"].values

        X_test = df_test[selected_features].values
        y_test = df_test["pCR (outcome)"].values

        #Training
        trained_model = model.fit(X_train, y_train)

        #Prediction probability for ROC
        probas = trained_model.predict_proba(X_test)[:, 1]

        all_probs.extend(probas)
        all_y_true.extend(y_test)

    # -> numpy
    all_probs = np.array(all_probs)
    all_y_true = np.array(all_y_true)

    #Save
    model_predicted_prob[model_name] = all_probs
    model_y_test[model_name] = all_y_true

    #ROC
    fpr, tpr, thresholds = roc_curve(all_y_true, all_probs)
    auc_score = auc(fpr, tpr)
    model_auc_scores[model_name] = auc_score

    #Find max Youden's J = TPR - FPR
    best_thr = thresholds[np.argmax(tpr - fpr)]
    model_best_threshold[model_name] = best_thr

    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.3f})")

    print(f"AUC = {auc_score:.4f}")
    print(f"Best threshold (max TPR-FPR) = {best_thr:.4f}")

    #Output classification(PCR) reports based on the optimal threshold
    y_pred = (all_probs > best_thr).astype(int)
    print("Classification Report:")
    print(classification_report(all_y_true, y_pred))

# Draw Image
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves of All Models (Feature Selection per Fold)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Summary
for name in models.keys():
    print(f"{name:<18}: AUC={model_auc_scores[name]:.4f}, BestThreshold={model_best_threshold[name]:.4f}")
