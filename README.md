# Predicting breast cancer treatment response (pCR and RFS)

Group coursework for COMP4139 Machine Learning (University of Nottingham, 2025-26), done by a team of five. Using clinical and MRI-based features of 400 breast cancer patients treated with chemotherapy, the team predicted two outcomes:

- **pCR** (pathological complete response, yes or no): a classification task.
- **RFS** (relapse-free survival time in months): a regression task.

The data came from a simplified I-SPY 2 trial dataset provided with the module (11 clinical features and 107 MRI-based radiomics features per patient).

**My part:** the feature selection for the RFS regression task (Pearson correlation, mutual information and Lasso), and the presentation of the regression results. The rest was done by teammates. The repository holds the final code for both tasks and the group report.

## How the code works

1. **Cleaning** (`dataPreProcessing.py`, `knnImputation.py`, `outlierDelete.py`, one copy in each folder): replaces missing values (coded as 999), using the mean for continuous features and KNN imputation for discrete ones, and removes outliers with the IQR rule.
2. **Feature selection**, done on the training data only:
   - rank every feature by absolute Pearson correlation and by mutual information, scale both to 0 to 1, add them, and keep the top 20,
   - always keep three clinically important features (ER, HER2 and Gene),
   - drop features with a zero coefficient in an L1-regularised model (Lasso for RFS, L1 logistic regression for pCR).
3. **Models:**
   - `classification/`: logistic regression (L2), random forest, gradient boosting and SVM for pCR, evaluated with 5-fold cross-validation, with the feature selection repeated inside each fold.
   - `regression/`: XGBoost, random forest, MLP and KNN for RFS, evaluated on a held-out 20% of the patients.

## Settings used

| Step | Settings |
|---|---|
| Missing values | Coded as 999. Continuous features: mean imputation. Discrete features: KNN imputation with 5 neighbours. |
| Outliers | Removed with the IQR rule (below Q1 - 1.5 x IQR or above Q3 + 1.5 x IQR). |
| Feature selection | Top 20 features by combined Pearson and mutual-information rank, plus ER, HER2 and Gene, then Lasso (5-fold `LassoCV`, `random_state=42`) for RFS. Always done on the training data only. |
| pCR evaluation | 5-fold stratified cross-validation (shuffled, `random_state=42`), features standardised inside each fold. |
| pCR models | Logistic regression (L2, `liblinear`, balanced class weights); random forest (300 trees, balanced class weights); gradient boosting (300 trees, learning rate 0.05); SVM (RBF kernel, C = 1, `gamma='scale'`, balanced class weights). |
| RFS evaluation | One 80/20 train/test split (`random_state=42`). |
| RFS models | XGBoost (1300 trees, depth 3, learning rate 0.03, column sample 0.8, gamma 0.5); random forest (500 trees, no depth limit); MLP (layers 128-64-32, ReLU, Adam, learning rate 0.001, early stopping, up to 500 iterations, standardised inputs); KNN (Manhattan distance, standardised inputs, k from 1 to 30 and uniform or distance weights searched: the best was k = 16 with distance weights). |
| Metrics | pCR: balanced accuracy, AUC, recall, F1. RFS: MAE, RMSE, MAPE, R². |

## Why these choices

- **Balanced accuracy and balanced class weights.** After cleaning, only 30 of the 166 patients have pCR (18%). A model that always says "no" would reach 82% plain accuracy, so balanced accuracy is the main pCR score and the models that support it use balanced class weights.
- **Simple models did best.** With 166 patients and over a hundred features before selection, flexible models overfit. The group report found L2 logistic regression the most balanced pCR model (balanced accuracy 0.74, AUC 0.79, recall 0.83, F1 0.48), while gradient boosting scored lower on balanced accuracy (0.62 in the re-run) and the SVM's AUC was close to chance (0.50 in the re-run). For RFS, the distance-based KNN had the lowest error (MAE 24.18 against 26.63 to 28.59 for the other models), although none of the models explains much of the variation (R² of 0.05 at best).
- **Feature selection inside the training data.** Choosing features using all the patients would let the test patients influence the choice, so selection is repeated on each training set. ER, HER2 and Gene are always kept because they are established clinical markers.
- **Filter methods first, then Lasso.** Pearson correlation finds linear links, mutual information finds non-linear ones, and Lasso then removes features that add nothing once the others are included.

## Results (from the group report)

**pCR classification** (Table II of the report): L2 logistic regression was the most balanced model.

| Model | Mean balanced accuracy | AUC | Recall | F1 |
|---|---|---|---|---|
| L2 logistic regression | 0.7414 | 0.7909 | 0.83 | 0.48 |
| Random forest | 0.5984 | 0.7862 | 0.77 | 0.45 |
| Gradient boosting | 0.6172 | 0.8049 | 0.80 | 0.48 |
| SVM (RBF) | 0.7635 | 0.5431 | 0.13 | 0.21 |

**RFS regression** (Table III, error in months): KNN had the lowest error.

| Model | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| XGBoost | 26.63 | 32.64 | 86.52 |
| Random forest | 26.72 | 31.68 | 91.24 |
| MLP | 28.59 | 33.12 | 87.35 |
| **KNN** | **24.18** | **28.99** | **86.34** |

The R² of KNN is only 0.05, and the R² of random forest and MLP is negative, so none of the models predicts RFS well.

## Limitations

- **Regression evaluation:** the RFS models are scored on a **single 80/20 split** (`random_state=42`), so only 33 patients (20% of the 166 left after cleaning) are used for testing, and the numbers are noisy. The report describes cross-validation for both tasks, but `regressionWithEvaluation.py` uses one hold-out split.
- KNN's number of neighbours and weighting were **chosen by R² on that same test split**, so the KNN result is probably optimistic.
- **Classification evaluation:** `classification.py` uses stratified 5-fold cross-validation, but `evaluation.py` (AUC, recall, F1) uses plain 5-fold, and picks the decision threshold from the out-of-fold predictions, which flatters recall and F1.
- **Small sample after cleaning:** of the 400 patients, 166 remain after the cleaning step (30 of them with pCR), for both tasks.
- Imputation and outlier removal were applied before the data was split.
- `classification.py` standardises the features, but `evaluation.py` (AUC, recall, F1) does not, so the two scripts test slightly different set-ups.
- **Not a ready-to-use workflow:** cleaning, feature selection and the models are separate scripts, with no saved pipeline that handles missing values by itself, so a new patient with missing values could not be scored directly.
- With an R² close to zero, the RFS models are only slightly better than predicting the average.

**What I would do next:** package the cleaning, feature selection and model into one scikit-learn pipeline that handles missing values by itself, choose KNN's settings with cross-validation on the training data only, use repeated cross-validation for every final score, drop the SVM from the comparison or report its AUC over many seeds, and compare every model with a simple baseline.

## Feedback and what I would improve

The marker described the development code as clearly commented, and the experimental design as well structured, with cross-validation and several evaluation metrics. The main criticisms were that the report needed more detail on parameter settings and a clearer link between modelling decisions and results, that there were no installation or run instructions, and that missing values were not handled inside a ready-to-use workflow.

This repository addresses the first three points with the "Settings used" and "Why these choices" sections above and the run instructions below. The last point is still open: I would turn the cleaning, feature selection and model into a single pipeline that accepts patients with missing values.

## Repository contents

| Path | What it is |
|---|---|
| `classification/` | Python code for the pCR classification task |
| `regression/` | Python code for the RFS regression task |
| `report/group-report.pdf` | The group's report (5 pages) |
| `data/` | Put the dataset here (see `data/README.md`) |
| `requirements.txt` | Python packages |

## Run it

1. Install Python 3 and the packages: `pip install -r requirements.txt`
2. Put `TrainDataset2025.csv` (provided with the module) in the `data/` folder.
3. From the main folder of this repository, run one of:
   - `python classification/classification.py` (balanced accuracy for each model: 0.7414, 0.7487, 0.6209, 0.7635)
   - `python classification/evaluation.py` (AUC and classification reports; it also opens a ROC plot. Expected AUCs: 0.7909, 0.7925, 0.8061, 0.5012)
   - `python regression/regressionWithEvaluation.py` (MAE, RMSE, MAPE and R² for each model; expected KNN MAE 24.18)
