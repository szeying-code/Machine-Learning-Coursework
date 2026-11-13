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

## Does the code reproduce the report?

The code in this repository was re-run on the training data with scikit-learn 1.9.1 and compared with the tables above. The report is included exactly as submitted. The code is the team's final code, with one change: a seed (`random_state=42`) was added to the SVM so that its results repeat from run to run.

| Result | Report | Re-run | Match |
|---|---|---|---|
| RFS: KNN (MAE / RMSE / MAPE) | 24.18 / 28.99 / 86.34 | 24.18 / 28.99 / 86.34 | exact |
| RFS: MLP | 28.59 / 33.12 / 87.35 | 28.59 / 33.12 / 87.35 | exact |
| RFS: random forest | 26.72 / 31.68 / 91.24 | 26.71 / 31.67 / 91.20 | within 0.04 |
| RFS: XGBoost | 26.63 / 32.64 / 86.52 | not re-run | not checked |
| pCR: L2 logistic regression (all four columns) | 0.7414 / 0.7909 / 0.83 / 0.48 | 0.7414 / 0.7909 / 0.83 / 0.48 | exact |
| pCR: SVM balanced accuracy | 0.7635 | 0.7635 | exact |
| pCR: gradient boosting | 0.6172 / 0.8049 / 0.80 / 0.48 | 0.6209 / 0.8061 / 0.80 / 0.48 | close |
| pCR: random forest | 0.5984 / 0.7862 / 0.77 / 0.45 | 0.7487 / 0.7925 / 0.80 / 0.46 | **does not match** |
| pCR: SVM AUC / recall / F1 | 0.5431 / 0.13 / 0.21 | 0.5012 / 0.10 / 0.17 | **does not match** |

Two rows do not reproduce:

- **SVM AUC, recall and F1.** The SVM's AUC is close to chance (0.5). Before the seed was added, its probabilities came from an unseeded step, and trying 300 different seeds gave AUCs from 0.46 to 0.61, so the report's 0.5431 is one possible outcome of that randomness and is not a stable result. No seed reproduced all three of the report's numbers.
- **Random forest balanced accuracy** (0.5984 in the report, 0.7487 here). The setting that produced the report's value was not found: neither the earlier version of the code in the team's folder nor a search over random forest settings gave it.

The report should therefore be read as the team's recorded results, and the table above shows which of them this code reproduces.

## Limitations

- **Regression evaluation:** the RFS models are scored on a **single 80/20 split** (`random_state=42`), so only 33 patients (20% of the 166 left after cleaning) are used for testing, and the numbers are noisy. The report describes cross-validation for both tasks, but `regressionWithEvaluation.py` uses one hold-out split.
- KNN's number of neighbours and weighting were **chosen by R² on that same test split**, so the KNN result is probably optimistic.
- **Classification evaluation:** `classification.py` uses stratified 5-fold cross-validation, but `evaluation.py` (AUC, recall, F1) uses plain 5-fold, and picks the decision threshold from the out-of-fold predictions, which flatters recall and F1.
- **Small sample after cleaning:** of the 400 patients, 166 remain after the cleaning step (30 of them with pCR), for both tasks.
- Imputation and outlier removal were applied before the data was split.
- With an R² close to zero, the RFS models are only slightly better than predicting the average.

**What I would do next:** choose KNN's settings with cross-validation on the training data only, use repeated cross-validation for every final score, drop the SVM from the comparison or report its AUC over many seeds, and compare every model with a simple baseline.

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
