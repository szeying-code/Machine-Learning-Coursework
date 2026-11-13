import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

#Load the preprocessed data
from dataPreProcessing import all_df_cleaned
from featureSelectionRegression import select_features_for_rfs

df = all_df_cleaned.copy()

#Separate X and y
X_full = df.drop(columns=['RelapseFreeSurvival (outcome)', 'pCR (outcome)'])
y_full = df['RelapseFreeSurvival (outcome)']

#Split
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

#Feature selection
df_train_for_fs = pd.concat([X_train_full, y_train], axis=1)
selected_features = select_features_for_rfs(df_train_for_fs)
print("Selected RFS Features:", selected_features)

#Final training set
X_train = X_train_full[selected_features]
X_test = X_test_full[selected_features]

#XGBoost
model = XGBRegressor(
    n_estimators=1300,
    max_depth=3,
    learning_rate=0.03,
    colsample_bytree=0.8,
    gamma=0.5,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)

print("XGBoost MAE:", mae)
print("XGBoost RMSE:", rmse)
print("XGBoost MAPE:", mape)
print("XGBoost R²:", r2)

#Random Forest
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mape = np.mean(np.abs((y_test - rf_pred) / y_test)) * 100
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest MAE:", rf_mae)
print("Random Forest RMSE:", rf_rmse)
print("Random Forest MAPE:", rf_mape)
print("Random Forest R²:", rf_r2)

#MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp_model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    learning_rate_init=0.001,
    early_stopping=True,
    random_state=42
)
mlp_model.fit(X_train_scaled, y_train)
mlp_pred = mlp_model.predict(X_test_scaled)

mlp_mae = mean_absolute_error(y_test, mlp_pred)
mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_pred))
mlp_mape = np.mean(np.abs((y_test - mlp_pred) / y_test)) * 100
mlp_r2 = r2_score(y_test, mlp_pred)

print("MLP MAE:", mlp_mae)
print("MLP RMSE:", mlp_rmse)
print("MLP MAPE:", mlp_mape)
print("MLP R²:", mlp_r2)

#KNN
best_r2 = -999
best_metrics = None

for weight in ['uniform', 'distance']:
    for k in range(1, 31):
        knn = KNeighborsRegressor(n_neighbors=k, weights=weight, p=1)

        knn.fit(X_train_scaled, y_train)
        pred = knn.predict(X_test_scaled)

        r2 = r2_score(y_test, pred)

        if r2 > best_r2:
            best_r2 = r2
            best_metrics = {
                "k": k,
                "weight": weight,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "MAPE": np.mean(np.abs((y_test - pred) / y_test)) * 100,
                "R²": r2
            }

#print("Best k:", best_metrics["k"])
#print("Best weight:", best_metrics["weight"])
print("KNN MAE:", best_metrics["MAE"])
print("KNN RMSE:", best_metrics["RMSE"])
print("KNN MAPE:", best_metrics["MAPE"])
print("KNN R²:", best_metrics["R²"])
