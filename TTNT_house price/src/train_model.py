import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "data/train.csv"
MODEL_PATH = "models/rf_pipeline.pkl"
OUTPUT_CORR_PATH = "outputs/correlation_plot.png"

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Id"], errors="ignore")

top_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
                'TotalBsmtSF', '1stFlrSF', 'ExterQual', 'FullBath',
                'BsmtQual', 'TotRmsAbvGrd']

y = df["SalePrice"]
X_raw = df[top_features].copy()

# phan loai
cat_cols = ['ExterQual', 'BsmtQual']
num_cols = list(set(top_features) - set(cat_cols))

# preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.1, random_state=42)

param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [None, 10, 20],
    "regressor__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nĐánh giá mô hình:")
print("-> Best Parameters:", grid_search.best_params_)
print(f"-> MSE: {mse:.2f}")
print(f"-> R^2: {r2:.2f}")

joblib.dump(best_model, MODEL_PATH)
print(f"\n Mô hình đã được lưu tại: {MODEL_PATH}")