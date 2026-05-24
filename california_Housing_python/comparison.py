from models_utils import create_preprocessing_pipeline

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

housing = pd.read_csv("housing.csv")
X = housing.drop("median_house_value", axis=1)
y = housing["median_house_value"].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



full_pipeline = create_preprocessing_pipeline(X_train)

# the models
models = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(kernel="rbf"),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# print(f"{'Model':<20} | {'Train RMSE':<15} | {'Test RMSE':<15}")
print("-" * 55)

# 4.  loop to find the winner
for name, model in models.items():
    # Glue pipeline to model
    current_pipeline = make_pipeline(full_pipeline, model)
    current_pipeline.fit(X_train, y_train)

    # Evaluate
    train_rmse = np.sqrt(mean_squared_error(y_train, current_pipeline.predict(X_train)))
    test_rmse = np.sqrt(mean_squared_error(y_test, current_pipeline.predict(X_test)))

   # print(f"{name:<20} | ${train_rmse:>12,.2f} | ${test_rmse:>12,.2f}")
