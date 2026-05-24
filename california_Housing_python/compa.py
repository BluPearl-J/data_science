import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# IMPORT NOTE: If this line causes the old table to pop up,
# go to comparison.py and delete/comment out the print statements there.
from comparison import full_pipeline, X_train, y_train

# 1. Define the models dictionary
models = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

# 2. Setup the table header - ONLY CV RMSE
print(f"\n{'Model':<20} | {'Train RMSE':<15} | {'CV RMSE (Avg)':<15}")
print("-" * 55)

# 3. The Loop (ONLY using Training data and CV to avoid Data Leakage)
for name, model in models.items():
    # Wrap the model in your preprocessing pipeline
    current_pipeline = make_pipeline(full_pipeline, model)

    # Calculate CV RMSE
    # This simulates unseen data by splitting X_train into 5 pieces
    scores = cross_val_score(current_pipeline, X_train, y_train,
                             scoring="neg_mean_squared_error", cv=5)
    cv_rmse = np.sqrt(-scores.mean())

    # Calculate Train RMSE
    # This shows if the model is "memorizing" (Overfitting)
    current_pipeline.fit(X_train, y_train)
    train_preds = current_pipeline.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))

    # Print results
    print(f"{name:<20} | ${train_rmse:>12,.2f} | ${cv_rmse:>12,.2f}")