from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

from california_Housing_python.comparison import full_pipeline, X_train, y_train

models = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

print(f"{'Model':<20} | {'Train RMSE':<15} | {'CV RMSE (Avg)':<15}")
print("-" * 55)

for name, model in models.items():
    current_pipeline = make_pipeline(full_pipeline, model)

    # 1. Calculate CV RMSE (This uses ONLY training data)
    # We use 5-fold cross-validation
    scores = cross_val_score(current_pipeline, X_train, y_train,
                             scoring="neg_mean_squared_error", cv=5)
    cv_rmse = np.sqrt(-scores.mean())

    # 2. Calculate Train RMSE
    current_pipeline.fit(X_train, y_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, current_pipeline.predict(X_train)))

    print(f"{name:<20} | ${train_rmse:>12,.2f} | ${cv_rmse:>12,.2f}")