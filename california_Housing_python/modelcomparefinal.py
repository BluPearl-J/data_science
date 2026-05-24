import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# --- 1. DATA LOADING & SPLITTING ---
# (Update the path to your CSV file as needed)
df = pd.read_csv("housing.csv")

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. THE FULL PIPELINE (Hard-written) ---
num_attribs = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_attribs = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

# --- 3. MODEL DICTIONARY ---
models = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

# --- 4. THE CLEAN COMPARISON LOOP ---
print(f"\n{'Model':<20} | {'Train RMSE':<15} | {'CV RMSE (Avg)':<15}")
print("-" * 55)

for name, model in models.items():
    # Construct the final pipeline for this specific model
    current_pipeline = Pipeline([
        ("preprocessing", full_pipeline),
        ("regressor", model)
    ])

    # 5-Fold Cross Validation (The 'Clean' Way)
    scores = cross_val_score(current_pipeline, X_train, y_train,
                             scoring="neg_mean_squared_error", cv=5)
    cv_rmse = np.sqrt(-scores.mean())

    # Training RMSE (To check for Overfitting)
    current_pipeline.fit(X_train, y_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, current_pipeline.predict(X_train)))

    print(f"{name:<20} | ${train_rmse:>12,.2f} | ${cv_rmse:>12,.2f}")