import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def explore_data(df):

    print("--- TASK 1: DATA EXPLORATION ---")
    print("\n1. First 5 Rows:")
    print(df.head())

    print("\n2. Data Info (Check for Nulls):")
    print(df.info())

    print("\n3. Category Counts (Ocean Proximity):")
    print(df["ocean_proximity"].value_counts())

    print("\n4. Summary Statistics:")
    print(df.describe())
    print("-" * 30 + "\n")

def create_preprocessing_pipeline(df):

    num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
                   "total_bedrooms", "population", "households", "median_income"]
    cat_attribs = ["ocean_proximity"]

    num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    return full_pipeline

def build_model(full_pipeline):

    model = make_pipeline(full_pipeline, RandomForestRegressor())
    return model

def evaluate_reliability(model, X_train, y_train, X_test, y_test):

    train_preds = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))

    test_preds = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    print(f"Training RMSE: ${train_rmse:,.2f}")
    print(f"Final Test RMSE: ${test_rmse:,.2f}")

    if test_rmse > train_rmse * 1.5:
        print("Warning: Model might be Overfitting!")
    else:
        print("Model is Reliable for the Real World.")




if __name__ == "__main__":

    housing = pd.read_csv("housing.csv")

    explore_data(housing)

    X = housing.drop("median_house_value", axis=1)
    y = housing["median_house_value"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = create_preprocessing_pipeline(X_train)
    forest_reg = build_model(pipeline)
    forest_reg.fit(X_train, y_train)


    evaluate_reliability(forest_reg, X_train, y_train, X_test, y_test)


    joblib.dump(forest_reg, "my_model.pkl")
    print("Model saved as my_model.pkl")