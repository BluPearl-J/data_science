import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from comparison import y_test, X_test, y_train, full_pipeline, X_train

# 1. Choose the model to tunetune Random Forest to fix that Overfitting!)
param_grid = [
    {
        'randomforestregressor__n_estimators': [50, 100, 200],
        'randomforestregressor__max_features': [4, 6, 8],
        'randomforestregressor__max_depth': [10, 20, None],
    }
]

# 2. Create the pipeline for the "Audition"
# We use the same 'fp' (full_pipeline) from before
rf_pipeline = make_pipeline(full_pipeline, RandomForestRegressor(random_state=42))

# 3. Start the Search (This will take a minute or two)
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

print("Tuning the model... this might take 2 minutes...")
grid_search.fit(X_train, y_train)

# 4. Get the Winner
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"Best Settings found: {best_params}")

# 5. Check the new RMSE
final_test_preds = best_model.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, final_test_preds))

print(f"New Optimized Test RMSE: ${final_rmse:,.2f}")