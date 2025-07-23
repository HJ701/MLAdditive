import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

# === Paths ===
INPUT_PATH = '/Users/hj/MLAdditive/data/preprocessed_smote.csv'
BASE_RESULTS_DIR = '/Users/hj/MLAdditive/results/regression'
# Added path for saving the best model checkpoint
MODEL_CHECKPOINT_DIR = '/Users/hj/MLAdditive/models'

# === Custom Scorers for CV ===
def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_score = make_scorer(rmse_scorer, greater_is_better=False)
mae_score = make_scorer(mean_absolute_error, greater_is_better=False)

def load_data(path):
    return pd.read_csv(path)

def evaluate_model(name, model, X_train, X_test, y_train, y_test, X_full, y_full):
    print(f"\n🚄 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Eval metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    out_dir = os.path.join(BASE_RESULTS_DIR, name.replace(" ", "_").lower())
    os.makedirs(out_dir, exist_ok=True)

    # --- Modification Start ---
    # Conditionally save only the Random Forest model to the specified checkpoint directory
    if name == "Random Forest":
        print(f"💾 Saving {name} model checkpoint...")
        os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_CHECKPOINT_DIR, 'best_rf_model.pkl')
        joblib.dump(model, model_path)
        print(f"✅ Model saved to {model_path}")
    # --- Modification End ---

    # Save test split metrics for all models
    test_metrics = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'RMSE', 'R2'],
        'Value': [mae, mse, rmse, r2]
    })
    test_metrics.to_csv(os.path.join(out_dir, 'evaluation_metrics.csv'), index=False)

    print(f"📊 {name} Test Results:\n{test_metrics.to_string(index=False)}")

    # Plot: Actual vs Predicted
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Lifetime_years')
    plt.ylabel('Predicted Lifetime_years')
    plt.title(f'{name}: Actual vs Predicted')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'actual_vs_predicted.png'))

    # Plot: Residual Distribution
    residuals = y_test - y_pred
    plt.figure()
    plt.hist(residuals, bins=20, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'{name}: Residuals Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'residuals_distribution.png'))

    # === Cross-validation ===
    print(f"🔄 Running 5-Fold Cross-Validation for {name}...")

    cv_r2 = cross_val_score(model, X_full, y_full, cv=5, scoring='r2')
    cv_mae = -cross_val_score(model, X_full, y_full, cv=5, scoring=mae_score)
    cv_rmse = -cross_val_score(model, X_full, y_full, cv=5, scoring=rmse_score)

    cv_df = pd.DataFrame({
        'Metric': ['CV_R2', 'CV_MAE', 'CV_RMSE'],
        'Mean': [cv_r2.mean(), cv_mae.mean(), cv_rmse.mean()],
        'Std': [cv_r2.std(), cv_mae.std(), cv_rmse.std()]
    })

    cv_df.to_csv(os.path.join(out_dir, 'crossval_metrics.csv'), index=False)

    print(f"📋 {name} Cross-Validation (5-Fold) Summary:\n{cv_df.to_string(index=False)}")

def train_all_models(df):
    X = df.drop('Lifetime_years', axis=1)
    y = df['Lifetime_years']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Ridge Regression": GridSearchCV(Ridge(), param_grid={'alpha': [0.1, 1.0, 10.0, 100.0]}, cv=5, scoring='neg_mean_squared_error'),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    }

    for name, model in models.items():
        evaluate_model(name, model, X_train, X_test, y_train, y_test, X, y)

if __name__ == "__main__":
    df = load_data(INPUT_PATH)
    train_all_models(df)