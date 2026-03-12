import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# ---------- Configuration ----------

BASE_DIR = r"D:\internship data analytics"

# Try common locations for the Housing dataset
HOUSING_CANDIDATES = [
    os.path.join(BASE_DIR, "Housing.csv"),
    os.path.join(BASE_DIR, "input", "Housing.csv"),
]

PLOT_ACTUAL_VS_PRED = os.path.join(BASE_DIR, "task1-L2_actual_vs_predicted.png")
PLOT_RESIDUALS = os.path.join(BASE_DIR, "task1-L2_residuals_hist.png")


# ---------- Utility functions ----------

def find_housing_path() -> str:
    for path in HOUSING_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find 'Housing.csv'. "
        "Place it either in 'D:\\internship data analytics' or in "
        "'D:\\internship data analytics\\input' and run again."
    )


def load_and_explore_data() -> pd.DataFrame:
    path = find_housing_path()
    print(f"Loading housing data from: {path}")
    housing = pd.read_csv(path)

    # Basic exploration
    print("\n=== Head of data ===")
    print(housing.head())

    print("\n=== Shape ===")
    print(housing.shape)

    print("\n=== Info ===")
    housing.info()

    print("\n=== Describe (numeric) ===")
    print(housing.describe())

    print("\n=== Missing values per column ===")
    print(housing.isna().mean() * 100)

    return housing


def clean_and_engineer_features(housing: pd.DataFrame):
    df = housing.copy()

    # In the original notebook, there are no missing values, but we handle it generically.
    # For numeric columns: fill with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # For categorical columns: fill with mode
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])

    # Target variable
    if "price" not in df.columns:
        raise ValueError("Expected target column 'price' not found in dataset.")

    y = df["price"].values

    # Feature engineering: one-hot encode categorical variables
    X = df.drop(columns=["price"])
    X = pd.get_dummies(X, drop_first=True)

    print("\n=== Feature matrix shape after encoding ===")
    print(X.shape)

    # Simple feature selection: show correlations of numeric features with price
    print("\n=== Correlation of numeric features with price ===")
    corr = df.select_dtypes(include=[np.number]).corr()["price"].sort_values(ascending=False)
    print(corr)

    return X, y, X.columns


def train_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTrain size:", X_train.shape[0], "Test size:", X_test.shape[0])

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\nModel training completed.")

    y_pred = model.predict(X_test)

    # Evaluation metrics
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)

    print("\n=== Evaluation Metrics (Test Set) ===")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.4f}")

    return model, X_train, X_test, y_train, y_test, y_pred


def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    max_val = max(max(y_test), max(y_pred))
    min_val = min(min(y_test), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal fit")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_ACTUAL_VS_PRED, dpi=120)
    plt.show()
    plt.close()
    print(f"Saved actual vs predicted plot to: {PLOT_ACTUAL_VS_PRED}")


def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.title("Residuals Distribution")
    plt.tight_layout()
    plt.savefig(PLOT_RESIDUALS, dpi=120)
    plt.show()
    plt.close()
    print(f"Saved residuals histogram to: {PLOT_RESIDUALS}")


def show_top_coefficients(model: LinearRegression, feature_names):
    coefs = pd.Series(model.coef_, index=feature_names)
    coefs_sorted = coefs.sort_values(key=lambda s: s.abs(), ascending=False)

    print("\n=== Top 10 features by absolute coefficient magnitude ===")
    print(coefs_sorted.head(10))


def main():
    print("Base directory:", BASE_DIR)

    # Data loading and exploration
    housing = load_and_explore_data()

    # Cleaning and feature engineering
    X, y, feature_names = clean_and_engineer_features(housing)

    # Model training and evaluation
    model, X_train, X_test, y_train, y_test, y_pred = train_linear_regression(X, y)

    # Interpret model
    show_top_coefficients(model, feature_names)

    # Visualizations (displayed at runtime and saved as PNGs)
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residuals(y_test, y_pred)

    print("\nAll linear regression tasks (task1-L2) completed.")


if __name__ == "__main__":
    main()