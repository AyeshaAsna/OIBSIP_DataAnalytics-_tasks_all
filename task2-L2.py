import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC


# ---------- Configuration ----------

BASE_DIR = r"D:\internship data analytics"
WINE_PATH = os.path.join(BASE_DIR, "WineQT.csv")


def load_and_inspect():
    if not os.path.exists(WINE_PATH):
        raise FileNotFoundError(f"Wine dataset not found at: {WINE_PATH}")

    print("Loading wine data...")
    df = pd.read_csv(WINE_PATH)

    print("\n=== Head ===")
    print(df.head())

    print("\n=== Shape ===")
    print(df.shape)

    print("\n=== Info ===")
    df.info()

    print("\n=== Missing values (%) ===")
    print(df.isna().mean() * 100)

    return df


def clean_and_prepare(df: pd.DataFrame):
    data = df.copy()

    # Drop obvious non-feature identifier if present
    if "Id" in data.columns:
        data = data.drop(columns=["Id"])

    if "quality" not in data.columns:
        raise ValueError("Expected target column 'quality' not found in WineQT.csv.")

    # No strong reason to impute here; dataset is usually clean, but handle generically
    for col in data.columns:
        if data[col].dtype.kind in "biufc":
            data[col] = data[col].fillna(data[col].median())
        else:
            mode_val = data[col].mode(dropna=True)
            if not mode_val.empty:
                data[col] = data[col].fillna(mode_val.iloc[0])

    # Basic distribution of target
    print("\n=== Quality value counts ===")
    print(data["quality"].value_counts().sort_index())

    # Correlation with quality
    corr = data.corr(numeric_only=True)["quality"].sort_values(ascending=False)
    print("\n=== Correlation with quality ===")
    print(corr)

    # Plot quality distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x="quality", data=data, palette="viridis")
    plt.title("Wine quality distribution")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(numeric_only=True), cmap="coolwarm", annot=False)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.show()
    plt.close()

    X = data.drop(columns=["quality"])
    y = data["quality"]

    return X, y


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTrain size:", X_train.shape[0], "Test size:", X_test.shape[0])

    models = {}

    # Random Forest (tree-based, no scaling needed)
    rf = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, max_depth=None
    )
    models["RandomForest"] = rf

    # SGD classifier (needs scaling)
    sgd = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)),
        ]
    )
    models["SGD"] = sgd

    # SVC (needs scaling)
    svc = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
        ]
    )
    models["SVC"] = svc

    results = {}

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"{name} accuracy: {acc:.4f}")

        print(f"\n{name} classification report:")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        results[name] = {"model": model, "y_test": y_test, "y_pred": y_pred, "cm": cm}

    return results


def plot_confusion_matrices(results):
    for name, res in results.items():
        cm = res["cm"]
        labels = np.unique(res["y_test"])

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.show()
        plt.close()


def main():
    print("Base directory:", BASE_DIR)
    df = load_and_inspect()
    X, y = clean_and_prepare(df)
    results = train_models(X, y)
    plot_confusion_matrices(results)
    print("\nAll wine-quality tasks (task2-L2) completed.")


if __name__ == "__main__":
    main()

