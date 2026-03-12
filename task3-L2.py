import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
)


# ---------- Configuration ----------

BASE_DIR = r"D:\internship data analytics"
CREDIT_PATH = os.path.join(BASE_DIR, "creditcard.csv.zip")


def load_and_inspect() -> pd.DataFrame:
    if not os.path.exists(CREDIT_PATH):
        raise FileNotFoundError(f"creditcard.csv.zip not found at: {CREDIT_PATH}")

    print("Loading credit card data...")
    df = pd.read_csv(CREDIT_PATH, compression="zip")

    print("\n=== Head ===")
    print(df.head())

    print("\n=== Shape ===")
    print(df.shape)

    print("\n=== Info ===")
    df.info()

    print("\n=== Class distribution (0 = legit, 1 = fraud) ===")
    print(df["Class"].value_counts(normalize=False))
    print("\nClass distribution (%):")
    print(df["Class"].value_counts(normalize=True) * 100)

    # Plot class distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(x="Class", data=df, palette="Set2")
    plt.title("Transaction class distribution")
    plt.xlabel("Class (0 = legitimate, 1 = fraud)")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Amount by class (log scale)
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Class", y="Amount", data=df)
    plt.yscale("log")
    plt.title("Transaction amount by class (log scale)")
    plt.tight_layout()
    plt.show()
    plt.close()

    return df


def prepare_features(df: pd.DataFrame):
    data = df.copy()

    if "Class" not in data.columns:
        raise ValueError("Expected target column 'Class' not found.")

    # Scale 'Amount' and 'Time' (other features are PCA components)
    scaler = StandardScaler()
    for col in ["Amount", "Time"]:
        if col in data.columns:
            data[col] = scaler.fit_transform(data[[col]])

    X = data.drop(columns=["Class"])
    y = data["Class"]

    print("\n=== Basic stats for scaled numeric features ===")
    print(X.describe().T[["mean", "std", "min", "max"]].head(10))

    return X, y


def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain size:", X_train.shape[0], "Test size:", X_test.shape[0])

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", n_jobs=-1
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=None, min_samples_split=4, class_weight="balanced", random_state=42
        ),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=20,
            random_state=42,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            # Fallback for models without predict_proba
            if hasattr(model, "decision_function"):
                decision = model.decision_function(X_test)
                # map decision scores to [0,1] via logistic
                y_proba = 1 / (1 + np.exp(-decision))
            else:
                y_proba = y_pred.astype(float)

        acc = (y_pred == y_test).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        auc = roc_auc_score(y_test, y_proba)

        print(f"{name} accuracy: {acc:.4f}")
        print(f"{name} precision (fraud class): {precision:.4f}")
        print(f"{name} recall (fraud class): {recall:.4f}")
        print(f"{name} F1-score (fraud class): {f1:.4f}")
        print(f"{name} ROC-AUC: {auc:.4f}")

        print(f"\n{name} classification report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "cm": cm,
            "auc": auc,
        }

    return results


def plot_confusion_matrices(results):
    for name, res in results.items():
        cm = res["cm"]
        labels = [0, 1]

        plt.figure(figsize=(5, 4))
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


def plot_roc_curves(results):
    plt.figure(figsize=(7, 6))

    for name, res in results.items():
        y_test = res["y_test"]
        y_proba = res["y_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Fraud Detection Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    print("Base directory:", BASE_DIR)
    df = load_and_inspect()
    X, y = prepare_features(df)
    results = train_and_evaluate_models(X, y)
    plot_confusion_matrices(results)
    plot_roc_curves(results)
    print("\nAll fraud-detection tasks (task3-L2) completed.")


if __name__ == "__main__":
    main()

