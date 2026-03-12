import os
import re
import string
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ---------- Configuration ----------

BASE_DIR = r"D:\internship data analytics"
TWITTER_PATH = os.path.join(BASE_DIR, "Twitter_Data.csv.zip")
NOTEBOOK_PATH = os.path.join(BASE_DIR, "play-store-sentiment-analysis-of-user-reviews.ipynb")

MODEL_PATH = os.path.join(BASE_DIR, "task4_sentiment_model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "task4_tfidf_vectorizer.joblib")
REPORT_PATH = os.path.join(BASE_DIR, "task4_classification_report.txt")

PLOT_CLASS_DIST = os.path.join(BASE_DIR, "task4_class_distribution.png")
PLOT_CONF_MAT = os.path.join(BASE_DIR, "task4_confusion_matrix.png")


# ---------- Text cleaning ----------

URL_PATTERN = re.compile(r"http\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(" ", text)

    text = text.translate(str.maketrans("", "", string.punctuation))
    text = MULTI_SPACE_PATTERN.sub(" ", text)
    return text.strip()


# ---------- Data loading & preprocessing ----------

def load_twitter_data():
    if not os.path.exists(TWITTER_PATH):
        raise FileNotFoundError(f"Twitter dataset not found at: {TWITTER_PATH}")

    print("Loading Twitter data...")
    df = pd.read_csv(TWITTER_PATH, compression="zip")
    print("Initial shape:", df.shape)
    print("Columns:", list(df.columns))

    # Try to identify text & label columns by common names
    text_candidates = ["clean_text", "text", "tweet", "message", "content"]
    label_candidates = ["label", "sentiment", "category", "polarity"]

    text_col = next((c for c in text_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not infer text/label columns. "
            f"Candidates text={text_candidates}, label={label_candidates}. "
            f"Found columns={list(df.columns)}"
        )

    print(f"Using text column: '{text_col}', label column: '{label_col}'")

    df = df[[text_col, label_col]].rename(
        columns={text_col: "text", label_col: "label"}
    )

    # Drop missing
    df = df.dropna(subset=["text", "label"]).copy()
    print("Shape after dropping NA:", df.shape)

    # Optionally normalize labels (example: map numeric to strings if needed)
    # Here we leave labels as-is to stay aligned with the dataset.

    # Clean text
    print("Cleaning text...")
    df["text_clean"] = df["text"].astype(str).apply(clean_text)

    # Drop rows with empty cleaned text
    before = df.shape[0]
    df = df[df["text_clean"].str.len() > 0].copy()
    after = df.shape[0]
    print(f"Dropped {before - after} rows with empty cleaned text.")
    print("Final usable rows:", df.shape[0])

    return df


# ---------- Model training & evaluation ----------

def train_and_evaluate(df: pd.DataFrame):
    X = df["text_clean"].values
    y = df["label"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Train size:", len(X_train), "Test size:", len(X_test))

    # TF-IDF + Logistic Regression pipeline
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 2),
                    stop_words="english"
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=-1)),
        ]
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)
    print("Training completed.")

    # Predictions & metrics
    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    report = classification_report(y_test, y_pred)
    print("Classification report:\n", report)

    # Extract vectorizer & classifier separately for saving
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    return {
        "pipeline": pipeline,
        "vectorizer": vectorizer,
        "clf": clf,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": acc,
        "report": report,
    }


# ---------- Visualization helpers ----------

def plot_class_distribution(labels, path):
    values, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(8, 5))
    plt.bar(values, counts, color="#4C72B0")
    plt.xlabel("Sentiment class")
    plt.ylabel("Count")
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.show()
    plt.close()
    print(f"Saved class distribution plot to: {path}")


def plot_confusion_matrix(y_true, y_pred, path):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha="right")
    plt.yticks(ticks=range(len(labels)), labels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix")

    # Write counts on cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.show()
    plt.close()
    print(f"Saved confusion matrix plot to: {path}")


# ---------- Main pipeline ----------

def main():
    print("Base directory:", BASE_DIR)
    df = load_twitter_data()

    # Visualize raw label distribution
    plot_class_distribution(df["label"].values, PLOT_CLASS_DIST)

    results = train_and_evaluate(df)

    # Confusion matrix plot
    plot_confusion_matrix(
        results["y_test"], results["y_pred"], PLOT_CONF_MAT
    )

    # Save model artifacts
    print("Saving model and vectorizer...")
    joblib.dump(results["clf"], MODEL_PATH)
    joblib.dump(results["vectorizer"], VECTORIZER_PATH)
    print(f"Saved classifier to: {MODEL_PATH}")
    print(f"Saved TF-IDF vectorizer to: {VECTORIZER_PATH}")

    # Save classification report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {results['accuracy']:.4f}\n\n")
        f.write(results["report"])
    print(f"Saved classification report to: {REPORT_PATH}")

    print("\nAll sentiment analysis tasks (task4) completed.")


if __name__ == "__main__":
    main()