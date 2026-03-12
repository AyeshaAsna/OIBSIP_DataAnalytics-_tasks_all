import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------- Configuration ----------

BASE_DIR = r"D:\internship data analytics"
APPS_PATH = os.path.join(BASE_DIR, "apps.csv")
REVIEWS_PATH = os.path.join(BASE_DIR, "user_reviews.csv.zip")


# ---------- Data loading & preparation ----------

def load_data():
    if not os.path.exists(APPS_PATH):
        raise FileNotFoundError(f"apps.csv not found at: {APPS_PATH}")
    if not os.path.exists(REVIEWS_PATH):
        raise FileNotFoundError(f"user_reviews.csv.zip not found at: {REVIEWS_PATH}")

    print("Loading apps data...")
    apps = pd.read_csv(APPS_PATH)
    print("Apps shape:", apps.shape)
    print(apps.head())

    print("\nLoading user reviews data...")
    reviews = pd.read_csv(REVIEWS_PATH, compression="zip")
    print("Reviews shape:", reviews.shape)
    print(reviews.head())

    return apps, reviews


def clean_apps(apps: pd.DataFrame) -> pd.DataFrame:
    df = apps.copy()

    print("\n=== Raw dtypes (apps) ===")
    print(df.dtypes)

    # Drop duplicate app entries based on App + Category
    before = df.shape[0]
    df = df.drop_duplicates(subset=["App", "Category"])
    print(f"Removed {before - df.shape[0]} duplicate app rows.")

    # Clean Rating
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    # Clean Reviews
    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

    # Clean Size: convert to MB, handle "Varies with device"
    def parse_size(val):
        if isinstance(val, str):
            val = val.strip()
            if val == "Varies with device" or val == "":
                return np.nan
            if val.endswith("k") or val.endswith("K"):
                try:
                    return float(val[:-1]) / 1024.0
                except ValueError:
                    return np.nan
            if val.endswith("M"):
                try:
                    return float(val[:-1])
                except ValueError:
                    return np.nan
            try:
                return float(val)
            except ValueError:
                return np.nan
        return np.nan

    df["Size_MB"] = df["Size"].apply(parse_size)

    # Clean Installs: remove '+' and ',' then to int
    df["Installs_clean"] = (
        df["Installs"]
        .astype(str)
        .str.replace("[+,]", "", regex=True)
        .replace("Free", np.nan)
    )
    df["Installs_clean"] = pd.to_numeric(df["Installs_clean"], errors="coerce")

    # Clean Price: remove '$', convert to float
    df["Price_clean"] = (
        df["Price"]
        .astype(str)
        .str.replace("$", "", regex=False)
    )
    df["Price_clean"] = pd.to_numeric(df["Price_clean"], errors="coerce")

    # Parse Last Updated
    df["Last_Updated_dt"] = pd.to_datetime(df["Last Updated"], errors="coerce")

    print("\n=== Cleaned dtypes (apps) ===")
    print(df[["Rating", "Reviews", "Size_MB", "Installs_clean", "Price_clean", "Last_Updated_dt"]].dtypes)

    print("\n=== Missing values (%) after cleaning (apps) ===")
    print(df[["Rating", "Reviews", "Size_MB", "Installs_clean", "Price_clean"]].isna().mean() * 100)

    return df


def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    df = reviews.copy()

    print("\n=== Raw dtypes (reviews) ===")
    print(df.dtypes)

    # Standard expected columns: App, Translated_Review, Sentiment, Sentiment_Polarity, Sentiment_Subjectivity
    if "Sentiment_Polarity" in df.columns:
        df["Sentiment_Polarity"] = pd.to_numeric(df["Sentiment_Polarity"], errors="coerce")
    if "Sentiment_Subjectivity" in df.columns:
        df["Sentiment_Subjectivity"] = pd.to_numeric(df["Sentiment_Subjectivity"], errors="coerce")

    print("\n=== Missing values (%) after cleaning (reviews) ===")
    print(df.isna().mean() * 100)

    return df


# ---------- Exploratory analysis & visualization ----------

def explore_categories(apps_clean: pd.DataFrame):
    print("\n=== App count by Category (top 10) ===")
    cat_counts = apps_clean["Category"].value_counts().head(10)
    print(cat_counts)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=cat_counts.index, y=cat_counts.values, palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of apps")
    plt.title("Top 10 Categories by App Count")
    plt.tight_layout()
    plt.show()
    plt.close()


def analyze_metrics(apps_clean: pd.DataFrame):
    # Ratings distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(apps_clean["Rating"].dropna(), bins=30, kde=True)
    plt.xlabel("Rating")
    plt.title("Distribution of App Ratings")
    plt.tight_layout()
    plt.show()
    plt.close()

    # App size vs Rating
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        x="Size_MB",
        y="Rating",
        data=apps_clean,
        alpha=0.3,
    )
    plt.xlabel("App Size (MB)")
    plt.ylabel("Rating")
    plt.title("App Size vs Rating")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Installs vs Rating (log scale)
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        x="Installs_clean",
        y="Rating",
        data=apps_clean,
        alpha=0.3,
    )
    plt.xscale("log")
    plt.xlabel("Installs (log scale)")
    plt.ylabel("Rating")
    plt.title("Installs vs Rating")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Free vs Paid comparison
    plt.figure(figsize=(6, 4))
    sns.boxplot(
        x="Type",
        y="Rating",
        data=apps_clean,
    )
    plt.title("Free vs Paid: Rating distribution")
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.boxplot(
        x="Type",
        y="Installs_clean",
        data=apps_clean,
    )
    plt.yscale("log")
    plt.title("Free vs Paid: Installs distribution (log)")
    plt.tight_layout()
    plt.show()
    plt.close()


def analyze_sentiment(apps_clean: pd.DataFrame, reviews_clean: pd.DataFrame):
    if "Sentiment" not in reviews_clean.columns:
        print("\nSentiment column not found in reviews dataset; skipping sentiment analysis.")
        return

    print("\n=== Sentiment counts ===")
    sent_counts = reviews_clean["Sentiment"].value_counts()
    print(sent_counts)

    plt.figure(figsize=(5, 4))
    sns.countplot(x="Sentiment", data=reviews_clean, order=sent_counts.index, palette="coolwarm")
    plt.title("Review Sentiment Distribution")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Average sentiment polarity by sentiment label
    if "Sentiment_Polarity" in reviews_clean.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(
            x="Sentiment",
            y="Sentiment_Polarity",
            data=reviews_clean,
            order=sent_counts.index,
        )
        plt.title("Sentiment Polarity by Sentiment Label")
        plt.tight_layout()
        plt.show()
        plt.close()

    # Merge reviews with apps to see sentiment by Category
    merged = pd.merge(
        reviews_clean,
        apps_clean[["App", "Category"]],
        on="App",
        how="left",
    )

    if "Sentiment_Polarity" in merged.columns:
        cat_sent = (
            merged.groupby("Category")["Sentiment_Polarity"]
            .mean()
            .dropna()
            .sort_values(ascending=False)
            .head(10)
        )
        print("\n=== Top 10 categories by average sentiment polarity ===")
        print(cat_sent)

        plt.figure(figsize=(10, 5))
        sns.barplot(x=cat_sent.index, y=cat_sent.values, palette="magma")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Average Sentiment Polarity")
        plt.title("Top 10 Categories by Average Review Sentiment")
        plt.tight_layout()
        plt.show()
        plt.close()


# ---------- Main ----------

def main():
    print("Base directory:", BASE_DIR)
    apps, reviews = load_data()
    apps_clean = clean_apps(apps)
    reviews_clean = clean_reviews(reviews)

    # Category exploration
    explore_categories(apps_clean)

    # Metrics analysis
    analyze_metrics(apps_clean)

    # Sentiment analysis
    analyze_sentiment(apps_clean, reviews_clean)

    print("\nAll Google Play Store analysis tasks (task4-L2) completed.")


if __name__ == "__main__":
    main()

