import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


def main():
    # -----------------------------
    # Load dataset
    # -----------------------------
    # We expect the CSV file (e.g. ifood_df.csv) to be inside a 'datasets' subfolder
    # next to this script: D:\internship data analytics\datasets\ifood_df.csv
    # If your file is elsewhere, update data_path accordingly.

    data_path = os.path.join("datasets", "ifood_df.csv")

    print(f"Loading data from: {os.path.abspath(data_path)}")
    df = pd.read_csv(data_path)

    print("Shape:", df.shape)
    print(df.head())

    # -----------------------------
    # Basic structure and stats
    # -----------------------------
    print("\n--- Data info ---")
    df.info()

    print("\n--- Descriptive statistics ---")
    print(df.describe().T)

    # -----------------------------
    # Missing values
    # -----------------------------
    print("\n--- Missing values per column (non-zero only) ---")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing[missing > 0])

    # -----------------------------
    # Simple missing value handling
    # -----------------------------
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # -----------------------------
    # Feature engineering
    # -----------------------------
    spend_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]

    purchase_cols = [
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
    ]

    # Keep only columns that actually exist in this dataset
    spend_cols = [c for c in spend_cols if c in df.columns]
    purchase_cols = [c for c in purchase_cols if c in df.columns]

    df["total_spent"] = df[spend_cols].sum(axis=1) if spend_cols else 0
    df["total_purchases"] = df[purchase_cols].sum(axis=1) if purchase_cols else 0

    # Avoid division by zero for customers with no purchases
    df["avg_purchase_value"] = df["total_spent"] / df["total_purchases"].replace(
        0, np.nan
    )

    # Example demographic features (if present)
    if {"Kidhome", "Teenhome"}.issubset(df.columns):
        df["family_size"] = 1 + df["Kidhome"] + df["Teenhome"]
    else:
        df["family_size"] = np.nan

    if "Income" in df.columns:
        # Winsorize extreme incomes by clipping at 1st and 99th percentiles
        low, high = df["Income"].quantile([0.01, 0.99])
        df["Income_clipped"] = df["Income"].clip(low, high)
    else:
        df["Income_clipped"] = np.nan

    # -----------------------------
    # Select features for clustering
    # -----------------------------
    candidate_features = [
        "total_spent",
        "total_purchases",
        "avg_purchase_value",
        "Income_clipped",  # clipped income
        "Recency",  # days since last purchase, if present
        "NumWebVisitsMonth",  # online engagement, if present
    ]

    features = [f for f in candidate_features if f in df.columns]
    print("\nUsing features for clustering:", features)

    X = df[features].copy()

    # Handle any remaining missing values by median imputation
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # Elbow method
    # -----------------------------
    inertias = []
    k_values = range(2, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    plt.figure()
    plt.plot(k_values, inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster SSE)")
    plt.title("Elbow Method for K-Means")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Silhouette scores
    # -----------------------------
    sil_scores = {}

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        sil_scores[k] = score

    print("\nSilhouette scores by k:")
    for k, v in sil_scores.items():
        print(f"k={k}: {v:.4f}")

    # -----------------------------
    # Final K-Means model
    # -----------------------------
    final_k = 4  # you can change this based on elbow + silhouette
    print(f"\nFitting final K-Means with k={final_k}")

    kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_scaled)

    df["cluster"] = cluster_labels

    print("\nCluster counts:")
    print(df["cluster"].value_counts().sort_index())

    # -----------------------------
    # Cluster-level descriptive statistics
    # -----------------------------
    cluster_profile_mean = df.groupby("cluster")[features + ["total_spent", "total_purchases"]].mean()
    cluster_profile_count = df["cluster"].value_counts().sort_index()

    print("\nCluster profile - customer counts:")
    print(cluster_profile_count.to_frame(name="num_customers"))

    print("\nCluster profile - mean values:")
    print(cluster_profile_mean)

    # -----------------------------
    # 2D PCA projection
    # -----------------------------
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    df["pc1"] = components[:, 0]
    df["pc2"] = components[:, 1]

    plt.figure()
    sns.scatterplot(data=df, x="pc1", y="pc2", hue="cluster", palette="tab10", alpha=0.7)
    plt.title("Customer Segments (PCA 2D Projection)")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Distribution of key metrics by cluster
    # -----------------------------
    metrics_to_plot = [
        c
        for c in ["total_spent", "total_purchases", "avg_purchase_value", "Income_clipped"]
        if c in df.columns
    ]

    for col in metrics_to_plot:
        plt.figure()
        sns.boxplot(data=df, x="cluster", y=col)
        plt.title(f"{col} by Cluster")
        plt.tight_layout()
        plt.show()

    print("\nScript finished. Check the printed stats and plots to interpret segments.")


if __name__ == "__main__":
    main()

