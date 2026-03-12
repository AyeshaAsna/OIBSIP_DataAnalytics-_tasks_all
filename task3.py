import os
import numpy as np
import pandas as pd


# ---------- Configuration ----------

BASE_DIR = r"D:\internship data analytics"
NYC_PATH = os.path.join(BASE_DIR, "AB_NYC_2019.csv.zip")
CA_PATH = os.path.join(BASE_DIR, "CA_category_id.json")

NYC_OUTPUT = os.path.join(BASE_DIR, "AB_NYC_2019_cleaned.csv")
CA_OUTPUT = os.path.join(BASE_DIR, "CA_category_id_cleaned.json")


# ---------- Utility functions ----------

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def remove_duplicates(df: pd.DataFrame, subset=None, name="") -> pd.DataFrame:
    before = df.shape[0]
    df = df.drop_duplicates(subset=subset).copy()
    after = df.shape[0]
    print(f"[{name}] Duplicates removed: {before - after}")
    return df


def handle_missing_values(df: pd.DataFrame, name="") -> pd.DataFrame:
    df = df.copy()
    print(f"[{name}] Missing values per column BEFORE:")
    print(df.isna().sum())

    # Numeric: fill with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Categorical: fill with mode (if it exists)
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if df[col].isna().any():
            modes = df[col].mode(dropna=True)
            if not modes.empty:
                df[col] = df[col].fillna(modes.iloc[0])

    print(f"[{name}] Missing values per column AFTER:")
    print(df.isna().sum())
    return df


def remove_outliers_iqr(df: pd.DataFrame, cols, name="") -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before = df.shape[0]
        df = df[(df[col] >= lower) & (df[col] <= upper)].copy()
        after = df.shape[0]
        print(f"[{name}] Outliers removed in '{col}': {before - after}")
    return df


# ---------- Cleaning AB_NYC_2019 ----------

def clean_ab_nyc():
    print("=== Cleaning AB_NYC_2019 ===")
    df = pd.read_csv(NYC_PATH)
    print("[NYC] Initial shape:", df.shape)

    # Standardize column names
    df = standardize_column_names(df)

    # Basic integrity checks and corrections
    for col in [
        "price",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]:
        if col in df.columns:
            df = df[df[col].ge(0) | df[col].isna()]

    # Parse date columns (if present)
    if "last_review" in df.columns:
        df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    # Handle missing data
    df = handle_missing_values(df, name="NYC")

    # Remove duplicate rows
    subset_cols = [c for c in ["id", "name", "host_id"] if c in df.columns]
    df = remove_duplicates(df, subset=subset_cols or None, name="NYC")

    # Outlier detection & removal using IQR on key numeric fields
    outlier_cols = [c for c in ["price", "minimum_nights", "number_of_reviews"] if c in df.columns]
    df = remove_outliers_iqr(df, cols=outlier_cols, name="NYC")

    print("[NYC] Final shape:", df.shape)

    # Save cleaned dataset
    df.to_csv(NYC_OUTPUT, index=False)
    print(f"[NYC] Cleaned data saved to: {NYC_OUTPUT}")


# ---------- Cleaning CA_category_id ----------

def clean_ca_category():
    print("=== Cleaning CA_category_id ===")
    df = pd.read_json(CA_PATH)
    print("[CA] Initial shape:", df.shape)

    # If the JSON has a top-level key like 'items', unwrap it
    if "items" in df.columns and df.shape[1] == 1:
        items = df["items"].explode().reset_index(drop=True)
        df = pd.json_normalize(items)
        print("[CA] Normalized nested 'items' JSON.")
        print("[CA] Shape after normalization:", df.shape)

    df = standardize_column_names(df)

    # Drop columns that contain dict-like objects (unhashable) to simplify cleaning
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, dict)).any():
            print(f"[CA] Dropping nested dict column: {col}")
            df = df.drop(columns=[col])

    # Try to ensure an id column exists and is integer-like
    for candidate in ["id", "category_id"]:
        if candidate in df.columns:
            df[candidate] = pd.to_numeric(df[candidate], errors="coerce")

    # Remove rows with missing critical identifiers
    id_col = "id" if "id" in df.columns else ("category_id" if "category_id" in df.columns else None)
    if id_col is not None:
        before = df.shape[0]
        df = df[df[id_col].notna()].copy()
        after = df.shape[0]
        print(f"[CA] Rows with missing '{id_col}' removed: {before - after}")

    # Remove duplicates on id (if present)
    df = remove_duplicates(df, subset=[id_col] if id_col else None, name="CA")

    # Handle missing values generically
    df = handle_missing_values(df, name="CA")

    print("[CA] Final shape:", df.shape)

    # Save cleaned data (JSON records)
    df.to_json(CA_OUTPUT, orient="records", force_ascii=False, indent=2)
    print(f"[CA] Cleaned data saved to: {CA_OUTPUT}")

    # Try to ensure an id column exists and is integer-like
    for candidate in ["id", "category_id"]:
        if candidate in df.columns:
            df[candidate] = pd.to_numeric(df[candidate], errors="coerce")

    # Remove rows with missing critical identifiers
    id_col = "id" if "id" in df.columns else ("category_id" if "category_id" in df.columns else None)
    if id_col is not None:
        before = df.shape[0]
        df = df[df[id_col].notna()].copy()
        after = df.shape[0]
        print(f"[CA] Rows with missing '{id_col}' removed: {before - after}")

    # Remove duplicates on id (if present)
    df = remove_duplicates(df, subset=[id_col] if id_col else None, name="CA")

    # Handle missing values generically
    df = handle_missing_values(df, name="CA")

    print("[CA] Final shape:", df.shape)

    # Save cleaned data (JSON records)
    df.to_json(CA_OUTPUT, orient="records", force_ascii=False, indent=2)
    print(f"[CA] Cleaned data saved to: {CA_OUTPUT}")


# ---------- Main ----------

def main():
    print("Base directory:", BASE_DIR)
    if not os.path.exists(NYC_PATH):
        raise FileNotFoundError(f"NYC dataset not found at: {NYC_PATH}")
    if not os.path.exists(CA_PATH):
        raise FileNotFoundError(f"CA dataset not found at: {CA_PATH}")

    clean_ab_nyc()
    print()
    clean_ca_category()
    print("\nAll cleaning tasks completed.")


if __name__ == "__main__":
    main()
    input("\nProcessing finished. Press Enter to close...")