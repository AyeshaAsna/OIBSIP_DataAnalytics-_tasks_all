import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------- Configuration ----------

BASE_DIR = r"D:\internship data analytics"

# Use a dedicated datasets subfolder
DATA_DIR = os.path.join(BASE_DIR, "datasets")

# Expected local filenames inside datasets folder:
# 1) Retail sales dataset (saved as retail_sales.csv)
RETAIL_PATH = os.path.join(DATA_DIR, "retail_sales.csv")
# 2) McDonald's nutrition facts (saved as mcdonalds_menu.csv)
MCD_PATH = os.path.join(DATA_DIR, "mcdonalds_menu.csv")


def check_files():
    missing = []
    if not os.path.exists(RETAIL_PATH):
        missing.append(RETAIL_PATH)
    if not os.path.exists(MCD_PATH):
        missing.append(MCD_PATH)
    if missing:
        print("The following expected CSV files are missing:")
        for p in missing:
            print("  -", p)
        print("\nPlease download and save:")
        print("  - Retail sales CSV as 'retail_sales.csv'")
        print("  - McDonald's menu CSV as 'mcdonalds_menu.csv'")
        print("into:", DATA_DIR)
        raise SystemExit(1)


# ---------- Retail sales EDA ----------

def load_retail():
    print(f"\nLoading retail sales data from: {RETAIL_PATH}")
    df = pd.read_csv(RETAIL_PATH)
    print("Retail shape:", df.shape)
    print(df.head())
    print("\nRetail info:")
    df.info()
    print("\nRetail missing values (%):")
    print(df.isna().mean() * 100)
    return df


def clean_and_prepare_retail(df: pd.DataFrame):
    data = df.copy()

    # Try to infer date and sales-like columns by common names
    date_cols = [c for c in data.columns if c.lower() in ["order_date", "date", "orderdate", "order date"]]
    sales_cols = [c for c in data.columns if c.lower() in ["sales", "revenue", "amount"]]

    date_col = date_cols[0] if date_cols else None
    sales_col = sales_cols[0] if sales_cols else None

    if date_col:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    if sales_col:
        data[sales_col] = pd.to_numeric(data[sales_col], errors="coerce")

    # Drop rows with missing core fields
    core_cols = [c for c in [date_col, sales_col] if c is not None]
    if core_cols:
        data = data.dropna(subset=core_cols)

    print("\nRetail descriptive stats (numeric):")
    print(data.describe())

    return data, date_col, sales_col


def retail_time_series_analysis(data: pd.DataFrame, date_col: str, sales_col: str):
    if not date_col or not sales_col:
        print("\nCould not infer date or sales columns; skipping time series plots.")
        return

    ts = (
        data.set_index(date_col)
        .sort_index()
        .resample("M")[sales_col]
        .sum()
    )

    plt.figure(figsize=(10, 4))
    ts.plot()
    plt.ylabel("Monthly sales")
    plt.title("Monthly sales over time")
    plt.tight_layout()
    plt.show()
    plt.close()


def retail_customer_product_analysis(data: pd.DataFrame, sales_col: str):
    # Try to guess customer and product columns
    customer_cols = [c for c in data.columns if "customer" in c.lower()]
    product_cols = [c for c in data.columns if any(x in c.lower() for x in ["product", "item", "sku"])]

    cust_col = customer_cols[0] if customer_cols else None
    prod_col = product_cols[0] if product_cols else None

    if cust_col and sales_col:
        cust_sales = (
            data.groupby(cust_col)[sales_col]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        print("\nTop 10 customers by sales:")
        print(cust_sales)

        plt.figure(figsize=(10, 4))
        sns.barplot(x=cust_sales.index, y=cust_sales.values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Total sales")
        plt.title("Top 10 customers by sales")
        plt.tight_layout()
        plt.show()
        plt.close()

    if prod_col and sales_col:
        prod_sales = (
            data.groupby(prod_col)[sales_col]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        print("\nTop 10 products by sales:")
        print(prod_sales)

        plt.figure(figsize=(10, 4))
        sns.barplot(x=prod_sales.index, y=prod_sales.values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Total sales")
        plt.title("Top 10 products by sales")
        plt.tight_layout()
        plt.show()
        plt.close()

    # Correlation heatmap for numeric fields
    plt.figure(figsize=(8, 6))
    corr = data.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Correlation heatmap (retail numeric features)")
    plt.tight_layout()
    plt.show()
    plt.close()


# ---------- McDonald's nutrition EDA ----------

def load_mcd():
    print(f"\nLoading McDonald's nutrition data from: {MCD_PATH}")
    df = pd.read_csv(MCD_PATH)
    print("McDonald's shape:", df.shape)
    print(df.head())
    print("\nMcDonald's info:")
    df.info()
    print("\nMcDonald's missing values (%):")
    print(df.isna().mean() * 100)
    return df


def mcd_descriptive_stats(df: pd.DataFrame):
    print("\nMcDonald's descriptive stats (numeric):")
    print(df.describe())

    # Try to detect calories column
    cal_cols = [c for c in df.columns if "cal" in c.lower()]
    cal_col = cal_cols[0] if cal_cols else None

    if cal_col:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[cal_col].dropna(), bins=30, kde=True)
        plt.xlabel(cal_col)
        plt.title("Distribution of calories")
        plt.tight_layout()
        plt.show()
        plt.close()


def mcd_category_analysis(df: pd.DataFrame):
    # Try to detect a category-like column
    cat_cols = [c for c in df.columns if any(x in c.lower() for x in ["category", "group", "type"])]
    cat_col = cat_cols[0] if cat_cols else None

    if not cat_col:
        print("\nNo obvious category column found in McDonald's data; skipping category plots.")
        return

    print(f"\nMenu item count by {cat_col}:")
    counts = df[cat_col].value_counts().head(10)
    print(counts)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title(f"Top {len(counts)} categories by item count")
    plt.tight_layout()
    plt.show()
    plt.close()


# ---------- Recommendations (printed) ----------

def print_recommendations():
    print("\n=== Example Recommendations from EDA ===")
    print("- Identify high-value customers and design loyalty programs targeting them.")
    print("- Promote top-selling products and consider bundling them with slower-moving items.")
    print("- Use time-series trends (e.g., seasonal peaks) to plan inventory and staffing.")
    print("- For menu analysis, highlight items with a better calorie-to-nutrition ratio.")
    print("- Use visual dashboards to track key KPIs (sales, average order value, category performance).")


# ---------- Main ----------

def main():
    print("Base directory:", BASE_DIR)
    check_files()

    # Retail EDA
    retail = load_retail()
    retail_clean, date_col, sales_col = clean_and_prepare_retail(retail)
    retail_time_series_analysis(retail_clean, date_col, sales_col)
    retail_customer_product_analysis(retail_clean, sales_col)

    # McDonald's EDA
    mcd = load_mcd()
    mcd_descriptive_stats(mcd)
    mcd_category_analysis(mcd)

    # Recommendations
    print_recommendations()

    print("\nAll EDA tasks (task1-L1) completed.")


if __name__ == "__main__":
    main()

