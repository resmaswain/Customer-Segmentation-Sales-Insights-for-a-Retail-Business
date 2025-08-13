"""
Customer Segmentation & Sales Insights for a Retail Business
-----------------------------------------------------------
Python Script (standalone) with clear comments.

Works with the classic "Online Retail" dataset (or any similar schema) having columns:
- InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

How to run:
    python analysis.py --data_path /path/to/online_retail.xlsx --sheet "Year 2009-2010"

Outputs:
- figures/ folder with PNG charts
- exports/ folder with CSVs (top products, monthly sales, top countries, rfm table)
- prints key KPIs in console

Author: (Your Name)
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Optional KMeans for clustering (can be skipped if not installed)
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def ensure_dirs():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("exports", exist_ok=True)


def load_data(data_path: str, sheet: str = None) -> pd.DataFrame:
    """
    Load Excel/CSV into a DataFrame.
    - If Excel: you may specify a sheet name like 'Year 2009-2010' or 'Year 2010-2011'.
    - If CSV: sheet is ignored.
    """
    ext = os.path.splitext(data_path)[-1].lower()
    if ext in [".xls", ".xlsx"]:
        df = pd.read_excel(data_path, sheet_name=sheet)
    elif ext == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Unsupported file type. Use .xls, .xlsx, or .csv")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning aligned with the notebook screenshots:
    - Drop NA CustomerID
    - Remove duplicates
    - Remove cancelled invoices (InvoiceNo starting with 'C')
    - Keep positive quantities and unit prices
    - Parse InvoiceDate to datetime
    - Create TotalPrice = Quantity * UnitPrice
    """
    # Drop rows with no CustomerID (cannot do RFM without it)
    df = df.dropna(subset=["CustomerID"]).copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove cancellations/credit memos that start with 'C'
    if "InvoiceNo" in df.columns:
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Keep only positive values
    if "Quantity" in df.columns:
        df = df[df["Quantity"] > 0]
    if "UnitPrice" in df.columns:
        df = df[df["UnitPrice"] > 0]

    # Parse date
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    # Monetary value
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


def top_products(df: pd.DataFrame, n=10) -> pd.DataFrame:
    """Top n products by Quantity."""
    top = (
        df.groupby("Description", dropna=False)["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    top.to_csv("exports/top_products.csv", index=False)
    # Plot
    plt.figure(figsize=(9, 5))
    bars = plt.barh(top["Description"][::-1], top["Quantity"][::-1])
    plt.xlabel("Quantity Sold")
    plt.title(f"Top {n} Selling Products by Quantity")
    # Add labels
    for bar in bars:
        w = bar.get_width()
        plt.text(w, bar.get_y() + bar.get_height()/2, f"{int(w):,}", va="center")
    plt.tight_layout()
    plt.savefig("figures/top_products.png", dpi=150)
    plt.close()
    return top


def monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly sales trend by total revenue."""
    m = (
        df.set_index("InvoiceDate")
        .resample("M")["TotalPrice"]
        .sum()
        .reset_index()
    )
    m["YearMonth"] = m["InvoiceDate"].dt.to_period("M").astype(str)
    m.to_csv("exports/monthly_sales.csv", index=False)

    # Plot
    plt.figure(figsize=(9, 5))
    plt.plot(m["YearMonth"], m["TotalPrice"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Total Revenue")
    plt.title("Monthly Sales Trends")
    # Add value labels
    for x, y in zip(m["YearMonth"], m["TotalPrice"]):
        plt.text(x, y, f"{int(y):,}", rotation=45, va="bottom")
    plt.tight_layout()
    plt.savefig("figures/monthly_sales.png", dpi=150)
    plt.close()
    return m


def top_countries(df: pd.DataFrame, n=10) -> pd.DataFrame:
    """Top n countries by total revenue."""
    c = (
        df.groupby("Country")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    c.to_csv("exports/top_countries.csv", index=False)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(c["Country"], c["TotalPrice"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Total Revenue")
    plt.title(f"Top {n} Countries by Sales")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h, f"{int(h):,}", ha="center", va="bottom", rotation=90)
    plt.tight_layout()
    plt.savefig("figures/top_countries.png", dpi=150)
    plt.close()
    return c


def compute_kpis(df: pd.DataFrame) -> dict:
    """Some quick KPIs used in the slides."""
    kpis = {}
    kpis["total_revenue"] = float(df["TotalPrice"].sum())
    orders = df.groupby("InvoiceNo").agg(order_value=("TotalPrice", "sum")).reset_index()
    kpis["avg_order_value"] = float(orders["order_value"].mean())
    cust_rev = df.groupby("CustomerID")["TotalPrice"].sum().reset_index()
    kpis["revenue_per_customer"] = float(cust_rev["TotalPrice"].mean())
    kpis["total_customers"] = int(cust_rev.shape[0])
    return kpis


def rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RFM metrics."""
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )
    rfm.to_csv("exports/rfm_raw.csv", index=False)
    return rfm


def rfm_segment_label(row):
    """Simple rule-based segmentation by RFM score (quartile ranks)."""
    r, f, m = row["R_quart"], row["F_quart"], row["M_quart"]
    score = r + f + m  # 3..12 (lower is better because Recency quartile 1 means most recent)
    # Map using simple heuristics similar to the notebook
    if score <= 6 and f <= 2 and m <= 2:
        return "Loyal"
    if score <= 7 and f <= 3:
        return "Potential"
    if r >= 3 and f >= 3:
        return "At Risk"
    return "Needs Attention"


def rfm_segment(rfm: pd.DataFrame) -> pd.DataFrame:
    """Create quartiles and segment labels; save chart of segment sizes."""
    # Quartiles: smaller Recency is better; larger Frequency/Monetary is better
    rfm["R_quart"] = pd.qcut(rfm["Recency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["F_quart"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["M_quart"] = pd.qcut(rfm["Monetary"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["Segment"] = rfm.apply(rfm_segment_label, axis=1)

    seg_counts = rfm["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    seg_counts.to_csv("exports/segment_counts.csv", index=False)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(seg_counts["Segment"], seg_counts["Count"])
    plt.title("Customer Segmentation (RFM)")
    plt.ylabel("Number of Customers")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h, f"{int(h):,}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("figures/rfm_segments.png", dpi=150)
    plt.close()

    rfm.to_csv("exports/rfm_table.csv", index=False)
    return rfm


def kmeans_on_rfm(rfm: pd.DataFrame, max_k: int = 10):
    """Optional: Standardize RFM and run KMeans; save elbow chart and scatter plots."""
    if not SKLEARN_AVAILABLE:
        print("sklearn not available—skipping KMeans clustering.")
        return None

    X = rfm[["Recency", "Frequency", "Monetary"]].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    wcss = []
    Ks = list(range(1, max_k + 1))
    for k in Ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(Ks, wcss, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for Optimal k")
    plt.tight_layout()
    plt.savefig("figures/elbow.png", dpi=150)
    plt.close()

    # Choose k=4 to mirror the screenshot and build a summary
    kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)
    rfm["Cluster"] = labels

    # Simple 2D scatter (Recency vs Monetary) to mimic screenshot feel
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(rfm["Recency"], rfm["Monetary"], c=rfm["Cluster"], s=20)
    plt.xlabel("Recency")
    plt.ylabel("Monetary")
    plt.title("Customer Clusters by RFM (KMeans)")
    plt.tight_layout()
    plt.savefig("figures/kmeans_rfm.png", dpi=150)
    plt.close()

    # Cluster summary
    summary = (
        rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .round(2)
        .reset_index()
    )
    summary.to_csv("exports/kmeans_cluster_summary.csv", index=False)
    return rfm, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to Online Retail dataset (.xls/.xlsx/.csv)")
    parser.add_argument("--sheet", type=str, default=None,
                        help="Excel sheet name (if applicable)")
    args = parser.parse_args()

    ensure_dirs()

    print("Loading data...")
    df = load_data(args.data_path, args.sheet)
    print(f"Raw shape: {df.shape}")

    print("Cleaning...")
    df = clean_data(df)
    print(f"Clean shape: {df.shape}")

    print("Top products...")
    tp = top_products(df, n=10)
    print(tp.head())

    print("Monthly sales...")
    ms = monthly_sales(df)
    print(ms.head())

    print("Top countries...")
    tc = top_countries(df, n=10)
    print(tc.head())

    print("Computing KPIs...")
    kpis = compute_kpis(df)
    for k, v in kpis.items():
        if isinstance(v, float):
            print(f"{k}: {v:,.2f}")
        else:
            print(f"{k}: {v}")

    print("RFM table & segments...")
    rfm = rfm_table(df)
    rfm = rfm_segment(rfm)

    if SKLEARN_AVAILABLE:
        print("KMeans clustering...")
        result = kmeans_on_rfm(rfm, max_k=10)
        if result is not None:
            rfm, summary = result
            print("Cluster summary:\n", summary)
    else:
        print("sklearn missing—skip clustering, but RFM segments ready.")

    print("All done. Charts saved to ./figures and CSVs to ./exports")


if __name__ == "__main__":
    main()
