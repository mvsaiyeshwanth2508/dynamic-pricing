import pandas as pd
import numpy as np
from datetime import datetime

# -------------------- AUTO HEADER MAP -------------------- #
def header_automap(columns):
    """
    Auto-detect standard column names even if user’s dataset uses variations.
    """
    std = {
        "Date": ["date", "order_date", "timestamp", "sales_date"],
        "Product_ID": ["product", "product_id", "sku", "item_id", "product name"],
        "Product_Category": ["category", "segment", "product_category"],
        "Price_Set": ["price", "unit_price", "selling_price", "list_price"],
        "Sales_Volume": ["quantity", "qty", "units_sold", "sales_volume", "sold_units"],
        "Revenue": ["revenue", "amount", "sales", "total_sales"],
        "Discount": ["discount", "markdown", "offer"]
    }

    header_map = {}
    warnings = []
    for key, aliases in std.items():
        for a in aliases:
            for col in columns:
                if a.lower() in str(col).lower():
                    header_map[key] = col
                    break
            if key in header_map:
                break
        if key not in header_map:
            warnings.append(f"⚠️ Could not auto-map column for {key}")
    return header_map, warnings

# -------------------- CLEANING -------------------- #
def parse_and_clean(df, header_map):
    """
    Standardize columns, convert numeric types, and remove invalid rows.
    """
    problems = []
    df = df.copy()

    # rename columns
    rename_map = {v: k for k, v in header_map.items()}
    df.rename(columns=rename_map, inplace=True)

    # remove duplicate columns (some Excel files cause this)
    df = df.loc[:, ~df.columns.duplicated()]

    # numeric conversions
    num_cols = ["Price_Set", "Sales_Volume", "Revenue", "Discount"]
    for col in num_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception as e:
                problems.append(f"Could not convert {col}: {e}")

    # date parsing
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # fill missing revenue if possible
    if "Revenue" in df.columns and "Price_Set" in df.columns and "Sales_Volume" in df.columns:
        df["Revenue"] = df["Revenue"].fillna(df["Price_Set"] * df["Sales_Volume"])

    # drop rows missing key info
    essential = [c for c in ["Product_ID", "Price_Set", "Sales_Volume"] if c in df.columns]
    df = df.dropna(subset=essential)

    # ensure positive values
    for c in ["Price_Set", "Sales_Volume", "Revenue"]:
        if c in df.columns:
            df = df[df[c] >= 0]

    # sort by date if exists
    if "Date" in df.columns:
        df = df.sort_values("Date")

    return df.to_dict("records"), problems

# -------------------- ANALYSIS -------------------- #
def generate_insights(df):
    """
    Generate summary, top products, charts, discount analysis, and recommendations.
    """
    report = {}
    df = df.copy()

    # Summary
    total_revenue = float(df["Revenue"].sum()) if "Revenue" in df.columns else 0
    total_volume = float(df["Sales_Volume"].sum()) if "Sales_Volume" in df.columns else 0
    num_products = df["Product_ID"].nunique() if "Product_ID" in df.columns else 0
    date_range = [str(df["Date"].min().date()) if "Date" in df.columns else "N/A",
                  str(df["Date"].max().date()) if "Date" in df.columns else "N/A"]

    report["summary"] = {
        "total_revenue": round(total_revenue, 2),
        "total_volume": int(total_volume),
        "num_products": int(num_products),
        "date_range": date_range,
    }

    # Time series
    if "Date" in df.columns:
        time_series = df.groupby("Date", as_index=False)["Revenue"].sum()
        report["time_series"] = time_series.to_dict("records")
    else:
        report["time_series"] = []

    # Sales by category
    if "Product_Category" in df.columns:
        by_cat = df.groupby("Product_Category", as_index=False)["Revenue"].sum()
        report["by_category"] = by_cat.to_dict("records")
    else:
        report["by_category"] = []

    # Discount effectiveness
    if "Discount" in df.columns and "Sales_Volume" in df.columns:
        df["Discount_Bucket"] = pd.cut(df["Discount"], bins=[-0.01, 0.1, 0.2, 0.3, 1.0],
                                       labels=["<10%", "10-20%", "20-30%", ">30%"])
        discount_effect = (
            df.groupby("Discount_Bucket", observed=False)["Sales_Volume"].mean().reset_index()
        )
        report["discount_effect"] = discount_effect.to_dict("records")
    else:
        report["discount_effect"] = []

    # Top products
    if {"Product_ID", "Revenue", "Sales_Volume", "Price_Set"}.issubset(df.columns):
        top_products = (
            df.groupby("Product_ID", as_index=False)
            .agg({"Revenue": "sum", "Sales_Volume": "sum", "Price_Set": "mean"})
            .sort_values("Revenue", ascending=False)
            .head(10)
        )
        report["top_products"] = top_products.to_dict("records")
    else:
        report["top_products"] = []

    # Recommendations
    recs = []
    if "Price_Set" in df.columns and "Sales_Volume" in df.columns:
        avg_price = df["Price_Set"].mean()
        avg_vol = df["Sales_Volume"].mean()
        for _, row in df.groupby("Product_ID").agg({"Price_Set": "mean", "Sales_Volume": "mean"}).iterrows():
            if row["Price_Set"] > avg_price and row["Sales_Volume"] < avg_vol:
                recs.append({
                    "product": _,
                    "action": "Consider reducing price",
                    "justification": "High price, low volume"
                })
            elif row["Price_Set"] < avg_price and row["Sales_Volume"] > avg_vol:
                recs.append({
                    "product": _,
                    "action": "Potential to increase price",
                    "justification": "Low price, high volume"
                })
    report["recommendations"] = recs

    return report
