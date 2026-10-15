"""
data_loader.py — Universal file parser with caching, type inference, and cleaning.
"""

import io
import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Dict, List


# ---------------------------------------------------------------------------
# Sample dataset builder
# ---------------------------------------------------------------------------

def build_sample_dataset() -> pd.DataFrame:
    """Return a rich sample dataset that exercises all dashboard features."""
    np.random.seed(42)
    n = 500

    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    categories = np.random.choice(["Electronics", "Clothing", "Furniture", "Food", "Sports"], n)
    regions = np.random.choice(["North", "South", "East", "West"], n)
    age = np.random.randint(18, 70, n)
    experience = np.clip(age - 18 + np.random.randint(-5, 5, n), 0, 50)
    base_salary = 30_000 + experience * 1_200 + np.random.normal(0, 5_000, n)
    units_sold = np.random.randint(1, 200, n)
    unit_price = np.random.uniform(5, 500, n).round(2)
    revenue = (units_sold * unit_price * np.random.uniform(0.8, 1.2, n)).round(2)
    churn = (revenue < revenue.mean() * 0.6).astype(int)

    # Introduce ~5 % missing values
    df = pd.DataFrame({
        "date": dates,
        "category": categories,
        "region": regions,
        "age": age,
        "experience_years": experience,
        "salary": base_salary.round(2),
        "units_sold": units_sold,
        "unit_price": unit_price,
        "revenue": revenue,
        "churn": churn,
    })

    rng = np.random.default_rng(7)
    for col in ["salary", "units_sold", "unit_price", "revenue"]:
        mask = rng.random(n) < 0.05
        df.loc[mask, col] = np.nan

    return df


# ---------------------------------------------------------------------------
# Column-type classifier
# ---------------------------------------------------------------------------

def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Partition the DataFrame's columns into:
    - 'numeric': int / float columns
    - 'categorical': object / bool / low-cardinality int columns
    - 'datetime': datetime columns
    """
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []

    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            # Low-cardinality integers are treated as categorical for modeling
            unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
            if df[col].nunique(dropna=True) <= 10 and unique_ratio < 0.05:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }


# ---------------------------------------------------------------------------
# File parser
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_file(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """
    Parse any supported file type into a DataFrame.
    Returns (df, error_message).  On success error_message is ''.
    """
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, low_memory=False)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        elif name.endswith(".json"):
            raw = uploaded_file.read()
            df = pd.read_json(io.BytesIO(raw))
        elif name.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(uploaded_file.read()))
        else:
            return pd.DataFrame(), f"Unsupported file type: {name.split('.')[-1]}"

        # Try to parse obvious datetime columns
        for col in df.select_dtypes(include="object").columns:
            sample = df[col].dropna().head(100)
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
                if parsed.notna().sum() / max(len(sample), 1) > 0.8:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            except Exception:
                pass

        return df, ""
    except Exception as exc:
        return pd.DataFrame(), str(exc)


# ---------------------------------------------------------------------------
# Preprocessing engine
# ---------------------------------------------------------------------------

def apply_preprocessing(
    df: pd.DataFrame,
    missing_strategy: str,
    remove_duplicates: bool,
    outlier_removal: bool,
    excluded_columns: List[str],
    col_types: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Apply the chosen cleaning operations and return a cleaned copy.
    """
    df = df.copy()

    # ── Excluded columns ─────────────────────────────────────────────────────
    cols_to_keep = [c for c in df.columns if c not in excluded_columns]
    df = df[cols_to_keep]

    # Recompute column lists after dropping columns
    numeric = [c for c in col_types["numeric"] if c in df.columns]
    categorical = [c for c in col_types["categorical"] if c in df.columns]

    # ── Missing value handling ────────────────────────────────────────────────
    if missing_strategy == "Drop rows with missing values":
        df = df.dropna()
    elif missing_strategy == "Impute — Mean (numeric) / Mode (categorical)":
        for col in numeric:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
        for col in categorical:
            if df[col].isna().any():
                mode_val = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) else "Unknown")
    elif missing_strategy == "Impute — Median (numeric) / Mode (categorical)":
        for col in numeric:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        for col in categorical:
            if df[col].isna().any():
                mode_val = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) else "Unknown")
    elif missing_strategy == "Forward Fill":
        df = df.ffill().bfill()

    # ── Duplicate removal ─────────────────────────────────────────────────────
    if remove_duplicates:
        df = df.drop_duplicates()

    # ── Outlier removal (IQR) ─────────────────────────────────────────────────
    if outlier_removal:
        for col in numeric:
            if col not in df.columns:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df = df[(df[col] >= lower) & (df[col] <= upper)]

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------

def profile_columns(df: pd.DataFrame, col_types: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Build a per-column summary table for the Data Profiling tab.
    """
    rows = []
    for col in df.columns:
        dtype_group = (
            "Datetime" if col in col_types["datetime"]
            else "Numeric" if col in col_types["numeric"]
            else "Categorical"
        )
        series = df[col]
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))

        if dtype_group == "Numeric":
            rows.append({
                "Column": col,
                "Type": dtype_group,
                "Missing": missing,
                "Unique": unique,
                "Min": round(series.min(), 4) if not series.dropna().empty else "—",
                "Max": round(series.max(), 4) if not series.dropna().empty else "—",
                "Mean": round(series.mean(), 4) if not series.dropna().empty else "—",
            })
        else:
            top = series.mode(dropna=True).iloc[0] if not series.dropna().empty else "—"
            rows.append({
                "Column": col,
                "Type": dtype_group,
                "Missing": missing,
                "Unique": unique,
                "Min": "—",
                "Max": "—",
                "Mean": f"Mode: {top}",
            })
    return pd.DataFrame(rows)
