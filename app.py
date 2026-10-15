"""
Enterprise Predictive Analytics Studio
Production-grade, beginner-friendly predictive modeling dashboard.
Engineered with clean corporate typography, minimal design, and zero jargon.
"""

from __future__ import annotations

import io
import re
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PAGE SETUP & CORPORATE ENTERPRISE STYLING
# ==============================================================================

st.set_page_config(
    page_title="Predictive Analytics Studio",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global Base */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #090d16;
        color: #f1f5f9;
    }
    
    /* Clean Top Header */
    header[data-testid="stHeader"] {
        background: rgba(9, 13, 22, 0.85);
        backdrop-filter: blur(12px);
    }
    
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }

    /* Enterprise Header */
    .corp-header {
        border-bottom: 1px solid #1e293b;
        padding-bottom: 1.25rem;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    }
    .corp-title-wrap {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }
    .corp-brand-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #60a5fa;
    }
    .corp-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 0;
    }
    .corp-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-top: 0.2rem;
    }
    .corp-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #0f172a;
        border: 1px solid #1e293b;
        padding: 5px 12px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #cbd5e1;
    }
    .corp-status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
    }

    /* Clean Section Containers */
    .corp-section {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.75rem;
    }
    .corp-section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.35rem;
    }
    .corp-section-idx {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: #3b82f6;
        background: rgba(59, 130, 246, 0.12);
        padding: 2px 7px;
        border-radius: 4px;
    }
    .corp-section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.01em;
    }
    .corp-section-desc {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-bottom: 1.25rem;
    }

    /* Professional Metric Cards */
    .metric-panel {
        background: #131d31;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1rem 1.2rem;
    }
    .metric-panel-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-panel-val {
        font-size: 1.65rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 0.2rem;
        letter-spacing: -0.01em;
    }
    .metric-panel-note {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Primary and Secondary Buttons */
    .stButton > button {
        background: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.4rem !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background: #1d4ed8 !important;
        border-color: #60a5fa !important;
    }
    .stButton > button * {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    .stDownloadButton > button {
        background: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.55rem 1.3rem !important;
        transition: all 0.15s ease !important;
    }
    .stDownloadButton > button:hover {
        background: #334155 !important;
        color: #ffffff !important;
    }
    .stDownloadButton > button * {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }

    /* Dropzone Refinement */
    [data-testid="stFileUploadDropzone"],
    [data-testid="stFileUploaderDropzone"] {
        background: #111a2e !important;
        border: 1px dashed #334155 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #3b82f6 !important;
    }
    [data-testid="stFileUploadDropzone"] button {
        background: #1e293b !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 6px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stFileUploadDropzone"] button * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    [data-testid="stFileUploadDropzone"] span {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }

    /* Executive Scenario Outcome Box */
    .scenario-outcome-panel {
        background: #0f231e;
        border: 1px solid #059669;
        border-radius: 10px;
        padding: 1.5rem 2rem;
        margin: 1.25rem 0;
        text-align: center;
    }
    .scenario-outcome-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        color: #34d399;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .scenario-outcome-val {
        font-size: 2.6rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.35rem 0;
        letter-spacing: -0.02em;
    }
    .scenario-outcome-sub {
        font-size: 0.85rem;
        color: #a7f3d0;
    }

    /* Clean native inputs */
    div[data-baseweb="select"] {
        background: #131d31 !important;
        border-color: #334155 !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="select"] * {
        color: #f1f5f9 !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: #111a2e !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
    }

    /* #1 Top Performer Card (Categorical Target) */
    .performer-card {
        background: linear-gradient(135deg, #111e38 0%, #0f172a 100%);
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .performer-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.35);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        width: fit-content;
        margin-bottom: 0.4rem;
    }
    .performer-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.01em;
        margin-bottom: 0.4rem;
        line-height: 1.2;
    }
    .performer-stats {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.88rem;
        color: #cbd5e1;
    }

    /* Big Summary Card for Numeric Target */
    .summary-hero-card {
        background: linear-gradient(135deg, #0d2820 0%, #0f172a 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .summary-hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        width: fit-content;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. FILE INGESTION & DATA CLASSIFICATION HELPERS
# ==============================================================================

def load_uploaded_spreadsheet(file_obj) -> pd.DataFrame:
    """Universally parse Excel (.xlsx, .xls), CSV, and HTML-based spreadsheets."""
    filename = file_obj.name.lower()
    content = file_obj.getvalue()

    # 1. Specialized engine based on file extension
    if filename.endswith(".xls") and not filename.endswith(".xlsx"):
        try:
            return pd.read_excel(io.BytesIO(content), engine="xlrd")
        except Exception:
            pass

    if filename.endswith(".xlsx"):
        try:
            return pd.read_excel(io.BytesIO(content), engine="openpyxl")
        except Exception:
            pass

    if filename.endswith(".csv"):
        for enc in ["utf-8", "latin1", "cp1252", "iso-8859-1"]:
            try:
                return pd.read_csv(io.BytesIO(content), encoding=enc)
            except Exception:
                pass

    # 2. Resilient fallbacks for misnamed files or alternative formats
    try:
        return pd.read_excel(io.BytesIO(content), engine="xlrd")
    except Exception:
        pass

    try:
        return pd.read_excel(io.BytesIO(content), engine="openpyxl")
    except Exception:
        pass

    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc)
        except Exception:
            pass

    try:
        tables = pd.read_html(io.BytesIO(content))
        if tables:
            return tables[0]
    except Exception:
        pass
    raise ValueError(
        f"Unable to parse '{getattr(file_obj, 'name', 'file')}'. Please ensure the spreadsheet is a valid .xlsx, .xls, or .csv file."
    )


DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_MAP = {d.lower(): i for i, d in enumerate(DAY_NAMES)}
DAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_ABBR_MAP = {d.lower(): i for i, d in enumerate(DAY_ABBR)}

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
MONTH_MAP = {m.lower(): i for i, m in enumerate(MONTH_NAMES)}
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_ABBR_MAP = {m.lower(): i for i, m in enumerate(MONTH_ABBR)}


def detect_calendar_order(items: Any) -> Tuple[bool, List[str]]:
    """Detect if items represent days of the week or months and return their canonical chronological sequence."""
    if hasattr(items, "tolist"):
        raw_list = items.tolist()
    else:
        raw_list = list(items)

    clean_items = [str(x).strip().lower() for x in raw_list if pd.notna(x) and str(x).strip() != ""]
    if not clean_items:
        return False, []

    # Check full day names
    match_days = sum(1 for x in clean_items if x in DAY_MAP)
    if match_days >= len(clean_items) * 0.5:
        present = set(clean_items)
        ordered = [d for d in DAY_NAMES if d.lower() in present]
        return True, ordered

    # Check abbreviated day names
    match_days_abbr = sum(1 for x in clean_items if x in DAY_ABBR_MAP)
    if match_days_abbr >= len(clean_items) * 0.5:
        present = set(clean_items)
        ordered = [d for d in DAY_ABBR if d.lower() in present]
        return True, ordered

    # Check full month names
    match_months = sum(1 for x in clean_items if x in MONTH_MAP)
    if match_months >= len(clean_items) * 0.5:
        present = set(clean_items)
        ordered = [m for m in MONTH_NAMES if m.lower() in present]
        return True, ordered

    # Check abbreviated month names
    match_months_abbr = sum(1 for x in clean_items if x in MONTH_ABBR_MAP)
    if match_months_abbr >= len(clean_items) * 0.5:
        present = set(clean_items)
        ordered = [m for m in MONTH_ABBR if m.lower() in present]
        return True, ordered

    return False, []


def is_id_column(col_name: str, series: pd.Series) -> bool:
    """Identify pure identity or record key columns that should never be used as chart bars or targets."""
    c_clean = str(col_name).strip()
    c_lower = c_clean.lower()

    # Strictly check ID keywords (do not falsely match descriptive names like CustomerName or Product)
    id_regex = r"(^id$|_id$|^id_|\bid\b|order_?id|cust_?id|customer_?id|account_?id|acc_?id|trans_?id|transaction_?id|invoice_?id|inv_?id|user_?id|employee_?id|emp_?id|^uuid|^guid|^index$|^row_num$|^row_id$|postal_?code|zip_?code)"
    if re.search(id_regex, c_lower):
        return True

    # Check if 100% unique alphanumeric codes (like ACC-1001, ORD-9923)
    n = len(series)
    if n > 15 and series.dtype == object:
        n_unique = series.nunique(dropna=True)
        if n_unique == n:
            sample_val = str(series.dropna().iloc[0]).strip()
            if re.match(r"^[A-Za-z0-9_-]+$", sample_val) and any(char.isdigit() for char in sample_val):
                return True

    return False


def is_date_column(col_name: str, series: pd.Series) -> bool:
    """Identify date / timestamp columns that should only be used for timeline trends."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    c_lower = str(col_name).strip().lower()
    date_keywords = ["date", "time", "timestamp", "datetime", "dob", "orderdate", "deliverydate"]
    if any(k in c_lower for k in date_keywords):
        return True
    if series.dtype == object and len(series) > 0:
        sample = series.dropna().head(20)
        if len(sample) > 5:
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() / len(sample) >= 0.8:
                    return True
            except Exception:
                pass
    return False


# ==============================================================================
# 3. BUSINESS DATA GENERATOR (ENTERPRISE RETAIL DEMO)
# ==============================================================================

def get_sample_business_data() -> pd.DataFrame:
    """Generate realistic commercial sales and customer engagement data."""
    np.random.seed(42)
    n = 600

    products = ["Smart Watch Pro", "Wireless ANC Headphones", "Ergonomic Office Chair", "Mechanical Keyboard", "USB-C Multiport Hub"]
    regions = ["North America", "Europe", "Asia-Pacific", "Latin America"]
    channels = ["Direct Online", "Enterprise B2B", "Retail Partner"]
    customers = [
        "Apex Global", "Vanguard Retail", "Nexus Enterprise", "Starlight Systems", 
        "Pinnacle Media", "Quantum Labs", "Summit Dynamics", "Horizon Logistics",
        "BlueSky Health", "Crestview Holdings", "Sterling Partners", "Optima Tech"
    ]

    chosen_products = np.random.choice(products, n, p=[0.25, 0.25, 0.15, 0.20, 0.15])
    chosen_regions = np.random.choice(regions, n)
    chosen_channels = np.random.choice(channels, n)
    chosen_customers = np.random.choice(customers, n)

    # Chronological timestamps for timeline analysis
    start_date = pd.Timestamp("2025-01-01")
    date_offsets = pd.to_timedelta(np.random.randint(0, 180, n), unit="D")
    order_dates = (start_date + date_offsets).strftime("%Y-%m-%d")

    price_map = {
        "Smart Watch Pro": 299,
        "Wireless ANC Headphones": 199,
        "Ergonomic Office Chair": 389,
        "Mechanical Keyboard": 149,
        "USB-C Multiport Hub": 79,
    }
    unit_price = np.array([price_map[p] for p in chosen_products])

    customer_age = np.random.randint(21, 65, n)
    marketing_spend = np.random.uniform(50, 850, n).round(2)
    discount_pct = np.random.choice([0, 5, 10, 15, 20], n, p=[0.3, 0.25, 0.2, 0.15, 0.1])

    expected_units = (marketing_spend / 110) + (discount_pct / 4.5) + np.random.normal(3.5, 1.2, n)
    units_sold = np.clip(np.round(expected_units), 1, 24).astype(int)
    total_sales = np.round(units_sold * unit_price * (1 - discount_pct / 100), 2)
    satisfaction = np.clip(np.round(4.8 - (discount_pct == 0) * 0.3 + np.random.normal(0, 0.4, n), 1), 1.0, 5.0)
    customer_tier = np.where(total_sales > np.median(total_sales), "Enterprise Tier", "Standard Tier")

    df = pd.DataFrame({
        "Account_ID": [f"ACC-{1000 + i}" for i in range(n)],
        "Order_Date": order_dates,
        "Customer_Name": chosen_customers,
        "Product_Line": chosen_products,
        "Sales_Region": chosen_regions,
        "Distribution_Channel": chosen_channels,
        "Customer_Age": customer_age,
        "Marketing_Budget": marketing_spend,
        "Discount_Rate": discount_pct,
        "Units_Purchased": units_sold,
        "Revenue": total_sales,
        "CSAT_Rating": satisfaction,
        "Account_Tier": customer_tier,
    })

    # Realistic missing values to verify background imputation
    rng = np.random.default_rng(88)
    for col in ["Marketing_Budget", "CSAT_Rating"]:
        mask = rng.random(n) < 0.03
        df.loc[mask, col] = np.nan

    return df


# ==============================================================================
# 4. BACKGROUND DATA CLEANER
# ==============================================================================

def clean_data_automatically(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Clean data silently without user friction, extract date components, and identify IDs/Dates."""
    df = raw_df.copy()
    ignored_id_columns: List[str] = []
    detected_date_columns: List[str] = []

    # 1. Parse date columns and extract Month / DayOfWeek
    cols_to_check = list(df.columns)
    for col in cols_to_check:
        if is_date_column(col, df[col]):
            detected_date_columns.append(col)
            try:
                parsed_all = pd.to_datetime(df[col], errors="coerce")
                clean_name = re.sub(r'(?i)(date|time)', '', str(col)).strip('_ ') or 'Event'
                month_col = f"{clean_name}_Month"
                dow_col = f"{clean_name}_DayOfWeek"
                if month_col not in df.columns:
                    df[month_col] = parsed_all.dt.month.fillna(1).astype(int)
                if dow_col not in df.columns:
                    df[dow_col] = parsed_all.dt.day_name().fillna("Unknown")
            except Exception:
                pass

    # 2. Identify pure ID / Index columns
    for col in df.columns:
        if col not in detected_date_columns and is_id_column(col, df[col]):
            ignored_id_columns.append(col)

    # 3. Impute missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean()
            df[col] = df[col].fillna(0 if pd.isna(mean_val) else mean_val)
        else:
            mode_series = df[col].mode(dropna=True)
            mode_val = mode_series.iloc[0] if len(mode_series) > 0 else "Standard"
            df[col] = df[col].fillna(mode_val).astype(str).str.strip()

    return df, ignored_id_columns, detected_date_columns


# ==============================================================================
# 4B. GLOBAL COLUMN FILTER (STRICT BUSINESS HYGIENE & ANTI-NOISE FILTER)
# ==============================================================================

EXCLUDED_ALWAYS_KEYWORDS = [
    "phone", "email", "address", "street", "postal", "zip", "code", "index",
    "serial", "url", "notes", "comment"
]

def has_id_keyword(col_lower: str) -> bool:
    """Identify ID columns by keyword (e.g. order_id, rowid, cust_id, product_id)."""
    if re.search(r'(^|[\s_\-/])id($|[\s_\-/0-9])', col_lower):
        return True
    if col_lower.endswith("_id") or col_lower.endswith(" id") or col_lower.startswith("id_") or col_lower.startswith("id "):
        return True
    if re.search(r'(order|cust|customer|client|user|product|trans|transaction|account|emp|employee|item|invoice|row|postal)id', col_lower):
        return True
    return False

def has_row_keyword(col_lower: str) -> bool:
    """Identify Row / Index columns while safely avoiding business words like 'growth' or 'borrow'."""
    return bool(re.search(r'(^|[\s_\-/])row($|[\s_\-/0-9]|_?id|_?num)', col_lower))

def has_name_keyword(col_lower: str) -> bool:
    """Identify Name columns (Customer Name, Product Name, Client Name, User Name, etc.)."""
    return "name" in col_lower

def get_global_filtered_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], Dict[str, str]]:
    """
    Global Column Filter Function:
    Filters out useless columns (IDs, names, customer personal info, addresses, zip codes,
    high-cardinality text, and zero-variance columns) across the entire platform.

    1. Globally excludes columns matching keywords: id, name, customer, client, user, phone, email,
       address, street, postal, zip, code, row, index, serial, url, notes, comment.
    2. Globally excludes text/categorical columns with nunique <= 1 or nunique > 15 (keeping clean
       grouping categories with 2-15 items, allowing Sub-Category with up to 20).
    3. Keeps only true business metrics for numeric columns, excluding IDs, years, zip codes, etc.
    """
    clean_numeric_cols: List[str] = []
    clean_category_cols: List[str] = []
    excluded_audit: Dict[str, str] = {}

    for col in df.columns:
        c_clean = str(col).strip()
        c_lower = c_clean.lower().replace("-", "_").replace(" ", "_")

        # Exclude internal date components and raw dates (reserved for timelines)
        if pd.api.types.is_datetime64_any_dtype(df[col]) or any(k in c_lower for k in ["date", "time", "timestamp", "dob"]):
            excluded_audit[col] = "Date / Timeline column"
            continue
        if c_lower.endswith("_month") or c_lower.endswith("_dayofweek"):
            continue

        # 1. Keyword Exclusions
        if has_id_keyword(str(col).lower()):
            excluded_audit[col] = "Contains ID keyword"
            continue
        if has_row_keyword(str(col).lower()):
            excluded_audit[col] = "Contains Row keyword"
            continue
        if has_name_keyword(str(col).lower()):
            excluded_audit[col] = "Contains Name keyword"
            continue

        matched_kw = next((kw for kw in EXCLUDED_ALWAYS_KEYWORDS if kw in str(col).lower()), None)
        if matched_kw:
            excluded_audit[col] = f"Contains excluded keyword '{matched_kw}'"
            continue

        matched_user_kw = next((kw for kw in ["customer", "client", "user"] if kw in str(col).lower()), None)
        if matched_user_kw:
            is_valid_grouping = any(term in c_lower for term in ["segment", "tier", "type", "category", "status", "group", "channel"])
            if not is_valid_grouping or pd.api.types.is_numeric_dtype(df[col]):
                excluded_audit[col] = f"Contains identity keyword '{matched_user_kw}'"
                continue

        series = df[col]
        n_unique = series.nunique(dropna=True)

        # 2. Text / Categorical columns
        is_num = pd.api.types.is_numeric_dtype(series)
        if not is_num:
            if n_unique <= 1:
                excluded_audit[col] = f"Zero/low variance text (nunique={n_unique} <= 1)"
                continue

            # Quasi-constant check (e.g. Country/Region where >= 95% is a single country)
            if len(series) > 0:
                top_freq = series.value_counts(normalize=True).iloc[0]
                if top_freq >= 0.95 and any(k in c_lower for k in ["country", "nation", "region", "state"]):
                    excluded_audit[col] = f"Quasi-constant ({top_freq:.1%} same value)"
                    continue

            # Clean grouping categories: 2 to 15 unique values (allow up to 20 for Sub-Category)
            is_subcat = any(k in c_lower for k in ["sub_category", "sub-category", "subcategory"])
            max_limit = 20 if is_subcat else 15

            if n_unique > max_limit:
                excluded_audit[col] = f"High cardinality text (nunique={n_unique} > {max_limit})"
                continue

            clean_category_cols.append(col)
        else:
            # 3. Numeric columns: Real business metrics
            if n_unique <= 1:
                excluded_audit[col] = "Zero variance numeric"
                continue
            if "year" in c_lower:
                excluded_audit[col] = "Year column"
                continue
            if series.dropna().between(1900, 2100).all() and n_unique <= 20 and ("yr" in c_lower or "year" in c_lower):
                excluded_audit[col] = "Year values"
                continue

            clean_numeric_cols.append(col)

    all_clean_cols = clean_numeric_cols + clean_category_cols
    return clean_numeric_cols, clean_category_cols, all_clean_cols, excluded_audit


# ==============================================================================
# 5. STATISTICAL MODELING PIPELINE
# ==============================================================================

def train_prediction_model(
    df: pd.DataFrame,
    target_col: str,
    ignored_cols: List[str],
    date_cols: List[str] | None = None,
    clean_candidate_pool: List[str] | None = None,
) -> Dict[str, Any]:
    """Train Random Forest model with automatic feature hygiene, dataset-wide inference, and calibrated scoring."""
    excluded_set = set(ignored_cols or []).union(set(date_cols or []))
    if clean_candidate_pool:
        date_features = [c for c in df.columns if c.endswith("_Month") or c.endswith("_DayOfWeek")]
        full_pool = list(clean_candidate_pool) + date_features
        candidate_cols = [c for c in full_pool if c in df.columns and c != target_col and c not in excluded_set]
    else:
        candidate_cols = [c for c in df.columns if c != target_col and c not in excluded_set]

    # Feature Hygiene & Leakage Prevention Rules:
    # 1. Backwards logic prevention: If predicting UnitPrice, exclude TotalPrice / Sales / Revenue
    target_clean = str(target_col).lower().replace(" ", "_").replace("-", "_")
    is_target_unit = "unit" in target_clean or "per_unit" in target_clean
    if is_target_unit:
        leakage_keywords = ["total", "sales", "revenue", "gross", "turnover", "profit", "amount"]
        candidate_cols = [c for c in candidate_cols if not any(k in str(c).lower() for k in leakage_keywords)]

    # 2. Strict ID, customer, order, contact, and high-cardinality exclusions:
    # Drop CustomerName, CustomerID, OrderID, etc., but preserve key product/category columns
    excluded_keywords = [
        "id", "date", "time", "phone", "email", "code", "customer", "cust",
        "order", "postal", "zip", "row_id", "rowid", "trans_id", "invoice"
    ]
    feature_cols = []

    for c in candidate_cols:
        c_lower = str(c).lower().strip().replace(" ", "_")
        # Rule (a): Exclude identity, customer, order keywords
        if any(w in c_lower for w in excluded_keywords):
            continue
        # Rule (b): Exclude generic 'name' columns (e.g. Customer Name, Client Name),
        # but allow Product/Category columns
        if "name" in c_lower and not any(k in c_lower for k in ["product", "category", "item"]):
            continue
        # Rule (c): Exclude text/categorical with > 30% unique values (high-cardinality)
        if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype):
            unique_ratio = df[c].nunique(dropna=True) / max(len(df), 1)
            if unique_ratio > 0.30 and df[c].nunique(dropna=True) > 10:
                continue
        feature_cols.append(c)

    # Safe fallback if rules filtered all columns
    if not feature_cols:
        feature_cols = [c for c in candidate_cols if not is_id_column(c, df[c]) and not is_date_column(c, df[c])]
    if not feature_cols:
        feature_cols = candidate_cols

    valid_data = df.dropna(subset=[target_col])
    if len(valid_data) < 10:
        raise ValueError("At least 10 valid observations are required for model training.")

    y = valid_data[target_col]
    X = valid_data[feature_cols]

    if y.nunique() <= 1:
        raise ValueError(f"Column '{target_col}' has zero variance. Please select another target.")

    is_numeric_target = pd.api.types.is_numeric_dtype(y) and y.nunique() > 10
    task_type = "number" if is_numeric_target else "category"

    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(X[c])]

    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_features,
        ))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if task_type == "number":
        estimator = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    else:
        y = y.astype(str)
        estimator = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=(y if task_type == "category" and y.value_counts().min() >= 2 else None)
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Score calculation & strict clamping
    if task_type == "number":
        raw_r2 = r2_score(y_test, y_pred)
        score_pct = max(0.0, min(float(raw_r2) * 100, 100.0))
    else:
        score_pct = float(accuracy_score(y_test, y_pred)) * 100

    score_pct = round(score_pct, 1)

    # Calibrated Thresholds
    if score_pct >= 75.0:
        score_label = "High Confidence (Reliable Pattern)"
        score_color = "#10b981"  # Green
    elif score_pct >= 50.0:
        score_label = "Moderate Confidence (Fair Pattern)"
        score_color = "#f59e0b"  # Yellow/Orange
    else:
        score_label = "Low Confidence (Weak Pattern - Try picking another column to predict)"
        score_color = "#ef4444"  # Red

    # Generate predictions across the full working dataset for business outputs
    all_preds = pipeline.predict(df[feature_cols])
    all_probas = None
    if task_type == "category" and hasattr(pipeline, "predict_proba"):
        try:
            all_probas = pipeline.predict_proba(df[feature_cols])
        except Exception:
            all_probas = None

    # Aggregating drivers for diagnostic transparency
    model = pipeline.named_steps["model"]
    raw_importances = model.feature_importances_
    col_importance_dict: Dict[str, float] = {col: 0.0 for col in feature_cols}

    feature_names = []
    if numeric_features:
        feature_names.extend(numeric_features)
    if categorical_features:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        feature_names.extend(cat_encoder.get_feature_names_out(categorical_features))

    for feat_name, imp in zip(feature_names, raw_importances):
        matched = False
        for orig in categorical_features:
            if feat_name.startswith(f"{orig}_"):
                col_importance_dict[orig] += imp
                matched = True
                break
        if not matched and feat_name in col_importance_dict:
            col_importance_dict[feat_name] += imp

    driver_df = (
        pd.DataFrame(list(col_importance_dict.items()), columns=["Factor", "Impact"])
        .sort_values(by="Impact", ascending=False)
        .head(5)
    )
    if driver_df["Impact"].sum() > 0:
        driver_df["Impact_Pct"] = (driver_df["Impact"] / driver_df["Impact"].sum() * 100).round(1)
    else:
        driver_df["Impact_Pct"] = 20.0

    driver_df = driver_df.sort_values(by="Impact_Pct", ascending=True)

    preview_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred,
    }).head(30).reset_index(drop=True)

    return {
        "pipeline": pipeline,
        "task_type": task_type,
        "score_pct": score_pct,
        "score_label": score_label,
        "score_color": score_color,
        "top_drivers": driver_df,
        "preview_data": preview_df,
        "feature_cols": feature_cols,
        "top_5_cols": driver_df["Factor"].tolist()[::-1],
        "all_preds": all_preds,
        "all_probas": all_probas,
    }


# ==============================================================================
# 5. USER INTERFACE
# ==============================================================================

# Enterprise Header Bar
st.markdown(
    """
    <div class="corp-header">
        <div class="corp-title-wrap">
            <span class="corp-brand-tag">Predictive Analytics Platform</span>
            <h1 class="corp-title">Decision Intelligence Studio</h1>
            <div class="corp-subtitle">Automated multivariate regression, classification, and real-time sensitivity analysis.</div>
        </div>
        <div>
            <span class="corp-status-pill">
                <span class="corp-status-dot"></span>
                ENGINE ACTIVE
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# SECTION 1: DATA INGESTION
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div class="corp-section">
        <div class="corp-section-header">
            <span class="corp-section-idx">01</span>
            <span class="corp-section-title">Data Ingestion & Profiling</span>
        </div>
        <div class="corp-section-desc">Connect an enterprise dataset in Excel or CSV format, or run with standard commercial sample data.</div>
    """,
    unsafe_allow_html=True,
)

col_upload, col_sample = st.columns([3, 1.2])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload dataset (.csv, .xlsx)",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

with col_sample:
    sample_btn = st.button("Load Sample Data", use_container_width=True)

if "active_data" not in st.session_state:
    st.session_state["active_data"] = None

if sample_btn:
    st.session_state["active_data"] = get_sample_business_data()
    st.toast("Sample dataset ingested successfully.", icon="✅")

elif uploaded_file is not None:
    try:
        raw_loaded = load_uploaded_spreadsheet(uploaded_file)
        if raw_loaded.empty or len(raw_loaded.columns) < 2:
            st.error("The uploaded file contains insufficient columns or data rows.")
        else:
            st.session_state["active_data"] = raw_loaded
    except Exception as e:
        st.error(f"Failed to parse file: {e}")

if st.session_state["active_data"] is None:
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem; color: #64748b; font-size: 0.9rem;">
            Awaiting data file. Upload a spreadsheet above or select <b>Load Sample Data</b>.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
raw_df = st.session_state.get("active_data")
if raw_df is None or not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
    st.stop()
    raise SystemExit(0)

clean_df, ignored_ids, detected_dates = clean_data_automatically(raw_df)

# Global Column Filter applied immediately after upload
clean_numeric_cols, clean_category_cols, global_clean_cols, excluded_audit = get_global_filtered_columns(clean_df)
ignored_ids = list(set(ignored_ids).union(set(excluded_audit.keys())))

total_rows = len(clean_df)
total_cols = len(clean_df.columns)
ready_cols = len(global_clean_cols)

# Clean Metric Tiles
st.write("")
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(
        f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Total Observations</div>
            <div class="metric-panel-val">{total_rows:,}</div>
            <div class="metric-panel-note">Validated rows in working memory</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Schema Dimensions</div>
            <div class="metric-panel-val">{ready_cols} Verified Columns</div>
            <div class="metric-panel-note">{len(clean_numeric_cols)} Metrics | {len(clean_category_cols)} Group Categories</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        """
        <div class="metric-panel">
            <div class="metric-panel-label">Data Pipeline Quality</div>
            <div class="metric-panel-val" style="color: #34d399;">Optimal</div>
            <div class="metric-panel-note">Auto-imputation & ID sanitization complete</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
with st.expander("View Uploaded Data (First 100 Rows)"):
    notes = []
    if excluded_audit:
        notes.append(f"Excluded non-predictive/identity columns ({len(excluded_audit)}): {', '.join(sorted(excluded_audit.keys()))}")
    if detected_dates:
        notes.append(f"Date columns reserved for timelines: {', '.join(detected_dates)}")
    if notes:
        st.caption(" | ".join(notes))
    st.dataframe(clean_df.head(100), use_container_width=True)

# ------------------------------------------------------------------------------
# QUICK DATA OVERVIEW (BUSINESS CHARTS & MONTHLY TRENDS)
# ------------------------------------------------------------------------------
st.write("")
st.markdown("<div style='font-size: 1.05rem; font-weight: 600; color: #f8fafc; margin-bottom: 0.2rem;'>Quick Data Overview</div>", unsafe_allow_html=True)
st.caption("Pick any column below to see its breakdown and trend:")

# Only show globally filtered clean business columns
overview_candidates = [c for c in global_clean_cols]
if not overview_candidates:
    overview_candidates = [c for c in clean_df.columns if not is_id_column(c, clean_df[c]) and not is_date_column(c, clean_df[c])]

# Default selected column to first numeric business metric (Sales, Profit, etc.), NEVER text
default_explore_idx = 0
for i, c in enumerate(overview_candidates):
    c_low = c.lower().replace(" ", "_")
    if any(k in c_low for k in ["sales", "revenue", "profit", "total_price", "totalprice", "amount"]):
        default_explore_idx = i
        break
else:
    for i, c in enumerate(overview_candidates):
        if c in clean_numeric_cols:
            default_explore_idx = i
            break

explore_col = st.selectbox("Select Column to View:", overview_candidates, index=default_explore_idx)

if explore_col:
    series = clean_df[explore_col]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    is_num = pd.api.types.is_numeric_dtype(series) and series.nunique() > 5

    if is_num:
        # Detect metric properties: currency vs rate/percentage vs volume
        is_rate = any(k in explore_col.lower() for k in ["discount", "rate", "pct", "percent", "margin", "ratio", "score"])
        is_currency = any(k in explore_col.lower() for k in ["sales", "revenue", "price", "profit", "spend", "cost", "budget", "amount"])
        is_decimal_rate = is_rate and series.max() <= 1.0

        # Best category from clean categories only (Category, Sub-Category, Segment, Region, etc.)
        cat_candidates = [c for c in clean_category_cols if c != explore_col]
        priority_cat_terms = ["category", "sub_category", "sub-category", "region", "segment", "product_line", "product", "channel", "department", "type"]
        best_cat = None
        for term in priority_cat_terms:
            for c in cat_candidates:
                if term in c.lower().replace(" ", "_"):
                    best_cat = c
                    break
            if best_cat:
                break
        if not best_cat and cat_candidates:
            best_cat = cat_candidates[0]

        # Calculate high-level summary KPIs
        if is_rate:
            avg_val = series.mean()
            stat_1_val = f"{avg_val * 100:.1f}%" if is_decimal_rate else f"{avg_val:.1f}%"
            stat_1_lbl = f"Average {explore_col}"
            stat_2_val = f"{series.min() * (100 if is_decimal_rate else 1):.0f}% – {series.max() * (100 if is_decimal_rate else 1):.0f}%"
            stat_2_lbl = "Observed Range"
        else:
            tot_val = series.sum()
            stat_1_val = f"${tot_val:,.0f}" if is_currency else f"{tot_val:,.0f}"
            stat_1_lbl = f"Total {explore_col}"
            avg_val = series.mean()
            stat_2_val = f"${avg_val:,.2f}" if is_currency else f"{avg_val:,.2f}"
            stat_2_lbl = "Average per Record"

        top_leader_str = "N/A"
        if best_cat:
            top_cat_agg = clean_df.groupby(best_cat)[explore_col].agg("mean" if is_rate else "sum").sort_values(ascending=False)
            if not top_cat_agg.empty:
                leader_name = str(top_cat_agg.index[0])
                leader_num = top_cat_agg.iloc[0]
                if is_rate:
                    fmt_leader = f"{leader_num * 100:.1f}%" if is_decimal_rate else f"{leader_num:.1f}%"
                elif is_currency:
                    fmt_leader = f"${leader_num:,.0f}"
                else:
                    fmt_leader = f"{leader_num:,.0f}"
                top_leader_str = f"{leader_name} ({fmt_leader})"

        # Clean KPI Summary Badges
        st.markdown(
            f"""
            <div style="display: flex; gap: 12px; margin: 0.85rem 0 1.25rem 0; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 170px; background: #131d31; border: 1px solid #1e293b; border-radius: 8px; padding: 0.75rem 1rem;">
                    <div style="font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: #94a3b8; text-transform: uppercase;">{stat_1_lbl}</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #f8fafc; margin-top: 2px;">{stat_1_val}</div>
                </div>
                <div style="flex: 1; min-width: 170px; background: #131d31; border: 1px solid #1e293b; border-radius: 8px; padding: 0.75rem 1rem;">
                    <div style="font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: #94a3b8; text-transform: uppercase;">{stat_2_lbl}</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #60a5fa; margin-top: 2px;">{stat_2_val}</div>
                </div>
                <div style="flex: 1.4; min-width: 220px; background: #131d31; border: 1px solid #1e293b; border-radius: 8px; padding: 0.75rem 1rem;">
                    <div style="font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: #94a3b8; text-transform: uppercase;">Top {best_cat or 'Group'} Leader</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #34d399; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{top_leader_str}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # TWO side-by-side charts
        col_left, col_right = st.columns(2)

        with col_left:
            if best_cat:
                agg_type = "mean" if is_rate else "sum"
                cat_summary = clean_df.groupby(best_cat)[explore_col].agg(agg_type).reset_index()
                cat_summary = cat_summary.sort_values(by=explore_col, ascending=False).head(10)

                chart_title = f"{'Average' if is_rate else 'Total'} {explore_col} by {best_cat} (Top 10)"

                fig_cat_breakdown = px.bar(
                    cat_summary,
                    y=best_cat,
                    x=explore_col,
                    orientation="h",
                    color_discrete_sequence=["#3b82f6"],
                    template="plotly_dark",
                )
                fig_cat_breakdown.update_layout(
                    title=dict(text=chart_title, font=dict(size=13, color="#f1f5f9")),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=20, t=35, b=20),
                    height=280,
                    yaxis=dict(showgrid=False, title="", autorange="reversed"),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.06)",
                        title=f"{'Average' if is_rate else 'Total'} {explore_col}",
                        tickformat=".0%" if is_decimal_rate else None,
                    ),
                )
                if is_rate:
                    if is_decimal_rate:
                        fig_cat_breakdown.update_traces(hovertemplate=f"<b>%{{y}}</b><br>Average {explore_col}: %{{x:.1%}}<extra></extra>")
                    else:
                        fig_cat_breakdown.update_traces(hovertemplate=f"<b>%{{y}}</b><br>Average {explore_col}: %{{x:.1f}}%<extra></extra>")
                elif is_currency:
                    fig_cat_breakdown.update_traces(hovertemplate=f"<b>%{{y}}</b><br>Total {explore_col}: $%{{x:,.2f}}<extra></extra>")
                else:
                    fig_cat_breakdown.update_traces(hovertemplate=f"<b>%{{y}}</b><br>Total {explore_col}: %{{x:,.0f}}<extra></extra>")
                st.plotly_chart(fig_cat_breakdown, use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown(
                    f"""
                    <div style="background: #131d31; border: 1px solid #1e293b; border-radius: 8px; padding: 1.25rem; height: 280px; display: flex; flex-direction: column; justify-content: center;">
                        <div style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #94a3b8; text-transform: uppercase;">Statistical Distribution</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-top: 4px;">{explore_col} Summary</div>
                        <div style="margin-top: 14px; font-size: 0.88rem; color: #cbd5e1; line-height: 1.9;">
                            • <b>Mean:</b> {series.mean():,.2f}<br>
                            • <b>Median:</b> {series.median():,.2f}<br>
                            • <b>Std Dev:</b> {series.std():,.2f}<br>
                            • <b>Min / Max:</b> {series.min():,.2f} / {series.max():,.2f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_right:
            # Check for Date column for Monthly Trend
            date_cols = [c for c in raw_df.columns if is_date_column(c, raw_df[c])]
            has_date_trend = False

            if date_cols:
                best_date = next((c for c in date_cols if "order" in c.lower()), date_cols[0])
                parsed_dt = pd.to_datetime(raw_df[best_date], errors="coerce")
                valid_mask = parsed_dt.notna() & clean_df[explore_col].notna()

                if valid_mask.sum() > 5:
                    trend_df = pd.DataFrame({
                        "Month": parsed_dt[valid_mask].dt.to_period("M").dt.to_timestamp(),
                        explore_col: clean_df.loc[valid_mask, explore_col],
                    })
                    agg_fn = "mean" if is_rate else "sum"
                    monthly_trend = trend_df.groupby("Month")[explore_col].agg(agg_fn).reset_index().sort_values("Month")

                    if len(monthly_trend) >= 2:
                        has_date_trend = True
                        fig_trend = px.line(
                            monthly_trend,
                            x="Month",
                            y=explore_col,
                            markers=True,
                            template="plotly_dark",
                        )
                        fig_trend.update_traces(
                            line=dict(color="#10b981", width=2.5),
                            marker=dict(size=6, color="#34d399"),
                            fill="tozeroy",
                            fillcolor="rgba(16, 185, 129, 0.08)",
                        )
                        if is_rate:
                            if is_decimal_rate:
                                fig_trend.update_traces(hovertemplate=f"<b>%{{x|%b %Y}}</b><br>Average {explore_col}: %{{y:.1%}}<extra></extra>")
                            else:
                                fig_trend.update_traces(hovertemplate=f"<b>%{{x|%b %Y}}</b><br>Average {explore_col}: %{{y:.1f}}%<extra></extra>")
                        elif is_currency:
                            fig_trend.update_traces(hovertemplate=f"<b>%{{x|%b %Y}}</b><br>Total {explore_col}: $%{{y:,.2f}}<extra></extra>")
                        else:
                            fig_trend.update_traces(hovertemplate=f"<b>%{{x|%b %Y}}</b><br>Total {explore_col}: %{{y:,.0f}}<extra></extra>")

                        fig_trend.update_layout(
                            title=dict(text=f"Monthly Trend: {'Average' if is_rate else 'Total'} {explore_col} ({best_date})", font=dict(size=13, color="#f1f5f9")),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=10, r=20, t=35, b=20),
                            height=280,
                            xaxis=dict(showgrid=False, title=""),
                            yaxis=dict(
                                showgrid=True,
                                gridcolor="rgba(255,255,255,0.06)",
                                title="",
                                tickformat=".0%" if is_decimal_rate else None,
                            ),
                        )
                        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

            if not has_date_trend:
                # If no date column exists, show Average per Category
                if best_cat:
                    avg_summary = clean_df.groupby(best_cat)[explore_col].mean().reset_index()
                    avg_summary = avg_summary.sort_values(by=explore_col, ascending=False).head(10)
                    fig_avg = px.bar(
                        avg_summary,
                        y=best_cat,
                        x=explore_col,
                        orientation="h",
                        color_discrete_sequence=["#10b981"],
                        template="plotly_dark",
                    )
                    fig_avg.update_layout(
                        title=dict(text=f"Average {explore_col} by {best_cat} (Top 10)", font=dict(size=13, color="#f1f5f9")),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=20, t=35, b=20),
                        height=280,
                        yaxis=dict(showgrid=False, title="", autorange="reversed"),
                        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title=f"Average {explore_col}", tickformat=".0%" if is_decimal_rate else None),
                    )
                    if is_rate:
                        fig_avg.update_traces(hovertemplate=f"<b>%{{y}}</b><br>Average {explore_col}: %{{x:.1%}}<extra></extra>" if is_decimal_rate else f"<b>%{{y}}</b><br>Average {explore_col}: %{{x:.1f}}%<extra></extra>")
                    elif is_currency:
                        fig_avg.update_traces(hovertemplate=f"<b>%{{y}}</b><br>Average {explore_col}: $%{{x:,.2f}}<extra></extra>")
                    else:
                        fig_avg.update_traces(hovertemplate=f"<b>%{{y}}</b><br>Average {explore_col}: %{{x:,.2f}}<extra></extra>")
                    st.plotly_chart(fig_avg, use_container_width=True, config={"displayModeBar": False})

    else:
        # Category / Text Column: Show clean Bar Chart of Top 10 most common items
        is_cal, cal_order = detect_calendar_order(series.unique())
        if is_cal and cal_order:
            top_counts = series.value_counts().reindex(cal_order).dropna().reset_index()
            top_counts.columns = [explore_col, "Count"]
        else:
            top_counts = series.value_counts().head(10).reset_index()
            top_counts.columns = [explore_col, "Count"]

        total_cnt = len(series)
        top_counts["Share"] = (top_counts["Count"] / total_cnt * 100).round(1)
        top_val = str(top_counts.iloc[0][explore_col]) if not top_counts.empty else "N/A"
        top_share = f"{top_counts.iloc[0]['Share']}%" if not top_counts.empty else "0%"

        # Metric Badges for Categorical Column
        st.markdown(
            f"""
            <div style="display: flex; gap: 12px; margin: 0.85rem 0 1.25rem 0; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 170px; background: #131d31; border: 1px solid #1e293b; border-radius: 8px; padding: 0.75rem 1rem;">
                    <div style="font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: #94a3b8; text-transform: uppercase;">Total Records</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #f8fafc; margin-top: 2px;">{total_cnt:,}</div>
                </div>
                <div style="flex: 1; min-width: 170px; background: #131d31; border: 1px solid #1e293b; border-radius: 8px; padding: 0.75rem 1rem;">
                    <div style="font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: #94a3b8; text-transform: uppercase;">Distinct Values</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #60a5fa; margin-top: 2px;">{series.nunique():,}</div>
                </div>
                <div style="flex: 1.4; min-width: 220px; background: #131d31; border: 1px solid #1e293b; border-radius: 8px; padding: 0.75rem 1rem;">
                    <div style="font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: #94a3b8; text-transform: uppercase;">#1 Dominant Group</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #34d399; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{top_val} ({top_share})</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_cat = px.bar(
            top_counts,
            y=explore_col,
            x="Count",
            orientation="h",
            color_discrete_sequence=["#3b82f6"],
            template="plotly_dark",
        )
        fig_cat.update_layout(
            title=dict(text=f"Top 10 Most Common in '{explore_col}'", font=dict(size=13, color="#f1f5f9")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=20, t=35, b=20),
            height=max(240, len(top_counts) * 26),
            yaxis=dict(showgrid=False, title="", autorange="reversed"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Record Count"),
        )
        fig_cat.update_traces(hovertemplate=f"<b>%{{y}}</b><br>Count: %{{x:,}}<extra></extra>")
        st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})

st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# SECTION 2: TARGET SELECTION & VALUE-BASED PREDICTIONS
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div class="corp-section">
        <div class="corp-section-header">
            <span class="corp-section-idx">02</span>
            <span class="corp-section-title">Predictive Intelligence & Value Insights</span>
        </div>
        <div class="corp-section-desc">Select the target metric or category. The model evaluates actual data values and forecasts performance outcomes.</div>
    """,
    unsafe_allow_html=True,
)

# Target selection: strictly use globally filtered clean business columns
eligible_targets = [c for c in global_clean_cols]
if not eligible_targets:
    eligible_targets = [
        c for c in clean_df.columns
        if not is_id_column(c, clean_df[c])
        and not is_date_column(c, clean_df[c])
        and not c.endswith("_Month")
        and not c.endswith("_DayOfWeek")
    ]

def get_default_target_column(candidates: List[str], df: pd.DataFrame) -> int:
    """Identify optimal default target column, prioritizing TotalPrice / Sales / Revenue and strictly avoiding UnitPrice."""
    def target_priority_score(col: str) -> int:
        c_clean = col.lower().replace(" ", "_").replace("-", "_")
        is_num = pd.api.types.is_numeric_dtype(df[col])
        is_unit = "unit" in c_clean or "per_unit" in c_clean

        # Strict rule: NEVER default to UnitPrice
        if is_unit:
            return 9999

        # Priority 1: Explicit Total Price / Sales / Revenue / Total Sales / Total Amount / Profit
        for i, term in enumerate([
            "total_price", "totalprice", "sales", "revenue", "total_sales",
            "totalsales", "total_revenue", "total_amount", "totalamount",
            "gross_sales", "turnover", "profit"
        ]):
            if term == c_clean or term in c_clean:
                return 10 + i

        # Priority 2: Generic 'total' or 'amount' (numeric only)
        if is_num and any(term in c_clean for term in ["total", "amount"]):
            return 50

        # Priority 3: Other business numeric targets (spend, budget, quantity)
        if is_num:
            for i, term in enumerate(["spend", "budget", "cost", "quantity"]):
                if term in c_clean:
                    return 100 + i

        # Priority 4: Any non-unit numeric column
        if is_num and df[col].nunique() > 1:
            return 200

        # Priority 5: High-level categories (Product, Category, Status)
        for i, term in enumerate(["category", "product", "status"]):
            if term in c_clean:
                return 300 + i

        return 500

    scores = [(target_priority_score(col), idx) for idx, col in enumerate(candidates)]
    scores.sort(key=lambda x: x[0])
    return scores[0][1] if scores else 0

default_target_idx = get_default_target_column(eligible_targets, clean_df)

target_column = st.selectbox(
    "What do you want to predict?",
    options=eligible_targets,
    index=default_target_idx,
)

with st.spinner("Calibrating predictive pipeline..."):
    try:
        model_results = train_prediction_model(clean_df, target_column, ignored_ids, detected_dates, clean_candidate_pool=global_clean_cols)
    except Exception as e:
        st.error(f"Estimation failure: {e}")
        st.stop()

task_type = model_results["task_type"]
score_pct = model_results["score_pct"]
score_label = model_results["score_label"]
score_color = model_results["score_color"]
all_preds = model_results["all_preds"]
pipeline = model_results["pipeline"]

st.write("")

# ==============================================================================
# CASE 1: CATEGORY / TEXT COLUMN SELECTED (e.g. Product, Category, City, Status)
# ==============================================================================
if task_type == "category":
    # 1. Calculate actual demand share and predicted volume for items inside this column
    if model_results["all_probas"] is not None and hasattr(pipeline, "classes_"):
        classes = pipeline.classes_
        mean_shares = model_results["all_probas"].mean(axis=0) * 100
    else:
        classes = np.unique(all_preds)
        val_counts = pd.Series(all_preds).value_counts(normalize=True)
        mean_shares = [val_counts.get(c, 0) * 100 for c in classes]

    pred_counts = pd.Series(all_preds).value_counts()
    item_stats = []
    for idx, cls in enumerate(classes):
        share_val = float(mean_shares[idx])
        vol_val = int(pred_counts.get(cls, 0))
        item_stats.append({
            "Item": str(cls),
            "Share_Pct": round(share_val, 1),
            "Predicted_Volume": vol_val,
        })
    cat_ranking_df = pd.DataFrame(item_stats).sort_values(by="Share_Pct", ascending=False).head(10)

    top_row = cat_ranking_df.iloc[0]
    top_item_name = top_row["Item"]
    top_item_share = top_row["Share_Pct"]
    top_item_vol = top_row["Predicted_Volume"]

    # Top Cards: Confidence + #1 Top Performer Card
    col_perf, col_top = st.columns([1, 1.8])

    with col_perf:
        st.markdown(
            f"""
            <div class="metric-panel" style="text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                <div class="metric-panel-label">Validation Confidence</div>
                <div style="font-size: 2.8rem; font-weight: 700; color: {score_color}; margin: 0.2rem 0; line-height: 1;">
                    {score_pct}%
                </div>
                <div style="font-size: 0.88rem; font-weight: 600; color: #f8fafc; margin-bottom: 0.5rem;">{score_label}</div>
                <div style="font-size: 0.75rem; color: #64748b;">
                    Pipeline Architecture: <span style="color: #cbd5e1;">Multi-Class Classifier</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_top:
        st.markdown(
            f"""
            <div class="performer-card">
                <div class="performer-badge">#1 Top Performer (Projected)</div>
                <div class="performer-title">{top_item_name}</div>
                <div class="performer-stats">
                    <div>Projected Demand Share: <b style="color: #38bdf8; font-size: 1.05rem;">{top_item_share}%</b></div>
                    <div style="color: #475569;">•</div>
                    <div>Projected Volume: <b style="color: #34d399; font-size: 1.05rem;">{top_item_vol:,} orders</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Actual Values Bar Chart
    st.write("")
    st.markdown(f"<div style='font-size: 0.95rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.2rem;'>Predicted Demand & Expected Share (%) by {target_column}</div>", unsafe_allow_html=True)
    st.caption(f"Top specific items inside '{target_column}' ranked by projected demand share:")

    is_cal, cal_order = detect_calendar_order(cat_ranking_df["Item"])
    if is_cal and cal_order:
        cat_ranking_df = cat_ranking_df[cat_ranking_df["Item"].isin(cal_order)]
        cat_ranking_df["__ord__"] = cat_ranking_df["Item"].map(lambda x: cal_order.index(x))
        cat_ranking_df = cat_ranking_df.sort_values(by="__ord__", ascending=True).drop(columns=["__ord__"])
        chart_cat_df = cat_ranking_df.iloc[::-1]
        y_axis_layout = dict(showgrid=False, title="", categoryorder="array", categoryarray=cal_order[::-1])
    else:
        chart_cat_df = cat_ranking_df.sort_values(by="Share_Pct", ascending=True)
        y_axis_layout = dict(showgrid=False, title="")

    max_share = float(chart_cat_df["Share_Pct"].max()) if not chart_cat_df.empty else 10.0
    x_upper = max(max_share * 1.35, 15.0)

    fig_cat = px.bar(
        chart_cat_df,
        x="Share_Pct",
        y="Item",
        orientation="h",
        text=chart_cat_df.apply(lambda r: f"{r['Share_Pct']}% ({r['Predicted_Volume']:,} orders)", axis=1),
        color_discrete_sequence=["#2563eb"],
        template="plotly_dark",
    )
    fig_cat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=90, t=10, b=10),
        height=max(220, len(chart_cat_df) * 38),
        xaxis=dict(range=[0, x_upper], showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Expected Demand Share (%)"),
        yaxis=y_axis_layout,
    )
    fig_cat.update_traces(textposition="outside", marker_line_width=0)
    st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})

# ==============================================================================
# CASE 2: NUMBER COLUMN SELECTED (e.g. Sales, Price, Quantity, Revenue)
# ==============================================================================
else:
    preds = np.array(all_preds)
    if clean_df[target_column].min() >= 0:
        preds = np.clip(preds, 0, None)

    pred_total = float(np.sum(preds))
    pred_avg = float(np.mean(preds))

    is_currency = any(k in target_column.lower() for k in ["sales", "revenue", "price", "spend", "cost", "profit", "budget", "amount"])
    prefix = "$" if is_currency else ""
    formatted_total = f"{prefix}{pred_total:,.2f}" if pred_total < 1e7 else f"{prefix}{pred_total:,.0f}"
    formatted_avg = f"{prefix}{pred_avg:,.2f}"

    col_perf, col_summary = st.columns([1, 1.8])

    with col_perf:
        st.markdown(
            f"""
            <div class="metric-panel" style="text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                <div class="metric-panel-label">Validation Confidence</div>
                <div style="font-size: 2.8rem; font-weight: 700; color: {score_color}; margin: 0.2rem 0; line-height: 1;">
                    {score_pct}%
                </div>
                <div style="font-size: 0.88rem; font-weight: 600; color: #f8fafc; margin-bottom: 0.5rem;">{score_label}</div>
                <div style="font-size: 0.75rem; color: #64748b;">
                    Pipeline Architecture: <span style="color: #cbd5e1;">Continuous Regressor</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_summary:
        st.markdown(
            f"""
            <div class="summary-hero-card">
                <div class="summary-hero-badge">Predicted Future Forecast</div>
                <div style="display: grid; grid-template-columns: 1.25fr 1fr; gap: 1.25rem; align-items: center;">
                    <div>
                        <div style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #a7f3d0; text-transform: uppercase;">Predicted Future Total</div>
                        <div style="font-size: 2.1rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em; margin-top: 2px; line-height: 1.1;">
                            {formatted_total}
                        </div>
                        <div style="font-size: 0.72rem; color: #6ee7b7; margin-top: 4px;">Model aggregate across all observations</div>
                    </div>
                    <div style="border-left: 1px solid rgba(255, 255, 255, 0.12); padding-left: 1.25rem;">
                        <div style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #93c5fd; text-transform: uppercase;">Predicted Future Average</div>
                        <div style="font-size: 1.65rem; font-weight: 700; color: #60a5fa; letter-spacing: -0.01em; margin-top: 2px; line-height: 1.1;">
                            {formatted_avg}
                        </div>
                        <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 4px;">Per observation unit</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Group Prediction strictly by clean business categories (Category, Sub-Category, Region, Segment, etc.)
    cat_group_candidates = [c for c in clean_category_cols if c != target_column]
    if not cat_group_candidates:
        cat_group_candidates = [
            c for c in global_clean_cols
            if c != target_column and not pd.api.types.is_numeric_dtype(clean_df[c])
        ]

    # Smart default grouping (Category, Sub-Category, Product, Region, Segment)
    group_pref_keywords = ["category", "sub_category", "sub-category", "product", "item", "segment", "region", "channel", "tier", "type"]
    default_group_idx = 0
    for idx, c in enumerate(cat_group_candidates):
        if any(k in c.lower() for k in group_pref_keywords):
            default_group_idx = idx
            break

    st.write("")
    col_grp_sel, _ = st.columns([1.5, 1])
    with col_grp_sel:
        group_by_col = st.selectbox(
            f"Group Predicted {target_column} by:",
            options=cat_group_candidates,
            index=default_group_idx,
            help="Choose a category to break down predicted values across its actual items."
        )

    temp_pred_df = pd.DataFrame({
        "Category_Item": clean_df[group_by_col].astype(str),
        "Predicted_Val": preds,
    })
    grouped_df = temp_pred_df.groupby("Category_Item")["Predicted_Val"].agg(
        Total="sum",
        Average="mean",
        Count="count"
    ).reset_index()
    grouped_df = grouped_df.sort_values(by="Total", ascending=False).head(10)
    grouped_df["Share_Pct"] = (grouped_df["Total"] / max(pred_total, 1) * 100).round(1)

    top_group_row = grouped_df.iloc[0]
    top_group_name = top_group_row["Category_Item"]
    top_group_total = top_group_row["Total"]
    top_group_share = top_group_row["Share_Pct"]

    # Top Contributor Highlight
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; background: #131d31; border: 1px solid #1e293b; border-left: 3px solid #3b82f6; border-radius: 6px; padding: 0.65rem 1.1rem; margin-bottom: 0.75rem;">
            <div>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">#1 Top Contributor ({group_by_col}):</span>
                <span style="font-weight: 700; color: #ffffff; margin-left: 6px; font-size: 0.95rem;">{top_group_name}</span>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.88rem;">
                <span style="color: #34d399; font-weight: 700;">{prefix}{top_group_total:,.2f}</span>
                <span style="color: #64748b; margin-left: 5px;">({top_group_share}% of predicted {target_column})</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    is_cal_grp, cal_order_grp = detect_calendar_order(grouped_df["Category_Item"])
    if is_cal_grp and cal_order_grp:
        grouped_df = grouped_df[grouped_df["Category_Item"].isin(cal_order_grp)]
        grouped_df["__ord__"] = grouped_df["Category_Item"].map(lambda x: cal_order_grp.index(x))
        grouped_df = grouped_df.sort_values(by="__ord__", ascending=True).drop(columns=["__ord__"])
        chart_num_df = grouped_df.iloc[::-1]
        y_axis_layout_num = dict(showgrid=False, title="", categoryorder="array", categoryarray=cal_order_grp[::-1])
    else:
        chart_num_df = grouped_df.sort_values(by="Total", ascending=True)
        y_axis_layout_num = dict(showgrid=False, title="")

    max_total = float(chart_num_df["Total"].max()) if not chart_num_df.empty else 100.0
    x_upper = max(max_total * 1.35, 10.0)

    fig_num = px.bar(
        chart_num_df,
        x="Total",
        y="Category_Item",
        orientation="h",
        text=chart_num_df.apply(lambda r: f"{prefix}{r['Total']:,.0f} ({r['Share_Pct']}%)", axis=1),
        color_discrete_sequence=["#2563eb"],
        template="plotly_dark",
    )
    fig_num.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=90, t=10, b=10),
        height=max(220, len(chart_num_df) * 38),
        xaxis=dict(range=[0, x_upper], showgrid=True, gridcolor="rgba(255,255,255,0.06)", title=f"Predicted Total {target_column}"),
        yaxis=y_axis_layout_num,
    )
    fig_num.update_traces(textposition="outside", marker_line_width=0)
    st.plotly_chart(fig_num, use_container_width=True, config={"displayModeBar": False})

# ==============================================================================
# TIMELINE TREND ANALYSIS (Requirement 3: Use Date columns ONLY for timeline trends)
# ==============================================================================
primary_date_col = next((c for c in detected_dates if c in raw_df.columns), None)
if primary_date_col is not None:
    try:
        parsed_dates = pd.to_datetime(raw_df[primary_date_col], errors="coerce")
        valid_date_mask = parsed_dates.notna()
        if valid_date_mask.sum() >= 5:
            st.write("")
            st.markdown(f"<div style='font-size: 0.95rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.2rem;'>Timeline Trend: {target_column} over Time ({primary_date_col})</div>", unsafe_allow_html=True)
            st.caption("Chronological trajectory tracking trend patterns over time:")

            time_df = pd.DataFrame({
                "Date": parsed_dates[valid_date_mask],
                "Actual": clean_df.loc[valid_date_mask, target_column] if task_type == "number" else 1,
                "Predicted": preds[valid_date_mask] if task_type == "number" else 1,
            })
            span_days = (time_df["Date"].max() - time_df["Date"].min()).days
            if span_days > 90:
                time_df["Period"] = time_df["Date"].dt.to_period("M").dt.to_timestamp()
            elif span_days > 28:
                time_df["Period"] = time_df["Date"].dt.to_period("W").dt.to_timestamp()
            else:
                time_df["Period"] = time_df["Date"].dt.floor("D")

            if task_type == "number":
                agg_time = time_df.groupby("Period")[["Actual", "Predicted"]].mean().reset_index()
                fig_time = go.Figure()
                fig_time.add_trace(go.Scatter(
                    x=agg_time["Period"],
                    y=agg_time["Actual"],
                    mode="lines+markers",
                    name=f"Actual {target_column}",
                    line=dict(color="#64748b", width=2),
                    marker=dict(size=5),
                ))
                fig_time.add_trace(go.Scatter(
                    x=agg_time["Period"],
                    y=agg_time["Predicted"],
                    mode="lines+markers",
                    name=f"Predicted {target_column}",
                    line=dict(color="#3b82f6", width=2, dash="dash"),
                    marker=dict(size=5),
                ))
                fig_time.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=220,
                    template="plotly_dark",
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Timeline"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title=target_column),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_time, use_container_width=True, config={"displayModeBar": False})
            else:
                agg_time = time_df.groupby("Period")["Date"].count().reset_index().rename(columns={"Date": "Transactions"})
                fig_time = px.line(
                    agg_time,
                    x="Period",
                    y="Transactions",
                    color_discrete_sequence=["#3b82f6"],
                    template="plotly_dark",
                )
                fig_time.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=10, b=20),
                    height=190,
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Timeline"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Transaction Volume"),
                )
                st.plotly_chart(fig_time, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        pass

# ==============================================================================
# DIAGNOSTIC INSPECTION (COLLAPSIBLE FOR TECHNICAL AUDITING)
# ==============================================================================
st.write("")
with st.expander("Diagnostic Inspection: Model Mechanics & Underlying Drivers"):
    col_d1, col_d2 = st.columns([1, 1.2])

    with col_d1:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #cbd5e1;'>Relative Feature Weights</div>", unsafe_allow_html=True)
        st.caption("Underlying mathematical weights assigned by the Random Forest model:")
        driver_df = model_results["top_drivers"]
        if not driver_df.empty:
            fig_drv = px.bar(
                driver_df,
                x="Impact_Pct",
                y="Factor",
                orientation="h",
                text=driver_df["Impact_Pct"].apply(lambda v: f"{v}%"),
                color_discrete_sequence=["#475569"],
                template="plotly_dark",
            )
            fig_drv.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=50, t=10, b=10),
                height=180,
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Impact (%)"),
                yaxis=dict(showgrid=False, title=""),
            )
            fig_drv.update_traces(textposition="outside", marker_line_width=0)
            st.plotly_chart(fig_drv, use_container_width=True, config={"displayModeBar": False})

    with col_d2:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #cbd5e1;'>Hold-Out Test Fit</div>", unsafe_allow_html=True)
        st.caption("Out-of-sample validation samples comparing actuals to predictions:")
        preview_df = model_results["preview_data"]
        if task_type == "number":
            fig_tst = go.Figure()
            fig_tst.add_trace(go.Scatter(y=preview_df["Actual"], mode="lines+markers", name="Actual", line=dict(color="#64748b", width=1.5)))
            fig_tst.add_trace(go.Scatter(y=preview_df["Predicted"], mode="lines+markers", name="Predicted", line=dict(color="#3b82f6", width=1.5, dash="dash")))
            fig_tst.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=180,
                template="plotly_dark",
                xaxis=dict(showgrid=False, title="Sample #"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_tst, use_container_width=True, config={"displayModeBar": False})
        else:
            matches = (preview_df["Actual"].astype(str) == preview_df["Predicted"].astype(str))
            preview_df["Status"] = np.where(matches, "Class Match", "Misclassification")
            fig_tst = px.histogram(
                preview_df,
                x="Actual",
                color="Status",
                barmode="group",
                color_discrete_map={"Class Match": "#10b981", "Misclassification": "#64748b"},
                template="plotly_dark",
            )
            fig_tst.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=180,
                xaxis=dict(showgrid=False, title=""),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Count"),
            )
            st.plotly_chart(fig_tst, use_container_width=True, config={"displayModeBar": False})

st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# SECTION 3: WHAT-IF CALCULATOR & BATCH PREDICTIONS
# ------------------------------------------------------------------------------
def select_calculator_columns(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    top_drivers: List[str],
    global_cols: List[str] | None = None,
) -> List[str]:
    """Select 3 to 5 intuitive business controls for the calculator from globally clean columns.
    Always includes important category dropdowns (Product/Category) and never Customer/Order/ID columns.
    If target is UnitPrice, strictly excludes TotalPrice / Sales / Revenue to prevent backwards logic."""
    forbidden_terms = [
        "id", "cust", "customer", "order", "name", "email", "phone",
        "code", "row", "postal", "zip", "transaction", "invoice", "timestamp"
    ]
    # If predicting UnitPrice, exclude TotalPrice / Sales / Revenue
    target_clean = target_col.lower().replace(" ", "_")
    if "unit" in target_clean or "per_unit" in target_clean:
        forbidden_terms.extend(["total", "sales", "revenue", "profit", "amount", "gross"])

    pool = global_cols if global_cols else feature_cols
    valid_candidates = []
    for c in pool:
        if c == target_col:
            continue
        c_low = c.lower().replace(" ", "_")
        if any(term in c_low for term in forbidden_terms):
            continue
        if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype):
            if df[c].nunique() > 20 or df[c].nunique() <= 1:
                continue
        valid_candidates.append(c)

    # 1. Priority categories: Product, Category, Sub-Category, Segment, Region, etc.
    cat_priority = ["product", "category", "sub_category", "sub-category", "item", "type", "segment", "region", "channel", "brand", "model", "tier"]
    selected_cats = []
    for term in cat_priority:
        for c in valid_candidates:
            if not pd.api.types.is_numeric_dtype(df[c]):
                if term in c.lower().replace(" ", "_") and c not in selected_cats:
                    selected_cats.append(c)
                    break
        if len(selected_cats) >= 2:
            break

    if not selected_cats:
        for c in valid_candidates:
            if not pd.api.types.is_numeric_dtype(df[c]):
                selected_cats.append(c)
                if len(selected_cats) >= 1:
                    break

    # 2. Priority numeric inputs: Quantity, Units, Discount, Price (if predicting Total/Sales), Spend, Budget
    num_priority = ["price", "unit_price", "quantity", "units", "discount", "budget", "spend", "cost"]
    selected_nums = []
    for term in num_priority:
        for c in valid_candidates:
            if pd.api.types.is_numeric_dtype(df[c]):
                if term in c.lower().replace(" ", "_") and c not in selected_nums:
                    selected_nums.append(c)

    # 3. Add top drivers from the model if space allows
    for d in top_drivers:
        if d in valid_candidates and d not in selected_cats and d not in selected_nums:
            if pd.api.types.is_numeric_dtype(df[d]):
                selected_nums.append(d)
            elif len(selected_cats) < 2:
                selected_cats.append(d)

    # Final combined list (categories first, then numeric sliders, max 5)
    final_cols = selected_cats[:2] + selected_nums
    final_cols = final_cols[:5]

    if not final_cols:
        final_cols = valid_candidates[:4]

    return final_cols

st.markdown(
    """
    <div class="corp-section">
        <div class="corp-section-header">
            <span class="corp-section-idx">03</span>
            <span class="corp-section-title">Price & Sales Calculator (What-If Tool)</span>
        </div>
        <div class="corp-section-desc">Change the values below to see how they affect your predicted result in real time.</div>
    """,
    unsafe_allow_html=True,
)

top_drivers = model_results["top_5_cols"]
feature_cols = model_results["feature_cols"]
pipeline = model_results["pipeline"]

calc_input_cols = select_calculator_columns(clean_df, target_column, feature_cols, top_drivers, global_clean_cols)
calc_cols = st.columns(max(1, min(len(calc_input_cols), 5)))
user_inputs: Dict[str, Any] = {}

for idx, col_name in enumerate(calc_input_cols):
    with calc_cols[idx % len(calc_cols)]:
        col_series = clean_df[col_name]
        is_num = pd.api.types.is_numeric_dtype(col_series)
        col_lower = col_name.lower().replace(" ", "_")

        if not is_num:
            # Category Dropdown (Product, Category, Segment, Region, etc.)
            options = sorted([str(x) for x in col_series.dropna().unique().tolist()])
            mode_val = str(col_series.mode().iloc[0]) if len(col_series.mode()) > 0 else options[0]
            default_idx = options.index(mode_val) if mode_val in options else 0
            user_inputs[col_name] = st.selectbox(
                col_name,
                options=options,
                index=default_idx,
            )
        else:
            min_raw = float(col_series.min())
            max_raw = float(col_series.max())
            mean_raw = float(col_series.mean())
            median_raw = float(col_series.median())

            is_count_col = any(k in col_lower for k in ["quantity", "unit", "count", "num", "item"]) or (
                col_series.dtype in [int, np.int64, np.int32] or np.all(col_series.dropna() % 1 == 0)
            )
            is_discount_col = any(k in col_lower for k in ["discount", "pct", "percent", "rate"])

            if is_discount_col:
                # Format Discount Slider clearly (e.g. 7% or 0.07 (7%))
                if max_raw <= 1.0:
                    # Data stored as decimal (0.00 to 1.00)
                    cur_pct = int(round(median_raw * 100))
                    max_pct = max(int(round(max_raw * 100)), 50)
                    min_pct = int(round(min_raw * 100))
                    chosen_pct = st.slider(
                        f"{col_name}",
                        min_value=min_pct,
                        max_value=max_pct,
                        value=cur_pct,
                        step=1,
                        format="%d%%",
                    )
                    user_inputs[col_name] = chosen_pct / 100.0
                    st.caption(f"Active: **{chosen_pct}%** ({chosen_pct / 100:.2f})")
                else:
                    # Data stored as percentage integers (0 to 100)
                    cur_pct = int(round(median_raw))
                    max_pct = int(round(max_raw))
                    min_pct = int(round(min_raw))
                    chosen_pct = st.slider(
                        f"{col_name}",
                        min_value=min_pct,
                        max_value=max_pct,
                        value=cur_pct,
                        step=1,
                        format="%d%%",
                    )
                    user_inputs[col_name] = chosen_pct
                    st.caption(f"Active: **{chosen_pct}%**")

            elif is_count_col:
                # Force Whole Numbers for Quantity, Units, Count (integers like 10, not 10.00)
                min_int = int(np.floor(min_raw))
                max_int = max(int(np.ceil(max_raw)), min_int + 1)
                val_int = int(round(median_raw))
                val_int = max(min_int, min(val_int, max_int))
                user_inputs[col_name] = st.slider(
                    col_name,
                    min_value=min_int,
                    max_value=max_int,
                    value=val_int,
                    step=1,
                )

            else:
                # Standard numeric slider
                if min_raw == max_raw:
                    step_val = 1.0
                else:
                    step_val = round((max_raw - min_raw) / 50, 2)
                    if step_val == 0:
                        step_val = 0.01

                user_inputs[col_name] = st.slider(
                    col_name,
                    min_value=float(min_raw),
                    max_value=float(max_raw),
                    value=float(median_raw),
                    step=float(step_val),
                    format="%.2f",
                )

# Lock non-calculator features to their medians/modes in the background
for col_name in feature_cols:
    if col_name not in user_inputs:
        col_series = clean_df[col_name]
        if pd.api.types.is_numeric_dtype(col_series):
            user_inputs[col_name] = col_series.mean()
        else:
            mode_series = col_series.mode()
            user_inputs[col_name] = mode_series.iloc[0] if len(mode_series) > 0 else "Standard"

# Real-time inference
scenario_df = pd.DataFrame([user_inputs])
scenario_pred = pipeline.predict(scenario_df)[0]

prob_text = ""
if task_type == "category" and hasattr(pipeline, "predict_proba"):
    try:
        probas = pipeline.predict_proba(scenario_df)[0]
        classes = pipeline.classes_
        class_idx = list(classes).index(scenario_pred)
        conf_pct = round(probas[class_idx] * 100, 1)
        prob_text = f"Class Probability: {conf_pct}%"
    except Exception:
        prob_text = ""

if task_type == "number":
    target_lower = target_column.lower()
    is_currency = any(k in target_lower for k in ["sales", "revenue", "price", "spend", "cost", "profit"])
    prefix = "$" if is_currency else ""
    formatted_value = f"{prefix}{scenario_pred:,.2f}"
else:
    formatted_value = str(scenario_pred)

# Outcome Panel with Simple English Text
st.markdown(
    f"""
    <div class="scenario-outcome-panel">
        <div class="scenario-outcome-label">Estimated Result ({target_column})</div>
        <div class="scenario-outcome-val">{formatted_value}</div>
        <div class="scenario-outcome-sub">
            {prob_text if prob_text else "Updated automatically based on your data patterns"}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Download Full Predictions (Immediate Download Button without broken <code> tags)
st.write("")
st.markdown("<div style='font-size: 0.95rem; font-weight: 600; color: #f8fafc; margin-bottom: 0.2rem;'>Download Full Predictions</div>", unsafe_allow_html=True)
st.caption(f"Score all {len(clean_df):,} rows with an appended `Predicted_{target_column}` column:")

export_df = raw_df.copy()
pred_col_name = f"Predicted_{target_column}"
export_df[pred_col_name] = np.round(all_preds, 2) if task_type == "number" else all_preds
csv_bytes = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label=f"Download Full Predictions (.csv)",
    data=csv_bytes,
    file_name=f"Predictions_{target_column}.csv",
    mime="text/csv",
    type="primary",
    use_container_width=True,
)

with st.expander("Preview Scored Records (First 25 Rows)"):
    st.dataframe(export_df.head(25), use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# Clean Enterprise Footer
st.markdown(
    """
    <div style="text-align: center; color: #475569; font-size: 0.78rem; margin-top: 3.5rem; padding-top: 1.5rem; border-top: 1px solid #1e293b;">
        Predictive Analytics Studio &nbsp;|&nbsp; Enterprise Machine Learning Engine
    </div>
    """,
    unsafe_allow_html=True,
)
