"""
ml_engine.py — Automated Scikit-Learn pipeline for regression & classification,
               time-series forecasting with statsmodels, and What-If prediction.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

PALETTE_DARK = "plotly_dark"
ACCENT = "#7c3aed"

# ---------------------------------------------------------------------------
# Task detection
# ---------------------------------------------------------------------------

def detect_task(series: pd.Series) -> str:
    """
    Return 'regression' or 'classification' based on the target column.
    """
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique(dropna=True)
        if n_unique <= 10:
            return "classification"
        return "regression"
    return "classification"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

REGRESSION_MODELS: Dict[str, Any] = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=200, random_state=42),
}

CLASSIFICATION_MODELS: Dict[str, Any] = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=200, random_state=42),
}


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    model_obj: Any,
) -> Pipeline:
    """Construct a ColumnTransformer → estimator Pipeline."""
    transformers = []
    if numeric_features:
        transformers.append(
            ("num", StandardScaler(), numeric_features)
        )
    if categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model_obj),
    ])


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    task: str,
    model_name: str,
    test_size: float,
    random_seed: int,
    col_types: Dict[str, List[str]],
) -> Tuple[Optional[Pipeline], Optional[pd.DataFrame], Optional[Dict]]:
    """
    Train the selected model. Returns (fitted_pipeline, result_df, metrics_dict).
    result_df has columns [y_true, y_pred].
    On error returns (None, None, None).
    """
    # ── Data validation ───────────────────────────────────────────────────────
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found.")
        return None, None, None

    df_model = df[feature_cols + [target_col]].dropna(subset=[target_col])

    if len(df_model) < 20:
        st.error("Need at least 20 non-null rows to train. Check your preprocessing settings.")
        return None, None, None

    y = df_model[target_col]
    if y.nunique() <= 1:
        st.error(f"Target column '{target_col}' has only 1 unique value — cannot train.")
        return None, None, None

    X = df_model[feature_cols]

    numeric_features = [c for c in feature_cols if c in col_types["numeric"]]
    categorical_features = [c for c in feature_cols if c in col_types["categorical"]]

    # ── Model selection ───────────────────────────────────────────────────────
    model_registry = REGRESSION_MODELS if task == "regression" else CLASSIFICATION_MODELS
    if model_name not in model_registry:
        st.error(f"Unknown model: {model_name}")
        return None, None, None

    model_obj = model_registry[model_name]

    # ── Train/test split ──────────────────────────────────────────────────────
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed,
            stratify=(y if task == "classification" and y.nunique() <= 20 else None),
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed,
        )

    pipeline = build_pipeline(numeric_features, categorical_features, model_obj)

    # ── Fit ───────────────────────────────────────────────────────────────────
    try:
        pipeline.fit(X_train, y_train)
    except Exception as exc:
        st.error(f"Training error: {exc}")
        return None, None, None

    y_pred = pipeline.predict(X_test)

    # ── Metrics ───────────────────────────────────────────────────────────────
    if task == "regression":
        metrics = {
            "R² Score": round(float(r2_score(y_test, y_pred)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            "MAE": round(float(mean_absolute_error(y_test, y_pred)), 4),
        }
    else:
        avg = "binary" if len(np.unique(y_test)) == 2 else "weighted"
        metrics = {
            "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "Precision": round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "F1 Score": round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "_y_test": y_test,
            "_y_pred": y_pred,
        }

    result_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})

    return pipeline, result_df, metrics


# ---------------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(result_df: pd.DataFrame) -> go.Figure:
    max_val = max(result_df["y_true"].max(), result_df["y_pred"].max())
    min_val = min(result_df["y_true"].min(), result_df["y_pred"].min())

    fig = px.scatter(
        result_df, x="y_true", y="y_pred",
        labels={"y_true": "Actual", "y_pred": "Predicted"},
        template=PALETTE_DARK,
        opacity=0.65,
        color_discrete_sequence=[ACCENT],
        title="Actual vs. Predicted",
    )
    fig.add_shape(
        type="line", x0=min_val, x1=max_val, y0=min_val, y1=max_val,
        line=dict(color="#f59e0b", dash="dash", width=2),
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_residuals(result_df: pd.DataFrame) -> go.Figure:
    residuals = result_df["y_true"] - result_df["y_pred"]
    fig = px.histogram(
        x=residuals, nbins=40,
        labels={"x": "Residual"},
        color_discrete_sequence=["#06b6d4"],
        template=PALETTE_DARK,
        title="Residuals Distribution",
    )
    fig.add_vline(x=0, line_color="#f59e0b", line_dash="dash", line_width=2)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_confusion_matrix(y_test, y_pred) -> go.Figure:
    labels = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig = px.imshow(
        cm,
        x=[str(l) for l in labels],
        y=[str(l) for l in labels],
        text_auto=True,
        color_continuous_scale="Viridis",
        template=PALETTE_DARK,
        labels=dict(x="Predicted", y="Actual"),
        title="Confusion Matrix",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def plot_feature_importance(
    pipeline: Pipeline,
    feature_cols: List[str],
    col_types: Dict[str, List[str]],
    top_n: int = 20,
) -> Optional[go.Figure]:
    """Extract feature importances and return a horizontal bar chart."""
    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        # Get feature names after transformation
        ohe_names: List[str] = []
        numeric_names: List[str] = []

        for name, transformer, cols in preprocessor.transformers_:
            if name == "num":
                numeric_names = list(cols)
            elif name == "cat" and hasattr(transformer, "get_feature_names_out"):
                ohe_names = list(transformer.get_feature_names_out(cols))

        all_feature_names = numeric_names + ohe_names

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importances = np.abs(coef.flatten()) if coef.ndim > 1 else np.abs(coef)
        else:
            return None

        if len(importances) != len(all_feature_names):
            all_feature_names = [f"Feature {i}" for i in range(len(importances))]

        fi_df = pd.DataFrame({
            "Feature": all_feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=True).tail(top_n)

        fig = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Viridis",
            template=PALETTE_DARK, title=f"Top {top_n} Feature Importances",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            margin=dict(l=20, r=20, t=50, b=40),
            yaxis_title="",
            showlegend=False,
        )
        return fig

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Time-series forecasting
# ---------------------------------------------------------------------------

def run_timeseries_forecast(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    freq: str,
    horizon: int,
) -> Optional[go.Figure]:
    """
    Aggregate the series and forecast using Holt-Winters / ExponentialSmoothing.
    Returns an interactive Plotly figure or None on failure.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        st.error("statsmodels is not installed. Run: pip install statsmodels")
        return None

    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}
    pd_freq = freq_map.get(freq, "D")

    try:
        ts = (
            df[[date_col, metric_col]]
            .dropna()
            .copy()
        )
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
        ts = ts.dropna(subset=[date_col])
        ts = ts.set_index(date_col)[metric_col].resample(pd_freq).mean().interpolate()

        if len(ts) < 8:
            st.error("Need at least 8 aggregated time periods to forecast.")
            return None

        # Determine trend / seasonality components
        seasonal_periods = {"D": 7, "W": 52, "MS": 12}.get(pd_freq, 12)
        use_seasonal = len(ts) >= 2 * seasonal_periods

        model = ExponentialSmoothing(
            ts,
            trend="add",
            seasonal="add" if use_seasonal else None,
            seasonal_periods=seasonal_periods if use_seasonal else None,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
        forecast = fit.forecast(horizon)

        # Confidence interval approximation (±1.96 * residual std)
        resid_std = float(np.std(fit.resid))
        ci_upper = forecast + 1.96 * resid_std
        ci_lower = forecast - 1.96 * resid_std

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts.index, y=ts.values, name="Historical", line=dict(color="#7c3aed", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=forecast.index, y=forecast.values, name="Forecast",
            line=dict(color="#f59e0b", width=2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=list(forecast.index) + list(forecast.index[::-1]),
            y=list(ci_upper.values) + list(ci_lower.values[::-1]),
            fill="toself",
            fillcolor="rgba(245,158,11,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
        ))
        fig.update_layout(
            template=PALETTE_DARK,
            title=f"{metric_col} — {freq} Forecast ({horizon} periods ahead)",
            xaxis_title="Date",
            yaxis_title=metric_col,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            margin=dict(l=40, r=20, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig, ts, forecast

    except Exception as exc:
        st.error(f"Forecasting error: {exc}")
        return None


# ---------------------------------------------------------------------------
# What-If single prediction
# ---------------------------------------------------------------------------

def predict_single(
    pipeline: Pipeline,
    input_dict: Dict[str, Any],
    task: str,
) -> Tuple[Any, Optional[Dict[str, float]]]:
    """
    Run a single-row prediction. Returns (prediction, probabilities | None).
    """
    row_df = pd.DataFrame([input_dict])
    pred = pipeline.predict(row_df)[0]

    proba = None
    if task == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            proba_arr = pipeline.predict_proba(row_df)[0]
            classes = pipeline.classes_
            proba = {str(c): round(float(p), 4) for c, p in zip(classes, proba_arr)}
        except Exception:
            pass

    return pred, proba


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def batch_predict(pipeline: Pipeline, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Generate predictions for all rows in df using the fitted pipeline.
    Returns df with an appended 'predicted' column.
    """
    X = df[[c for c in feature_cols if c in df.columns]]
    preds = pipeline.predict(X)
    result = df.copy()
    result["predicted"] = preds
    return result
