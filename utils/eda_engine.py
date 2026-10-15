"""
eda_engine.py — Dynamic Plotly Express visualizations that adapt to any dataset.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
PALETTE = px.colors.qualitative.Vivid

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------

def render_kpi_cards(df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    """Render 5 summary KPI metric cards."""
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_pct = (df.isna().sum().sum() / max(df.size, 1) * 100)
    duplicates = int(raw_df.duplicated().sum())
    mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("📊 Total Rows", f"{total_rows:,}")
    with c2:
        st.metric("📋 Total Columns", f"{total_cols:,}")
    with c3:
        st.metric("❓ Missing %", f"{missing_pct:.2f}%")
    with c4:
        st.metric("🔁 Duplicate Rows", f"{duplicates:,}")
    with c5:
        st.metric("💾 Memory Usage", f"{mem_mb:.2f} MB")


# ---------------------------------------------------------------------------
# Univariate / Bivariate chart builder
# ---------------------------------------------------------------------------

def build_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: Optional[str],
    color_col: Optional[str],
    col_types: Dict[str, List[str]],
) -> Optional[go.Figure]:
    """
    Build a Plotly figure based on user selections.
    Returns None if inputs are invalid.
    """
    if x_col not in df.columns:
        return None

    kwargs = {"data_frame": df, "x": x_col, "color": color_col or None,
              "color_discrete_sequence": PALETTE, "template": "plotly_dark"}

    try:
        if chart_type == "Histogram":
            fig = px.histogram(nbins=40, **kwargs)

        elif chart_type == "Box Plot":
            fig = px.box(y=y_col or x_col, **kwargs)

        elif chart_type == "Violin Plot":
            fig = px.violin(y=y_col or x_col, box=True, **kwargs)

        elif chart_type == "Scatter Plot":
            if not y_col or y_col not in df.columns:
                st.warning("Scatter Plot requires both X and Y columns.")
                return None
            trend = "ols" if x_col in col_types["numeric"] else None
            fig = px.scatter(y=y_col, trendline=trend, **kwargs)

        elif chart_type == "Line Chart":
            if not y_col or y_col not in df.columns:
                st.warning("Line Chart requires both X and Y columns.")
                return None
            fig = px.line(y=y_col, **kwargs)

        elif chart_type == "Bar Chart":
            if y_col and y_col in df.columns:
                agg = df.groupby(x_col, dropna=False)[y_col].mean().reset_index()
                fig = px.bar(agg, x=x_col, y=y_col,
                             color_discrete_sequence=PALETTE, template="plotly_dark")
            else:
                counts = df[x_col].value_counts().reset_index()
                counts.columns = [x_col, "count"]
                fig = px.bar(counts, x=x_col, y="count",
                             color_discrete_sequence=PALETTE, template="plotly_dark")
        else:
            return None

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig

    except Exception as exc:
        st.error(f"Chart error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]) -> Optional[go.Figure]:
    """Interactive Pearson correlation heatmap for numeric columns."""
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < 2:
        return None

    corr = df[cols].corr(numeric_only=True).round(3)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        template="plotly_dark",
        title="Pearson Correlation Matrix",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Distribution overview grid
# ---------------------------------------------------------------------------

def numeric_distribution_grid(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    """Render a grid of small histograms for each numeric column."""
    cols_to_plot = [c for c in numeric_cols if c in df.columns][:12]
    if not cols_to_plot:
        st.info("No numeric columns found for distribution plots.")
        return

    n_cols = 3
    rows = [cols_to_plot[i: i + n_cols] for i in range(0, len(cols_to_plot), n_cols)]

    for row in rows:
        grid_cols = st.columns(n_cols)
        for idx, col_name in enumerate(row):
            with grid_cols[idx]:
                fig = px.histogram(
                    df, x=col_name, nbins=30,
                    color_discrete_sequence=["#7c3aed"],
                    template="plotly_dark",
                    title=col_name,
                )
                fig.update_layout(
                    showlegend=False,
                    margin=dict(l=20, r=10, t=40, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0e0", size=11),
                    height=220,
                )
                st.plotly_chart(fig, use_container_width=True)
