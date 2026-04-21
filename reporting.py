# Plotly charts + CSV export

from __future__ import annotations

import os
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from database import get_ticker_mentions, get_volume_snapshots


def generate_time_series_chart(conn, ticker: str, output_dir: str) -> str:
    """
    Build a dual-axis Plotly chart:
      - Left y-axis  : daily mention count (bar trace)
      - Right y-axis : daily trading volume (line trace)

    Saves to {output_dir}/{ticker}_timeseries.html.
    Returns the output file path, or an empty string if either series is empty.
    """
    mentions = get_ticker_mentions(conn, ticker)
    snapshots = get_volume_snapshots(conn, ticker)

    if not mentions or not snapshots:
        return ""

    # --- mention counts per day ------------------------------------------------
    daily_mentions: Counter = Counter(
        m["published_date"][:10] for m in mentions
    )
    mention_dates = sorted(daily_mentions.keys())
    mention_counts = [daily_mentions[d] for d in mention_dates]

    # --- volume series ---------------------------------------------------------
    snap_sorted = sorted(snapshots, key=lambda s: s["snapshot_date"])
    volume_dates = [s["snapshot_date"] for s in snap_sorted]
    volumes = [s["volume"] for s in snap_sorted]

    # --- build figure ----------------------------------------------------------
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=mention_dates,
            y=mention_counts,
            name="Daily Mentions",
            marker_color="steelblue",
            opacity=0.7,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=volume_dates,
            y=volumes,
            name="Trading Volume",
            mode="lines+markers",
            line=dict(color="darkorange", width=2),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title_text=f"{ticker} — Mention Frequency vs. Trading Volume",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Mention Count", secondary_y=False)
    fig.update_yaxes(title_text="Trading Volume", secondary_y=True)

    # --- save ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{ticker}_timeseries.html")
    fig.write_html(out_path)
    return out_path


def generate_scatter_chart(df: pd.DataFrame, output_dir: str) -> str:
    """
    Scatter plot: mention_count (x) vs. volume_spike_ratio (y), labelled by ticker.

    Only includes rows where mention_count >= 3 and volume_spike_ratio is not NaN.
    Saves to {output_dir}/scatter_mention_vs_spike.html.
    Returns the output file path, or an empty string if the filtered DataFrame is empty.
    """
    if df.empty:
        return ""

    filtered = df[
        (df["mention_count"] >= 3) & df["volume_spike_ratio"].notna()
    ].copy()

    if filtered.empty:
        return ""

    fig = go.Figure(
        go.Scatter(
            x=filtered["mention_count"],
            y=filtered["volume_spike_ratio"],
            mode="markers+text",
            text=filtered["ticker"],
            textposition="top center",
            marker=dict(
                size=10,
                color=filtered["pearson_r"] if "pearson_r" in filtered.columns else "steelblue",
                colorscale="RdYlGn",
                showscale="pearson_r" in filtered.columns,
                colorbar=dict(title="Pearson r") if "pearson_r" in filtered.columns else None,
                line=dict(width=1, color="darkgrey"),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Mentions: %{x}<br>"
                "Volume Spike Ratio: %{y:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title_text="Mention Count vs. Volume Spike Ratio",
        xaxis_title="Mention Count",
        yaxis_title="Volume Spike Ratio",
        plot_bgcolor="white",
        hovermode="closest",
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "scatter_mention_vs_spike.html")
    fig.write_html(out_path)
    return out_path


def export_csv(df: pd.DataFrame, output_dir: str) -> str:
    """
    Save the full correlations DataFrame to {output_dir}/correlation_results.csv.
    Float columns are rounded to 4 decimal places.
    Returns the output file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "correlation_results.csv")

    # Round all float columns to 4 decimal places without touching other dtypes.
    rounded = df.copy()
    float_cols = rounded.select_dtypes(include="float").columns
    rounded[float_cols] = rounded[float_cols].round(4)

    rounded.to_csv(out_path, index=False)
    return out_path
