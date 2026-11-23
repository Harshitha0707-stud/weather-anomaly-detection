"""
Visualizer Module for Weather Anomaly Detection.
Provides professional, interactive Plotly charts for time-series analysis,
multivariate 3D scatter, correlation heatmaps, and distribution comparisons.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Clean theme palette
PRIMARY_BLUE = "#1E88E5"
ACCENT_RED = "#E53935"
ACCENT_GREEN = "#43A047"
ACCENT_AMBER = "#FB8C00"
ACCENT_CYAN = "#00ACC1"
BG_CARD = "rgba(255, 255, 255, 0.05)"


def plot_metric_timeseries(
    df: pd.DataFrame,
    metric: str,
    title: str,
    unit: str,
    line_color: str = PRIMARY_BLUE
) -> go.Figure:
    """
    Create a clean, interactive time-series chart with anomaly points highlighted in red.
    """
    fig = go.Figure()
    
    # Base time series line
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df[metric],
        mode="lines",
        name=f"Observed {title}",
        line=dict(color=line_color, width=1.8),
        hovertemplate=f"<b>Date:</b> %{{x|%Y-%m-%d %H:%M}}<br><b>{title}:</b> %{{y:.1f}} {unit}<extra></extra>"
    ))
    
    # Anomaly points
    anomalies = df[df["is_anomaly"]]
    if not anomalies.empty:
        custom_data = np.stack((
            anomalies["anomaly_type"],
            anomalies["severity_index"]
        ), axis=-1)
        
        fig.add_trace(go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies[metric],
            mode="markers",
            name="Detected Anomaly",
            marker=dict(
                color=ACCENT_RED,
                size=8,
                symbol="circle",
                line=dict(width=1.5, color="#FFFFFF")
            ),
            customdata=custom_data,
            hovertemplate=(
                f"<b>⚠️ ANOMALY DETECTED</b><br>"
                f"<b>Date:</b> %{{x|%Y-%m-%d %H:%M}}<br>"
                f"<b>{title}:</b> %{{y:.1f}} {unit}<br>"
                f"<b>Category:</b> %{{customdata[0]}}<br>"
                f"<b>Severity:</b> %{{customdata[1]}} / 100"
                f"<extra></extra>"
            )
        ))
        
    fig.update_layout(
        title=f"<b>{title} Time Series with Anomaly Markers</b>",
        title_font=dict(size=16),
        xaxis_title="Timeline",
        yaxis_title=f"{title} ({unit})",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(
            rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=30, label="1m", step="day", stepmode="backward"),
                    dict(count=90, label="3m", step="day", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        )
    )
    
    return fig


def plot_all_metrics_dashboard(df: pd.DataFrame) -> go.Figure:
    """
    Subplots displaying all 4 core weather metrics synchronized along the time axis.
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Temperature (°C)",
            "Relative Humidity (%)",
            "Precipitation (mm)",
            "Wind Speed (km/h)"
        )
    )
    
    configs = [
        ("temperature", "Temperature", "°C", "#E65100", 1),
        ("humidity", "Humidity", "%", "#0288D1", 2),
        ("precipitation", "Precipitation", "mm", "#00897B", 3),
        ("wind_speed", "Wind Speed", "km/h", "#6D4C41", 4)
    ]
    
    anomalies = df[df["is_anomaly"]]
    
    for metric, name, unit, color, row_idx in configs:
        # Base line
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[metric],
                mode="lines",
                name=name,
                line=dict(color=color, width=1.5),
                hovertemplate=f"%{{x|%Y-%m-%d %H:%M}}<br>{name}: %{{y:.1f}} {unit}<extra></extra>"
            ),
            row=row_idx, col=1
        )
        # Anomalies
        if not anomalies.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomalies["timestamp"],
                    y=anomalies[metric],
                    mode="markers",
                    name=f"{name} Anomaly" if row_idx == 1 else None,
                    showlegend=(row_idx == 1),
                    marker=dict(color=ACCENT_RED, size=6, symbol="circle")
                ),
                row=row_idx, col=1
            )
            
    fig.update_layout(
        height=850,
        title_text="<b>Multi-Metric Atmospheric Profile & Anomalies</b>",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_multivariate_3d(df: pd.DataFrame) -> go.Figure:
    """
    Generate an interactive 3D scatter plot of Temperature vs Humidity vs Wind Speed,
    colored by anomaly status.
    """
    df_plot = df.copy()
    df_plot["Status"] = df_plot["is_anomaly"].map({True: "Anomaly", False: "Normal"})
    
    fig = px.scatter_3d(
        df_plot,
        x="temperature",
        y="humidity",
        z="wind_speed",
        color="Status",
        color_discrete_map={"Normal": PRIMARY_BLUE, "Anomaly": ACCENT_RED},
        size="severity_index",
        size_max=14,
        opacity=0.75,
        hover_data={
            "timestamp": "|%Y-%m-%d %H:%M",
            "anomaly_type": True,
            "severity_index": True,
            "precipitation": ":.1f"
        },
        title="<b>3D Multivariate Space (Temp vs Humidity vs Wind Speed)</b>"
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title="Temperature (°C)",
            yaxis_title="Humidity (%)",
            zaxis_title="Wind Speed (km/h)"
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> go.Figure:
    """
    Generate correlation heatmap of weather metrics and engineered features.
    """
    if feature_cols is None:
        feature_cols = [
            "temperature", "humidity", "precipitation", "wind_speed",
            "temp_delta_3h", "temp_diurnal_dev", "precip_sum_6h", "wind_roll_max_6h"
        ]
        
    cols = [c for c in feature_cols if c in df.columns]
    corr = df[cols].corr().round(2)
    
    labels = [c.replace("_", " ").title() for c in cols]
    
    fig = px.imshow(
        corr.values,
        x=labels,
        y=labels,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=True,
        title="<b>Correlation Heatmap of Meteorological Variables</b>"
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_metric_distribution_comparison(df: pd.DataFrame, metric: str, title: str, unit: str) -> go.Figure:
    """
    Box and violin comparison plot of normal vs anomalous weather conditions.
    """
    df_plot = df.copy()
    df_plot["Status"] = df_plot["is_anomaly"].map({True: "Anomaly", False: "Normal"})
    
    fig = px.box(
        df_plot,
        x="Status",
        y=metric,
        color="Status",
        color_discrete_map={"Normal": PRIMARY_BLUE, "Anomaly": ACCENT_RED},
        points="outliers",
        title=f"<b>Distribution Comparison: {title} ({unit})</b>"
    )
    fig.update_layout(
        yaxis_title=f"{title} ({unit})",
        xaxis_title="Classification",
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig


def plot_anomaly_types_donut(df: pd.DataFrame) -> go.Figure:
    """
    Donut chart summarizing distribution of detected anomaly categories.
    """
    anomalies = df[df["is_anomaly"]]
    if anomalies.empty:
        fig = go.Figure()
        fig.add_annotation(text="No anomalies detected with current parameters", showarrow=False)
        return fig
        
    counts = anomalies["anomaly_type"].value_counts().reset_index()
    counts.columns = ["Type", "Count"]
    
    fig = px.pie(
        counts,
        values="Count",
        names="Type",
        hole=0.45,
        title="<b>Anomaly Breakdown by Meteorological Driver</b>",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )
    return fig


def plot_diurnal_anomaly_pattern(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing the frequency of anomalies across the 24 hours of the day.
    """
    if "hour" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        
    anomalies = df[df["is_anomaly"]]
    hourly_counts = anomalies.groupby("hour").size().reindex(range(24), fill_value=0).reset_index()
    hourly_counts.columns = ["Hour", "Anomaly Count"]
    
    fig = px.bar(
        hourly_counts,
        x="Hour",
        y="Anomaly Count",
        title="<b>Hourly Distribution of Anomalies (Diurnal Pattern)</b>",
        color="Anomaly Count",
        color_continuous_scale="Tealgrn"
    )
    fig.update_layout(
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
        xaxis_title="Hour of Day (00:00 - 23:00)",
        yaxis_title="Detected Anomaly Frequency",
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig
