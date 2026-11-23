"""
Weather Anomaly Detection Dashboard.
Built with Streamlit, Scikit-learn (Isolation Forest), Plotly, and Open-Meteo API.
"""

from datetime import datetime, timedelta
import os
import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import (
    PRESET_CITIES,
    search_city,
    fetch_historical_weather,
    generate_synthetic_weather
)
from src.preprocessor import clean_weather_data, engineer_features, get_default_feature_names
from src.model import WeatherAnomalyDetector
from src.visualizer import (
    plot_metric_timeseries,
    plot_all_metrics_dashboard,
    plot_multivariate_3d,
    plot_correlation_heatmap,
    plot_metric_distribution_comparison,
    plot_anomaly_types_donut,
    plot_diurnal_anomaly_pattern
)

# ---------------------------------------------------------
# Streamlit App Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Weather Anomaly Detection",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .badge-anomaly {
        background-color: #ffebee;
        color: #c62828;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("⛅ Weather Anomaly Detection")
st.sidebar.markdown("Unsupervised Machine Learning with **Isolation Forest**")

st.sidebar.header("1. Data Source")
data_source_mode = st.sidebar.radio(
    "Select Source:",
    ["🌍 Open-Meteo API (Live/Historical)", "📁 Sample Dataset (Instant / Offline)"],
    index=0
)

# Coordinates and selection defaults
lat, lon = 51.5074, -0.1278
location_name = "London, United Kingdom"
start_date_str = ""
end_date_str = ""

if data_source_mode.startswith("🌍"):
    location_input_type = st.sidebar.selectbox(
        "Location Method:",
        ["Preset Major Cities", "Search Any Global City", "Custom Coordinates"]
    )
    
    if location_input_type == "Preset Major Cities":
        selected_city = st.sidebar.selectbox("Choose City:", list(PRESET_CITIES.keys()), index=0)
        coords = PRESET_CITIES[selected_city]
        lat, lon = coords["lat"], coords["lon"]
        location_name = selected_city
        
    elif location_input_type == "Search Any Global City":
        search_query = st.sidebar.text_input("Enter City Name (e.g., Berlin, Miami, Delhi):", "Madrid")
        if search_query:
            results = search_city(search_query)
            if results:
                city_options = {r["label"]: r for r in results}
                chosen_label = st.sidebar.selectbox("Matching Cities:", list(city_options.keys()))
                chosen_data = city_options[chosen_label]
                lat, lon = chosen_data["lat"], chosen_data["lon"]
                location_name = chosen_data["label"]
            else:
                st.sidebar.warning("No cities found. Defaulting to London.")
                lat, lon = 51.5074, -0.1278
                location_name = "London"
                
    else:  # Custom Coordinates
        lat = st.sidebar.number_input("Latitude:", value=40.7128, format="%.4f")
        lon = st.sidebar.number_input("Longitude:", value=-74.0060, format="%.4f")
        location_name = f"Coordinates ({lat:.2f}, {lon:.2f})"

    # Date Range Selection
    st.sidebar.subheader("Date Range")
    default_end = datetime.now().date() - timedelta(days=5)
    default_start = default_end - timedelta(days=90)
    
    date_col1, date_col2 = st.sidebar.columns(2)
    start_date = date_col1.date_input("Start Date", default_start)
    end_date = date_col2.date_input("End Date", default_end)
    
    if start_date >= end_date:
        st.sidebar.error("Start date must be earlier than End date.")
        
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
else:
    location_name = "Simulated Meteorological Station"
    sample_duration_days = st.sidebar.slider("Sample Data Length (Days):", 30, 365, 120, step=30)

# Model Settings
st.sidebar.header("2. Model Configuration")
contamination = st.sidebar.slider(
    "Contamination Rate (Expected Anomaly %):",
    min_value=0.01,
    max_value=0.15,
    value=0.05,
    step=0.01,
    help="The proportion of outliers expected in the dataset. Higher values flag more points."
)

n_estimators = st.sidebar.slider(
    "Isolation Forest Trees (n_estimators):",
    min_value=50,
    max_value=300,
    value=150,
    step=25
)

# Feature Selection
st.sidebar.subheader("Feature Engineering Selection")
use_core_features = st.sidebar.checkbox("Core Weather Metrics (Temp, Hum, Rain, Wind)", value=True)
use_rate_of_change = st.sidebar.checkbox("Rate of Change (3h Deltas)", value=True)
use_diurnal_dev = st.sidebar.checkbox("Diurnal Cycle Deviation", value=True)
use_rolling_extremes = st.sidebar.checkbox("Rolling Accumulations (6h Rain/Wind)", value=True)

# Build feature column list based on choices
active_features = []
if use_core_features:
    active_features.extend(["temperature", "humidity", "precipitation", "wind_speed"])
if use_rate_of_change:
    active_features.extend(["temp_delta_3h", "wind_delta_3h", "humidity_delta_3h"])
if use_diurnal_dev:
    active_features.append("temp_diurnal_dev")
if use_rolling_extremes:
    active_features.extend(["precip_sum_6h", "wind_roll_max_6h"])

if not active_features:
    st.sidebar.warning("Select at least one feature. Using core metrics.")
    active_features = ["temperature", "humidity", "precipitation", "wind_speed"]


# ---------------------------------------------------------
# Data Loading & Processing Pipeline (Cached)
# ---------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_and_process_data(
    source_mode: str,
    lat: float,
    lon: float,
    start_str: str,
    end_str: str,
    location_label: str
) -> pd.DataFrame:
    """Fetch raw data, clean it, and compute engineered features."""
    if source_mode.startswith("🌍"):
        raw_df = fetch_historical_weather(lat, lon, start_str, end_str)
        raw_df["location"] = location_label
    else:
        raw_df = generate_synthetic_weather(days=120, city_name=location_label)
        
    df_clean = clean_weather_data(raw_df)
    df_engineered = engineer_features(df_clean)
    return df_engineered


# ---------------------------------------------------------
# Main Application Header
# ---------------------------------------------------------
st.title("⛅ Intelligent Weather Anomaly Detection")
st.markdown(
    f"Analyzing atmospheric patterns for **{location_name}** using unsupervised learning "
    f"to flag sudden thermal shocks, torrential precipitation, squalls, and atypical climate extremes."
)

# Trigger Data Load
data_load_error = None
df_processed = None

with st.spinner("Fetching and processing meteorological records..."):
    try:
        df_processed = load_and_process_data(
            data_source_mode,
            lat,
            lon,
            start_date_str,
            end_date_str,
            location_name
        )
    except Exception as e:
        data_load_error = str(e)

if data_load_error:
    st.error(f"⚠️ Failed to retrieve data from Open-Meteo: {data_load_error}")
    st.info("💡 Switching automatically to high-fidelity simulated weather dataset so you can continue exploring.")
    df_processed = engineer_features(clean_weather_data(generate_synthetic_weather(120, city_name=location_name)))

if df_processed is None or df_processed.empty:
    st.error("No data available to display.")
    st.stop()


# ---------------------------------------------------------
# Train Isolation Forest Model & Detect Anomalies
# ---------------------------------------------------------
detector = WeatherAnomalyDetector(
    contamination=contamination,
    n_estimators=n_estimators,
    random_state=42
)

df_result = detector.fit_predict(df_processed, feature_cols=active_features)
stats = detector.get_summary_stats(df_result)


# ---------------------------------------------------------
# High-Level KPI Summary Cards
# ---------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric(
    label="Observations",
    value=f"{stats['total_records']:,} hrs",
    help="Total hourly records analyzed"
)
kpi2.metric(
    label="Anomalies Flagged",
    value=f"{stats['anomaly_count']} ({stats['anomaly_pct']}%)",
    delta="Outlier Events",
    delta_color="inverse"
)
kpi3.metric(
    label="Max Temperature",
    value=f"{stats['max_temp']} °C",
    help=f"Min: {stats['min_temp']} °C"
)
kpi4.metric(
    label="Max Wind Speed",
    value=f"{stats['max_wind']} km/h",
    help="Peak sustained gust"
)
kpi5.metric(
    label="Peak Precipitation",
    value=f"{stats['max_precip']} mm",
    help=f"Total Volume: {stats['total_precip']} mm"
)

st.markdown("---")


# ---------------------------------------------------------
# Dashboard Tabs
# ---------------------------------------------------------
tab_timeseries, tab_eda, tab_model, tab_records = st.tabs([
    "📈 Time Series & Anomaly Highlights",
    "🔍 Exploratory Data Analysis (EDA)",
    "🧠 Machine Learning Diagnostics",
    "📋 Anomaly Records & CSV Export"
])

# ---------------------------------------------------------
# Tab 1: Time Series & Anomaly Highlights
# ---------------------------------------------------------
with tab_timeseries:
    st.subheader("Interactive Atmospheric Timeline")
    st.markdown(
        "Hover over highlighted **red dots** to inspect detected anomaly events, "
        "their meteorological category, and severity score."
    )
    
    view_option = st.radio(
        "Select Visualization View:",
        ["All Metrics Stacked", "Temperature (°C)", "Relative Humidity (%)", "Precipitation (mm)", "Wind Speed (km/h)"],
        horizontal=True
    )
    
    if view_option == "All Metrics Stacked":
        fig_multi = plot_all_metrics_dashboard(df_result)
        st.plotly_chart(fig_multi, use_container_width=True)
    elif view_option == "Temperature (°C)":
        fig_temp = plot_metric_timeseries(df_result, "temperature", "Temperature", "°C", line_color="#E65100")
        st.plotly_chart(fig_temp, use_container_width=True)
    elif view_option == "Relative Humidity (%)":
        fig_hum = plot_metric_timeseries(df_result, "humidity", "Relative Humidity", "%", line_color="#0288D1")
        st.plotly_chart(fig_hum, use_container_width=True)
    elif view_option == "Precipitation (mm)":
        fig_precip = plot_metric_timeseries(df_result, "precipitation", "Precipitation", "mm", line_color="#00897B")
        st.plotly_chart(fig_precip, use_container_width=True)
    elif view_option == "Wind Speed (km/h)":
        fig_wind = plot_metric_timeseries(df_result, "wind_speed", "Wind Speed", "km/h", line_color="#5D4037")
        st.plotly_chart(fig_wind, use_container_width=True)

# ---------------------------------------------------------
# Tab 2: Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis")
    
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        st.markdown("#### Correlation Heatmap")
        fig_corr = plot_correlation_heatmap(df_result, feature_cols=active_features)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with eda_col2:
        st.markdown("#### Diurnal Anomaly Distribution")
        fig_diurnal = plot_diurnal_anomaly_pattern(df_result)
        st.plotly_chart(fig_diurnal, use_container_width=True)
        
    st.markdown("#### Normal vs Anomalous Distribution Comparisons")
    dist_metric = st.selectbox(
        "Select metric to inspect distributions:",
        ["temperature", "humidity", "wind_speed", "precipitation"],
        format_func=lambda x: x.replace("_", " ").title()
    )
    unit_map = {"temperature": "°C", "humidity": "%", "wind_speed": "km/h", "precipitation": "mm"}
    fig_dist = plot_metric_distribution_comparison(
        df_result,
        dist_metric,
        dist_metric.replace("_", " ").title(),
        unit_map[dist_metric]
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ---------------------------------------------------------
# Tab 3: Machine Learning Diagnostics
# ---------------------------------------------------------
with tab_model:
    st.subheader("Isolation Forest Diagnostics & Feature Space")
    
    ml_col1, ml_col2 = st.columns([3, 2])
    
    with ml_col1:
        st.markdown("#### 3D Multivariate Feature Space")
        st.markdown(
            "Interact with the 3D projection below to observe how normal records cluster "
            "together while anomalous weather states diverge into outer geometric subspaces."
        )
        fig_3d = plot_multivariate_3d(df_result)
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with ml_col2:
        st.markdown("#### Anomaly Categorization")
        fig_donut = plot_anomaly_types_donut(df_result)
        st.plotly_chart(fig_donut, use_container_width=True)
        
        st.markdown("#### Model Parameters")
        st.json({
            "Algorithm": "Isolation Forest (scikit-learn)",
            "Contamination Rate": contamination,
            "Estimator Trees": n_estimators,
            "Features Used": active_features,
            "Total Anomalies Flagged": stats["anomaly_count"]
        })

# ---------------------------------------------------------
# Tab 4: Anomaly Records & CSV Export
# ---------------------------------------------------------
with tab_records:
    st.subheader("Detailed Anomaly Log & Data Export")
    
    df_anomalies_only = df_result[df_result["is_anomaly"]].copy()
    
    display_cols = [
        "timestamp", "location", "temperature", "humidity",
        "precipitation", "wind_speed", "anomaly_type", "severity_index"
    ]
    present_display_cols = [c for c in display_cols if c in df_result.columns]
    
    if df_anomalies_only.empty:
        st.success("No anomalies detected based on the current threshold and contamination rate.")
    else:
        st.markdown(f"**Found {len(df_anomalies_only)} anomalous hourly intervals:**")
        
        # Filter by Anomaly Category
        unique_types = ["All"] + sorted(list(df_anomalies_only["anomaly_type"].unique()))
        selected_type = st.selectbox("Filter by Anomaly Type:", unique_types)
        
        if selected_type != "All":
            filtered_view = df_anomalies_only[df_anomalies_only["anomaly_type"] == selected_type]
        else:
            filtered_view = df_anomalies_only
            
        st.dataframe(
            filtered_view[present_display_cols].sort_values("severity_index", ascending=False),
            use_container_width=True,
            height=320
        )
        
    st.markdown("---")
    st.subheader("📥 Export Data to CSV")
    
    exp_col1, exp_col2 = st.columns(2)
    
    # 1. Download Anomalies Only
    csv_anomalies = df_anomalies_only.to_csv(index=False).encode("utf-8")
    exp_col1.download_button(
        label="⬇️ Download Anomalies Only (CSV)",
        data=csv_anomalies,
        file_name=f"weather_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # 2. Download Full Processed Dataset
    csv_full = df_result.to_csv(index=False).encode("utf-8")
    exp_col2.download_button(
        label="⬇️ Download Full Dataset with Predictions (CSV)",
        data=csv_full,
        file_name=f"weather_full_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
