"""
Preprocessor Module for Weather Anomaly Detection.
Handles data cleaning, missing value imputation, and feature engineering.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


CORE_METRICS = ["temperature", "humidity", "precipitation", "wind_speed"]


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw weather dataframe:
    - Sort chronologically by timestamp
    - Remove duplicates
    - Impute missing values with forward/backward fill and linear interpolation
    - Enforce valid domain bounds (e.g., humidity [0, 100], precipitation >= 0)
    """
    df_clean = df.copy()
    
    # Ensure timestamp format
    if not pd.api.types.is_datetime64_any_dtype(df_clean["timestamp"]):
        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
        
    df_clean = df_clean.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    
    # Impute missing values
    for col in CORE_METRICS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].interpolate(method="linear").bfill().ffill()
            
    # Domain constraints
    if "humidity" in df_clean.columns:
        df_clean["humidity"] = df_clean["humidity"].clip(0.0, 100.0)
    if "precipitation" in df_clean.columns:
        df_clean["precipitation"] = df_clean["precipitation"].clip(lower=0.0)
    if "wind_speed" in df_clean.columns:
        df_clean["wind_speed"] = df_clean["wind_speed"].clip(lower=0.0)
        
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate domain-specific features for anomaly detection:
    - Rolling means and std deviations (short-term 6h and medium-term 24h)
    - Cumulative rainfall sums (6h and 24h)
    - Rate of change / delta metrics (3h)
    - Diurnal temperature deviations (deviation from historical mean for that specific hour)
    """
    df_feat = df.copy()
    
    # Temporal features
    df_feat["hour"] = df_feat["timestamp"].dt.hour
    df_feat["month"] = df_feat["timestamp"].dt.month
    df_feat["day_of_week"] = df_feat["timestamp"].dt.dayofweek
    
    # Rolling Statistics
    # Temperature
    df_feat["temp_roll_mean_6h"] = df_feat["temperature"].rolling(window=6, min_periods=1).mean()
    df_feat["temp_roll_std_6h"] = df_feat["temperature"].rolling(window=6, min_periods=1).std().fillna(0)
    df_feat["temp_roll_mean_24h"] = df_feat["temperature"].rolling(window=24, min_periods=1).mean()
    
    # Rapid temperature change (delta over 3 hours)
    df_feat["temp_delta_3h"] = (df_feat["temperature"] - df_feat["temperature"].shift(3)).fillna(0)
    
    # Wind Speed
    df_feat["wind_roll_mean_6h"] = df_feat["wind_speed"].rolling(window=6, min_periods=1).mean()
    df_feat["wind_roll_max_6h"] = df_feat["wind_speed"].rolling(window=6, min_periods=1).max()
    df_feat["wind_delta_3h"] = (df_feat["wind_speed"] - df_feat["wind_speed"].shift(3)).fillna(0)
    
    # Cumulative Precipitation
    df_feat["precip_sum_6h"] = df_feat["precipitation"].rolling(window=6, min_periods=1).sum()
    df_feat["precip_sum_24h"] = df_feat["precipitation"].rolling(window=24, min_periods=1).sum()
    
    # Humidity
    df_feat["humidity_roll_mean_6h"] = df_feat["humidity"].rolling(window=6, min_periods=1).mean()
    df_feat["humidity_delta_3h"] = (df_feat["humidity"] - df_feat["humidity"].shift(3)).fillna(0)
    
    # Diurnal deviation (how much current temperature departs from the average temperature for this specific hour of the day)
    hourly_means = df_feat.groupby("hour")["temperature"].transform("mean")
    df_feat["temp_diurnal_dev"] = df_feat["temperature"] - hourly_means
    
    return df_feat


def get_default_feature_names() -> List[str]:
    """
    Return the standard list of features used for training the Isolation Forest model.
    """
    return [
        "temperature",
        "humidity",
        "precipitation",
        "wind_speed",
        "temp_delta_3h",
        "temp_diurnal_dev",
        "precip_sum_6h",
        "wind_roll_max_6h"
    ]
