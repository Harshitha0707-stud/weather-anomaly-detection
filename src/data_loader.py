"""
Data Loader Module for Weather Anomaly Detection.
Fetches historical weather data from Open-Meteo Archive API and provides geocoding search.
Includes offline fallback data generator.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests

PRESET_CITIES = {
    "London, United Kingdom": {"lat": 51.5074, "lon": -0.1278, "timezone": "Europe/London"},
    "New York, United States": {"lat": 40.7128, "lon": -74.0060, "timezone": "America/New_York"},
    "Tokyo, Japan": {"lat": 35.6762, "lon": 139.6503, "timezone": "Asia/Tokyo"},
    "Paris, France": {"lat": 48.8566, "lon": 2.3522, "timezone": "Europe/Paris"},
    "Mumbai, India": {"lat": 19.0760, "lon": 72.8777, "timezone": "Asia/Kolkata"},
    "Sydney, Australia": {"lat": -33.8688, "lon": 151.2093, "timezone": "Australia/Sydney"},
    "Cairo, Egypt": {"lat": 30.0444, "lon": 31.2357, "timezone": "Africa/Cairo"},
    "São Paulo, Brazil": {"lat": -23.5505, "lon": -46.6333, "timezone": "America/Sao_Paulo"},
    "Singapore": {"lat": 1.3521, "lon": 103.8198, "timezone": "Asia/Singapore"},
    "Toronto, Canada": {"lat": 43.6532, "lon": -79.3832, "timezone": "America/Toronto"}
}


def search_city(query: str) -> List[Dict]:
    """
    Search for cities using Open-Meteo Geocoding API.
    Returns list of matching locations with name, country, lat, lon.
    """
    if not query or len(query.strip()) < 2:
        return []
    
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": query.strip(),
        "count": 5,
        "language": "en",
        "format": "json"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("results", []):
            label = f"{item.get('name')}, {item.get('admin1', '')} ({item.get('country', '')})".replace(",  (", " (")
            results.append({
                "label": label,
                "name": item.get("name"),
                "country": item.get("country"),
                "lat": item.get("latitude"),
                "lon": item.get("longitude"),
                "timezone": item.get("timezone", "UTC")
            })
        return results
    except Exception as e:
        print(f"Error querying Open-Meteo Geocoding API: {e}")
        return []


def fetch_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str = "auto"
) -> pd.DataFrame:
    """
    Fetch hourly historical weather data from Open-Meteo Historical Weather Archive API.
    
    Parameters:
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        start_date (str): Format 'YYYY-MM-DD'
        end_date (str): Format 'YYYY-MM-DD'
        timezone (str): Timezone string (default 'auto')
        
    Returns:
        pd.DataFrame: Cleaned dataframe with hourly weather metrics.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": timezone
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if "hourly" not in data:
            raise ValueError("No hourly weather data found in Open-Meteo API response.")
            
        hourly_data = data["hourly"]
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly_data["time"]),
            "temperature": hourly_data.get("temperature_2m", []),
            "humidity": hourly_data.get("relative_humidity_2m", []),
            "precipitation": hourly_data.get("precipitation", []),
            "wind_speed": hourly_data.get("wind_speed_10m", [])
        })
        
        df["latitude"] = latitude
        df["longitude"] = longitude
        return df

    except Exception as e:
        # Fallback to forecast API past_days or raise error
        print(f"Error fetching from Open-Meteo Archive API: {e}. Attempting forecast archive fallback...")
        try:
            # Open-Meteo Forecast API can provide historical data for recent days
            forecast_url = "https://api.open-meteo.com/v1/forecast"
            forecast_params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                "timezone": timezone
            }
            res = requests.get(forecast_url, params=forecast_params, timeout=15)
            res.raise_for_status()
            data = res.json()
            if "hourly" in data:
                hourly = data["hourly"]
                df = pd.DataFrame({
                    "timestamp": pd.to_datetime(hourly["time"]),
                    "temperature": hourly.get("temperature_2m", []),
                    "humidity": hourly.get("relative_humidity_2m", []),
                    "precipitation": hourly.get("precipitation", []),
                    "wind_speed": hourly.get("wind_speed_10m", [])
                })
                df["latitude"] = latitude
                df["longitude"] = longitude
                return df
        except Exception as fallback_err:
            print(f"Forecast fallback failed: {fallback_err}")
            
        raise RuntimeError(f"Failed to fetch weather data: {e}")


def generate_synthetic_weather(
    days: int = 180,
    end_date: Optional[datetime] = None,
    city_name: str = "Simulated Station"
) -> pd.DataFrame:
    """
    Generate realistic synthetic hourly weather data with simulated anomalies
    (e.g., heat waves, torrential cloudbursts, severe windstorms, dry drops).
    Used for instant offline testing and fallback demonstration.
    """
    np.random.seed(42)
    
    if end_date is None:
        end_date = datetime.now() - timedelta(days=5)
    start_date = end_date - timedelta(days=days)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq="h")
    n = len(date_range)
    
    # Base diurnal cycle and seasonal wave
    day_of_year = date_range.dayofyear.values
    hour_of_day = date_range.hour.values
    
    seasonal_temp = 18 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    diurnal_temp = 5 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
    noise_temp = np.random.normal(0, 1.8, n)
    temp = seasonal_temp + diurnal_temp + noise_temp
    
    # Humidity (inversely correlated with temperature)
    humidity = 70 - 1.2 * diurnal_temp + np.random.normal(0, 6, n)
    humidity = np.clip(humidity, 15, 100)
    
    # Wind speed (Gamma or Rayleigh distributed)
    wind_speed = np.random.gamma(shape=2.5, scale=4.0, size=n)
    
    # Precipitation (Mostly 0 with occasional showers)
    rain_prob = np.random.uniform(0, 1, n)
    precipitation = np.where(rain_prob > 0.88, np.random.exponential(scale=2.2, size=n), 0.0)
    
    # Inject Synthetic Weather Anomalies
    # 1. Extreme Heatwave (days 40-42)
    heatwave_start = int(n * 0.22)
    heatwave_duration = 36
    temp[heatwave_start:heatwave_start + heatwave_duration] += 12.5
    humidity[heatwave_start:heatwave_start + heatwave_duration] -= 25.0
    
    # 2. Severe Cloudburst / Flash Flood (day 80)
    rain_burst = int(n * 0.45)
    precipitation[rain_burst:rain_burst + 8] = np.random.uniform(35.0, 75.0, 8)
    humidity[rain_burst:rain_burst + 8] = 98.0
    temp[rain_burst:rain_burst + 8] -= 7.0
    
    # 3. Severe Gale / Windstorm (day 115)
    gale_start = int(n * 0.65)
    wind_speed[gale_start:gale_start + 12] = np.random.uniform(65.0, 95.0, 12)
    precipitation[gale_start:gale_start + 12] += np.random.uniform(8.0, 20.0, 12)
    
    # 4. Sudden Cold Snap (day 150)
    cold_start = int(n * 0.85)
    temp[cold_start:cold_start + 48] -= 16.0
    
    humidity = np.clip(humidity, 10, 100)
    precipitation = np.clip(precipitation, 0, 150)
    wind_speed = np.clip(wind_speed, 0, 140)
    
    df = pd.DataFrame({
        "timestamp": date_range,
        "temperature": np.round(temp, 2),
        "humidity": np.round(humidity, 1),
        "precipitation": np.round(precipitation, 2),
        "wind_speed": np.round(wind_speed, 2),
        "location": city_name
    })
    
    return df
