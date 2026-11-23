"""
Model Module for Weather Anomaly Detection using Isolation Forest.
Handles model training, anomaly prediction, severity scoring, and anomaly type classification.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


class WeatherAnomalyDetector:
    """
    Isolation Forest based detector for identifying meteorological anomalies.
    """
    
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 150,
        random_state: int = 42
    ):
        """
        Initialize the detector.
        
        Parameters:
            contamination (float): Expected proportion of outliers in the data (0.01 to 0.20).
            n_estimators (int): Number of decision trees in the ensemble.
            random_state (int): Seed for reproducibility.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.feature_cols: List[str] = []
        self.is_fitted: bool = False
        self.threshold: float = 0.0

    def fit_predict(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit the Isolation Forest model on selected features and predict anomalies.
        
        Parameters:
            df (pd.DataFrame): Preprocessed weather dataframe.
            feature_cols (List[str], optional): List of column names to train on.
            
        Returns:
            pd.DataFrame: Augmented dataframe with `is_anomaly`, `anomaly_score`,
                          `severity_index`, and `anomaly_type`.
        """
        df_res = df.copy()
        
        if feature_cols is None:
            # Default feature selection
            available = [
                "temperature", "humidity", "precipitation", "wind_speed",
                "temp_delta_3h", "temp_diurnal_dev", "precip_sum_6h", "wind_roll_max_6h"
            ]
            self.feature_cols = [c for c in available if c in df_res.columns]
        else:
            self.feature_cols = [c for c in feature_cols if c in df_res.columns]

        if not self.feature_cols:
            raise ValueError("No valid feature columns found for anomaly detection.")

        X = df_res[self.feature_cols].values
        
        # Scale features using RobustScaler (less sensitive to extreme outliers than StandardScaler)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest and predict
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        # Predictions: -1 for anomaly, 1 for normal
        raw_preds = self.model.predict(X_scaled)
        df_res["is_anomaly"] = (raw_preds == -1)
        
        # Anomaly score: raw score from decision_function
        # Note: In scikit-learn, lower score = more anomalous. Negative values are anomalies.
        raw_scores = self.model.decision_function(X_scaled)
        df_res["raw_score"] = raw_scores
        
        # Convert to an intuitive 0-100 severity index where 100 is most anomalous
        min_s = raw_scores.min()
        max_s = raw_scores.max()
        if max_s > min_s:
            # Invert: lower raw score -> higher severity
            severity = 100.0 * (max_s - raw_scores) / (max_s - min_s)
        else:
            severity = np.zeros(len(raw_scores))
            
        df_res["severity_index"] = np.round(severity, 1)
        
        # Tag primary anomaly driver / type
        df_res["anomaly_type"] = self._classify_anomaly_types(df_res)
        
        return df_res

    def _classify_anomaly_types(self, df: pd.DataFrame) -> pd.Series:
        """
        Assign an informative meteorological category to detected anomalies
        based on statistical threshold exceedances.
        """
        anomaly_types = pd.Series("Normal", index=df.index)
        
        if not df["is_anomaly"].any():
            return anomaly_types
            
        anom_mask = df["is_anomaly"]
        
        # Baseline percentiles
        temp_p95 = df["temperature"].quantile(0.95)
        temp_p05 = df["temperature"].quantile(0.05)
        wind_p95 = df["wind_speed"].quantile(0.95)
        precip_p95 = df["precipitation"].quantile(0.95)
        
        # Classify each anomalous record
        for idx in df[anom_mask].index:
            row = df.loc[idx]
            reasons = []
            
            # Temperature extremes
            if row.get("temperature", 0) >= temp_p95:
                reasons.append("Heatwave / High Temp")
            elif row.get("temperature", 0) <= temp_p05:
                reasons.append("Cold Snap / Freeze")
                
            # Precipitation extremes
            if row.get("precipitation", 0) > max(15.0, precip_p95) or row.get("precip_sum_6h", 0) > 25.0:
                reasons.append("Heavy Rainfall")
                
            # Wind extremes
            if row.get("wind_speed", 0) >= max(35.0, wind_p95):
                reasons.append("Severe Wind / Gale")
                
            # Rapid shift
            if abs(row.get("temp_delta_3h", 0)) >= 8.0:
                reasons.append("Abrupt Temp Shift")
                
            if abs(row.get("humidity_delta_3h", 0)) >= 30.0:
                reasons.append("Sudden Humidity Jump")
                
            if len(reasons) == 1:
                anomaly_types.loc[idx] = reasons[0]
            elif len(reasons) > 1:
                anomaly_types.loc[idx] = f"Compound ({' + '.join(reasons[:2])})"
            else:
                anomaly_types.loc[idx] = "Multivariate Pattern Anomaly"
                
        return anomaly_types

    def get_summary_stats(self, df: pd.DataFrame) -> dict:
        """
        Generate high-level metrics for dashboard cards.
        """
        total_records = len(df)
        anomaly_count = int(df["is_anomaly"].sum())
        anomaly_pct = (anomaly_count / total_records * 100) if total_records > 0 else 0.0
        
        max_temp = float(df["temperature"].max())
        min_temp = float(df["temperature"].min())
        max_wind = float(df["wind_speed"].max())
        max_precip = float(df["precipitation"].max())
        total_precip = float(df["precipitation"].sum())
        
        return {
            "total_records": total_records,
            "anomaly_count": anomaly_count,
            "anomaly_pct": round(anomaly_pct, 2),
            "max_temp": round(max_temp, 1),
            "min_temp": round(min_temp, 1),
            "max_wind": round(max_wind, 1),
            "max_precip": round(max_precip, 1),
            "total_precip": round(total_precip, 1)
        }
