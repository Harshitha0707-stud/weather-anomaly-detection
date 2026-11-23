# ⛅ Weather Anomaly Detection Dashboard

An end-to-end Machine Learning web application built in Python that automatically flags unusual meteorological events (such as extreme heatwaves, unseasonal cold snaps, torrential cloudbursts, and severe gale-force wind gusts) using **Isolation Forest** and the **Open-Meteo Historical Weather API**.

---

## 🌟 Key Features

- **🌐 Live & Historical Weather Data**:
  - Direct integration with the Open-Meteo Archive API (no API key required).
  - Preset major global cities (London, New York, Tokyo, Paris, Mumbai, Sydney, Cairo, São Paulo, etc.).
  - Dynamic city geocoding search for any city, town, or province across the globe.
  - Custom Latitude / Longitude input mode.
  - Built-in synthetic fallback generator for offline testing.

- **⚙️ Feature Engineering & Preprocessing**:
  - Automated timestamp alignment and missing value imputation.
  - Multi-scale rolling statistical metrics (6-hour and 24-hour rolling averages, rolling max).
  - Rate-of-change indicators (3-hour temperature, humidity, and wind speed deltas).
  - Diurnal cycle baseline deviations (detecting atypical departures from normal hour-of-day temperatures).

- **🤖 Unsupervised Machine Learning (Isolation Forest)**:
  - Scikit-learn's `IsolationForest` algorithm with `RobustScaler` preprocessing.
  - Dynamic contamination slider (adjust expected outlier percentage from 1% to 15%).
  - Configurable ensemble size (`n_estimators`).
  - Feature selection controls (choose which variables drive anomaly detection).
  - Automated categorization of anomalies into meteorological triggers (Heatwave, Cold Snap, Torrential Rain, Severe Wind, Compound Anomalies).
  - Continuous Severity Index (0–100 scale).

- **📊 Rich Interactive Visualizations (Plotly)**:
  - Multi-metric synchronized timeline with interactive zoom and range selector.
  - Individual parameter timelines with highlighted red anomaly markers and detailed hover tooltips.
  - Exploratory Data Analysis (EDA) correlation matrix heatmap.
  - Diurnal anomaly distribution analysis (hourly frequency pattern).
  - Distribution comparisons (Normal vs Anomaly boxplots).
  - 3D multivariate scatter plot in the Temperature–Humidity–Wind feature space.

- **📥 CSV Data Export**:
  - One-click export of detected anomaly records.
  - Full dataset export with computed features, severity scores, and anomaly classifications.

---

## 📁 Project Structure

```
weather anomoly detection/
├── app.py                      # Streamlit dashboard application
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation and guide
├── src/
│   ├── __init__.py             # Package initializer
│   ├── data_loader.py          # Open-Meteo API client and synthetic data generator
│   ├── preprocessor.py         # Data cleaning and feature engineering pipeline
│   ├── model.py                # Isolation Forest model and anomaly classification
│   └── visualizer.py           # Plotly interactive charting functions
└── data/
    └── sample_weather.csv      # Offline fallback dataset
```

---

## 🚀 Getting Started in VS Code

### Step 1: Open the Project in VS Code
1. Launch **Visual Studio Code**.
2. Click **File > Open Folder...** and select:
   ```
   c:\Users\Pooja C\Desktop\weather anomoly detection
   ```

### Step 2: Open the Integrated Terminal
- Press <kbd>Ctrl</kbd> + <kbd>`</kbd> (backtick) or go to **Terminal > New Terminal** in the top menu.

### Step 3: (Recommended) Create & Activate a Virtual Environment
In your VS Code terminal (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If you see an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Launch the Streamlit Application
Run the following command in the terminal:
```powershell
streamlit run app.py
```

The terminal will display the local URL:
```
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```
Streamlit will automatically open the dashboard in your default web browser!

---

## 🛠️ How to Use the Dashboard

1. **Select Data Source**:
   - Pick **Open-Meteo API** to analyze real historical climate records, or select **Sample Dataset** for an instant offline demo.
   - Choose a preset city, or search for any location (e.g., `Paris`, `Tokyo`, `Mumbai`, `New York`).
   - Pick the start and end dates (e.g., past 3 to 6 months).

2. **Adjust Machine Learning Settings**:
   - **Contamination Rate**: Increase to flag more subtle deviations, or decrease to flag only extreme outliers.
   - **Estimators**: Adjust the number of Isolation Forest decision trees.
   - **Features**: Toggle whether rate-of-change, diurnal deviations, or rolling precipitation are fed into the model.

3. **Explore Dashboard Tabs**:
   - **📈 Time Series**: Inspect temperature, humidity, rainfall, and wind timelines with highlighted anomaly points.
   - **🔍 EDA**: Analyze correlation heatmaps and boxplot distributions between normal and anomalous intervals.
   - **🧠 ML Diagnostics**: Rotate the 3D feature space and view the breakdown of anomaly triggers.
   - **📋 Anomaly Records & CSV Export**: Review anomalies in a table and click **Download CSV** to export the data.

---

## 🔬 Machine Learning Methodology

### Why Isolation Forest?
Unlike classification algorithms that require historical ground-truth labels (which are rarely available for unusual weather events), **Isolation Forest** is an unsupervised learning algorithm that exploits two quantitative properties of anomalies:
1. They are few in number.
2. They possess attribute-values distinct from nominal instances.

Isolation Forest recursively isolates observations by randomly selecting a feature and a split value. Anomalous weather states (such as an unprecedented temperature spike or combined gale winds with cloudbursts) have shorter average path lengths in the tree structure, allowing them to be identified efficiently without assuming a normal distribution.

---

## 📜 License
This project is open-source and free to use for educational, research, and production applications.
