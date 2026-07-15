"""
src/eda/eda_config.py
======================
Centralised EDA styling, colour palette, and shared constants.
All plots import from here to guarantee publication consistency.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Publication style ───────────────────────────────────────
STYLE = {
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "axes.edgecolor":      "#444444",
    "axes.linewidth":      0.8,
    "axes.grid":           True,
    "grid.color":          "#e0e0e0",
    "grid.linewidth":      0.5,
    "grid.linestyle":      "--",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "font.family":         "DejaVu Sans",
    "font.size":           10,
    "axes.titlesize":      12,
    "axes.titleweight":    "bold",
    "axes.labelsize":      10,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "legend.fontsize":     9,
    "legend.framealpha":   0.9,
    "figure.dpi":          150,
    "savefig.dpi":         150,
    "savefig.bbox":        "tight",
    "savefig.facecolor":   "white",
}
plt.rcParams.update(STYLE)

# ── Station colours ────────────────────────────────────────
STATION_COLORS = {
    "noida_sector_1":  "#1a6faf",   # blue
    "noida_sector_62": "#c0392b",   # red
}
STATION_LABELS = {
    "noida_sector_1":  "Sector-1 (UPPCB)",
    "noida_sector_62": "Sector-62 (CAAQMS)",
}
STATION_MARKERS = {
    "noida_sector_1":  "o",
    "noida_sector_62": "s",
}

# ── AQI category colours (CPCB standard) ──────────────────
AQI_COLORS = {
    "Good":         "#009966",
    "Satisfactory": "#ffde33",
    "Moderate":     "#ff9933",
    "Poor":         "#cc0033",
    "Very Poor":    "#660099",
    "Severe":       "#7e0023",
}
AQI_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# ── Pollutant metadata ────────────────────────────────────
POLLUTANTS = {
    "pm25":        {"label": "PM₂.₅ (µg/m³)",   "color": "#8e44ad", "limit": 60},
    "pm10":        {"label": "PM₁₀ (µg/m³)",    "color": "#e67e22", "limit": 100},
    "no2":         {"label": "NO₂ (µg/m³)",     "color": "#2980b9", "limit": 80},
    "nh3":         {"label": "NH₃ (µg/m³)",     "color": "#27ae60", "limit": 400},
    "so2":         {"label": "SO₂ (µg/m³)",     "color": "#c0392b", "limit": 80},
    "co":          {"label": "CO (mg/m³)",       "color": "#7f8c8d", "limit": 10},
    "o3":          {"label": "O₃ (µg/m³)",      "color": "#16a085", "limit": 180},
    "pb":          {"label": "Pb (µg/m³)",       "color": "#d35400", "limit": 1.0},
    "temperature": {"label": "Temperature (°C)", "color": "#e74c3c", "limit": None},
    "humidity":    {"label": "Humidity (%)",     "color": "#3498db", "limit": None},
    "wind_speed":  {"label": "Wind Speed (km/h)","color": "#1abc9c", "limit": None},
    "pressure":    {"label": "Pressure (hPa)",   "color": "#95a5a6", "limit": None},
}

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

SEASON_MAP = {
    1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Summer",
    6:"Summer",7:"Monsoon",8:"Monsoon",9:"Monsoon",10:"Post-Monsoon",
    11:"Winter",12:"Winter",
}
SEASON_COLORS = {
    "Winter":       "#2980b9",
    "Spring":       "#27ae60",
    "Summer":       "#e67e22",
    "Monsoon":      "#16a085",
    "Post-Monsoon": "#8e44ad",
}

FIG_DIR = "/home/claude/aqi_noida_project/outputs/figures/eda"

def save_fig(fig, name, tight=True):
    path = f"{FIG_DIR}/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight" if tight else None,
                facecolor="white")
    plt.close(fig)
    return path
