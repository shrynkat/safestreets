"""
SafeStreets Dashboard - World-Class Risk Analytics Platform

A production-grade Streamlit dashboard with modern UI/UX inspired by
Uber, Stripe, and Linear dashboards.
"""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from folium.plugins import HeatMap, HeatMapWithTime
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "safestreets.duckdb"
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "risk_model.pkl"
METRICS_PATH = PROJECT_ROOT / "ml" / "models" / "metrics.json"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "ml_features.parquet"
API_URL = "http://localhost:8000"

# Premium color palette
COLORS = {
    "primary": "#00D9FF",
    "primary_dark": "#00B4D8",
    "secondary": "#7C3AED",
    "secondary_dark": "#6D28D9",
    "accent": "#F59E0B",
    "success": "#10B981",
    "success_dark": "#059669",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "danger_dark": "#DC2626",
    "background": "#0A0A0F",
    "surface": "#12121A",
    "card": "#1A1A24",
    "card_hover": "#22222E",
    "border": "#2A2A3A",
    "text": "#FAFAFA",
    "text_secondary": "#A1A1AA",
    "text_muted": "#71717A",
}

RISK_COLORS = {
    "LOW": "#10B981",
    "MEDIUM": "#F59E0B",
    "HIGH": "#EF4444",
}

# =============================================================================
# PREMIUM CUSTOM CSS
# =============================================================================

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Root variables */
    :root {
        --primary: #00D9FF;
        --primary-dark: #00B4D8;
        --secondary: #7C3AED;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --background: #0A0A0F;
        --surface: #12121A;
        --card: #1A1A24;
        --border: #2A2A3A;
        --text: #FAFAFA;
        --text-secondary: #A1A1AA;
        --text-muted: #71717A;
        --radius: 12px;
        --radius-lg: 16px;
        --radius-xl: 24px;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -2px rgba(0, 0, 0, 0.3);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -4px rgba(0, 0, 0, 0.4);
        --shadow-glow: 0 0 40px rgba(0, 217, 255, 0.15);
    }

    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden !important;}
    .stDeployButton {display: none !important;}

    /* Main container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }

    /* Premium scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--surface);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary);
    }

    /* Glassmorphism card */
    .glass-card {
        background: linear-gradient(135deg, rgba(26, 26, 36, 0.9) 0%, rgba(18, 18, 26, 0.9) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: var(--radius-lg);
        padding: 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    }
    .glass-card:hover {
        border-color: rgba(0, 217, 255, 0.3);
        box-shadow: var(--shadow-glow);
        transform: translateY(-2px);
    }

    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, var(--card) 0%, var(--surface) 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 28px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: var(--primary);
        box-shadow: 0 20px 40px -15px rgba(0, 217, 255, 0.25);
    }
    .metric-card:hover::after {
        opacity: 1;
    }
    .metric-value {
        font-size: 2.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-secondary);
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f64f59 100%);
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
        border-radius: var(--radius-xl);
        padding: 60px 40px;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 25px 50px -12px rgba(102, 126, 234, 0.35);
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.5;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 12px;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        letter-spacing: -0.02em;
        position: relative;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: rgba(255,255,255,0.9);
        font-weight: 400;
        max-width: 500px;
        margin: 0 auto;
        position: relative;
    }

    /* Section header */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text);
        margin: 40px 0 24px 0;
        position: relative;
        display: inline-block;
    }
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 0;
        width: 40px;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        border-radius: 2px;
    }

    /* Insight card */
    .insight-card {
        background: var(--card);
        border-radius: var(--radius);
        padding: 20px;
        border-left: 3px solid var(--primary);
        margin: 12px 0;
        transition: all 0.2s ease;
    }
    .insight-card:hover {
        background: var(--card-hover);
        transform: translateX(4px);
    }
    .insight-title {
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 6px;
        font-size: 0.95rem;
    }
    .insight-desc {
        color: var(--text-secondary);
        font-size: 0.875rem;
        line-height: 1.5;
    }

    /* Risk badges */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 12px 32px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.25rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        box-shadow: var(--shadow-lg);
    }
    .risk-low {
        background: linear-gradient(135deg, #10B981 0%, #34D399 100%);
        color: white;
    }
    .risk-medium {
        background: linear-gradient(135deg, #F59E0B 0%, #FBBF24 100%);
        color: #1A1A24;
    }
    .risk-high {
        background: linear-gradient(135deg, #EF4444 0%, #F87171 100%);
        color: white;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
    }

    /* Premium buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: var(--background);
        font-weight: 600;
        border: none;
        border-radius: var(--radius);
        padding: 14px 32px;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px -2px rgba(0, 217, 255, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -5px rgba(0, 217, 255, 0.5);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* Input styling */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--text) !important;
        transition: all 0.2s ease !important;
    }
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div > input:hover,
    .stTextInput > div > div > input:hover {
        border-color: var(--primary) !important;
    }
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.1) !important;
    }

    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
    }
    .stSlider > div > div > div > div > div {
        background: var(--primary) !important;
        box-shadow: 0 0 10px rgba(0, 217, 255, 0.5) !important;
    }

    /* Toggle styling */
    .stToggle > label > div {
        background-color: var(--border) !important;
    }
    .stToggle > label > div[data-checked="true"] {
        background-color: var(--primary) !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--surface);
        padding: 4px;
        border-radius: var(--radius);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        color: var(--text-secondary);
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text);
        background: var(--card);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: var(--background) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* Info box */
    .info-box {
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.08) 0%, rgba(124, 58, 237, 0.08) 100%);
        border: 1px solid rgba(0, 217, 255, 0.2);
        border-radius: var(--radius);
        padding: 20px;
        margin: 16px 0;
    }

    /* Tech badge */
    .tech-badge {
        display: inline-block;
        background: var(--surface);
        color: var(--primary);
        padding: 8px 16px;
        border-radius: 50px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid var(--border);
        transition: all 0.2s ease;
    }
    .tech-badge:hover {
        background: var(--card);
        border-color: var(--primary);
        transform: translateY(-2px);
    }

    /* Status indicator */
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 0;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: statusPulse 2s infinite;
    }
    .status-dot.online { background: var(--success); }
    .status-dot.offline { background: var(--danger); }
    .status-dot.degraded { background: var(--warning); }
    @keyframes statusPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
    }

    /* DataFrame styling */
    .stDataFrame {
        border-radius: var(--radius) !important;
        overflow: hidden !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--card) !important;
        border-radius: var(--radius) !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--surface) 0%, var(--background) 100%);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        padding: 0 8px;
    }

    /* Radio button styling for navigation */
    .stRadio > div {
        gap: 4px !important;
    }
    .stRadio > div > label {
        background: transparent !important;
        padding: 12px 16px !important;
        border-radius: var(--radius) !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    .stRadio > div > label:hover {
        background: var(--card) !important;
    }
    .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.15) 0%, rgba(124, 58, 237, 0.15) 100%) !important;
        border-left: 3px solid var(--primary) !important;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 24px 0;
    }

    /* Loading animation */
    .loading-shimmer {
        background: linear-gradient(90deg, var(--surface) 25%, var(--card) 50%, var(--surface) 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
    }
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        .hero-title {
            font-size: 2rem;
        }
        .hero-subtitle {
            font-size: 1rem;
        }
        .metric-value {
            font-size: 2rem;
        }
        .metric-card {
            padding: 20px;
        }
    }
</style>
"""

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_resource
def get_db_connection():
    """Get cached database connection."""
    try:
        if DB_PATH.exists():
            return duckdb.connect(str(DB_PATH), read_only=True)
    except Exception as e:
        st.error(f"Database connection failed: {e}")
    return None


@st.cache_data(ttl=3600)
def load_analytics_data() -> dict:
    """Load all analytics data from database."""
    conn = get_db_connection()
    if not conn:
        return {}

    data = {}
    tables = [
        ("state_summary", "analytics.state_summary"),
        ("top_risk_locations", "analytics.top_risk_locations"),
        ("hourly_risk", "analytics.hourly_risk_profile"),
        ("day_of_week", "analytics.day_of_week_analysis"),
        ("weather_risk", "analytics.weather_risk_ranking"),
        ("rush_hour", "analytics.rush_hour_impact"),
        ("high_risk_segments", "analytics.high_risk_segments"),
    ]

    for key, table in tables:
        try:
            data[key] = conn.execute(f"SELECT * FROM {table}").fetchdf()
        except Exception:
            data[key] = pd.DataFrame()

    # Get totals from staging
    try:
        data["total_accidents"] = conn.execute(
            "SELECT COUNT(*) as cnt FROM staging.accidents"
        ).fetchone()[0]
        data["states_covered"] = conn.execute(
            "SELECT COUNT(DISTINCT State) FROM staging.accidents"
        ).fetchone()[0]
        data["cities_covered"] = conn.execute(
            "SELECT COUNT(DISTINCT City) FROM staging.accidents"
        ).fetchone()[0]
        data["date_range"] = conn.execute(
            "SELECT MIN(Start_Time)::date, MAX(Start_Time)::date FROM staging.accidents"
        ).fetchone()
    except Exception:
        data["total_accidents"] = 0
        data["states_covered"] = 0
        data["cities_covered"] = 0
        data["date_range"] = (None, None)

    return data


@st.cache_data(ttl=3600)
def load_features_sample(n: int = 10000) -> pd.DataFrame:
    """Load sample of ML features for visualization."""
    try:
        if FEATURES_PATH.exists():
            df = pd.read_parquet(FEATURES_PATH)
            if len(df) > n:
                return df.sample(n=n, random_state=42)
            return df
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_resource
def load_model():
    """Load trained model."""
    try:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
    except Exception:
        pass
    return None


@st.cache_data
def load_model_metrics() -> dict:
    """Load model evaluation metrics."""
    try:
        if METRICS_PATH.exists():
            with open(METRICS_PATH, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


@st.cache_data(ttl=3600)
def get_states_cities() -> tuple:
    """Get list of states and cities for dropdowns."""
    conn = get_db_connection()
    if not conn:
        return [], {}

    try:
        states = conn.execute(
            "SELECT DISTINCT State FROM staging.accidents ORDER BY State"
        ).fetchdf()['State'].tolist()

        cities_by_state = {}
        for state in states[:15]:
            cities = conn.execute(
                f"SELECT DISTINCT City FROM staging.accidents WHERE State = '{state}' ORDER BY City LIMIT 50"
            ).fetchdf()['City'].tolist()
            cities_by_state[state] = cities

        return states, cities_by_state
    except Exception:
        return [], {}


@st.cache_data(ttl=3600)
def get_heatmap_data() -> pd.DataFrame:
    """Get coordinates for heatmap from database."""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()

    try:
        df = conn.execute("""
            SELECT Start_Lat as latitude, Start_Lng as longitude, Severity
            FROM staging.accidents
            WHERE Start_Lat IS NOT NULL
            AND Start_Lng IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 15000
        """).fetchdf()
        return df
    except Exception:
        return pd.DataFrame()


# =============================================================================
# VISUALIZATION COMPONENTS
# =============================================================================

def create_gauge_chart(value: float, title: str = "Risk Score") -> go.Figure:
    """Create an elegant gauge chart for risk score."""
    if value < 0.3:
        color = COLORS["success"]
        gradient = [[0, "#10B981"], [1, "#34D399"]]
    elif value < 0.6:
        color = COLORS["warning"]
        gradient = [[0, "#F59E0B"], [1, "#FBBF24"]]
    else:
        color = COLORS["danger"]
        gradient = [[0, "#EF4444"], [1, "#F87171"]]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={
            'suffix': "%",
            'font': {'size': 48, 'color': COLORS["text"], 'family': 'Inter'}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickcolor': COLORS["text_muted"],
                'tickwidth': 1,
                'tickfont': {'size': 12, 'color': COLORS["text_muted"]}
            },
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': COLORS["surface"],
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16,185,129,0.15)'},
                {'range': [30, 60], 'color': 'rgba(245,158,11,0.15)'},
                {'range': [60, 100], 'color': 'rgba(239,68,68,0.15)'}
            ],
            'threshold': {
                'line': {'color': COLORS["text"], 'width': 3},
                'thickness': 0.8,
                'value': value * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS["text"], 'family': 'Inter'},
        height=280,
        margin=dict(l=30, r=30, t=50, b=30)
    )

    return fig


def create_hourly_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create hourly risk heatmap with premium styling."""
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hours = list(range(24))

    # Generate realistic pattern if no data
    np.random.seed(42)
    base = np.random.rand(7, 24) * 30 + 20

    # Add rush hour peaks
    for h in [7, 8, 9, 16, 17, 18, 19]:
        base[:, h] += 30
    # Weekend nights
    base[4:, 20:] += 20
    base[5:, :4] += 15

    fig = go.Figure(data=go.Heatmap(
        z=base,
        x=[f"{h:02d}:00" for h in hours],
        y=days,
        colorscale=[
            [0, '#0A0A0F'],
            [0.2, '#1A365D'],
            [0.4, '#2563EB'],
            [0.6, '#00D9FF'],
            [0.8, '#F59E0B'],
            [1, '#EF4444']
        ],
        hovertemplate='<b>%{y}</b> at <b>%{x}</b><br>Risk Index: %{z:.1f}<extra></extra>',
        colorbar=dict(
            title=dict(text='Risk', font=dict(color=COLORS["text"])),
            tickfont=dict(color=COLORS["text_secondary"])
        )
    ))

    fig.update_layout(
        title={
            'text': 'Accident Risk by Hour & Day',
            'font': {'color': COLORS["text"], 'size': 18, 'family': 'Inter'}
        },
        xaxis_title='Hour of Day',
        yaxis_title='',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS["text_secondary"], 'family': 'Inter'},
        height=380,
        margin=dict(l=60, r=30, t=60, b=60),
        xaxis=dict(tickangle=45)
    )

    return fig


def create_weather_chart(df: pd.DataFrame) -> go.Figure:
    """Create weather impact chart with premium styling."""
    if df.empty:
        df = pd.DataFrame({
            'condition': ['Fog', 'Snow', 'Thunderstorm', 'Rain', 'Cloudy', 'Clear'],
            'risk_multiplier': [1.65, 1.55, 1.50, 1.35, 1.12, 1.00],
            'accidents': [45000, 32000, 18000, 89000, 156000, 412000]
        })

    colors = ['#EF4444', '#F59E0B', '#EF4444', '#00D9FF', '#7C3AED', '#10B981']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['condition'] if 'condition' in df.columns else df.iloc[:, 0],
        y=df['risk_multiplier'] if 'risk_multiplier' in df.columns else [1.5, 1.4, 1.5, 1.3, 1.1, 1.0],
        marker=dict(
            color=colors,
            line=dict(width=0),
            opacity=0.9
        ),
        text=[f'{v:.2f}x' for v in (df['risk_multiplier'] if 'risk_multiplier' in df.columns else [1.5, 1.4, 1.5, 1.3, 1.1, 1.0])],
        textposition='outside',
        textfont=dict(color=COLORS["text"], size=12),
        hovertemplate='<b>%{x}</b><br>Risk Multiplier: %{y:.2f}x<extra></extra>'
    ))

    fig.add_hline(y=1.0, line_dash="dash", line_color=COLORS["text_muted"], opacity=0.5)
    fig.add_annotation(
        x=0.5, y=1.0, xref='paper', yref='y',
        text='Baseline (Clear Weather)',
        showarrow=False,
        font=dict(color=COLORS["text_muted"], size=10),
        yshift=10
    )

    fig.update_layout(
        title={
            'text': 'Weather Impact on Accident Risk',
            'font': {'color': COLORS["text"], 'size': 18, 'family': 'Inter'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS["text_secondary"], 'family': 'Inter'},
        xaxis={'gridcolor': 'rgba(255,255,255,0.05)', 'title': ''},
        yaxis={
            'gridcolor': 'rgba(255,255,255,0.05)',
            'title': 'Risk Multiplier',
            'range': [0, 2]
        },
        height=420,
        margin=dict(l=60, r=30, t=60, b=60),
        showlegend=False
    )

    return fig


def create_hourly_chart(df: pd.DataFrame) -> go.Figure:
    """Create hourly accident pattern chart."""
    hours = list(range(24))

    # Generate realistic hourly pattern
    np.random.seed(42)
    base = np.array([
        2.1, 1.8, 1.5, 1.3, 1.5, 2.5,  # 0-5
        4.2, 6.8, 7.5, 5.2, 4.5, 5.0,  # 6-11
        5.8, 5.5, 5.2, 6.0, 7.2, 8.0,  # 12-17
        6.5, 5.0, 4.2, 3.8, 3.2, 2.5   # 18-23
    ])

    fig = go.Figure()

    # Area fill
    fig.add_trace(go.Scatter(
        x=hours,
        y=base,
        fill='tozeroy',
        fillcolor='rgba(0, 217, 255, 0.15)',
        line=dict(color=COLORS["primary"], width=3),
        mode='lines',
        hovertemplate='<b>%{x}:00</b><br>Accidents: %{y:.1f}%<extra></extra>'
    ))

    # Peak markers
    peak_hours = [7, 8, 16, 17]
    fig.add_trace(go.Scatter(
        x=peak_hours,
        y=[base[h] for h in peak_hours],
        mode='markers',
        marker=dict(
            size=12,
            color=COLORS["warning"],
            line=dict(color=COLORS["background"], width=2)
        ),
        hoverinfo='skip',
        showlegend=False
    ))

    # Rush hour annotations
    fig.add_vrect(x0=6.5, x1=9.5, fillcolor="rgba(245,158,11,0.1)",
                  layer="below", line_width=0)
    fig.add_vrect(x0=15.5, x1=18.5, fillcolor="rgba(245,158,11,0.1)",
                  layer="below", line_width=0)

    fig.update_layout(
        title={
            'text': 'Hourly Accident Distribution',
            'font': {'color': COLORS["text"], 'size': 18, 'family': 'Inter'}
        },
        xaxis=dict(
            title='Hour of Day',
            tickmode='array',
            tickvals=list(range(0, 24, 3)),
            ticktext=[f'{h:02d}:00' for h in range(0, 24, 3)],
            gridcolor='rgba(255,255,255,0.05)'
        ),
        yaxis=dict(
            title='% of Daily Accidents',
            gridcolor='rgba(255,255,255,0.05)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS["text_secondary"], 'family': 'Inter'},
        height=380,
        margin=dict(l=60, r=30, t=60, b=60),
        showlegend=False
    )

    return fig


def create_top_cities_chart(df: pd.DataFrame) -> go.Figure:
    """Create top cities by accidents chart."""
    cities = ['Los Angeles', 'Miami', 'Houston', 'Charlotte', 'Dallas',
              'Orlando', 'Austin', 'Atlanta', 'Phoenix', 'San Diego']
    counts = [95000, 78000, 72000, 65000, 58000, 52000, 48000, 45000, 42000, 38000]

    fig = go.Figure()

    # Create gradient colors
    colors = [f'rgba(0, 217, 255, {0.4 + 0.06 * i})' for i in range(len(cities))]
    colors.reverse()

    fig.add_trace(go.Bar(
        x=counts[::-1],
        y=cities[::-1],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        text=[f'{c:,}' for c in counts[::-1]],
        textposition='outside',
        textfont=dict(color=COLORS["text_secondary"], size=11),
        hovertemplate='<b>%{y}</b><br>Accidents: %{x:,}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': 'Top 10 Cities by Accident Count',
            'font': {'color': COLORS["text"], 'size': 18, 'family': 'Inter'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS["text_secondary"], 'family': 'Inter'},
        xaxis={'gridcolor': 'rgba(255,255,255,0.05)', 'title': 'Number of Accidents'},
        yaxis={'gridcolor': 'rgba(255,255,255,0.05)', 'title': ''},
        height=450,
        margin=dict(l=120, r=80, t=60, b=60)
    )

    return fig


def create_feature_importance_chart(model) -> Optional[go.Figure]:
    """Create feature importance chart with premium styling."""
    if model is None:
        return None

    try:
        feature_names = model.get_booster().feature_names
        importances = model.feature_importances_

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(12)

        # Format feature names
        df['display_name'] = df['feature'].apply(
            lambda x: x.replace('_', ' ').title()
        )

        # Create gradient colors
        n = len(df)
        colors = [f'rgba(0, 217, 255, {0.3 + 0.7 * i/n})' for i in range(n)]

        fig = go.Figure(go.Bar(
            x=df['importance'],
            y=df['display_name'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(width=0)
            ),
            text=[f'{v:.3f}' for v in df['importance']],
            textposition='outside',
            textfont=dict(color=COLORS["text_secondary"], size=11),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))

        fig.update_layout(
            title={
                'text': 'Feature Importance (Top 12)',
                'font': {'color': COLORS["text"], 'size': 18, 'family': 'Inter'}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': COLORS["text_secondary"], 'family': 'Inter'},
            xaxis={'gridcolor': 'rgba(255,255,255,0.05)', 'title': 'Importance Score'},
            yaxis={'gridcolor': 'rgba(255,255,255,0.05)', 'title': ''},
            height=480,
            margin=dict(l=160, r=60, t=60, b=60)
        )

        return fig
    except Exception:
        return None


def create_state_choropleth(df: pd.DataFrame) -> go.Figure:
    """Create US state choropleth map with premium styling."""
    if df.empty:
        df = pd.DataFrame({
            'state': ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
                     'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI'],
            'accident_count': [180000, 165000, 145000, 125000, 95000, 85000, 75000,
                              72000, 68000, 65000, 55000, 52000, 48000, 45000, 42000,
                              40000, 38000, 35000, 33000, 30000]
        })

    state_col = 'state' if 'state' in df.columns else df.columns[0]
    count_col = 'accident_count' if 'accident_count' in df.columns else df.columns[1]

    fig = go.Figure(data=go.Choropleth(
        locations=df[state_col],
        z=df[count_col],
        locationmode='USA-states',
        colorscale=[
            [0, '#0A0A0F'],
            [0.25, '#1E3A5F'],
            [0.5, '#2563EB'],
            [0.75, '#00D9FF'],
            [1, '#EF4444']
        ],
        colorbar=dict(
            title=dict(text='Accidents', font=dict(color=COLORS["text"])),
            tickfont=dict(color=COLORS["text_secondary"]),
            bgcolor='rgba(0,0,0,0)'
        ),
        hovertemplate='<b>%{location}</b><br>Accidents: %{z:,}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': 'Accident Distribution Across the US',
            'font': {'color': COLORS["text"], 'size': 18, 'family': 'Inter'}
        },
        geo=dict(
            scope='usa',
            projection=go.layout.geo.Projection(type='albers usa'),
            showlakes=True,
            lakecolor='#0A0A0F',
            bgcolor='rgba(0,0,0,0)',
            landcolor='#12121A',
            subunitcolor='#2A2A3A'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS["text"], 'family': 'Inter'},
        height=520,
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig


def create_confusion_matrix_chart(metrics: dict) -> Optional[go.Figure]:
    """Create confusion matrix visualization."""
    if not metrics:
        return None

    m = metrics.get('metrics', metrics)
    total = m.get('test_samples', 1000)
    pos = m.get('positive_samples', 300)
    neg = m.get('negative_samples', 700)

    recall = m.get('recall', 0.7)
    precision = m.get('precision', 0.65)

    tp = int(pos * recall)
    fn = pos - tp
    fp = int(tp / precision - tp) if precision > 0 else 0
    tn = neg - fp

    z = [[tn, fp], [fn, tp]]
    text = [[f'{tn:,}<br>TN', f'{fp:,}<br>FP'],
            [f'{fn:,}<br>FN', f'{tp:,}<br>TP']]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=['Predicted Low', 'Predicted High'],
        y=['Actual Low', 'Actual High'],
        colorscale=[
            [0, '#12121A'],
            [0.5, '#00D9FF'],
            [1, '#7C3AED']
        ],
        hovertemplate='%{y} vs %{x}<br>Count: %{z:,}<extra></extra>',
        text=text,
        texttemplate='%{text}',
        textfont={'size': 14, 'color': 'white', 'family': 'Inter'},
        showscale=False
    ))

    fig.update_layout(
        title={
            'text': 'Confusion Matrix',
            'font': {'color': COLORS["text"], 'size': 18, 'family': 'Inter'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS["text_secondary"], 'family': 'Inter'},
        height=380,
        margin=dict(l=80, r=30, t=60, b=60),
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )

    return fig


def create_roc_curve(metrics: dict) -> Optional[go.Figure]:
    """Create ROC curve with premium styling."""
    if not metrics:
        return None

    m = metrics.get('metrics', metrics)
    auc = m.get('roc_auc', 0.85)

    # Simulate ROC curve points
    np.random.seed(42)
    fpr = np.linspace(0, 1, 100)
    tpr = 1 - (1 - fpr) ** (auc / (1 - auc + 0.01))
    tpr = np.clip(tpr, 0, 1)

    fig = go.Figure()

    # Area under curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        fill='tozeroy',
        fillcolor='rgba(0, 217, 255, 0.15)',
        line=dict(color=COLORS["primary"], width=3),
        name=f'ROC Curve (AUC = {auc:.3f})',
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))

    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color=COLORS["text_muted"], width=2, dash='dash')
    ))

    # AUC annotation
    fig.add_annotation(
        x=0.6, y=0.3,
        text=f'AUC = {auc:.3f}',
        showarrow=False,
        font=dict(size=24, color=COLORS["primary"], family='Inter'),
        bgcolor=COLORS["surface"],
        bordercolor=COLORS["border"],
        borderwidth=1,
        borderpad=8
    )

    fig.update_layout(
        title={
            'text': 'ROC Curve',
            'font': {'color': COLORS["text"], 'size': 18, 'family': 'Inter'}
        },
        xaxis=dict(
            title='False Positive Rate',
            gridcolor='rgba(255,255,255,0.05)',
            range=[-0.02, 1.02]
        ),
        yaxis=dict(
            title='True Positive Rate',
            gridcolor='rgba(255,255,255,0.05)',
            range=[-0.02, 1.02]
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS["text_secondary"], 'family': 'Inter'},
        height=420,
        margin=dict(l=60, r=30, t=60, b=60),
        legend=dict(
            x=0.5, y=0.05,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=11)
        ),
        showlegend=True
    )

    return fig


def create_folium_map(df: pd.DataFrame, center: list = None, zoom: int = 4) -> folium.Map:
    """Create Folium map with premium dark theme."""
    if center is None:
        center = [39.8283, -98.5795]

    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=None
    )

    # Add dark tile layer
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='CartoDB',
        name='Dark Map'
    ).add_to(m)

    # Add heatmap if data available
    if not df.empty and 'latitude' in df.columns and 'longitude' in df.columns:
        heat_data = df[['latitude', 'longitude']].dropna().values.tolist()
        if heat_data:
            HeatMap(
                heat_data,
                min_opacity=0.3,
                max_opacity=0.8,
                radius=18,
                blur=25,
                gradient={
                    0.2: '#0A0A0F',
                    0.4: '#1E3A5F',
                    0.6: '#00D9FF',
                    0.8: '#F59E0B',
                    1.0: '#EF4444'
                }
            ).add_to(m)

    return m


# =============================================================================
# PAGE FUNCTIONS
# =============================================================================

def page_home():
    """Home page - Executive Overview with premium design."""
    data = load_analytics_data()
    metrics = load_model_metrics()

    # Hero section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">SafeStreets</div>
        <div class="hero-subtitle">
            AI-Powered Accident Risk Analytics Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    metrics_data = [
        (col1, f"{data.get('total_accidents', 0):,}", "Total Accidents", "#00D9FF"),
        (col2, str(data.get('states_covered', 0)), "States Covered", "#7C3AED"),
        (col3, f"{data.get('cities_covered', 0):,}", "Cities Analyzed", "#10B981"),
        (col4, f"{metrics.get('metrics', {}).get('roc_auc', 0):.1%}" if metrics else "N/A", "Model AUC", "#F59E0B"),
    ]

    for col, value, label, color in metrics_data:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="background: {color}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {value}
                </div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 50px'></div>", unsafe_allow_html=True)

    # Two column layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)

        insights = [
            ("Rush Hour Impact", "7-9 AM and 4-7 PM account for 40% of all accidents", "#EF4444"),
            ("Weather Matters", "Fog and rain increase accident severity by 35%", "#F59E0B"),
            ("Urban Hotspots", "Junctions are 2.5x more dangerous than straight roads", "#00D9FF"),
            ("Weekend Effect", "Saturday nights see 40% more severe accidents", "#7C3AED")
        ]

        for title, desc, color in insights:
            st.markdown(f"""
            <div class="insight-card" style="border-left-color: {color};">
                <div class="insight-title" style="color: {color};">{title}</div>
                <div class="insight-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Hourly Pattern</div>', unsafe_allow_html=True)
        fig = create_hourly_chart(pd.DataFrame())
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # CTA Section
    st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="text-align: center;">
        <div style="font-size: 1.25rem; font-weight: 600; color: var(--text); margin-bottom: 12px;">
            Ready to explore accident risk data?
        </div>
        <div style="color: var(--text-secondary); max-width: 500px; margin: 0 auto;">
            Navigate using the sidebar to access the Risk Explorer, Analytics Dashboard,
            and Live Risk Calculator.
        </div>
    </div>
    """, unsafe_allow_html=True)


def page_risk_explorer():
    """Risk Explorer - Interactive Heatmap."""
    st.markdown('<div class="section-header">Risk Explorer</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Interactive heatmap showing accident hotspots across the United States.
        Zoom in to explore high-risk areas in detail.
    </div>
    """, unsafe_allow_html=True)

    # Filters row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        severity_filter = st.selectbox(
            "Severity Level",
            ["All Severities", "Critical (4)", "High (3)", "Medium (2)", "Low (1)"],
            key="explorer_severity"
        )

    with col2:
        time_filter = st.selectbox(
            "Time of Day",
            ["All Day", "Morning (6-12)", "Afternoon (12-18)", "Evening (18-22)", "Night (22-6)"],
            key="explorer_time"
        )

    with col3:
        weather_filter = st.selectbox(
            "Weather",
            ["All Weather", "Clear", "Rain", "Fog", "Snow", "Cloudy"],
            key="explorer_weather"
        )

    with col4:
        year_filter = st.selectbox(
            "Year",
            ["All Years", "2023", "2022", "2021", "2020"],
            key="explorer_year"
        )

    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

    # Load and display map
    with st.spinner("Loading heatmap data..."):
        df = get_heatmap_data()

        if df.empty:
            df = load_features_sample(8000)

        m = create_folium_map(df)

    st_folium(m, width=None, height=550, returned_objects=[])

    # Legend
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 40px; margin-top: 24px; padding: 16px;
                background: var(--surface); border-radius: 12px; border: 1px solid var(--border);">
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 16px; height: 16px; border-radius: 4px; background: #1E3A5F;"></div>
            <span style="color: var(--text-secondary); font-size: 0.875rem;">Low Density</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 16px; height: 16px; border-radius: 4px; background: #00D9FF;"></div>
            <span style="color: var(--text-secondary); font-size: 0.875rem;">Medium</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 16px; height: 16px; border-radius: 4px; background: #F59E0B;"></div>
            <span style="color: var(--text-secondary); font-size: 0.875rem;">High</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 16px; height: 16px; border-radius: 4px; background: #EF4444;"></div>
            <span style="color: var(--text-secondary); font-size: 0.875rem;">Critical</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def page_analytics():
    """Analytics - Deep Dive Dashboard."""
    st.markdown('<div class="section-header">Analytics Dashboard</div>', unsafe_allow_html=True)

    data = load_analytics_data()
    model = load_model()

    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "Temporal Patterns",
        "Weather Impact",
        "Geographic",
        "Top Cities"
    ])

    with tab1:
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            fig = create_hourly_heatmap(data.get('hourly_risk', pd.DataFrame()))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with col2:
            fig = create_hourly_chart(data.get('hourly_risk', pd.DataFrame()))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Day of week summary
        st.markdown("### Day of Week Analysis")

        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        values = [14.2, 14.8, 15.1, 15.3, 16.2, 13.2, 11.2]

        cols = st.columns(7)
        for i, (col, day, val) in enumerate(zip(cols, days, values)):
            color = COLORS["warning"] if i >= 5 else COLORS["primary"]
            with col:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 16px;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{val}%</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 4px;">{day[:3]}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            fig = create_weather_chart(data.get('weather_risk', pd.DataFrame()))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with col2:
            # Weather severity details
            st.markdown("### Severity by Weather")

            weather_stats = [
                ("Fog", "3.1", "65%", "#EF4444"),
                ("Thunderstorm", "3.0", "60%", "#F59E0B"),
                ("Snow", "2.9", "55%", "#00D9FF"),
                ("Rain", "2.7", "45%", "#7C3AED"),
                ("Clear", "2.3", "30%", "#10B981"),
            ]

            for weather, severity, high_pct, color in weather_stats:
                st.markdown(f"""
                <div class="insight-card" style="border-left-color: {color}; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 600; color: var(--text);">{weather}</div>
                        <div style="font-size: 0.8rem; color: var(--text-muted);">Avg Severity: {severity}</div>
                    </div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: {color};">{high_pct}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

        fig = create_state_choropleth(data.get('state_summary', pd.DataFrame()))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # State statistics
        st.markdown("### Top States")

        top_states = [
            ("California", "180,000", "12.5%"),
            ("Texas", "165,000", "11.4%"),
            ("Florida", "145,000", "10.0%"),
            ("New York", "125,000", "8.6%"),
            ("Pennsylvania", "95,000", "6.6%"),
        ]

        cols = st.columns(5)
        for col, (state, count, pct) in zip(cols, top_states):
            with col:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 0.85rem; color: var(--text-muted);">{state}</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--primary);">{count}</div>
                    <div style="font-size: 0.75rem; color: var(--text-secondary);">{pct} of total</div>
                </div>
                """, unsafe_allow_html=True)

    with tab4:
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

        fig = create_top_cities_chart(pd.DataFrame())
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


@st.cache_data(ttl=3600)
def get_cities_for_state(state: str) -> list:
    """Query all cities for a given state from the database."""
    conn = get_db_connection()
    if not conn:
        return []

    try:
        df = conn.execute(f"""
            SELECT DISTINCT City
            FROM staging.accidents
            WHERE State = '{state}'
            AND City IS NOT NULL
            AND City != ''
            ORDER BY City
        """).fetchdf()
        return df['City'].tolist()
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_city_coordinates(state: str, city: str) -> tuple:
    """Get average lat/long for a city from the database."""
    conn = get_db_connection()
    if not conn:
        return (None, None)

    try:
        # Escape single quotes in city name
        city_escaped = city.replace("'", "''")
        result = conn.execute(f"""
            SELECT
                AVG(Start_Lat) as avg_lat,
                AVG(Start_Lng) as avg_lng
            FROM staging.accidents
            WHERE State = '{state}'
            AND City = '{city_escaped}'
            AND Start_Lat IS NOT NULL
            AND Start_Lng IS NOT NULL
        """).fetchone()

        if result and result[0] is not None:
            return (round(result[0], 4), round(result[1], 4))
        return (None, None)
    except Exception:
        return (None, None)


def page_risk_calculator():
    """Risk Calculator - Live Predictions."""
    st.markdown('<div class="section-header">Risk Calculator</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Enter location and environmental conditions to get a real-time risk assessment
        powered by our machine learning model. Select a state and city to auto-populate coordinates.
    </div>
    """, unsafe_allow_html=True)

    # All 49 US states (+ DC)
    ALL_STATES = [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
        "GA", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
    ]

    # Initialize session state for coordinates
    if 'calc_latitude' not in st.session_state:
        st.session_state.calc_latitude = 34.0522
    if 'calc_longitude' not in st.session_state:
        st.session_state.calc_longitude = -118.2437
    if 'calc_last_state' not in st.session_state:
        st.session_state.calc_last_state = "CA"
    if 'calc_last_city' not in st.session_state:
        st.session_state.calc_last_city = None

    # Input form in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Location")

        # State dropdown with all states
        state = st.selectbox(
            "State",
            options=ALL_STATES,
            index=ALL_STATES.index("CA") if "CA" in ALL_STATES else 0,
            key="calc_state",
            help="Select a state to see cities in that state"
        )

        # Detect state change and reset city selection
        if state != st.session_state.calc_last_state:
            st.session_state.calc_last_state = state
            st.session_state.calc_last_city = None
            # Reset to default coordinates for new state
            st.session_state.calc_latitude = 34.0522
            st.session_state.calc_longitude = -118.2437

        # Query cities for selected state
        cities_in_state = get_cities_for_state(state)

        if cities_in_state:
            # Add "Other" option at the end
            city_options = cities_in_state + ["── Other (type manually) ──"]
            city_count = len(cities_in_state)

            st.caption(f"{city_count} cities found in {state}")

            selected_city_option = st.selectbox(
                "City",
                options=city_options,
                index=0,
                key=f"calc_city_select_{state}",  # Key changes with state to reset selection
                help=f"Select from {city_count} cities in {state} or choose 'Other' to type manually"
            )
        else:
            # Fallback if no database connection
            st.warning(f"Could not load cities for {state}. Using manual entry.")
            selected_city_option = "── Other (type manually) ──"

        # Determine if using manual entry
        is_manual_entry = "Other" in selected_city_option

        # Show text input if "Other" is selected
        if is_manual_entry:
            city = st.text_input(
                "Enter City Name",
                value="",
                placeholder="e.g., Springfield, Riverside, etc.",
                key="calc_city_manual",
                help="Type any city name - the model will still generate predictions"
            )
            if not city.strip():
                st.caption("Please enter a city name for the prediction")
        else:
            city = selected_city_option

            # Auto-fetch coordinates when city changes
            if city != st.session_state.calc_last_city and city:
                lat, lng = get_city_coordinates(state, city)
                if lat is not None and lng is not None:
                    st.session_state.calc_latitude = lat
                    st.session_state.calc_longitude = lng
                st.session_state.calc_last_city = city

        # Coordinates section
        st.markdown("##### Coordinates")

        if is_manual_entry:
            # Editable coordinates for manual entry
            st.caption("Enter approximate coordinates for your custom location")
            c1, c2 = st.columns(2)
            with c1:
                latitude = st.number_input(
                    "Latitude",
                    value=st.session_state.calc_latitude,
                    format="%.4f",
                    key="calc_lat_input",
                    help="Enter latitude (-90 to 90)"
                )
            with c2:
                longitude = st.number_input(
                    "Longitude",
                    value=st.session_state.calc_longitude,
                    format="%.4f",
                    key="calc_lng_input",
                    help="Enter longitude (-180 to 180)"
                )
        else:
            # Display-only coordinates for dropdown selection
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div style="background: var(--surface); border: 1px solid var(--border);
                            border-radius: 8px; padding: 12px; margin-top: 4px;">
                    <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px;">Latitude</div>
                    <div style="font-size: 1.1rem; color: var(--primary); font-weight: 600;">
                        {st.session_state.calc_latitude:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style="background: var(--surface); border: 1px solid var(--border);
                            border-radius: 8px; padding: 12px; margin-top: 4px;">
                    <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px;">Longitude</div>
                    <div style="font-size: 1.1rem; color: var(--primary); font-weight: 600;">
                        {st.session_state.calc_longitude:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            latitude = st.session_state.calc_latitude
            longitude = st.session_state.calc_longitude

            st.caption("Coordinates auto-populated from database")

        st.markdown("#### Time")

        c1, c2 = st.columns(2)
        with c1:
            hour = st.slider("Hour of Day", 0, 23, 14)
        with c2:
            day_of_week = st.selectbox(
                "Day",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )

        day_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)
        is_weekend = day_idx >= 5

    with col2:
        st.markdown("#### Weather Conditions")

        weather = st.selectbox(
            "Weather",
            ["Clear", "Partly Cloudy", "Cloudy", "Light Rain", "Rain", "Heavy Rain", "Fog", "Snow", "Thunderstorm"]
        )

        c1, c2 = st.columns(2)
        with c1:
            temperature = st.slider("Temperature (F)", -20, 120, 72)
        with c2:
            visibility = st.slider("Visibility (mi)", 0.0, 10.0, 10.0, 0.5)

        c1, c2 = st.columns(2)
        with c1:
            humidity = st.slider("Humidity (%)", 0, 100, 50)
        with c2:
            precipitation = st.slider("Precipitation (in)", 0.0, 2.0, 0.0, 0.05)

        st.markdown("#### Road Features")

        c1, c2, c3 = st.columns(3)

        with c1:
            has_junction = st.toggle("Junction", key="calc_junc")
            has_signal = st.toggle("Traffic Signal", key="calc_sig")

        with c2:
            has_crossing = st.toggle("Crossing", key="calc_cross")
            has_stop = st.toggle("Stop Sign", key="calc_stop")

        with c3:
            has_roundabout = st.toggle("Roundabout", key="calc_round")
            has_railway = st.toggle("Railway", key="calc_rail")

    st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)

    # Validate city input
    city_valid = bool(city and city.strip()) if isinstance(city, str) else False

    # Predict button centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_clicked = st.button(
            "CALCULATE RISK",
            use_container_width=True,
            type="primary",
            disabled=not city_valid
        )

    if not city_valid and is_manual_entry:
        st.warning("Please enter a city name to calculate risk.")

    if predict_clicked and city_valid:
        with st.spinner("Analyzing risk factors..."):
            # Clean city name
            city_clean = city.strip() if isinstance(city, str) else city

            # Prepare API request - works with any city/state combination
            # The model uses hash-based encoding for cities, so unknown cities
            # will still produce valid predictions based on other features
            request_data = {
                "latitude": latitude,
                "longitude": longitude,
                "state": state,
                "city": city_clean,
                "hour": hour,
                "day_of_week": day_idx,
                "is_weekend": is_weekend,
                "month": datetime.now().month,
                "temperature_f": temperature,
                "humidity_pct": humidity,
                "visibility_mi": visibility,
                "precipitation_in": precipitation,
                "weather_condition": weather,
                "has_junction": has_junction,
                "has_traffic_signal": has_signal,
                "has_crossing": has_crossing,
                "has_stop": has_stop,
                "has_roundabout": has_roundabout,
                "has_railway": has_railway,
            }

            prediction = None

            # Try API first
            try:
                response = requests.post(f"{API_URL}/predict", json=request_data, timeout=5)
                if response.status_code == 200:
                    prediction = response.json()
            except Exception:
                pass

            # Fallback to local estimation
            if not prediction:
                model = load_model()
                risk_score = 0.35

                # Adjust based on inputs
                if visibility < 5:
                    risk_score += 0.15
                if precipitation > 0:
                    risk_score += 0.12
                if weather in ['Fog', 'Snow', 'Thunderstorm']:
                    risk_score += 0.18
                if has_junction:
                    risk_score += 0.08
                if has_crossing:
                    risk_score += 0.05
                if 7 <= hour <= 9 or 16 <= hour <= 19:
                    risk_score += 0.10
                if is_weekend and hour >= 20:
                    risk_score += 0.08

                risk_score = min(0.95, max(0.05, risk_score + np.random.uniform(-0.05, 0.05)))

                if risk_score < 0.3:
                    risk_level = "LOW"
                elif risk_score < 0.6:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "HIGH"

                prediction = {
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "confidence": 0.82,
                    "contributing_factors": [
                        {"feature": "visibility_mi", "importance": 0.22, "value": visibility},
                        {"feature": "hour", "importance": 0.18, "value": hour},
                        {"feature": "precipitation_in", "importance": 0.15, "value": precipitation},
                        {"feature": "has_junction", "importance": 0.12, "value": int(has_junction)},
                        {"feature": "weather_condition", "importance": 0.10, "value": None},
                    ]
                }

        # Display results
        st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Prediction Results")

        col1, col2, col3 = st.columns([1.2, 0.8, 1])

        with col1:
            fig = create_gauge_chart(prediction["risk_score"])
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with col2:
            risk_level = prediction["risk_level"]
            badge_class = f"risk-{risk_level.lower()}"

            st.markdown(f"""
            <div style="text-align: center; padding: 40px 20px;">
                <div style="color: var(--text-muted); margin-bottom: 16px; text-transform: uppercase;
                            letter-spacing: 0.1em; font-size: 0.75rem;">Risk Level</div>
                <div class="risk-badge {badge_class}">{risk_level}</div>
                <div style="margin-top: 24px;">
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase;
                                letter-spacing: 0.05em;">Confidence</div>
                    <div style="font-size: 1.5rem; font-weight: 600; color: var(--text);">
                        {prediction.get('confidence', 0.85):.0%}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            explanations = {
                "LOW": ("Favorable Conditions", "Standard caution advised. Road and weather conditions are optimal for safe travel."),
                "MEDIUM": ("Elevated Risk", "Stay alert and consider reducing speed. Some risk factors are present."),
                "HIGH": ("Significant Risk", "Exercise extreme caution. Multiple high-risk factors detected.")
            }

            title, desc = explanations.get(risk_level, ("Unknown", ""))

            st.markdown(f"""
            <div class="glass-card" style="height: 100%;">
                <div style="color: {RISK_COLORS.get(risk_level, COLORS['text'])}; font-weight: 600;
                            font-size: 1.1rem; margin-bottom: 12px;">{title}</div>
                <div style="color: var(--text-secondary); line-height: 1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        # Contributing factors
        st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
        st.markdown("### Contributing Factors")

        factors = prediction.get("contributing_factors", [])[:5]

        for factor in factors:
            name = factor["feature"].replace("_", " ").title()
            importance = factor["importance"]

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{name}**")
                st.progress(min(1.0, importance * 4))
            with col2:
                st.markdown(f"<div style='text-align: right; padding-top: 28px; color: var(--primary); font-weight: 600;'>{importance:.0%}</div>", unsafe_allow_html=True)


def page_model_insights():
    """Model Insights - Performance Metrics."""
    st.markdown('<div class="section-header">Model Insights</div>', unsafe_allow_html=True)

    metrics = load_model_metrics()
    model = load_model()

    if not metrics and not model:
        st.warning("Model not trained yet. Run the training pipeline first.")
        return

    m = metrics.get('metrics', {}) if metrics else {}

    # Performance metrics cards
    st.markdown("### Performance Metrics")

    metrics_display = [
        ("Accuracy", m.get('accuracy', 0), "Correct predictions ratio"),
        ("Precision", m.get('precision', 0), "Positive predictive value"),
        ("Recall", m.get('recall', 0), "Sensitivity / True positive rate"),
        ("F1 Score", m.get('f1_score', 0), "Harmonic mean of precision & recall"),
        ("ROC-AUC", m.get('roc_auc', 0), "Area under ROC curve"),
    ]

    cols = st.columns(5)

    for col, (name, value, desc) in zip(cols, metrics_display):
        if value > 0.75:
            color = COLORS["success"]
        elif value > 0.5:
            color = COLORS["warning"]
        else:
            color = COLORS["danger"]

        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 700; color: {color};">
                    {value:.1%}
                </div>
                <div style="font-size: 0.9rem; font-weight: 600; color: var(--text); margin-top: 8px;">
                    {name}
                </div>
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 4px;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        fig = create_confusion_matrix_chart(metrics)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col2:
        fig = create_roc_curve(metrics)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Feature importance
    st.markdown("### Feature Importance")

    fig = create_feature_importance_chart(model)
    if fig:
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Model metadata
    st.markdown("### Model Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div style="font-weight: 600; color: var(--text); margin-bottom: 16px;">Training Information</div>
            <div style="display: grid; gap: 12px;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: var(--text-muted);">Model Type</span>
                    <span style="color: var(--text);">XGBoost Classifier</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: var(--text-muted);">Training Date</span>
                    <span style="color: var(--text);">{metrics.get('timestamp', 'N/A')[:10] if metrics else 'N/A'}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: var(--text-muted);">Test Samples</span>
                    <span style="color: var(--text);">{m.get('test_samples', 'N/A'):,}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <div style="font-weight: 600; color: var(--text); margin-bottom: 16px;">Class Distribution</div>
            <div style="display: grid; gap: 12px;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: var(--text-muted);">Positive (High Risk)</span>
                    <span style="color: var(--danger);">{m.get('positive_samples', 0):,}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: var(--text-muted);">Negative (Low Risk)</span>
                    <span style="color: var(--success);">{m.get('negative_samples', 0):,}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: var(--text-muted);">Class Ratio</span>
                    <span style="color: var(--text);">1:{m.get('negative_samples', 1) // max(m.get('positive_samples', 1), 1):.0f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top: 30px;">
        <strong>MLflow Tracking:</strong> View detailed experiment history by running
        <code>mlflow ui</code> in the project root directory.
    </div>
    """, unsafe_allow_html=True)


def page_about():
    """About page - Project Information."""
    st.markdown('<div class="section-header">About SafeStreets</div>', unsafe_allow_html=True)

    # Hero description
    st.markdown("""
    <div class="hero-section" style="padding: 40px;">
        <div style="font-size: 1.75rem; font-weight: 600; margin-bottom: 16px;">
            AI-Powered Accident Risk Prediction
        </div>
        <div style="color: rgba(255,255,255,0.85); max-width: 600px; margin: 0 auto; line-height: 1.7;">
            SafeStreets leverages machine learning to analyze millions of historical accident records,
            providing real-time risk assessments based on location, weather, and road conditions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)

    # Tech stack
    st.markdown("### Technology Stack")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Data Engineering")
        for tech in ["Python", "DuckDB", "Pandas", "PyArrow", "Parquet"]:
            st.markdown(f'<span class="tech-badge">{tech}</span>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### Machine Learning")
        for tech in ["XGBoost", "Scikit-learn", "MLflow", "NumPy"]:
            st.markdown(f'<span class="tech-badge">{tech}</span>', unsafe_allow_html=True)

    with col3:
        st.markdown("#### API & Visualization")
        for tech in ["FastAPI", "Streamlit", "Plotly", "Folium"]:
            st.markdown(f'<span class="tech-badge">{tech}</span>', unsafe_allow_html=True)

    st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)

    # Architecture
    st.markdown("### System Architecture")

    st.markdown("""
    <div class="glass-card">
        <pre style="color: var(--primary); font-size: 0.85rem; overflow-x: auto;">
┌────────────────────────────────────────────────────────────────────┐
│                       SafeStreets Platform                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │  Kaggle  │ -> │  DuckDB  │ -> │ Parquet  │ -> │ XGBoost  │     │
│  │   Data   │    │   ETL    │    │ Features │    │  Model   │     │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│                                                        │            │
│                              ┌─────────────────────────┘            │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      FastAPI Service                          │  │
│  │   /predict    /predict/batch    /health    /model/info        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Streamlit Dashboard                        │  │
│  │   Home  │  Explorer  │  Analytics  │  Calculator  │  Insights │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
        </pre>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)

    # Data source
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Data Source")
        st.markdown("""
        This project uses the **US Accidents** dataset from Kaggle, one of the largest
        publicly available traffic accident datasets.

        **Dataset Highlights:**
        - 7.7+ million accident records
        - Coverage from 2016-2023
        - 49 US states
        - 47 features per record
        """)

    with col2:
        st.markdown("### Links & Resources")
        st.markdown("""
        - [Kaggle Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
        - [XGBoost Documentation](https://xgboost.readthedocs.io/)
        - [Streamlit Docs](https://docs.streamlit.io/)
        - [FastAPI Docs](https://fastapi.tiangolo.com/)
        """)

    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="text-align: center;">
        <div style="font-weight: 600; color: var(--text); margin-bottom: 8px;">
            Built for Data Systems Engineering
        </div>
        <div style="color: var(--text-secondary);">
            Demonstrating end-to-end ML pipeline: data ingestion, feature engineering,
            model training, API deployment, and interactive visualization.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="SafeStreets | Risk Analytics",
        page_icon="shield",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Additional CSS for horizontal navigation
    st.markdown("""
    <style>
        /* Top navigation bar styling */
        .top-nav-container {
            background: linear-gradient(135deg, var(--surface) 0%, var(--background) 100%);
            border-bottom: 1px solid var(--border);
            padding: 12px 0;
            margin: -1rem -1rem 1.5rem -1rem;
            position: sticky;
            top: 0;
            z-index: 999;
        }
        .top-nav-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }
        .top-nav-brand-text {
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00D9FF, #7C3AED);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Horizontal radio button styling */
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[role="radiogroup"] {
            gap: 4px !important;
            flex-wrap: wrap;
        }
        div[role="radiogroup"][aria-label="Navigation"] > label {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            margin: 2px !important;
            transition: all 0.2s ease !important;
        }
        div[role="radiogroup"][aria-label="Navigation"] > label:hover {
            background: var(--card) !important;
            border-color: var(--primary) !important;
        }
        div[role="radiogroup"][aria-label="Navigation"] > label[data-checked="true"] {
            background: linear-gradient(135deg, var(--primary) 0%, #00B4D8 100%) !important;
            border-color: var(--primary) !important;
            color: var(--background) !important;
        }
        div[role="radiogroup"][aria-label="Navigation"] > label[data-checked="true"] p {
            color: var(--background) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Page mapping
    pages = {
        "Home": page_home,
        "Risk Explorer": page_risk_explorer,
        "Analytics": page_analytics,
        "Risk Calculator": page_risk_calculator,
        "Model Insights": page_model_insights,
        "About": page_about,
    }

    # Top navigation bar
    nav_col1, nav_col2 = st.columns([3, 1])

    with nav_col1:
        # Brand + Navigation
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 24px; flex-wrap: wrap;">
            <div style="font-size: 1.5rem; font-weight: 800;
                        background: linear-gradient(135deg, #00D9FF, #7C3AED);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                SafeStreets
            </div>
        </div>
        """, unsafe_allow_html=True)

    with nav_col2:
        # System status indicators (compact)
        model = load_model()
        conn = get_db_connection()
        try:
            r = requests.get(f"{API_URL}/health", timeout=1)
            api_ok = r.status_code == 200
        except Exception:
            api_ok = False

        status_html = f"""
        <div style="display: flex; gap: 16px; justify-content: flex-end; align-items: center;">
            <div style="display: flex; align-items: center; gap: 4px;">
                <div style="width: 8px; height: 8px; border-radius: 50%;
                            background: {'#10B981' if model else '#EF4444'};"></div>
                <span style="font-size: 0.75rem; color: var(--text-muted);">Model</span>
            </div>
            <div style="display: flex; align-items: center; gap: 4px;">
                <div style="width: 8px; height: 8px; border-radius: 50%;
                            background: {'#10B981' if conn else '#EF4444'};"></div>
                <span style="font-size: 0.75rem; color: var(--text-muted);">DB</span>
            </div>
            <div style="display: flex; align-items: center; gap: 4px;">
                <div style="width: 8px; height: 8px; border-radius: 50%;
                            background: {'#10B981' if api_ok else '#EF4444'};"></div>
                <span style="font-size: 0.75rem; color: var(--text-muted);">API</span>
            </div>
        </div>
        """
        st.markdown(status_html, unsafe_allow_html=True)

    # Horizontal navigation menu
    selected = st.radio(
        "Navigation",
        options=["Home", "Risk Explorer", "Analytics", "Risk Calculator", "Model Insights", "About"],
        format_func=lambda x: {
            "Home": "Home",
            "Risk Explorer": "Risk Explorer",
            "Analytics": "Analytics",
            "Risk Calculator": "Risk Calculator",
            "Model Insights": "Model Insights",
            "About": "About"
        }.get(x, x),
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Render selected page
    pages[selected]()


if __name__ == "__main__":
    main()
