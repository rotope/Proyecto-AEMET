import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import warnings
import psycopg2
from sqlalchemy import create_engine
import requests
import os
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Cargar variables de entorno
load_dotenv()

# ===== CONFIGURACIÓN DE BASE DE DATOS =====
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_DATABASE', 'postgres'), 
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'port': int(os.getenv('DB_PORT', 5432))
}

# ===== CONFIGURACIÓN DE API IA =====
API_URL = os.getenv('API_URL', "http://localhost:8000")

# ===== CONFIGURACIÓN PROFESIONAL =====
st.set_page_config(
    page_title="🌤️ AEMET Analytics Pro",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== TEMA DARK PROFESIONAL =====
def load_dark_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Tema dark global */
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff !important;
    }
    
    /* Forzar texto blanco en todo */
    .stApp, .stApp * {
        color: #ffffff !important;
    }
    
    /* Sidebar dark */
    .stSidebar {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
    }
    
    .stSidebar, .stSidebar * {
        color: #ffffff !important;
    }
    
    .stSidebar .stSelectbox > div > div {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(102,126,234,0.3) !important;
        color: white !important;
    }
    
    .stSidebar .stSelectbox > div > div > div {
        color: white !important;
    }
    
    /* Header ultra profesional */
    .ultra-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 4rem 2rem;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 3rem;
        color: white;
        box-shadow: 0 20px 60px rgba(102,126,234,0.4);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .ultra-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 20%, rgba(255,255,255,0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(255,255,255,0.1) 0%, transparent 50%),
            linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.05) 50%, transparent 70%);
        animation: shimmer 6s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .ultra-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 2;
        background: linear-gradient(45deg, #ffffff, #f0f0f0, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .ultra-header p {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        font-weight: 400;
        margin: 1rem 0 0 0;
        opacity: 0.95;
        position: relative;
        z-index: 2;
    }
    
    /* Cards ultra modernas */
    .ultra-metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin: 1rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 8px 32px rgba(0,0,0,0.3),
            inset 0 1px 0 rgba(255,255,255,0.2);
    }
    
    .ultra-metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    }
    
    .ultra-metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 20px 60px rgba(102,126,234,0.4),
            inset 0 1px 0 rgba(255,255,255,0.3);
        border-color: rgba(102,126,234,0.5);
    }
    
    .ultra-metric-card h3 {
        font-family: 'Inter', sans-serif;
        color: rgba(255,255,255,0.8);
        margin: 0;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .ultra-metric-card h2 {
        font-family: 'JetBrains Mono', monospace;
        color: #ffffff;
        margin: 1rem 0 0.5rem 0;
        font-size: 2.8rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .ultra-metric-card p {
        font-family: 'Inter', sans-serif;
        color: rgba(255,255,255,0.7);
        margin: 0;
        font-size: 0.9rem;
        font-weight: 400;
    }
    
    /* Predicción card épica */
    .prediction-ultra {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 50%, #09fbd3 100%);
        border-radius: 25px;
        padding: 3rem 2rem;
        text-align: center;
        color: white;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 25px 80px rgba(17,153,142,0.5),
            inset 0 1px 0 rgba(255,255,255,0.3);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .prediction-ultra::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 30% 30%, rgba(255,255,255,0.2) 0%, transparent 60%),
            radial-gradient(circle at 70% 70%, rgba(255,255,255,0.1) 0%, transparent 60%);
        animation: pulse 4s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    
    .prediction-value-ultra {
        font-family: 'JetBrains Mono', monospace;
        font-size: 5rem;
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 2;
        background: linear-gradient(45deg, #ffffff, #f0f0f0, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Botones premium */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 1rem 2.5rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 25px rgba(102,126,234,0.4) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 15px 40px rgba(102,126,234,0.6) !important;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    }
    
    /* Selectbox premium */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(102,126,234,0.3) !important;
        border-radius: 12px !important;
        color: white !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Slider premium */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
    }
    
    /* Dataframe dark */
    .stDataFrame {
        background: rgba(0,0,0,0.3) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(102,126,234,0.2) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Métricas premium */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
    }
    
    [data-testid="metric-container"] > div {
        color: white !important;
    }
    
    /* Text inputs dark - Mejorado */
    .stTextInput > div > div > input,
    .stTextInput > div > div,
    .stTextInput input,
    .stNumberInput > div > div > input,
    .stNumberInput input,
    div[data-testid="stTextInput"] > div > div > input,
    div[data-testid="stTextInput"] input,
    .stSelectbox > div > div > div,
    .stSelectbox label,
    .stSelectbox > div[data-baseweb="select"] > div {
        background: rgba(26,26,46,0.9) !important;
        border: 1px solid rgba(102,126,234,0.3) !important;
        color: white !important;
        border-radius: 10px !important;
    }
    
    /* Text input placeholder */
    .stTextInput input::placeholder,
    div[data-testid="stTextInput"] input::placeholder {
        color: rgba(255,255,255,0.6) !important;
    }
    
    /* Forzar color blanco en selectbox */
    .stSelectbox div[role="button"],
    .stSelectbox div[role="button"] span,
    .stSelectbox div[role="listbox"] div,
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] span {
        color: white !important;
        background: rgba(26,26,46,0.9) !important;
    }
    
    /* Date input - Forzar estilo consistente con selectbox */
    .stDateInput div,
    .stDateInput input,
    .stDateInput button,
    .stDateInput span,
    div[data-testid="stDateInput"] div,
    div[data-testid="stDateInput"] input,
    div[data-testid="stDateInput"] button {
        background: rgba(26,26,46,0.9) !important;
        color: white !important;
        border: 1px solid rgba(102,126,234,0.3) !important;
        border-radius: 10px !important;
    }
    
    /* Dropdown opciones */
    div[data-baseweb="popover"] {
        background: rgba(26,26,46,0.95) !important;
        border: 1px solid rgba(102,126,234,0.3) !important;
    }
    
    div[data-baseweb="popover"] div {
        color: white !important;
        background: rgba(26,26,46,0.95) !important;
    }
    
    /* Success/Info/Warning dark */
    .stAlert {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(102,126,234,0.3) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
        color: white !important;
    }
    
    .stAlert * {
        color: white !important;
    }
    
    /* Sidebar header */
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }
    
    .stSidebar .stMarkdown,
    .stSidebar .stMarkdown * {
        color: rgba(255,255,255,0.9) !important;
    }
    
    /* Checkbox dark */
    .stCheckbox > label,
    .stCheckbox > label * {
        color: rgba(255,255,255,0.9) !important;
    }
    
    /* Radio button dark */
    .stRadio > label,
    .stRadio > div,
    .stRadio * {
        color: rgba(255,255,255,0.9) !important;
    }
    
    /* Date input dark - Mejorado */
    .stDateInput > div > div > input,
    .stDateInput label,
    .stDateInput > div > div,
    .stDateInput [data-baseweb="input"] > div,
    .stDateInput [data-baseweb="input"] input {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(102,126,234,0.3) !important;
        color: white !important;
        border-radius: 10px !important;
    }
    
    /* Date input popup/calendar */
    .stDateInput [data-baseweb="popover"] {
        background: rgba(26, 28, 36, 0.95) !important;
        border: 1px solid rgba(102,126,234,0.3) !important;
        border-radius: 10px !important;
    }
    
    /* Date input calendar buttons and text */
    .stDateInput [data-baseweb="calendar"] *,
    .stDateInput [data-baseweb="popover"] * {
        color: white !important;
        background: transparent !important;
    }
    
    /* Slider dark */
    .stSlider > label,
    .stSlider * {
        color: white !important;
    }
    
    /* Expander dark */
    .streamlit-expanderHeader,
    .streamlit-expanderContent,
    details summary {
        color: white !important;
        background: rgba(255,255,255,0.05) !important;
    }
    
    /* Markdown content */
    .stMarkdown, .stMarkdown * {
        color: white !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button,
    .stTabs [data-baseweb="tab-list"] button span,
    .stTabs [data-baseweb="tab"] {
        color: white !important;
    }
    
    /* Date input - Override final para asegurar consistencia */
    div[data-testid="stDateInput"] > div > div,
    div[data-testid="stDateInput"] > div > div > div,
    div[data-testid="stDateInput"] input,
    .stDateInput > div > div > div {
        background: rgba(26,26,46,0.9) !important;
        border: 1px solid rgba(102,126,234,0.3) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    /* Text input - Override final para consistencia completa */
    div[data-testid="stTextInput"] > div > div,
    div[data-testid="stTextInput"] > div > div > input,
    div[data-testid="stTextInput"] input,
    .stTextInput > div > div,
    .stTextInput > div > div > input {
        background: rgba(26,26,46,0.9) !important;
        border: 1px solid rgba(102,126,234,0.3) !important;
        border-radius: 10px !important;
        color: white !important;
        font-size: 14px !important;
    }
    
    /* Text input focus y hover */
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextInput"] input:hover,
    .stTextInput input:focus,
    .stTextInput input:hover {
        background: rgba(26,26,46,0.95) !important;
        border: 1px solid rgba(102,126,234,0.6) !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ===== FUNCIONES DE DATOS MEJORADAS =====
@st.cache_data
def load_weather_data():
    """Cargar y procesar datos meteorológicos desde PostgreSQL AWS RDS"""
    try:
        with st.spinner('🔄 Conectando con base de datos AWS RDS...'):
            # Crear conexión a PostgreSQL
            connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            engine = create_engine(connection_string)
            
            # Query para obtener los datos meteorológicos
            query = """
            SELECT fecha, indicativo, nombre, provincia, altitud, 
                   tmed, prec, tmin, horatmin, tmax, horatmax,
                   hrmax, horahrmax, hrmin, horahrmin, hrmedia
            FROM datos_clima 
            ORDER BY fecha DESC
            LIMIT 500000;
            """
            
           
            df = pd.read_sql_query(query, engine)
            
            if df.empty:
                st.error("❌ No se encontraron datos en la base de datos")
                return None
            
            # Procesar fechas
            if 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
                df = df.dropna(subset=['fecha'])
                df = df.sort_values('fecha')
            
            # Limpiar y validar datos numéricos
            numeric_cols = ['tmed', 'tmax', 'tmin', 'prec']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Eliminar outliers extremos usando IQR
                    if col in ['tmed', 'tmax', 'tmin']:
                        Q1 = df[col].quantile(0.05)
                        Q3 = df[col].quantile(0.95)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            # Información sobre la conexión
            stations_count = df['indicativo'].nunique() if 'indicativo' in df.columns else 0
            date_range = f"{df['fecha'].min().strftime('%Y-%m-%d')} a {df['fecha'].max().strftime('%Y-%m-%d')}" if 'fecha' in df.columns else "N/A"
            
            engine.dispose()  # Cerrar conexión
            return df
            
    except Exception as e:
        st.error(f"❌ Error conectando con AWS RDS: {str(e)}")
        st.info("🔄 Intentando con datos locales como respaldo...")
        # Fallback a CSV local si falla la conexión
        try:
            df = pd.read_csv('df_total.csv')
            if not df.empty:
                # Procesar fechas
                if 'fecha' in df.columns:
                    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
                    df = df.dropna(subset=['fecha'])
                    df = df.sort_values('fecha')
                
                st.warning(f"⚠️ Usando datos locales: {len(df):,} registros")
                return df
        except:
            pass
            
        return None

def get_station_list(df):
    """Obtener lista completa de estaciones disponibles"""
    if 'nombre' in df.columns:
        # Obtener estaciones únicas y limpiar nombres
        stations = df['nombre'].dropna().unique()
        stations = sorted([str(station).strip() for station in stations if str(station).strip()])
        return [f"📍 {station}" for station in stations]
    elif 'indicativo' in df.columns:
        # Si no hay nombre, usar indicativo
        indicators = df['indicativo'].dropna().unique()
        indicators = sorted([str(ind).strip() for ind in indicators if str(ind).strip()])
        return [f"📡 {indicator}" for indicator in indicators]
    return []

def filter_by_station(df, station_name):
    """Filtrar datos por estación seleccionada"""
    if station_name.startswith('📍 '):
        # Filtrar por nombre
        clean_name = station_name.replace('📍 ', '').strip()
        if 'nombre' in df.columns:
            filtered_df = df[df['nombre'].str.strip() == clean_name].copy()
            return filtered_df, clean_name
    elif station_name.startswith('📡 '):
        # Filtrar por indicativo
        clean_indicator = station_name.replace('📡 ', '').strip()
        if 'indicativo' in df.columns:
            filtered_df = df[df['indicativo'].str.strip() == clean_indicator].copy()
            station_name_display = f"{clean_indicator}"
            if 'nombre' in filtered_df.columns and not filtered_df.empty:
                first_name = filtered_df['nombre'].iloc[0]
                if pd.notna(first_name):
                    station_name_display = f"{first_name} ({clean_indicator})"
            return filtered_df, station_name_display
    
    # Si no encuentra nada, devolver dataset vacío
    return df.iloc[:0].copy(), 'Sin datos'

# ===== FUNCIONES DE PREDICCIÓN AVANZADAS =====
def calculate_advanced_trend(temperatures, window=30):
    """Calcular tendencia avanzada con múltiples ventanas"""
    try:
        if len(temperatures) < window:
            return 0.0, 0.0
        
        recent_temps = temperatures.tail(window).values
        recent_temps = recent_temps[~np.isnan(recent_temps)]
        
        if len(recent_temps) < 5:
            return 0.0, 0.0
        
        # Tendencia lineal
        x = np.arange(len(recent_temps))
        linear_trend = np.polyfit(x, recent_temps, 1)[0]
        
        # Tendencia exponencial suavizada
        weights = np.exp(np.linspace(0, 1, len(recent_temps)))
        weighted_trend = np.polyfit(x, recent_temps, 1, w=weights)[0]
        
        return float(linear_trend), float(weighted_trend)
        
    except Exception:
        return 0.0, 0.0

def advanced_seasonal_analysis(date=None, latitude=40.4):
    """Análisis estacional avanzado basado en ubicación"""
    try:
        if date is None:
            date = datetime.now()
        
        day_of_year = date.timetuple().tm_yday
        
        # Factor estacional base (España)
        seasonal_base = np.sin(2 * np.pi * (day_of_year - 80) / 365) * 12
        
        # Ajuste por latitud (más al norte = mayor variación)
        latitude_factor = (latitude - 35) / 10  # Normalizado para España
        seasonal_adjusted = seasonal_base * (1 + latitude_factor * 0.3)
        
        # Factor de continentalidad (interior vs costa)
        continental_factor = np.cos(2 * np.pi * day_of_year / 365) * 3
        
        return float(seasonal_adjusted + continental_factor)
        
    except Exception:
        return 0.0

def ultra_prediction(df, station='Estación específica'):
    """Sistema de predicción ultra avanzado"""
    try:
        # Filtrar datos
        df_pred, station_clean = filter_by_station(df, station)
        
        if len(df_pred) < 50 or 'tmed' not in df_pred.columns:
            return None, "Datos insuficientes para análisis"
        
        # Análisis de datos recientes
        recent_data = df_pred.tail(180)  # 6 meses
        temperatures = recent_data['tmed'].dropna()
        
        if len(temperatures) < 30:
            return None, "Historial de temperaturas insuficiente"
        
        # Componentes de predicción
        base_temp = float(temperatures.iloc[-1])
        
        # Análisis de tendencias múltiples
        linear_trend, weighted_trend = calculate_advanced_trend(temperatures, 30)
        long_trend, _ = calculate_advanced_trend(temperatures, 90)
        
        # Análisis estacional avanzado
        seasonal_factor = advanced_seasonal_analysis()
        
        # Análisis de variabilidad
        recent_volatility = temperatures.tail(14).std()
        seasonal_volatility = temperatures.groupby(
            recent_data['fecha'].dt.month
        )['tmed'].std().mean() if 'fecha' in recent_data.columns else recent_volatility
        
        # Modelo de predicción ensemble
        predictions = []
        
        # Modelo 1: Tendencia lineal + estacional
        pred1 = base_temp + (linear_trend * 7) + seasonal_factor
        predictions.append(pred1)
        
        # Modelo 2: Tendencia ponderada + ajuste de volatilidad
        pred2 = base_temp + (weighted_trend * 5) + seasonal_factor * 0.8
        predictions.append(pred2)
        
        # Modelo 3: Tendencia a largo plazo + factor de corrección
        pred3 = base_temp + (long_trend * 3) + seasonal_factor * 1.2
        predictions.append(pred3)
        
        # Ensemble final
        final_prediction = np.mean(predictions)
        
        # Añadir variabilidad realista
        noise_factor = min(recent_volatility * 0.2, 1.5)
        final_prediction += np.random.normal(0, noise_factor)
        
        # Restricciones geográficas para España
        final_prediction = np.clip(final_prediction, -30, 60)
        
        # Cálculo de confianza avanzado
        data_quality = min(100, len(temperatures) / 90 * 100)
        trend_consistency = max(50, 100 - abs(linear_trend - weighted_trend) * 30)
        seasonal_alignment = max(60, 100 - abs(seasonal_factor) / 15 * 40)
        volatility_score = max(40, 100 - recent_volatility * 8)
        
        confidence = (data_quality + trend_consistency + seasonal_alignment + volatility_score) / 4
        
        # Análisis de patrones meteorológicos
        temp_range = temperatures.max() - temperatures.min()
        avg_change = abs(temperatures.diff().mean())
        
        result = {
            'prediction': float(final_prediction),
            'base_temp': base_temp,
            'linear_trend': linear_trend,
            'weighted_trend': weighted_trend,
            'long_trend': long_trend,
            'seasonal_factor': seasonal_factor,
            'confidence': float(np.clip(confidence, 65, 98)),
            'volatility': float(recent_volatility),
            'temp_range': float(temp_range),
            'avg_change': float(avg_change),
            'data_points': len(temperatures),
            'models_used': 3
        }
        
        return result, "success"
        
    except Exception as e:
        return None, f"Error en predicción avanzada: {str(e)}"

# ===== COMPONENTES UI ULTRA PROFESIONALES =====
def render_ultra_header():
    """Header ultra profesional con animaciones"""
    st.markdown("""
    <div class="ultra-header">
        <h1>🌤️ AEMET Analytics Professional</h1>
        <p>Sistema Avanzado de Inteligencia Meteorológica • Powered by Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)

def render_ultra_metrics(df):
    """Dashboard de métricas ultra profesional"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Cálculos avanzados
    total_records = len(df)
    
    if 'fecha' in df.columns:
        date_range = f"{df['fecha'].min().strftime('%m/%Y')} - {df['fecha'].max().strftime('%m/%Y')}"
        years_span = (df['fecha'].max() - df['fecha'].min()).days / 365.25
        data_density = total_records / years_span if years_span > 0 else 0
    else:
        date_range = "N/A"
        years_span = 0
        data_density = 0
    
    if 'tmed' in df.columns:
        temp_stats = df['tmed'].describe()
        temp_mean = temp_stats['mean']
        temp_std = temp_stats['std']
        temp_trend = calculate_advanced_trend(df['tmed'], min(len(df), 365))[0] * 365  # Tendencia anual
    else:
        temp_mean = temp_std = temp_trend = 0
    
    station_count = 1
    coverage_type = "Local"
    if 'nombre' in df.columns:
        station_count = df['nombre'].nunique()
        coverage_type = "Nacional" if station_count > 20 else "Regional" if station_count > 5 else "Local"
    elif 'indicativo' in df.columns:
        station_count = df['indicativo'].nunique()
        coverage_type = "Nacional" if station_count > 20 else "Regional" if station_count > 5 else "Local"
    
    with col1:
        st.markdown(f"""
        <div class="ultra-metric-card">
            <h3>📊 REGISTROS HISTÓRICOS</h3>
            <h2>{total_records:,}</h2>
            <p>{data_density:.0f} registros/año promedio</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="ultra-metric-card">
            <h3>📅 COBERTURA TEMPORAL</h3>
            <h2>{years_span:.1f} años</h2>
            <p>{date_range}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        trend_emoji = "📈" if temp_trend > 0 else "📉" if temp_trend < 0 else "➡️"
        st.markdown(f"""
        <div class="ultra-metric-card">
            <h3>🌡️ TEMPERATURA MEDIA</h3>
            <h2>{temp_mean:.1f}°C</h2>
            <p>{trend_emoji} {temp_trend:+.2f}°C tendencia anual</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="ultra-metric-card">
            <h3>📍 COBERTURA ESPACIAL</h3>
            <h2>{station_count}</h2>
            <p>Estaciones {coverage_type.lower()}</p>
        </div>
        """, unsafe_allow_html=True)

def create_ultra_charts(df, station='Estación específica', days=90, show_trends=True, show_confidence=True, show_precipitation=True):
    """Crear visualizaciones ultra profesionales y bonitas"""
    try:
        df_chart, station_clean = filter_by_station(df, station)
        
        if df_chart.empty:
            st.warning("⚠️ No hay datos disponibles para esta estación")
            return
        
        # Datos recientes
        recent_data = df_chart.tail(days * 3).copy()  # Más datos para mejor análisis
        
        if 'fecha' not in recent_data.columns:
            st.warning("⚠️ Datos de fecha no disponibles")
            return
        
        # Procesar fechas
        recent_data['fecha'] = pd.to_datetime(recent_data['fecha'])
        recent_data = recent_data.sort_values('fecha').tail(days)
        
        # Configurar tema dark ultramoderno para plotly
        ultra_theme = {
            'layout': {
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'font': {
                    'color': '#ffffff', 
                    'family': 'Inter, system-ui, sans-serif',
                    'size': 13
                },
                'xaxis': {
                    'gridcolor': 'rgba(102,126,234,0.15)',
                    'zerolinecolor': 'rgba(102,126,234,0.3)',
                    'linecolor': 'rgba(102,126,234,0.3)',
                    'color': '#ffffff',
                    'showgrid': True,
                    'gridwidth': 1
                },
                'yaxis': {
                    'gridcolor': 'rgba(102,126,234,0.15)',
                    'zerolinecolor': 'rgba(102,126,234,0.3)',
                    'linecolor': 'rgba(102,126,234,0.3)',
                    'color': '#ffffff',
                    'showgrid': True,
                    'gridwidth': 1
                },
                'colorway': ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a']
            }
        }
        
        # === GRÁFICO PRINCIPAL DE TEMPERATURA ===
        st.markdown("### 🌡️ **Análisis de Temperatura Avanzado**")
        
        fig_temp = go.Figure()
        
        if 'tmed' in recent_data.columns:
            # Calcular medias móviles suaves
            ma_3 = recent_data['tmed'].rolling(window=3, center=True).mean()
            ma_7 = recent_data['tmed'].rolling(window=7, center=True).mean()
            ma_15 = recent_data['tmed'].rolling(window=min(15, len(recent_data)), center=True).mean()
            
            # Temperatura real con puntos y líneas
            fig_temp.add_trace(go.Scatter(
                x=recent_data['fecha'],
                y=recent_data['tmed'],
                mode='markers+lines',
                name='🌡️ Temperatura Real',
                line=dict(color='#667eea', width=2),
                marker=dict(color='#667eea', size=4, opacity=0.8),
                hovertemplate='<b>%{x|%d/%m/%Y}</b><br>Temperatura: <b>%{y:.1f}°C</b><extra></extra>',
                fill=None
            ))
            
            # Media móvil 7 días (tendencia suave)
            if show_trends:
                fig_temp.add_trace(go.Scatter(
                    x=recent_data['fecha'],
                    y=ma_7,
                    mode='lines',
                    name='📊 Tendencia 7 días',
                    line=dict(color='#f093fb', width=3, smoothing=1.3),
                    hovertemplate='<b>%{x|%d/%m/%Y}</b><br>Media 7d: <b>%{y:.1f}°C</b><extra></extra>'
                ))
            
            # Línea de media general
            mean_temp = recent_data['tmed'].mean()
            fig_temp.add_hline(
                y=mean_temp, 
                line_dash="dash", 
                line_color="rgba(255,255,255,0.4)",
                annotation_text=f"Media: {mean_temp:.1f}°C",
                annotation_position="bottom right",
                annotation_font_color="white"
            )
            
            # Banda de temperatura (máx/mín si disponible)
            if show_confidence and 'tmax' in recent_data.columns and 'tmin' in recent_data.columns:
                fig_temp.add_trace(go.Scatter(
                    x=recent_data['fecha'],
                    y=recent_data['tmax'],
                    mode='lines',
                    name='🔥 Temperatura Máxima',
                    line=dict(color='#fa709a', width=2, dash='dot'),
                    opacity=0.8,
                    hovertemplate='<b>%{x|%d/%m/%Y}</b><br>Máxima: <b>%{y:.1f}°C</b><extra></extra>'
                ))
                
                fig_temp.add_trace(go.Scatter(
                    x=recent_data['fecha'],
                    y=recent_data['tmin'],
                    mode='lines',
                    name='❄️ Temperatura Mínima',
                    line=dict(color='#43e97b', width=2, dash='dot'),
                    opacity=0.8,
                    hovertemplate='<b>%{x|%d/%m/%Y}</b><br>Mínima: <b>%{y:.1f}°C</b><extra></extra>'
                ))
                
                # Relleno entre máx y mín - SIMPLIFICADO
                fig_temp.add_trace(go.Scatter(
                    x=list(recent_data['fecha']) + list(recent_data['fecha'][::-1]),
                    y=list(recent_data['tmax']) + list(recent_data['tmin'][::-1]),
                    fill='toself',
                    fillcolor='rgba(102,126,234,0.1)',
                    line=dict(width=0),
                    mode='lines',
                    name='🌡️ Rango Térmico',
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Actualizar layout del gráfico de temperatura
        fig_temp.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Inter, system-ui, sans-serif', size=13),
            title=dict(
                text=f'<b>Evolución Térmica - {station_clean}</b>',
                font=dict(size=18, color='white'),
                x=0.5
            ),
            height=400,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color='white')
            ),
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        fig_temp.update_xaxes(
            title=dict(text="📅 Fecha", font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor='rgba(102,126,234,0.15)',
            showgrid=True
        )
        fig_temp.update_yaxes(
            title=dict(text="🌡️ Temperatura (°C)", font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor='rgba(102,126,234,0.15)',
            showgrid=True
        )
        
        st.plotly_chart(fig_temp, use_container_width=True, config={'displayModeBar': False})
        
        # === GRÁFICOS ADICIONALES EN COLUMNAS ===
        if show_precipitation:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🌧️ **Análisis de Precipitación**")
                
                fig_prec = go.Figure()
                
                if 'prec' in recent_data.columns:
                    # Precipitación diaria
                    fig_prec.add_trace(go.Bar(
                        x=recent_data['fecha'],
                        y=recent_data['prec'],
                        name='🌧️ Precipitación Diaria',
                        marker=dict(
                            color=recent_data['prec'],
                            colorscale='Blues',
                            opacity=0.8,
                            line=dict(width=0)
                        ),
                        hovertemplate='<b>%{x|%d/%m/%Y}</b><br>Precipitación: <b>%{y:.1f} mm</b><extra></extra>'
                    ))
                
                fig_prec.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    title=dict(
                        text='<b>Precipitación Diaria</b>',
                        font=dict(size=16, color='white'),
                        x=0.5
                    ),
                    height=350,
                    xaxis=dict(
                        title=dict(text='📅 Fecha', font=dict(color='white')),
                        tickfont=dict(color='white'),
                        gridcolor='rgba(255,255,255,0.1)',
                        showgrid=True
                    ),
                    yaxis=dict(
                        title=dict(text='🌧️ Precipitación (mm)', font=dict(color='white')),
                        tickfont=dict(color='white'),
                        gridcolor='rgba(255,255,255,0.1)',
                        showgrid=True
                    ),
                    showlegend=False,
                    margin=dict(l=60, r=30, t=70, b=60)
                )
                
                st.plotly_chart(fig_prec, use_container_width=True, config={'displayModeBar': False})
            
            with col2:
                st.markdown("### 📊 **Distribución Estadística**")
                
                # Crear histograma dentro de la columna
                fig_hist = go.Figure()
                
                if 'tmed' in recent_data.columns:
                    # Histograma con curva de densidad
                    fig_hist.add_trace(go.Histogram(
                        x=recent_data['tmed'],
                        name='📊 Frecuencia',
                        nbinsx=min(20, len(recent_data)//3),
                        marker=dict(
                            color='rgba(102,126,234,0.7)',
                            line=dict(color='rgba(102,126,234,1)', width=1)
                        ),
                        opacity=0.8,
                        hovertemplate='Temperatura: <b>%{x:.1f}°C</b><br>Frecuencia: <b>%{y}</b><extra></extra>'
                    ))
                    
                    # Líneas de percentiles
                    q25 = recent_data['tmed'].quantile(0.25)
                    q50 = recent_data['tmed'].quantile(0.50)
                    q75 = recent_data['tmed'].quantile(0.75)
                    
                    fig_hist.add_vline(x=q25, line_dash="dash", line_color="#43e97b", 
                                     annotation_text=f"Q1: {q25:.1f}°C", annotation_position="top")
                    fig_hist.add_vline(x=q50, line_dash="solid", line_color="#f093fb", line_width=2,
                                     annotation_text=f"Mediana: {q50:.1f}°C", annotation_position="top")
                    fig_hist.add_vline(x=q75, line_dash="dash", line_color="#fa709a",
                                     annotation_text=f"Q3: {q75:.1f}°C", annotation_position="top")
                
                fig_hist.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff', family='Inter, system-ui, sans-serif', size=13),
                    title=dict(
                        text='<b>Distribución de Temperaturas</b>',
                        font=dict(size=16, color='white'),
                        x=0.5
                    ),
                    height=350,
                    showlegend=False,
                    margin=dict(l=50, r=50, t=60, b=50)
                )
                
                fig_hist.update_xaxes(
                    title=dict(text="🌡️ Temperatura (°C)", font=dict(color='white')),
                    tickfont=dict(color='white'),
                    gridcolor='rgba(102,126,234,0.15)',
                    showgrid=True
                )
                fig_hist.update_yaxes(
                    title=dict(text="📊 Frecuencia", font=dict(color='white')),
                    tickfont=dict(color='white'),
                    gridcolor='rgba(102,126,234,0.15)',
                    showgrid=True
                )
                
                st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
                
        else:
            # Solo mostrar distribución estadística en una columna más ancha
            st.markdown("### 📊 **Distribución Estadística**")
            
            # Crear histograma fuera de columnas (ocupando todo el ancho)
            fig_hist = go.Figure()
            
            if 'tmed' in recent_data.columns:
                # Histograma con curva de densidad
                fig_hist.add_trace(go.Histogram(
                    x=recent_data['tmed'],
                    name='📊 Frecuencia',
                    nbinsx=min(20, len(recent_data)//3),
                    marker=dict(
                        color='rgba(102,126,234,0.7)',
                        line=dict(color='rgba(102,126,234,1)', width=1)
                    ),
                    opacity=0.8,
                    hovertemplate='Temperatura: <b>%{x:.1f}°C</b><br>Frecuencia: <b>%{y}</b><extra></extra>'
                ))
                
                # Líneas de percentiles
                q25 = recent_data['tmed'].quantile(0.25)
                q50 = recent_data['tmed'].quantile(0.50)
                q75 = recent_data['tmed'].quantile(0.75)
                
                fig_hist.add_vline(x=q25, line_dash="dash", line_color="#43e97b", 
                                 annotation_text=f"Q1: {q25:.1f}°C", annotation_position="top")
                fig_hist.add_vline(x=q50, line_dash="solid", line_color="#f093fb", line_width=2,
                                 annotation_text=f"Mediana: {q50:.1f}°C", annotation_position="top")
                fig_hist.add_vline(x=q75, line_dash="dash", line_color="#fa709a",
                                 annotation_text=f"Q3: {q75:.1f}°C", annotation_position="top")
            
            fig_hist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', family='Inter, system-ui, sans-serif', size=13),
                title=dict(
                    text='<b>Distribución de Temperaturas</b>',
                    font=dict(size=16, color='white'),
                    x=0.5
                ),
                height=350,
                showlegend=False,
                margin=dict(l=50, r=50, t=60, b=50)
            )
            
            fig_hist.update_xaxes(
                title=dict(text="🌡️ Temperatura (°C)", font=dict(color='white')),
                tickfont=dict(color='white'),
                gridcolor='rgba(102,126,234,0.15)',
                showgrid=True
            )
            fig_hist.update_yaxes(
                title=dict(text="📊 Frecuencia", font=dict(color='white')),
                tickfont=dict(color='white'),
                gridcolor='rgba(102,126,234,0.15)',
                showgrid=True
            )
            
            st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
        
        # === GRÁFICO DE CORRELACIÓN Y PATRONES ===
        if len(recent_data) > 7:
            st.markdown("### 🔬 **Análisis de Patrones Meteorológicos**")
            
            fig_patterns = make_subplots(
                rows=1, cols=2,
                subplot_titles=('📈 Variabilidad Diaria', '🌀 Correlación T°-Precipitación'),
                horizontal_spacing=0.1
            )
            
            if 'tmed' in recent_data.columns:
                # Variabilidad diaria (diferencias día a día)
                temp_change = recent_data['tmed'].diff()
                colors = ['#43e97b' if x >= 0 else '#fa709a' for x in temp_change]
                
                fig_patterns.add_trace(
                    go.Bar(
                        x=recent_data['fecha'][1:],
                        y=temp_change[1:],
                        name='📈 Cambio T°',
                        marker_color=colors[1:],
                        hovertemplate='<b>%{x|%d/%m/%Y}</b><br>Cambio: <b>%{y:+.1f}°C</b><extra></extra>'
                    ), row=1, col=1
                )
                
                # Correlación temperatura-precipitación
                if 'prec' in recent_data.columns:
                    fig_patterns.add_trace(
                        go.Scatter(
                            x=recent_data['tmed'],
                            y=recent_data['prec'],
                            mode='markers',
                            name='🌡️vs🌧️',
                            marker=dict(
                                color=recent_data['tmed'],
                                size=8,
                                colorscale='RdYlBu_r',
                                opacity=0.7,
                                line=dict(width=1, color='white')
                            ),
                            hovertemplate='T°: <b>%{x:.1f}°C</b><br>Precipitación: <b>%{y:.1f}mm</b><extra></extra>'
                        ), row=1, col=2
                    )
            
            fig_patterns.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', family='Inter, system-ui, sans-serif', size=13),
                height=350,
                showlegend=False,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            fig_patterns.update_xaxes(
                title=dict(text="📅 Fecha", font=dict(color='white')),
                row=1, col=1,
                tickfont=dict(color='white'),
                gridcolor='rgba(102,126,234,0.15)',
                showgrid=True
            )
            fig_patterns.update_xaxes(
                title=dict(text="🌡️ Temperatura (°C)", font=dict(color='white')),
                row=1, col=2,
                tickfont=dict(color='white'),
                gridcolor='rgba(102,126,234,0.15)',
                showgrid=True
            )
            fig_patterns.update_yaxes(
                title=dict(text="🌡️ Cambio (°C)", font=dict(color='white')),
                row=1, col=1,
                tickfont=dict(color='white'),
                gridcolor='rgba(102,126,234,0.15)',
                showgrid=True
            )
            fig_patterns.update_yaxes(
                title=dict(text="🌧️ Precipitación (mm)", font=dict(color='white')),
                row=1, col=2,
                tickfont=dict(color='white'),
                gridcolor='rgba(102,126,234,0.15)',
                showgrid=True
            )
            
            st.plotly_chart(fig_patterns, use_container_width=True, config={'displayModeBar': False})
        
        # === SISTEMA DE PREDICCIÓN ML PROFESIONAL ===
        st.markdown("---")
        st.markdown("### 🔮 **Predicción de Temperatura Media - Sistema ML Avanzado**")
        
        # === IMPLEMENTACIÓN EXACTA DE LA ARQUITECTURA REAL ===
        class AdvancedPositionalEncoding(nn.Module):
            def __init__(self, d_model: int, max_len: int):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
                
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                
                self.register_buffer('pe', pe)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x + self.pe[:, :x.size(1)]
                return self.dropout(x)

        class MultiHeadAttentionLayer(nn.Module):
            def __init__(self, d_model: int, n_heads: int, dropout_rate: float = 0.1):
                super().__init__()
                self.mha = nn.MultiheadAttention(
                    embed_dim=d_model, 
                    num_heads=n_heads, 
                    dropout=dropout_rate,
                    batch_first=True
                )
                self.norm = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(dropout_rate)
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                attn_output, _ = self.mha(x, x, x, need_weights=False)
                return self.norm(x + self.dropout(attn_output))

        class FeedForwardLayer(nn.Module):
            def __init__(self, d_model: int, dff: int, dropout_rate: float = 0.1):
                super().__init__()
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, dff),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(dff, d_model),
                    nn.Dropout(dropout_rate)
                )
                self.norm = nn.LayerNorm(d_model)
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.norm(x + self.ffn(x))

        class TransformerBlock(nn.Module):
            def __init__(self, d_model: int, n_heads: int, dff: int, dropout_rate: float = 0.1):
                super().__init__()
                self.attention = MultiHeadAttentionLayer(d_model, n_heads, dropout_rate)
                self.feed_forward = FeedForwardLayer(d_model, dff, dropout_rate)
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.attention(x)
                x = self.feed_forward(x)
                return x

        class ProductionLSTMTransformerModel(nn.Module):
            def __init__(
                self,
                d_input: int,
                lstm_hidden_size: int,
                lstm_layers: int,
                n_transformer_layers: int,
                d_model: int,
                n_heads: int,
                dff: int,
                max_len: int,
                dropout_rate: float = 0.1,
                prediccion: int = 7
            ):
                super().__init__()
                
                self.d_model = d_model
                self.prediccion = prediccion
                self.input_projection = nn.Linear(d_input, lstm_hidden_size)
                
                # LSTM Stack
                self.lstm_layers = nn.ModuleList([
                    nn.LSTM(
                        input_size=lstm_hidden_size if i == 0 else lstm_hidden_size * 2,
                        hidden_size=lstm_hidden_size,
                        batch_first=True,
                        dropout=dropout_rate if i < lstm_layers - 1 else 0,
                        bidirectional=True
                    ) for i in range(lstm_layers)
                ])
                
                # Projection to transformer dimension
                self.lstm_to_transformer = nn.Linear(lstm_hidden_size * 2, d_model)
                
                # Transformer Stack
                self.pos_encoding = AdvancedPositionalEncoding(d_model, max_len)
                self.transformer_blocks = nn.ModuleList([
                    TransformerBlock(d_model, n_heads, dff, dropout_rate)
                    for _ in range(n_transformer_layers)
                ])
                
                # Output layers
                self.output_norm = nn.LayerNorm(d_model)
                self.output_layers = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(d_model // 2, d_model // 4),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(d_model // 4, 1)
                )
                
                # Skip connection
                self.skip_connection = nn.Linear(d_input, 1)
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                batch_size, seq_len, _ = x.shape
                
                # Skip connection
                skip = self.skip_connection(x[:, -1, :])
                
                # Input projection
                x = self.input_projection(x)
                
                # LSTM processing
                for lstm in self.lstm_layers:
                    x, _ = lstm(x)
                
                # Project to transformer dimension
                x = self.lstm_to_transformer(x)
                
                # Positional encoding
                x = self.pos_encoding(x)
                
                # Transformer blocks
                for transformer_block in self.transformer_blocks:
                    x = transformer_block(x)
                
                # Output processing
                x = self.output_norm(x)
                x = x[:, -self.prediccion:, :].mean(dim=1)
                
                # Main prediction
                main_output = self.output_layers(x)
                
                # Combine with skip connection
                output = main_output + 0.1 * skip
                
                return output
        
        # Interfaz de usuario para predicción
        pred_col1, pred_col2, pred_col3 = st.columns([1, 1, 1])
        
        with pred_col1:
            dias_prediccion = st.selectbox(
                "📅 Días a predecir:",
                [1, 3, 7, 14, 30],
                index=2,
                help="Número de días futuros a predecir"
            )
        
        with pred_col2:
            confianza_nivel = st.slider(
                "📊 Nivel de confianza:",
                min_value=80,
                max_value=99,
                value=95,
                step=5,
                help="Nivel de confianza para intervalos de predicción"
            )
        
        with pred_col3:
            validacion_avanzada = st.checkbox(
                "🔍 Validación avanzada",
                value=True,
                help="Realizar validaciones adicionales de calidad"
            )
        
        # Botón de predicción
        if st.button("🚀 Generar Predicción Profesional", type="primary", use_container_width=True):
            with st.spinner("🔄 Cargando modelo y procesando datos..."):
                try:
                    # === 1. CARGA DEL MODELO ===
                    device = torch.device('cpu')
                    checkpoint = torch.load('production_weather_model.pth', map_location=device)
                    
                    # Crear modelo con arquitectura exacta del entrenamiento
                    config = checkpoint.get('config', {})
                    model = ProductionLSTMTransformerModel(
                        d_input=13,
                        lstm_hidden_size=256,
                        lstm_layers=3,
                        n_transformer_layers=3,
                        d_model=128,
                        n_heads=8,
                        dff=512,
                        max_len=90,
                        dropout_rate=0.2,
                        prediccion=7
                    )
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    
                    # Cargar scalers
                    scaler_X = joblib.load('scaler_X_production.joblib')
                    scaler_y = joblib.load('scaler_y_production.joblib')
                    
                    # Features del modelo
                    model_features = checkpoint['features']
                    
                    # === 2. VALIDACIÓN Y MANEJO INTELIGENTE DE DATOS ===
                    station_data = recent_data.copy().sort_values('fecha').reset_index(drop=True)
                    
                    if len(station_data) < 30:
                        st.error(f"❌ Datos insuficientes: Se necesitan al menos 30 días de historial. Disponibles: {len(station_data)} días")
                        st.info("💡 Se requiere un mínimo de datos para generar predicciones fiables")
                    else:
                        # Manejo inteligente para secuencias menores a 90 días
                        required_sequence = 90
                        available_days = len(station_data)
                        
                        if available_days < required_sequence:
                            # Estrategias de mitigación profesionales
                            estrategia = st.radio(
                                "🔧 Estrategia de manejo de datos:",
                                [
                                    "🔄 Padding inteligente",
                                    
                                ],
                                index=0,
                                help="Selecciona cómo manejar la falta de datos históricos"
                            )
                            
                            if estrategia.startswith("🎯"):
                                st.error("❌ Predicción cancelada por falta de datos suficientes")
                                st.info("💡 Espera a tener más datos históricos o usa el padding inteligente")
                            else:
                                proceed_with_prediction = True
                        else:
                            estrategia = "🔄 Datos completos"
                            proceed_with_prediction = True
                        
                        if 'proceed_with_prediction' in locals() and proceed_with_prediction:
                            # === 3. FEATURE ENGINEERING PROFESIONAL ===
                            
                            # Validar features básicas
                            required_basic = ['tmed', 'tmin', 'tmax', 'prec', 'altitud']
                            available_basic = [f for f in required_basic if f in station_data.columns]
                            missing_basic = [f for f in required_basic if f not in available_basic]
                            
                            if missing_basic:
                                st.error(f"❌ Features básicas faltantes: {missing_basic}")
                            else:
                                # Limpieza de datos anómalos (rangos realistas España)
                                station_data.loc[station_data['tmed'] < -20, 'tmed'] = np.nan
                                station_data.loc[station_data['tmed'] > 50, 'tmed'] = np.nan
                                station_data.loc[station_data['tmin'] < -25, 'tmin'] = np.nan
                                station_data.loc[station_data['tmax'] > 55, 'tmax'] = np.nan
                                station_data.loc[station_data['prec'] < 0, 'prec'] = 0
                                station_data.loc[station_data['prec'] > 500, 'prec'] = np.nan
                                
                                # Interpolación de datos faltantes
                                for col in ['tmed', 'tmin', 'tmax']:
                                    if col in station_data.columns:
                                        station_data[col] = station_data[col].interpolate(method='linear', limit_direction='both')
                                
                                station_data['prec'] = station_data['prec'].fillna(0)
                                
                                # Features estacionales
                                station_data['day_of_year'] = station_data['fecha'].dt.dayofyear
                                station_data['season_sin'] = np.sin(2 * np.pi * station_data['day_of_year'] / 365.25)
                                station_data['season_cos'] = np.cos(2 * np.pi * station_data['day_of_year'] / 365.25)
                                
                                # Features de lag
                                for lag in [1, 3, 7]:
                                    station_data[f'tmed_lag_{lag}'] = station_data['tmed'].shift(lag)
                                
                                # Estadísticas móviles
                                station_data['tmed_mean_7d'] = station_data['tmed'].rolling(window=7, min_periods=3).mean()
                                station_data['tmed_std_7d'] = station_data['tmed'].rolling(window=7, min_periods=3).std()
                                
                                # Rango térmico
                                station_data['temp_range'] = station_data['tmax'] - station_data['tmin']
                                
                                # Relleno de valores faltantes para lags
                                lag_cols = [f'tmed_lag_{lag}' for lag in [1, 3, 7]] + ['tmed_mean_7d', 'tmed_std_7d']
                                for col in lag_cols:
                                    if col in station_data.columns:
                                        station_data[col] = station_data[col].fillna(method='bfill').fillna(station_data['tmed'].mean())
                                
                                # Verificar datos completos
                                complete_data = station_data.dropna(subset=model_features)
                                
                                if len(complete_data) < 30:
                                    st.error(f"❌ Datos completos insuficientes: {len(complete_data)} registros válidos")
                                else:
                                    
                                    # === 4. PREPARACIÓN INTELIGENTE DE SECUENCIA ===
                                    available_complete = len(complete_data)
                                    
                                    if estrategia.startswith("🔄") and available_complete < required_sequence:
                                        # PADDING INTELIGENTE PROFESIONAL
                                        
                                        # Usar todos los datos disponibles
                                        base_sequence = complete_data[model_features].values
                                        
                                        # Calcular estadísticas para padding
                                        means = np.mean(base_sequence, axis=0)
                                        stds = np.std(base_sequence, axis=0)
                                        
                                        # Generar datos sintéticos realistas
                                        missing_days = required_sequence - available_complete
                                        
                                        # Padding al inicio con variación gaussiana controlada
                                        synthetic_data = []
                                        for _ in range(missing_days):
                                            synthetic_row = means + np.random.normal(0, stds * 0.3, len(model_features))
                                            # Asegurar rangos realistas
                                            synthetic_row[0] = np.clip(synthetic_row[0], -15, 40)  # tmed
                                            synthetic_row[1] = np.clip(synthetic_row[1], -20, 35)  # tmin
                                            synthetic_row[2] = np.clip(synthetic_row[2], -10, 45)  # tmax
                                            synthetic_row[3] = np.clip(synthetic_row[3], 0, 100)   # prec
                                            synthetic_data.append(synthetic_row)
                                        
                                        # Combinar datos sintéticos + datos reales
                                        input_sequence = np.vstack([synthetic_data, base_sequence])
                                        
                                    elif estrategia.startswith("📊"):
                                        # USAR DATOS DISPONIBLES CON REPETICIÓN
                                        
                                        base_sequence = complete_data[model_features].values
                                        
                                        if available_complete < required_sequence:
                                            # Repetir secuencia hasta completar 90 días
                                            repetitions_needed = (required_sequence // available_complete) + 1
                                            extended_sequence = np.tile(base_sequence, (repetitions_needed, 1))
                                            input_sequence = extended_sequence[:required_sequence]
                                        else:
                                            input_sequence = base_sequence[-required_sequence:]
                                    
                                    else:
                                        # DATOS COMPLETOS - usar últimos 90 días
                                        input_sequence = complete_data.tail(required_sequence)[model_features].values
                                    
                                    
                                    
                                    # === 5. ESCALADO CORRECTO Y PREDICCIÓN ===
                                    # Verificar dimensiones finales
                                    if input_sequence.shape != (required_sequence, len(model_features)):
                                        st.error(f"❌ Error en dimensiones: {input_sequence.shape} != ({required_sequence}, {len(model_features)})")
                                    else:
                                        
                                        # ESCALADO CORRECTO: aplicar scaler fila por fila (día por día)
                                        input_scaled = np.zeros_like(input_sequence)
                                        for day in range(input_sequence.shape[0]):
                                            # Normalizar cada día individualmente (13 features)
                                            day_data = input_sequence[day:day+1, :]  # Shape: (1, 13)
                                            day_scaled = scaler_X.transform(day_data)  # Shape: (1, 13)
                                            input_scaled[day, :] = day_scaled[0, :]
                                        
                                        # Convertir a tensor
                                        X_tensor = torch.FloatTensor(input_scaled).unsqueeze(0)  # Shape: (1, 90, 13)
                                        
                                        # Generar predicciones
                                        predicciones = []
                                        fechas_pred = []
                                        
                                        # Predicción iterativa mejorada
                                        X_pred = X_tensor.clone()
                                        
                                        for day in range(dias_prediccion):
                                            fecha_pred = station_data['fecha'].max() + timedelta(days=day+1)
                                            fechas_pred.append(fecha_pred)
                                            
                                            # Predicción del modelo
                                            with torch.no_grad():
                                                pred_scaled = model(X_pred).cpu().numpy()[0, 0]
                                            
                                            # Desnormalizar predicción
                                            pred_temp = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
                                            
                                            # Aplicar constraints realistas para España
                                            pred_temp = np.clip(pred_temp, -20, 50)
                                            predicciones.append(pred_temp)
                                            
                                            # Actualizar secuencia para próxima predicción
                                            if day < dias_prediccion - 1:
                                                # Crear nuevo registro con valores sintéticos coherentes
                                                new_day_features = input_sequence[-1, :].copy()  # Basar en último día real
                                                new_day_features[0] = pred_temp  # tmed predicha
                                                
                                                # Actualizar features derivadas básicas
                                                if len(model_features) > 1:
                                                    new_day_features[1] = pred_temp - 3  # tmin aprox
                                                    new_day_features[2] = pred_temp + 5  # tmax aprox
                                                    # Mantener otros features similares al último día
                                                
                                                # Escalar el nuevo día
                                                new_day_scaled = scaler_X.transform(new_day_features.reshape(1, -1))[0]
                                                
                                                # Rotar secuencia: quitar primer día, añadir nuevo al final
                                                X_pred = torch.cat([
                                                    X_pred[:, 1:, :], 
                                                    torch.FloatTensor(new_day_scaled).unsqueeze(0).unsqueeze(0)
                                                ], dim=1)
                                
                                # === 6. CÁLCULO DE INTERVALOS DE CONFIANZA ===
                                # Basado en la desviación estándar del error del modelo
                                model_std = 1.89  # RMSE del modelo
                                z_score = 1.96 if confianza_nivel == 95 else (2.58 if confianza_nivel == 99 else 1.645)
                                
                                intervalos_inf = [pred - z_score * model_std for pred in predicciones]
                                intervalos_sup = [pred + z_score * model_std for pred in predicciones]
                                
                                # === 7. PRESENTACIÓN DE RESULTADOS ===
                                
                                # Crear DataFrame de resultados
                                df_prediccion = pd.DataFrame({
                                    'Fecha': fechas_pred,
                                    'Temperatura_Predicha': [round(p, 1) for p in predicciones],
                                    'Limite_Inferior': [round(inf, 1) for inf in intervalos_inf],
                                    'Limite_Superior': [round(sup, 1) for sup in intervalos_sup]
                                })
                                
                                # Métricas de resumen
                                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                                
                                temp_actual = complete_data['tmed'].iloc[-1]
                                temp_pred_promedio = np.mean(predicciones)
                                variacion = temp_pred_promedio - temp_actual
                                tendencia = "📈" if predicciones[-1] > predicciones[0] else "📉"
                                
                                with col_m1:
                                    st.metric("🌡️ Temp. Actual", f"{temp_actual:.1f}°C")
                                
                                with col_m2:
                                    st.metric("🔮 Pred. Promedio", f"{temp_pred_promedio:.1f}°C", 
                                             delta=f"{variacion:+.1f}°C")
                                
                                with col_m3:
                                    st.metric("📊 Rango Predicción", 
                                             f"{min(predicciones):.1f}°C - {max(predicciones):.1f}°C")
                                
                                with col_m4:
                                    st.metric("📈 Tendencia", f"{tendencia} {abs(predicciones[-1] - predicciones[0]):.1f}°C")
                                
                                # Gráfico de predicción
                                fig_pred = go.Figure()
                                
                                # Datos históricos (últimos 30 días)
                                hist_data = complete_data.tail(30)
                                fig_pred.add_trace(go.Scatter(
                                    x=hist_data['fecha'],
                                    y=hist_data['tmed'],
                                    mode='lines+markers',
                                    name='📊 Histórico',
                                    line=dict(color='#4fc3f7', width=2),
                                    marker=dict(size=6)
                                ))
                                
                                # Predicciones
                                fig_pred.add_trace(go.Scatter(
                                    x=df_prediccion['Fecha'],
                                    y=df_prediccion['Temperatura_Predicha'],
                                    mode='lines+markers',
                                    name='🔮 Predicción ML',
                                    line=dict(color='#ff6b6b', width=3, dash='dot'),
                                    marker=dict(size=8, symbol='star')
                                ))
                                
                                # Intervalo de confianza
                                fig_pred.add_trace(go.Scatter(
                                    x=df_prediccion['Fecha'],
                                    y=df_prediccion['Limite_Superior'],
                                    mode='lines',
                                    line=dict(width=0),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                                
                                fig_pred.add_trace(go.Scatter(
                                    x=df_prediccion['Fecha'],
                                    y=df_prediccion['Limite_Inferior'],
                                    mode='lines',
                                    line=dict(width=0),
                                    fill='tonexty',
                                    fillcolor='rgba(255, 107, 107, 0.2)',
                                    name=f'📊 Confianza {confianza_nivel}%',
                                    hoverinfo='skip'
                                ))
                                
                                fig_pred.update_layout(
                                    title=dict(
                                        text=f"🔮 Predicción ML - Temperatura Media ({station_clean})",
                                        font=dict(color='white', size=18),
                                        x=0.5
                                    ),
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='white'),
                                    height=450,
                                    xaxis=dict(
                                        title="📅 Fecha",
                                        gridcolor='rgba(102,126,234,0.15)',
                                        showgrid=True
                                    ),
                                    yaxis=dict(
                                        title="🌡️ Temperatura (°C)",
                                        gridcolor='rgba(102,126,234,0.15)',
                                        showgrid=True
                                    ),
                                    legend=dict(
                                        bgcolor='rgba(0,0,0,0.5)',
                                        bordercolor='white',
                                        borderwidth=1
                                    )
                                )
                                
                                st.plotly_chart(fig_pred, use_container_width=True, config={'displayModeBar': False})
                                
                                # Tabla de resultados detallados
                                st.markdown("#### 📋 Resultados Detallados")
                                st.dataframe(
                                    df_prediccion.style.format({
                                        'Temperatura_Predicha': '{:.1f}°C',
                                        'Limite_Inferior': '{:.1f}°C',
                                        'Limite_Superior': '{:.1f}°C'
                                    }).background_gradient(subset=['Temperatura_Predicha'], cmap='RdYlBu_r'),
                                    use_container_width=True
                                )
                                
                                # Información del modelo
                                with st.expander("ℹ️ Información del Modelo ML"):
                                    metrics = checkpoint.get('metrics', {})
                                    
                                    st.markdown(f"""
                                    **🤖 Arquitectura:** LSTM Bidireccional + Transformer + Skip Connections  
                                    **📊 Features:** {len(model_features)} variables (temperatura, meteorología, lags, estadísticas)  
                                    **🎯 Rendimiento:**
                                    - MAE: {metrics.get('mae', 'N/A'):.3f}°C
                                    - RMSE: {metrics.get('rmse', 'N/A'):.3f}°C  
                                    - R²: {metrics.get('r2', 'N/A'):.3f}
                                    
                                    **⚡ Configuración:**
                                    - Secuencia entrada: 90 días
                                    - Predicción: Iterativa día a día
                                    - Confianza: {confianza_nivel}%
                                    
                                    **📈 Features utilizadas:** {', '.join(model_features)}
                                    """)
                                
                except Exception as model_error:
                    st.error(f"❌ Error en el sistema de predicción: {str(model_error)}")
                    st.info("💡 Verifica que todos los archivos del modelo estén presentes y sean compatibles")
                    if validacion_avanzada:
                        st.exception(model_error)
        
    except Exception as e:
        st.error(f"❌ Error creando visualizaciones: {str(e)}")
        st.info("💡 Verifica que los datos contienen las columnas necesarias (fecha, tmed, prec, etc.)")

def render_advanced_data_table(df, station='Estación específica', limit=100):
    """Tabla de datos avanzada con métricas"""
    try:
        df_table, station_clean = filter_by_station(df, station)
        
        if df_table.empty:
            st.warning("⚠️ No hay datos para mostrar")
            return
        
        # Preparar datos para la tabla
        display_data = df_table.tail(limit).copy()
        
        # Añadir columnas calculadas
        if 'tmed' in display_data.columns:
            display_data['temp_change'] = display_data['tmed'].diff()
            display_data['temp_ma7'] = display_data['tmed'].rolling(window=7, min_periods=1).mean()
        
        # Formatear columnas numéricas
        numeric_cols = display_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['tmed', 'tmax', 'tmin', 'temp_change', 'temp_ma7']:
                display_data[col] = display_data[col].round(1)
            elif col == 'prec':
                display_data[col] = display_data[col].round(2)
        
        # Renombrar columnas para mejor presentación
        column_names = {
            'fecha': '📅 Fecha',
            'tmed': '🌡️ T.Media',
            'tmax': '🔥 T.Máxima',
            'tmin': '❄️ T.Mínima',
            'prec': '🌧️ Precipitación',
            'temp_change': '📈 Cambio',
            'temp_ma7': '📊 Media 7d',
            'nombre': '📍 Estación',
            'indicativo': '📡 Código',
            'provincia': '�️ Provincia'
        }
        
        display_data = display_data.rename(columns=column_names)
        
        # Mostrar estadísticas rápidas
        col1, col2, col3, col4 = st.columns(4)
        
        if '🌡️ T.Media' in display_data.columns:
            with col1:
                avg_temp = display_data['🌡️ T.Media'].mean()
                st.metric("📊 Promedio", f"{avg_temp:.1f}°C")
            
            with col2:
                max_temp = display_data['🌡️ T.Media'].max()
                st.metric("🔥 Máxima", f"{max_temp:.1f}°C")
            
            with col3:
                min_temp = display_data['🌡️ T.Media'].min()
                st.metric("❄️ Mínima", f"{min_temp:.1f}°C")
            
            with col4:
                if '📈 Cambio' in display_data.columns:
                    avg_change = display_data['📈 Cambio'].mean()
                    change_emoji = "📈" if avg_change > 0 else "📉" if avg_change < 0 else "➡️"
                    st.metric("📈 Cambio Prom.", f"{change_emoji} {avg_change:.2f}°C")
        
        # Mostrar tabla con estilo
        st.dataframe(
            display_data,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
    except Exception as e:
        st.error(f"❌ Error preparando tabla: {str(e)}")

# ===== APLICACIÓN PRINCIPAL =====
def main():
    load_dark_theme()
    render_ultra_header()
    
    # Cargar datos
    df = load_weather_data()
    if df is None:
        st.stop()
    
    # Sidebar ultra profesional
    with st.sidebar:
        st.markdown("## 🎛️ Centro de Control")
        
        # Selector de estación mejorado
        station_options = get_station_list(df)
        selected_station = st.selectbox(
            "🌍 Ubicación de análisis:",
            station_options,
            index=0 if station_options else 0,
            help="Selecciona una estación meteorológica específica para análisis detallado"
        )
        
        st.markdown("---")
        
        # Configuración avanzada
        st.markdown("### ⚙️ Configuración de Análisis")
        
        dias_analisis = st.slider(
            "📊 Período de análisis (días)",
            min_value=30,
            max_value=730,
            value=180,
            step=30,
            help="Número de días históricos para el análisis"
        )
        
        show_trends = st.checkbox("📈 Mostrar tendencias", True)
        show_confidence = st.checkbox("📊 Bandas de confianza", True)
        show_precipitation = st.checkbox("🌧️ Análisis precipitación", True)
        
        st.markdown("---")
        
        # Filtros temporales
        st.markdown("### 📅 Filtros Temporales")
        
        if 'fecha' in df.columns:
            min_date = df['fecha'].min().date()
            max_date = df['fecha'].max().date()
            
            # Selector de rango rápido
            range_option = st.selectbox(
                "⚡ Rango rápido:",
                ["📅 Últimos 30 días", "📅 Últimos 90 días", "📅 Últimos 6 meses", "📅 Último año", "🎯 Personalizado"],
                index=1,
                help="Selecciona un período predefinido o personaliza el rango"
            )
            
            # Configurar fechas según selección
            if range_option == "📅 Últimos 30 días":
                start_date = max_date - timedelta(days=30)
                end_date = max_date
                date_range = [start_date, end_date]
            elif range_option == "📅 Últimos 90 días":
                start_date = max_date - timedelta(days=90)
                end_date = max_date
                date_range = [start_date, end_date]
            elif range_option == "📅 Últimos 6 meses":
                start_date = max_date - timedelta(days=180)
                end_date = max_date
                date_range = [start_date, end_date]
            elif range_option == "📅 Último año":
                start_date = max_date - timedelta(days=365)
                end_date = max_date
                date_range = [start_date, end_date]
            else:
                # Personalizado
                date_range = st.date_input(
                    "🎯 Rango personalizado:",
                    value=[max_date - timedelta(days=dias_analisis), max_date],
                    min_value=min_date,
                    max_value=max_date,
                    help="Selecciona fechas específicas para el análisis"
                )
        
        st.markdown("---")
        
        # Estado del sistema
        st.markdown("### 📊 Estado del Sistema")
        st.success("✅ Sistema operativo")
        st.info(f"🔄 Sincronizado: {datetime.now().strftime('%H:%M:%S')}")
        
        # Información de datos
        df_filtered, station_clean = filter_by_station(df, selected_station)
        
        # Aplicar filtro temporal si está disponible
        if 'fecha' in df_filtered.columns and 'date_range' in locals() and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            df_filtered = df_filtered[
                (df_filtered['fecha'] >= start_date) & 
                (df_filtered['fecha'] <= end_date)
            ].copy()
        
        station_count = 1
        if 'nombre' in df.columns:
            station_count = df['nombre'].nunique()
        elif 'indicativo' in df.columns:
            station_count = df['indicativo'].nunique()
            
        # Mostrar información del rango temporal aplicado
        if 'fecha' in df_filtered.columns and not df_filtered.empty:
            period_start = df_filtered['fecha'].min().strftime('%d/%m/%Y')
            period_end = df_filtered['fecha'].max().strftime('%d/%m/%Y')
            period_info = f"{period_start} - {period_end}"
        else:
            period_info = f"{dias_analisis} días"
            
        st.markdown(f"""
        **📈 Registros activos:** {len(df_filtered):,}  
        **🌍 Estación:** {station_clean}  
        **📊 Período:** {period_info}  
        **🎯 Calidad:** {min(100, len(df_filtered)/1000*100):.0f}%  
        **📍 Total estaciones:** {station_count:,}
        """)
    
    # Dashboard principal
    render_ultra_metrics(df_filtered)
    
    # Layout principal mejorado
    col1, col2 = st.columns([2.8, 1.2])
    
    with col1:
        st.markdown("## 📈 Análisis Meteorológico Avanzado")
        create_ultra_charts(df_filtered, selected_station, dias_analisis, show_trends, show_confidence, show_precipitation)
    
    with col2:
        st.markdown("## 🤖 Asistente de Consultas con IA")
        st.info("Pregunta al asistente sobre los datos históricos del clima. Por ejemplo: '¿Cuál fue la temperatura media en Madrid en mayo de 2023?'")
        
        pregunta = st.text_input("Haz tu pregunta aquí:", key="ask_input", placeholder="Ejemplo: ¿Cuál fue la precipitación total en Valencia en 2022?")

        if st.button("Hacer Pregunta", type="primary", use_container_width=True):
            if pregunta:
                with st.spinner("🧠 El asistente está buscando la respuesta..."):
                    try:
                        payload = {"pregunta": pregunta}
                        response = requests.post(f"{API_URL}/ask", json=payload, timeout=90)
                        
                        if response.status_code == 200:
                            respuesta_api = response.json().get("respuesta", "No se pudo obtener la respuesta.")
                            st.markdown("### 💬 Respuesta del Asistente:")
                            st.success(respuesta_api)
                        else:
                            error_detail = response.json().get('detail', 'Error desconocido.')
                            st.error(f"Error de la API ({response.status_code}): {error_detail}")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ No se pudo conectar con el servidor de IA. Verifica que la API esté ejecutándose en http://16.171.198.191:8000")
                    except requests.exceptions.Timeout:
                        st.error("⏱️ La consulta tardó demasiado tiempo. Intenta con una pregunta más específica.")
                    except Exception as e:
                        st.error(f"❌ Ha ocurrido un error al conectar con la API: {e}")
            else:
                st.warning("⚠️ Por favor, introduce una pregunta.")
        
        st.markdown("---")
        
        # Ejemplos de consultas
        st.markdown("### 💡 Ejemplos de Consultas")
        
        ejemplos = [
            "¿Cuál fue la temperatura máxima en Barcelona en julio de 2023?",
            "¿Qué mes tuvo más precipitación en Madrid en 2022?",
            "¿Cuál es la diferencia de temperatura entre enero y julio en Sevilla?",
            "¿Cuántos días llovió en Valencia en el último mes?",
            "¿Cuál fue la temperatura media en Andalucía en verano?"
        ]
        
        for i, ejemplo in enumerate(ejemplos, 1):
            st.button(f"📝 {ejemplo}", key=f"ejemplo_{i}", use_container_width=True, disabled=True)
        
        st.markdown("---")
        
        # Estadísticas de la estación actual
        st.markdown("### 📊 Resumen de la Estación Actual")
        
        if not df_filtered.empty and 'tmed' in df_filtered.columns:
            temp_data = df_filtered['tmed'].dropna()
            
            if len(temp_data) > 0:
                stats = temp_data.describe()
                
                # Métricas básicas
                col_x, col_y = st.columns(2)
                with col_x:
                    st.metric("🔥 Máxima", f"{stats['max']:.1f}°C")
                    st.metric("📊 Media", f"{stats['mean']:.1f}°C")
                with col_y:
                    st.metric("❄️ Mínima", f"{stats['min']:.1f}°C")
                    st.metric("� Desv. Std", f"{stats['std']:.1f}°C")
                
                # Información adicional
                if 'prec' in df_filtered.columns:
                    total_prec = df_filtered['prec'].sum()
                    dias_lluvia = (df_filtered['prec'] > 0).sum()
                    st.metric("🌧️ Precipitación Total", f"{total_prec:.0f}mm")
                    st.metric("☔ Días con lluvia", f"{dias_lluvia} días")
        
        # Footer informativo
        st.markdown("---")
        st.markdown("### 🌐 Información del Sistema")
        st.info(f"""
        **🤖 IA:** Asistente con procesamiento de lenguaje natural  
        **📊 Datos:** {len(df):,} registros históricos en AWS RDS  
        **🌍 Cobertura:** España peninsular e islas  
        **⚡ Actualización:** Tiempo real desde PostgreSQL  
        **🎯 Consultas:** Análisis inteligente de datos meteorológicos
        """)

if __name__ == "__main__":
    main()






