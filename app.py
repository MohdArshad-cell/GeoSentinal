import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk
from plotly.subplots import make_subplots
import index_calculator
import data_ingestion
import analysis_engine
import config
import json
import os
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 1. PAGE CONFIGURATION & REFRESH
# ==========================================
st.set_page_config(
    layout="wide", 
    page_title="GeoSentinel Commander",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 60 seconds to pull new live intel
st_autorefresh(interval=60 * 1000, key="data_refresh")

# Custom CSS for "Military-Grade" Aesthetic
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #E63946; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    .stAlert { background-color: #1E1E1E; border: 1px solid #333; color: white; }
    .metric-card { background-color: #1A1A1A; border: 1px solid #333; padding: 15px; border-radius: 8px; }
    .console-text { font-family: 'Courier New', monospace; color: #4CC9F0; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING UTILITIES
# ==========================================
def load_live_intel():
    log_file = "live_intelligence_log.json"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

# ==========================================
# 3. SIDEBAR CONTROL PANEL
# ==========================================
with st.sidebar:
    st.title("📡 Control Panel")
    
    # NEW: Operational Mode Toggle for Validation
    st.subheader("🕹️ Operational Mode")
    op_mode = st.radio("Select Mode:", ["Live Intelligence", "Historical Validation"])
    
    st.markdown("---")
    
    if op_mode == "Live Intelligence":
        st.subheader("📍 Target Zone")
        selected_dyad = st.selectbox(
            "Monitor Active Conflict:",
            ["India-Pakistan", "Russia-Ukraine", "Israel-Palestine", "Iran-Israel-USA"],
            index=0
        )
    else:
        st.subheader("⌛ Validation Scenario")
        selected_benchmark = st.selectbox(
            "Select Benchmark Event:",
            ["India-Pakistan 2019", "Iran-Israel 2026 (Epic Fury)"]
        )

    st.markdown("---")
    if st.button("🔄 Force Data Refresh"):
        st.rerun()
    
    st.caption(f"System Time: {datetime.now().strftime('%H:%M:%S')}")

# ==========================================
# 4. MAIN DASHBOARD LOGIC
# ==========================================
live_intel_data = load_live_intel()

if op_mode == "Historical Validation":
    st.title(f"🛡️ Benchmark Analysis: {selected_benchmark}")
    df = data_ingestion.get_validation_data(selected_benchmark)
    # Process through our Two-Pillar Calculator
    calculator = index_calculator.IndexCalculator()
    final_df = calculator.process_index(df)
else:
    st.title(f"🛡️ GeoSentinel Live: {selected_dyad}")
    # Load Real-time News Analytics
    if live_intel_data:
        latest = live_intel_data[0]
        with st.expander("🚨 COMMANDER'S CONSOLE (LIVE AI INTEL)", expanded=True):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("📝 LATEST SITREP")
                st.markdown(f"**SOURCE:** {latest['source']} | **TIME:** {latest['timestamp']}")
                st.info(f"**BRIEF:** {latest['sitrep']}")
                st.markdown("### 🎯 STRATEGIC OPTIONS")
                for i, opt in enumerate(latest['options'], 1):
                    st.write(f"**{i}.** {opt}")
            with c2:
                risk_val = latest['risk_score'] * 100
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=risk_val,
                    title={'text': "HOSTILITY INDEX"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#EF233C" if risk_val > 70 else "#F4A261"},
                           'steps': [{'range': [0, 40], 'color': "rgba(0, 255, 0, 0.2)"},
                                     {'range': [70, 100], 'color': "rgba(255, 0, 0, 0.2)"}]}))
                st.plotly_chart(fig_gauge, use_container_width=True)

    # Load Trend Data
    df = data_ingestion.generate_synthetic_data(selected_dyad)
    calculator = index_calculator.IndexCalculator()
    final_df = calculator.process_index(df)

# ==========================================
# 5. VISUALIZATION LAYOUT
# ==========================================
st.markdown("---")
tab1, tab2 = st.tabs(["📈 Tension Timeline", "🗺️ War Room Map"])

with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    # Pillar 1 & 2 Combined (GPTI)
    fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['GPTI'], name='GPTI Index', 
                             line=dict(color='#EF233C', width=3), fill='tozeroy'), row=1, col=1)
    
    # Dynamic Weights Display
    fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['weight_kinetic'], 
                             name='MCT Weight (Kinetic)', line=dict(color='#4CC9F0')), row=2, col=1)
    
    fig.update_layout(height=600, template="plotly_dark", 
                      title_text="Two-Pillar Conflict Evolution Index",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📍 Geolocation of Kinetic Events")
    map_df, lat, lon, zoom = data_ingestion.generate_location_data(
        selected_dyad if op_mode == "Live Intelligence" else "India-Pakistan", 500)
    
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/dark-v10',
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=45),
        layers=[pdk.Layer('HexagonLayer', data=map_df, get_position='[lon, lat]', 
                          radius=5000, elevation_scale=100, extruded=True)]
    ))