import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk
from plotly.subplots import make_subplots
import index_calculator
import data_ingestion
import advanced_modules 
import os
from datetime import datetime

# ==========================================
# 1. CYBER-PUNK MILITARY UI CONFIG
# ==========================================
st.set_page_config(layout="wide", page_title="GeoSentinel Command", page_icon="🛡️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    .main { background-color: #0b0d11; color: #e0e0e0; }
    h1, h2, h3 { font-family: 'Orbitron', sans-serif; color: #ff4b4b !important; text-transform: uppercase; letter-spacing: 2px; }
    .stMetric { background-color: #1a1c23; border: 1px solid #3e4451; padding: 15px; border-radius: 10px; }
    .sidebar .sidebar-content { background-image: linear-gradient(#1a1c23, #0b0d11); }
    .css-10trblm { color: #ff4b4b !important; } /* Radio button color */
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ACQUISITION & PROCESSING
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/nolan/512/radar.png", width=100)
    st.title("🛡️ GEOSENTINEL")
    st.markdown("---")
    
    # [FIX] Matching all 5 scenarios from generate_complete_benchmarks.py
    scenarios = [
        "India-Pakistan 2019", 
        "Russia-Ukraine 2022", 
        "Israel-Palestine 2023", 
        "Sudan Conflict 2023", 
        "Iran-Israel-US 2026"
    ]
    
    selected_zone = st.selectbox("🎯 TARGET ACQUISITION:", scenarios)
    
    st.markdown("### 🖥️ SYSTEM VIEW")
    selected_module = st.selectbox("SELECT MODULE:", [
        "Strategic Overview",
        "Lead-Lag Intelligence",
        "Economic & Cyber Fallout",
        "War Room Map"
    ])
    
    st.markdown("---")
    st.warning("⚠️ SYSTEM STATUS: DEFCON 3 (Elevated)")

# Data Loading
df_raw = data_ingestion.get_validation_data(selected_zone)
calc = index_calculator.IndexCalculator()
final_df = calc.process_index(df_raw)

# Global Metrics
current_gpti = final_df['GPTI'].iloc[-1]
trend = final_df['GPTI_Trend'].iloc[-1]
adv_features = advanced_modules.AdvancedFeatures()

# ==========================================
# 3. DYNAMIC MODULES
# ==========================================

# --- MODULE 1: STRATEGIC OVERVIEW ---
if selected_module == "Strategic Overview":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GPTI INDEX", f"{current_gpti:.2f}", f"{trend:+.3f}", delta_color="inverse")
    col2.metric("KINETIC WEIGHT", f"{final_df['weight_kinetic'].iloc[-1]:.2f}")
    col3.metric("NARRATIVE WEIGHT", f"{final_df['weight_narrative'].iloc[-1]:.2f}")
    col4.metric("ALERT LEVEL", "CRITICAL" if current_gpti > 0.75 else "ELEVATED" if current_gpti > 0.4 else "NOMINAL")

    st.markdown("### 📉 MULTI-PILLAR TENSION TRAJECTORY")
    fig = go.Figure()
    # GPTI Line
    fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['GPTI'], name='GPTI (Aggregated)', 
                             line=dict(color='#ff4b4b', width=4), fill='tozeroy'))
    # MCT & INT Areas
    fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['MCT_norm'], name='MCT (Military)', 
                             line=dict(color='#4CC9F0', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['INT_norm'], name='INT (Narrative)', 
                             line=dict(color='#00FF41', width=1, dash='dot')))
    
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

# --- MODULE 2: LEAD-LAG INTELLIGENCE (NEW! BEST FOR VIVA) ---
elif selected_module == "Lead-Lag Intelligence":
    st.markdown("### 🧠 INFO-WAR LEAD-LAG ANALYSIS")
    st.info("This module visualizes how Narrative Tension (INT) predicts Kinetic Escalation (MCT).")
    
    # 7-day Rolling correlation
    final_df['correlation'] = final_df['INT_norm'].rolling(window=7).corr(final_df['MCT_norm'])
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(x=final_df['date'], y=final_df['correlation'], name='INT-MCT Correlation',
                                      line=dict(color='#FFD700', width=2), fill='tozeroy'))
        fig_corr.update_layout(template="plotly_dark", title="Dynamic Coupling Score", height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col_b:
        st.markdown("#### 🚩 Strategic Briefing")
        high_corr_dates = final_df[final_df['correlation'] > 0.8]['event'].unique()
        if len(high_corr_dates) > 1:
            st.write(f"System detected **Strong Coupling** during: *{', '.join([e for e in high_corr_dates if e != 'Routine Monitoring'][:3])}*")
        else:
            st.write("No strong lead-lag coupling detected in the current window.")

# --- MODULE 3: ECONOMIC & CYBER FALLOUT ---
elif selected_module == "Economic & Cyber Fallout":
    st.markdown("### 💸 SECONDARY DOMAIN IMPACT")
    impact = adv_features.get_economic_impact(current_gpti, selected_zone)
    
    c1, c2, c3 = st.columns(3)
    for col, (key, data) in zip([c1, c2, c3], impact.items()):
        with col:
            st.metric(data['name'], f"{data['symbol']}{data['value']:,.2f}", f"{data['change']:+.2f}%", delta_color="inverse")
            st.progress(abs(data['change']) / 10) # Visualizing risk
            
    st.markdown("---")
    st.markdown("### ⚡ CYBER & INFRASTRUCTURE VULNERABILITY")
    threat_matrix = adv_features.generate_threat_matrix(current_gpti, selected_zone)
    cyber_val = threat_matrix["Cyber & Grid"]
    st.slider("System Probe Intensity", 0, 100, int(cyber_val), disabled=True)
    st.code(f"STATUS: {'CRITICAL OVERLOAD' if cyber_val > 80 else 'UNUSUAL ACTIVITY'} detected in regional data-centers.")

# --- MODULE 4: WAR ROOM MAP ---
elif selected_module == "War Room Map":
    st.markdown("### 📍 TACTICAL GEOLOCATION (HEATMAP)")
    map_df, lat, lon, zoom = data_ingestion.generate_location_data(selected_zone, 300)
    
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/satellite-v9',
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=50),
        layers=[
            pdk.Layer('HeatmapLayer', data=map_df, get_position='[lon, lat]', get_weight='intensity', radius_pixels=60),
            pdk.Layer('ScatterplotLayer', data=map_df, get_position='[lon, lat]', get_color='[255, 75, 75, 160]', get_radius=2000)
        ]
    ))

# ==========================================
# 4. FOOTER / LIVE FEED
# ==========================================
st.markdown("---")
st.caption(f"🛡️ GeoSentinel Framework | Operator: Mohd Arshad | System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")