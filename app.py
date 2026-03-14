import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk
from plotly.subplots import make_subplots
import index_calculator
import data_ingestion
import config
import json
import os
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import advanced_modules 

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="GeoSentinel Commander", page_icon="🛡️", initial_sidebar_state="expanded")
st_autorefresh(interval=60 * 1000, key="data_refresh")

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #E63946; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    .nav-header { color: #4CC9F0; font-weight: bold; margin-top: 20px;}
    .cyber-text { color: #00FF41; font-family: 'Courier New', monospace; }
</style>
""", unsafe_allow_html=True)

def load_live_intel():
    log_file = "live_intelligence_log.json"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                return json.load(f)
        except: return []
    return []

adv_features = advanced_modules.AdvancedFeatures()
live_intel_data = load_live_intel()

# ==========================================
# 2. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.title("📡 Command Center")
    st.markdown('<p class="nav-header">TARGET ACQUISITION</p>', unsafe_allow_html=True)
    op_mode = st.radio("Mode:", ["Live Intelligence", "Historical Validation"], horizontal=True)
    
    if op_mode == "Live Intelligence":
        selected_dyad = st.selectbox("Active Conflict Zone:", ["India-Pakistan", "Russia-Ukraine", "Israel-Palestine", "Iran-Israel-USA"])
    else:
        selected_benchmark = st.selectbox("Benchmark Scenario:", ["India-Pakistan 2019", "Iran-Israel 2026 (Epic Fury)"])

    st.markdown("---")
    st.markdown('<p class="nav-header">SYSTEM MODULES</p>', unsafe_allow_html=True)
    selected_module = st.radio("Select View:", [
        "🗺️ 1. Multi-Domain Dashboard",
        "🚨 2. Early Warning & Trajectory",
        "📉 3. Economic & Cyber Impact",
        "🧠 4. Info-War & PsyOps"
    ])

    st.markdown("---")
    if st.button("🔄 Force Data Refresh", use_container_width=True): st.rerun()

# ==========================================
# 3. GLOBAL DATA PROCESSING
# ==========================================
if op_mode == "Historical Validation":
    df = data_ingestion.get_validation_data(selected_benchmark)
    current_zone = selected_benchmark
else:
    df = data_ingestion.generate_synthetic_data(selected_dyad)
    current_zone = selected_dyad

calculator = index_calculator.IndexCalculator()
final_df = calculator.process_index(df)

current_gpti = final_df['GPTI'].iloc[-1]
gpti_trend = final_df['GPTI_Trend'].iloc[-1]
latest_intel = live_intel_data[0] if (live_intel_data and op_mode == "Live Intelligence") else None

st.title(f"{selected_module.split('.')[1].strip()} - {current_zone}")
st.markdown("---")

# ------------------------------------------
# MODULE 1: MULTI-DOMAIN DASHBOARD (With Radar Chart)
# ------------------------------------------
if selected_module.startswith("🗺️ 1"):
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("### 📈 Time-Series Tension Index (GPTI)")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['GPTI'], name='GPTI Index', line=dict(color='#EF233C', width=3), fill='tozeroy'), row=1, col=1)
        fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['weight_kinetic'], name='MCT Weight (Kinetic)', line=dict(color='#4CC9F0')), row=2, col=1)
        fig.update_layout(height=500, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### 🕸️ Threat Matrix")
        st.caption("Real-time distribution of conflict domains.")
        matrix_data = adv_features.generate_threat_matrix(current_gpti, current_zone)
        
        categories = list(matrix_data.keys())
        values = list(matrix_data.values())
        # Close the loop for the radar chart
        categories.append(categories[0])
        values.append(values[0])
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values, theta=categories, fill='toself', line_color='#EF233C', fillcolor='rgba(239, 35, 60, 0.4)'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False, template="plotly_dark", height=450, margin=dict(l=40, r=40, t=30, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("### 📍 Kinetic Geolocation Layer")
    map_df, lat, lon, zoom = data_ingestion.generate_location_data(selected_dyad if op_mode == "Live Intelligence" else "India-Pakistan", 500)
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/dark-v10',
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=45),
        layers=[pdk.Layer('HexagonLayer', data=map_df, get_position='[lon, lat]', radius=5000, elevation_scale=100, extruded=True)]
    ))

# ------------------------------------------
# MODULE 2: EARLY WARNING & TRAJECTORY
# ------------------------------------------
elif selected_module.startswith("🚨 2"):
    alerts, status = adv_features.generate_alerts(current_gpti, gpti_trend)
    c1, c2, c3 = st.columns(3)
    c1.metric("Current GPTI Score", f"{current_gpti:.2f}/1.00", f"{gpti_trend:.3f} slope")
    c2.metric("System Trajectory", status)
    
    if latest_intel:
        is_leading = adv_features.analyze_leading_indicators(latest_intel['headline'], current_zone)
        c3.metric("OSINT Pre-War Indicators", "Detected" if is_leading else "Clear", "- Troop/Logistics Move" if is_leading else "")
    else:
        c3.metric("OSINT Pre-War Indicators", "Awaiting Data", "")

    st.markdown("### 🔔 Automated Action Alerts")
    if alerts:
        for alert in alerts: st.error(alert)
    else:
        st.success("✅ System Nominal. No critical escalation thresholds breached.")

# ------------------------------------------
# MODULE 3: ECONOMIC & CYBER IMPACT
# ------------------------------------------
elif selected_module.startswith("📉 3"):
    st.markdown("Simulating secondary conflict damage across Financial and Infrastructure domains.")
    
    # 1. Economic Data
    st.markdown("### 🌐 Macro-Economic Fallout")
    impact = adv_features.get_economic_impact(current_gpti, current_zone)
    c1, c2, c3 = st.columns(3)
    for col, key in zip([c1, c2, c3], impact.keys()):
        data = impact[key]
        with col:
            st.info(f"**{data['name']}**")
            st.metric(label="Current Est. Value", value=f"{data['symbol']}{data['value']:,.2f}", 
                      delta=f"{data['change']:+.2f}%", delta_color="inverse" if "NIFTY" in data['name'] or "KSE" in data['name'] else "normal")
            st.caption(f"**72-Hr Risk Projection:** {data['symbol']}{data['worst_case']:,.2f}") # Monte carlo projection

    st.markdown("---")
    
    # 2. Cyber Threat Matrix
    st.markdown("### 💻 Cyber Warfare & Critical Infrastructure")
    matrix = adv_features.generate_threat_matrix(current_gpti, current_zone)
    cyber_threat = matrix["Cyber & Grid"]
    
    st.progress(cyber_threat / 100)
    st.markdown(f'<p class="cyber-text">CURRENT INFRASTRUCTURE PROBING SEVERITY: {cyber_threat:.1f}%</p>', unsafe_allow_html=True)
    if cyber_threat > 75:
        st.error("CRITICAL: Elevated DDoS activity detected on regional banking and ATC (Air Traffic Control) networks.")
    elif cyber_threat > 50:
        st.warning("ELEVATED: Unverified state-sponsored phishing campaigns targeting civilian grid operators.")

# ------------------------------------------
# MODULE 4: INFO-WAR & PSYOPS
# ------------------------------------------
elif selected_module.startswith("🧠 4"):
    panic_idx, top_searches = adv_features.get_public_panic_index(current_gpti)
    st.markdown("### 🕵️‍♂️ Narrative Integrity & Ground Pulse (PsyOps Detection)")
    
    colA, colB = st.columns(2)
    with colA:
        st.metric("Civilian Panic Index (Google Trends)", f"{panic_idx:.1f} / 100", f"{'Critical' if panic_idx > 75 else 'Elevated' if panic_idx > 50 else 'Normal'}")
        st.caption(f"**Top Search Spikes:** '{top_searches[0]}', '{top_searches[1]}'")
        
    with colB:
        if latest_intel:
            bot_check = adv_features.analyze_information_integrity(latest_intel['sitrep'])
            integrity_score = bot_check['integrity_score']
            st.metric("OSINT Bot & Deepfake Check", f"{integrity_score}/100", bot_check['narrative_status'], delta_color="normal" if integrity_score > 70 else "inverse")
            if bot_check['flags']: st.warning(f"🚩 **Flags:** {', '.join(bot_check['flags'])}")
        else:
            st.metric("OSINT Bot Check", "Awaiting Data", "")

    if latest_intel:
        st.markdown("---")
        st.markdown("### 📡 Raw Intelligence Feed (LLM Processed)")
        st.code(json.dumps(latest_intel, indent=4), language="json")