import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime
import json
import os

# --- 1. COMMANDER'S HUD (Heads-Up Display) CONFIG ---
st.set_page_config(layout="wide", page_title="GEOSENTINEL C2", page_icon="📡")

# Custom CSS for Elite "Military" Feel
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    
    .main { background-color: #05070a; color: #00ff41; font-family: 'Share Tech Mono', monospace; }
    .stMetric { background-color: #0d1117; border: 1px solid #00ff41; border-radius: 5px; box-shadow: 0 0 10px #00ff4133; }
    .css-10trblm { color: #00ff41 !important; }
    
    /* Global Ticker Animation */
    .ticker-wrapper { width: 100%; overflow: hidden; background: #1a1c23; border-bottom: 2px solid #ff4b4b; padding: 5px 0; }
    .ticker-text { display: inline-block; white-space: nowrap; animation: ticker 30s linear infinite; color: #ff4b4b; font-weight: bold; }
    @keyframes ticker { 0% { transform: translateX(100%); } 100% { transform: translateX(-100%); } }
    
    /* Glassmorphism Cards */
    .intel-card { background: rgba(255, 255, 255, 0.05); border-left: 5px solid #ff4b4b; padding: 15px; margin: 10px 0; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- 2. THE LIVE INTEL TICKER (Top of the Page) ---
if os.path.exists("live_intelligence_log.json"):
    with open("live_intelligence_log.json", 'r') as f:
        ticker_data = json.load(f)
        ticker_str = " | ".join([f"⚠️ {i['zone'].upper()}: {i['sitrep']}" for i in ticker_data[:5]])
        st.markdown(f'<div class="ticker-wrapper"><div class="ticker-text">{ticker_str}</div></div>', unsafe_allow_html=True)

# --- 3. DYNAMIC HEADER & DEFCON STATUS ---
st.title("🛡️ GEOSENTINEL GLOBAL COMMAND & CONTROL")
st.markdown(f"**SYSTEM STATUS:** OPERATIONAL | **ORBITAL TIME:** {datetime.now().strftime('%H:%M:%S')} UTC")

# Logic to calculate DEFCON
# Assuming final_df and current_gpti are already calculated as per previous steps
# (Placeholder logic for the snippet)
current_gpti = 0.82 # Example spike
defcon_level = "1" if current_gpti > 0.8 else "2" if current_gpti > 0.6 else "3"

c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1: st.metric("CURRENT GPTI", f"{current_gpti:.2f}", "+12% escalation")
with c2: st.markdown(f"### DEFCON: <span style='color:#ff4b4b;'>{defcon_level}</span>", unsafe_allow_html=True)
with c3: st.metric("OSINT CONFIDENCE", "91%", "Optimal")
with c4: st.metric("ACTIVE THEATERS", "5", "Global Scan")

st.markdown("---")

# --- 4. THE ELITE "SITUATIONAL AWARENESS" GRID ---
col_map, col_intel = st.columns([2, 1])

with col_map:
    st.subheader("📍 TACTICAL DEPLOYMENT HEATMAP")
    # Using Pydeck for a high-end 3D visual
    # (data_ingestion logic assumed to be present)
    view_state = pdk.ViewState(latitude=34.08, longitude=74.79, zoom=6, pitch=45)
    layer = pdk.Layer(
        "HexagonLayer",
        data=pd.DataFrame(np.random.randn(100, 2) / [20, 20] + [34.08, 74.79], columns=['lat', 'lon']),
        get_position=["lon", "lat"],
        auto_highlight=True,
        elevation_scale=1000,
        pickable=True,
        elevation_range=[0, 3000],
        extruded=True,
        coverage=1
    )
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/satellite-v9"))

with col_intel:
    st.subheader("🧠 STRATEGIC INTELLIGENCE")
    if os.path.exists("live_intelligence_log.json"):
        with open("live_intelligence_log.json", 'r') as f:
            logs = json.load(f)
            for log in logs[:3]: # Only top 3 for elite look
                st.markdown(f"""
                <div class="intel-card">
                    <small style='color:#ff4b4b;'>{log['timestamp']}</small><br>
                    <strong>{log['headline']}</strong><br>
                    <span style='color:#00ff41;'>SitRep: {log['sitrep']}</span>
                </div>
                """, unsafe_allow_html=True)

# --- 5. THE "GRAVITY" CHART (Dual Pillar Analysis) ---
st.markdown("---")
st.subheader("📉 CONFLICT GRAVITY & PREDICTION HORIZON")
fig = go.Figure()

# Plotting the main GPTI vs pillars
# (Using placeholder dates/values for structure)
fig.add_trace(go.Scatter(y=[0.2, 0.4, 0.5, 0.85, 0.82], name="Overall GPTI", line=dict(color='#ff4b4b', width=4), fill='tozeroy'))
fig.add_trace(go.Bar(y=[0.1, 0.2, 0.3, 0.7, 0.6], name="Kinetic Load", marker_color='#4CC9F0', opacity=0.4))
fig.add_trace(go.Bar(y=[0.3, 0.5, 0.6, 0.9, 0.8], name="Narrative Pressure", marker_color='#00FF41', opacity=0.4))

fig.update_layout(
    template="plotly_dark", 
    barmode='group',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, t=20, b=0)
)
st.plotly_chart(fig, use_container_width=True)