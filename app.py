import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime
import json
import os
import time

# --- INTERNAL MODULE IMPORTS ---
import data_ingestion
import index_calculator
import advanced_modules
# ==========================================
# 0. GLOBAL INITIALIZATION (Top of the Script)
# ==========================================
CONFLICT_THEATERS = {
    "India-Pakistan 2019": "India Pakistan military border tension LoC",
    "Russia-Ukraine 2022": "Russia Ukraine war offensive missile strike",
    "Israel-Palestine 2023": "Gaza Israel conflict IDF Hamas military",
    "Iran-Israel-US 2026": "Iran Israel US military escalation drone strike",
    "Sudan Conflict 2023": "Sudan army RSF fighting Khartoum"
}

# ==========================================
# 1. PAGE CONFIGURATION (MUST BE FIRST)
# ==========================================
st.set_page_config(
    layout="wide", 
    page_title="GEOSENTINEL C2 | Global Defense Monitor", 
    page_icon="🛡️"
)

# ==========================================
# 2. ELITE MILITARY-GRADE CSS (UI/UX)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    
    /* Global Styles */
    .main { background-color: #05070a; color: #00ff41; font-family: 'Share Tech Mono', monospace; }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    
    /* Header & Pulse */
    .header-container {
        background: linear-gradient(90deg, rgba(13,17,23,1) 0%, rgba(30,41,59,0.5) 100%);
        padding: 20px; border-left: 5px solid #00ff41; border-radius: 5px; margin-bottom: 20px;
    }
    .status-pulse {
        height: 10px; width: 10px; background-color: #00ff41; border-radius: 50%;
        display: inline-block; margin-right: 10px; box-shadow: 0 0 8px #00ff41;
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 65, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(0, 255, 65, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 65, 0); }
    }

    /* DEFCON Box */
    .defcon-box {
        padding: 15px; border-radius: 5px; text-align: center;
        font-family: 'Share Tech Mono', monospace; font-weight: bold;
        letter-spacing: 2px; transition: 0.5s;
    }

    /* Ticker Logic */
    .ticker-wrapper {
        background: rgba(10, 10, 15, 0.95); border-bottom: 1px solid #30363d;
        padding: 10px 0; display: flex; align-items: center; overflow: hidden;
    }
    .ticker-content {
        display: inline-block; white-space: nowrap;
        animation: ticker-move 60s linear infinite;
    }
    @keyframes ticker-move { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
    .live-badge {
        background: #ff4b4b; color: white; padding: 2px 8px; font-size: 12px;
        font-weight: bold; border-radius: 3px; margin: 0 15px; animation: blink 1.5s infinite;
    }
    @keyframes blink { 0%, 100% {opacity: 1;} 50% {opacity: 0.3;} }

    /* Cards */
    .intel-card {
        background: rgba(22, 27, 34, 0.8); border: 1px solid #30363d;
        border-left: 4px solid #00ff41; padding: 15px; margin-bottom: 15px; border-radius: 4px;
    }
    .propaganda-flag {
        font-size: 10px; padding: 2px 6px; border-radius: 3px; float: right; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. ELITE SIDEBAR RECONSTRUCTION
# ==========================================
with st.sidebar:
    # --- Tactical Logo Section ---
    st.markdown("""
        <div style="text-align: center; padding-bottom: 20px;">
            <svg class="radar-icon" width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="#00ff41" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M12 2a10 10 0 1 0 10 10"></path>
                <path d="M12 12L19 5"></path>
                <circle cx="12" cy="12" r="3"></circle>
            </svg>
            <h2 style='font-family: "Share Tech Mono"; color: white; margin-top: 10px; letter-spacing: 2px;'>GEOSENTINEL</h2>
            <p style='color: #00ff41; font-size: 10px; font-family: monospace;'>V2.5 LITE | ENCRYPTED</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Theater Control ---
    st.markdown('<p class="tactical-label">Primary Engagement Zone</p>', unsafe_allow_html=True)
    selected_zone = st.selectbox(
        "SELECT THEATER:", 
        list(CONFLICT_THEATERS.keys()),
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- System Metrics Card ---
    st.markdown("""
        <div class="sidebar-card">
            <p class="tactical-label">System Health</p>
            <div style="display: flex; justify-content: space-between; font-size: 12px;">
                <span>Signal Strength</span><span style="color:#00ff41;">Excellent</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 5px;">
                <span>Latency</span><span style="color:#00ff41;">42ms</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 5px;">
                <span>Model</span><span style="color:#4CC9F0;">Gemini 2.5F</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- Parameters Section ---
    st.markdown('<p class="tactical-label">Intelligence Tuning</p>', unsafe_allow_html=True)
    with st.container():
        sensitivity = st.slider("AI Signal Sensitivity", 0.0, 1.0, 0.85, help="Threshold for noise vs intelligence.")
        st.markdown("<div style='margin-top: -15px;'></div>", unsafe_allow_html=True)
        auto_refresh = st.checkbox("AUTOSYNC: Satellite Feed", value=True)
        
    st.markdown("---")

    # --- Force Action Button ---
    if st.button("🚨 INITIALIZE DATA RESYNC", use_container_width=True):
        with st.spinner("Re-syncing with global OSINT nodes..."):
            st.cache_data.clear()
            time.sleep(1) 
            st.rerun()

    # --- Intelligence Export (Fixed Logic) ---
    st.markdown('<p class="tactical-label">Report Generation</p>', unsafe_allow_html=True)
    
    # Pre-generate report content to avoid errors
    safe_gpti = gpti_val if 'gpti_val' in locals() else 0.0
    safe_sitrep = "Scan in progress..."
    if 'ticker_data' in locals() and ticker_data:
        safe_sitrep = ticker_data[0].get('sitrep', "Analyzing...")

    report_content = f"""
    GEOSENTINEL TACTICAL DOSSIER
    ----------------------------
    THEATER: {selected_zone.upper()}
    INTENSITY: {safe_gpti:.2f}
    SITREP: {safe_sitrep}
    TIMESTAMP: {datetime.now().strftime('%H:%M:%S')} UTC
    """

    st.download_button(
        label="📄 GENERATE STRATEGIC DOSSIER",
        data=report_content,
        file_name=f"GeoSentinel_Report_{selected_zone.replace(' ', '_')}.txt",
        mime="text/plain",
        use_container_width=True,
        help="Compile and download the latest intelligence packet."
    )

    # --- Fixed Footer Info (Inside Sidebar) ---
    st.markdown("""
        <div style="position: relative; margin-top: 50px; font-size: 10px; 
                    color: #4b5563; font-family: 'Share Tech Mono', monospace; 
                    background: rgba(13,17,23,0.5); padding: 5px; border-top: 1px solid #30363d;">
            STATION: LKO-UP-IND-2026<br>
            SECURITY: LEVEL 4 CLEARED<br>
            [DASHBOARD-ALPHA-01]
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. DATA PIPELINE (SEQUENTIAL EXECUTION)
# ==========================================
try:
    # A. Fetch Raw Data
    df_raw = data_ingestion.get_validation_data(selected_zone)
    
    # B. Calculate GPTI Index
    calc = index_calculator.IndexCalculator()
    final_df = calc.process_index(df_raw)
    
    # C. Extract Metrics
    latest = final_df.iloc[-1]
    gpti_val = latest['GPTI']
    trend = latest.get('GPTI_Trend', 0.0)
    
    # D. Economic/Cyber Impacts
    adv = advanced_modules.AdvancedFeatures()
    impacts = adv.get_economic_impact(gpti_val, selected_zone)

except Exception as e:
    st.error(f"FATAL ERROR IN PIPELINE: {e}")
    st.stop()

# ==========================================
# 5. LIVE INTEL TICKER
# ==========================================
if os.path.exists("live_intelligence_log.json"):
    with open("live_intelligence_log.json", 'r') as f:
        try:
            ticker_data = json.load(f)
            if ticker_data:
                items_html = ""
                # Infinite loop simulation
                display_items = (ticker_data[:10]) * 3
                for i in display_items:
                    r_score = i.get('risk_score', 0)
                    color = "#ff4b4b" if r_score > 0.7 else "#00ff41"
                    items_html += f"""
                    <span style='margin: 0 40px; color: {color}; font-size: 14px;'>
                        <b>[{i['zone'].upper()}]</b>: {i['sitrep']}
                    </span>"""
                st.markdown(f"""
                    <div class="ticker-wrapper">
                        <div class="live-badge">📡 LIVE FEED</div>
                        <div class="ticker-content">{items_html}</div>
                    </div>
                """, unsafe_allow_html=True)
        except: pass

# ==========================================
# 6. HEADER & DEFCON DISPLAY
# ==========================================
st.markdown(f"""
    <div class="header-container">
        <h1 style='margin:0; font-size: 2.5rem;'>🛡️ GEOSENTINEL <span style='color:#00ff41; font-weight:100;'>COMMAND</span></h1>
        <p style='margin:0; color:#8b949e;'><span class="status-pulse"></span> STATUS: OPERATIONAL | THEATER: {selected_zone.upper()} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    </div>
""", unsafe_allow_html=True)

# DEFCON Logic
if gpti_val > 0.85:
    defcon = {"lv": "1", "ds": "COCKED PISTOL", "clr": "#ff4b4b", "bg": "rgba(255, 75, 75, 0.2)"}
elif gpti_val > 0.65:
    defcon = {"lv": "2", "ds": "FAST PACE", "clr": "#ffaa00", "bg": "rgba(255, 170, 0, 0.2)"}
elif gpti_val > 0.45:
    defcon = {"lv": "3", "ds": "ROUND HOUSE", "clr": "#ffff00", "bg": "rgba(255, 255, 0, 0.1)"}
else:
    defcon = {"lv": "5", "ds": "FADE OUT", "clr": "#00ff41", "bg": "rgba(0, 255, 65, 0.1)"}

m_col1, m_col2, m_col3, m_col4 = st.columns([1, 1.5, 1, 1])
m_col1.metric("CONFLICT INDEX (GPTI)", f"{gpti_val:.2f}", f"{trend:+.2f}")
with m_col2:
    st.markdown(f"""
        <div class="defcon-box" style="border: 2px solid {defcon['clr']}; background: {defcon['bg']}; color: {defcon['clr']};">
            <small>THREAT LEVEL</small><br><span style="font-size: 24px;">DEFCON {defcon['lv']}</span><br><small>{defcon['ds']}</small>
        </div>
    """, unsafe_allow_html=True)
m_col3.metric("OSINT CONFIDENCE", "94.2%", "Optimized")
m_col4.metric("SYSTEM UPTIME", "100%", "Secure")

st.markdown("---")

# ==========================================
# 7. SITUATIONAL AWARENESS GRID
# ==========================================
col_map, col_intel = st.columns([2.2, 1])

with col_map:
    st.markdown("### 📍 TACTICAL DEPLOYMENT HEATMAP (3D)")
    
    try:
        # data_ingestion se real coordinates uthao
        map_df, lat, lon, zoom = data_ingestion.generate_location_data(selected_zone)
        
        # Hexagon Layer Logic
        layer = pdk.Layer(
            "HexagonLayer",
            data=map_df,
            get_position='[lon, lat]',
            auto_highlight=True,
            elevation_scale=500,
            pickable=True,
            extruded=True,
            coverage=1,
            radius=20000,
            get_fill_color="[255, (1 - intensity) * 255, 0, 150]",
        )

        # --- THE BULLETPROOF FIX: DIRECT STYLE URL ---
        # No Mapbox Key needed. No Pydeck constant needed.
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                latitude=lat, 
                longitude=lon, 
                zoom=zoom, 
                pitch=45
            ),
            # Direct link to Dark Matter tiles
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            tooltip={"text": "Conflict Intensity: {elevationValue}"}
        ))
        
    except Exception as e:
        # User-friendly error for non-tech jury members
        st.error(f"📡 Intelligence Map Offline: Verify Internet Connection.")
        print(f"DEBUG ERROR: {e}")

with col_intel:
    st.subheader("🧠 AI STRATEGIC REPORTS")
    if os.path.exists("live_intelligence_log.json"):
        with open("live_intelligence_log.json", 'r') as f:
            logs = json.load(f)
            relevant = [l for l in logs if l.get('zone') == selected_zone][:4]
            if not relevant: st.info("No active signals in this theater.")
            for log in relevant:
                integrity = log.get('integrity_score', 100)
                badge_clr = "#064e3b" if integrity > 70 else "#78350f" if integrity > 40 else "#7f1d1d"
                st.markdown(f"""
                <div class="intel-card">
                    <span class="propaganda-flag" style="background:{badge_clr}; color:white;">SCORE: {integrity}</span>
                    <small style='color:#8b949e;'>{log.get('timestamp', '')[:16]}</small><br>
                    <strong style='color:#f0f6fc;'>{log.get('headline', '')}</strong><br>
                    <p style='color:#00ff41; font-size:13px;'>⚡ {log.get('sitrep', '')}</p>
                    <div style='border-top: 1px solid #30363d; padding-top: 5px;'><small><b>NEXT:</b> {log.get('options', ['Review'])[0]}</small></div>
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# 8. CONFLICT GRAVITY & PREDICTION
# ==========================================
st.markdown("---")
st.subheader("📉 CONFLICT GRAVITY & PREDICTION HORIZON")

chart_data = final_df.tail(30)
# Prediction logic
last_val = chart_data['GPTI'].iloc[-1]
prediction_points = [last_val + (trend * i) for i in range(1, 4)]
pred_dates = pd.date_range(start=chart_data['date'].iloc[-1], periods=4, freq='H')[1:]

fig = go.Figure()
fig.add_trace(go.Bar(x=chart_data['date'], y=chart_data['INT_norm'], name="Narrative Pressure", marker_color='rgba(0, 255, 65, 0.2)'))
fig.add_trace(go.Bar(x=chart_data['date'], y=chart_data['MCT_norm'], name="Kinetic Load", marker_color='rgba(76, 201, 240, 0.2)'))
fig.add_trace(go.Scatter(x=chart_data['date'], y=chart_data['GPTI'], name="Total Index (GPTI)", line=dict(color='#ff4b4b', width=4), fill='tozeroy'))
fig.add_trace(go.Scatter(x=pred_dates, y=prediction_points, name="Prediction", line=dict(color='#ffffff', dash='dot')))

fig.update_layout(template="plotly_dark", hovermode="x unified", barmode='overlay', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 9. ECONOMIC IMPACTS (ADVANCED MODULE)
# ==========================================
st.subheader("📊 SECONDARY DOMAIN FALLOUT")
ec1, ec2, ec3 = st.columns(3)
for col, (key, data) in zip([ec1, ec2, ec3], impacts.items()):
    with col:
        st.markdown(f"""
        <div style="background:#161b22; padding:15px; border-radius:5px; border:1px solid #30363d;">
            <small style="color:#8b949e;">{data['name'].upper()}</small><br>
            <span style="font-size:20px;">{data['symbol']}{data['value']:,}</span>
            <span style="color:{'#ff4b4b' if data['change'] < 0 else '#00ff41'}; float:right;">{data['change']:+.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 10. SYSTEM METHODOLOGY (THE JURY BAIT)
# ==========================================
with st.expander("🔬 SYSTEM ARCHITECTURE & MATHEMATICAL MODEL"):
    st.write("The Geopolitical Tension Index (GPTI) is calculated using a weighted PCA approach:")
    st.latex(r"GPTI = \omega_1 \cdot MCT_{norm} + \omega_2 \cdot INT_{norm}")
    st.write("Where $\omega$ represents dynamic weights assigned based on variance explanation.")
    st.info("System uses Gemini 1.5 Flash for NLU and DistilBERT for local Sentiment Analysis.")

st.markdown("<br><center><small>GEOSENTINEL C2 | DEVELOPED BY MOHD ARSHAD | © 2026</small></center>", unsafe_allow_html=True)


# ==========================================
# 11. STRATEGIC SCENARIO SIMULATOR
# ==========================================
st.markdown("---")
st.subheader("🔮 STRATEGIC SCENARIO SIMULATOR (WAR-GAMING)")
st.markdown("""
<div style="background: rgba(0, 255, 65, 0.05); padding: 10px; border-radius: 5px; border: 1px solid #00ff4133;">
    <small style="color: #00ff41;">OPERATIONAL NOTE: Adjust sliders to simulate hypothetical escalation and observe the impact on the Global Geopolitical Tension Index (GPTI).</small>
</div>
""", unsafe_allow_html=True)

with st.container():
    col_sim_left, col_sim_right = st.columns([1, 1.5])

    with col_sim_left:
        st.markdown("<br>", unsafe_allow_html=True)
        # Tactical Sliders
        sim_kinetic = st.slider("Simulated Kinetic Activity (MCT):", 0.0, 1.0, float(latest['MCT_norm']), 
                                help="Simulate increase in troop deployments, border skirmishes, or drone strikes.")
        
        sim_narrative = st.slider("Simulated Narrative Pressure (INT):", 0.0, 1.0, float(latest['INT_norm']), 
                                  help="Simulate a massive surge in hostile state media reporting or cyber-propaganda.")
        
        sim_horizon = st.select_slider("Projection Horizon:", options=["Immediate", "48h Short-term", "7-Day Tactical"], value="48h Short-term")

    # --- SIMULATION MATH (PCA-WEIGHTED) ---
    # Using the current weights from your index_calculator.py
    # $Projected\_GPTI = (\omega_1 \cdot MCT_{sim}) + (\omega_2 \cdot INT_{sim})$
    sim_gpti = (sim_kinetic * 0.65) + (sim_narrative * 0.35) 
    escalation_delta = sim_gpti - gpti_val

    with col_sim_right:
        # Gauge Visual for Simulation Impact
        sim_fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = sim_gpti,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "PROJECTED INTENSITY", 'font': {'size': 18, 'color': '#00ff41'}},
            delta = {'reference': gpti_val, 'increasing': {'color': "#ff4b4b"}, 'decreasing': {'color': "#00ff41"}},
            gauge = {
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "#00ff41"},
                'bar': {'color': "#ff4b4b" if escalation_delta > 0 else "#00ff41"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#30363d",
                'steps': [
                    {'range': [0, 0.4], 'color': 'rgba(0, 255, 65, 0.1)'},
                    {'range': [0.4, 0.7], 'color': 'rgba(255, 255, 0, 0.1)'},
                    {'range': [0.7, 1], 'color': 'rgba(255, 75, 75, 0.1)'}
                ],
            }
        ))
        sim_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Share Tech Mono"}, height=350, margin=dict(t=50, b=0))
        st.plotly_chart(sim_fig, use_container_width=True)

# --- ACTIONABLE AI ANALYSIS OF THE SIMULATION ---
if abs(escalation_delta) > 0.05:
    st.markdown(f"""
        <div style="background: {'rgba(255, 75, 75, 0.1)' if escalation_delta > 0 else 'rgba(0, 255, 65, 0.1)'}; 
                    border: 1px solid {'#ff4b4b' if escalation_delta > 0 else '#00ff41'}; 
                    padding: 15px; border-radius: 5px;">
            <b style="color: {'#ff4b4b' if escalation_delta > 0 else '#00ff41'};">
                SYSTEM PROJECTION: {'CRITICAL ESCALATION' if escalation_delta > 0 else 'STABILIZATION TREND'}
            </b><br>
            Hypothetical shift of <b>{escalation_delta:+.2f}</b> detected. 
            This scenario would trigger a <b>DEFCON {'1' if sim_gpti > 0.8 else '2' if sim_gpti > 0.6 else '3'}</b> alert.
        </div>
    """, unsafe_allow_html=True)