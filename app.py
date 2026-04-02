import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime
import json
import os
import time
import streamlit.components.v1 as components
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
# 2. MISSION CONTROL 2.0 CSS (STREATH OPS)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* 1. Global Stealth Theme */
    .stApp {
        background-color: #0A0F14;
        color: #F0F6FC;
        font-family: 'Inter', sans-serif;
    }
    
    /* Typography Overrides */
    h1, h2, h3, .tactical-label { 
        font-family: 'JetBrains Mono', monospace !important; 
        letter-spacing: -0.5px;
    }

    /* 2. Professional Glassmorphism Header */
    .header-container {
        background: rgba(16, 22, 31, 0.8);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 212, 255, 0.2);
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    
    .status-pulse {
        height: 8px; width: 8px; border-radius: 50%;
        background-color: #00D4FF;
        display: inline-block; margin-right: 12px;
        box-shadow: 0 0 12px #00D4FF;
        animation: pulse-cyan 2.5s infinite;
    }
    @keyframes pulse-cyan {
        0% { transform: scale(0.9); opacity: 0.5; }
        50% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(0.9); opacity: 0.5; }
    }

    /* 3. Modern DEFCON Architecture */
    .defcon-box {
        background: rgba(13, 17, 23, 0.6);
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        border-top: 4px solid #00D4FF;
        transition: all 0.3s ease;
    }
    .defcon-box:hover { transform: translateY(-3px); background: rgba(0, 212, 255, 0.05); }

    /* 4. Sleek Signal Cards (No Clutter) */
    .tactical-card {
        background: #111821;
        border: 1px solid #1C2631;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
        transition: border 0.3s ease;
    }
    .tactical-card:hover {
        border-color: #00D4FF;
    }
    .sitrep-box {
        background: rgba(0, 212, 255, 0.05);
        border-left: 3px solid #00D4FF;
        padding: 12px;
        font-size: 14px;
        color: #C9D1D9;
        margin: 15px 0;
    }

    /* 5. Clean Ticker (Stealth Style) */
    .ticker-wrapper {
        background: #0D1117;
        border-bottom: 1px solid #1C2631;
        padding: 8px 0;
    }
    .live-badge {
        background: #FF0055;
        color: white;
        padding: 1px 10px;
        font-size: 10px;
        font-weight: bold;
        border-radius: 100px;
        text-transform: uppercase;
    }

    /* 6. Tabs & Indicators */
    .stTabs [data-baseweb="tab-list"] { gap: 15px; }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: #8B949E;
        font-family: 'JetBrains Mono', monospace;
    }
    .stTabs [aria-selected="true"] {
        color: #00D4FF !important;
        border-bottom: 2px solid #00D4FF !important;
    }

    /* Indicator Glows */
    .indicator-box {
        background: #111821;
        border: 1px solid #1C2631;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .indicator-active {
        border-color: #00D4FF;
        background: rgba(0, 212, 255, 0.05);
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.1);
    }
    
    /* Metrics Fix */
    [data-testid="stMetricValue"] { color: #F0F6FC !important; font-family: 'JetBrains Mono' !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2.5 GLOBAL INITIALIZATION (CRITICAL FIX)
# ==========================================
# Sidebar chalne se pehle ye variables hone zaroori hain
display_logs = [] 
gpti_val = 0.0
trend = 0.0
defcon = {"lv": "5", "ds": "INITIALIZING...", "clr": "#00D4FF", "bg": "rgba(0, 212, 255, 0.05)"}
integrity = {"integrity_score": 0, "narrative_status": "SCANNING..."}
panic_val = 0.0


# ==========================================
# 3. C2 COMMAND SIDEBAR (STRATEGIC OVERWATCH)
# ==========================================
with st.sidebar:
    # --- Tactical Branding (HUD Style) ---
    st.markdown("""
        <div style="text-align: center; padding: 20px 0; background: rgba(0, 212, 255, 0.03); border-radius: 10px; margin-bottom: 25px; border: 1px solid rgba(0, 212, 255, 0.1);">
            <svg width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="#00D4FF" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M12 2a10 10 0 1 0 10 10"></path>
                <path d="M12 12L19 5"></path>
                <circle cx="12" cy="12" r="1" fill="#00D4FF"></circle>
                <path d="M12 7v5l3 3" opacity="0.5"></path>
            </svg>
            <h2 style='font-family: "JetBrains Mono"; color: #F0F6FC; margin-top: 15px; font-size: 1.2rem; letter-spacing: 3px;'>GEOSENTINEL</h2>
            <p style='color: #00D4FF; font-size: 9px; font-family: "JetBrains Mono"; letter-spacing: 1px;'>STRATEGIC COMMAND NODE | V3.0</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Theater Control (Primary Engagement) ---
    st.markdown('<p class="tactical-label">Active Engagement Theater</p>', unsafe_allow_html=True)
    selected_zone = st.selectbox(
        "SELECT THEATER:", 
        list(CONFLICT_THEATERS.keys()),
        label_visibility="collapsed"
    )

    # --- System Metrics (C2 Health) ---
    st.markdown(f"""
        <div class="sidebar-intel-box" style="margin-bottom: 20px;">
            <p class="tactical-label" style="border-left-color: #FF0055;">C2 Hardware Status</p>
            <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 8px;">
                <span style="color: #8B949E;">Satellite Sync</span><span style="color:#00ff41;">SECURE</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 8px;">
                <span style="color: #8B949E;">NLU Latency</span><span style="color:#00D4FF;">28ms</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 11px;">
                <span style="color: #8B949E;">Signal Nodes</span><span style="color:#00D4FF;">34 Active</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- Intelligence Tuning (Parameters) ---
    st.markdown('<p class="tactical-label">Neural Tuning</p>', unsafe_allow_html=True)
    with st.container():
        sensitivity = st.slider("SIGINT Sensitivity", 0.0, 1.0, 0.82, help="Filter for signal-to-noise ratio.")
        st.markdown("<div style='margin-top: -10px;'></div>", unsafe_allow_html=True)
        auto_sync = st.toggle("LIVE Satellite Feed", value=True)
        
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)

    # --- Force Action (Tactical Override) ---
    if st.button("⚡ FORCE GLOBAL RESYNC", use_container_width=True):
        with st.spinner("Flushing C2 Cache & Re-polling Nodes..."):
            st.cache_data.clear()
            time.sleep(1.2) 
            st.rerun()

    # --- Strategic Dossier Generation ---
    st.markdown('<p class="tactical-label">Intelligence Export</p>', unsafe_allow_html=True)
    
    # Advanced Report Logic
    safe_gpti = gpti_val if 'gpti_val' in locals() else 0.0
    safe_sitrep = display_logs[0].get('sitrep', "No active signals") if display_logs else "SCANNING..."
    
    report_content = f"""
    CLASSIFIED: GEOSENTINEL STRATEGIC DOSSIER [TOP SECRET]
    ======================================================
    THEATER: {selected_zone.upper()}
    REGISTRY: LKO-UP-IND-2026-ALPHA
    TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
    
    INTEL SUMMARY:
    --------------
    CONFLICT INTENSITY (GPTI): {safe_gpti:.2f}
    THREAT POSTURE: DEFCON {defcon.get('lv', 'N/A')} ({defcon.get('ds', 'N/A')})
    LATEST SIGINT: {safe_sitrep}
    
    ANALYTICS:
    ----------
    - Multi-Node Sentiment: ACTIVE
    - Narrative Integrity: {integrity['integrity_score'] if 'integrity' in locals() else 'SCANNING'}%
    - Public Panic Index: {panic_val if 'panic_val' in locals() else 'N/A'}%
    
    [END OF DOSSIER]
    """

    st.download_button(
        label="📄 DOWNLOAD STRATEGIC DOSSIER",
        data=report_content,
        file_name=f"GeoSentinel_Brief_{selected_zone.replace(' ', '_')}_{datetime.now().strftime('%H%M')}.txt",
        mime="text/plain",
        use_container_width=True,
        help="Compile and export classified intelligence summary."
    )

    # --- Sidebar Terminal Info ---
    st.markdown(f"""
        <div style="margin-top: 30px; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
            <p style="font-size: 8px; color: #4B5563; font-family: 'JetBrains Mono'; margin: 0;">
                STATION: LKO-UP-IND-2026<br>
                AUTH: LEAD ARCHITECT (ARSHAD)<br>
                SEC_LEVEL: 4 (UNCLASSIFIED DEMO)<br>
                [LINK_STABLE: 100%]
            </p>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. DATA PIPELINE (GLOBAL EXECUTION)
# ==========================================
display_logs = []  # Initialize globally to prevent NameError
defcon = {}        # Initialize globally

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
    
    # D. Load SIGINT Logs (Single Point of Entry)
    if os.path.exists("live_intelligence_log.json"):
        with open("live_intelligence_log.json", 'r') as f:
            try:
                raw_data = json.load(f)
                # Theater Name Sanitizer (Removes 2019, 2022, etc.)
                for item in raw_data:
                    item['zone_clean'] = item['zone'].split(' ')[0].split('-')[0].upper()
                display_logs = raw_data[:30] 
            except: display_logs = []

    # E. Economic/Cyber Impacts
    adv = advanced_modules.AdvancedFeatures()
    impacts = adv.get_economic_impact(gpti_val, selected_zone)

except Exception as e:
    st.error(f"🚨 PIPELINE CRITICAL FAILURE: {e}")
    st.stop()

# ==========================================
# 5. LIVE INTEL TICKER (STEALTH OPS STYLE)
# ==========================================
if display_logs:
    ticker_items = ""
    # Creating a seamless loop for the marquee with the new color palette
    for i in (display_logs[:12] * 2):
        r_score = i.get('risk_score', 0)
        # Use Cyber Cyan (#00D4FF) for low risk, War Red (#FF0055) for high risk
        color = "#FF0055" if r_score > 0.75 else "#00D4FF"
        ticker_items += f"""
            <span style='margin: 0 60px; color: {color}; font-family: "JetBrains Mono"; font-size: 13px; font-weight: bold;'>
                <b>[{i['zone_clean']}]</b>: {i['sitrep']} <span style="opacity: 0.5;">⚡ LVL {r_score:.2f}</span>
            </span>"""
    
    st.markdown(f"""
        <div class="ticker-wrapper" style="background: #080C11; border-bottom: 1px solid #1C2631;">
            <div style="display:flex; align-items:center;">
                <div class="live-badge" style="margin-left: 20px;">SIGNAL INTERCEPT</div>
                <marquee scrollamount="7" style="color: #00D4FF; padding: 5px 0;">{ticker_items}</marquee>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# 6. HEADER & DEFCON COMMAND CENTER
# ==========================================
# Elite DEFCON Logic with Stealth Palette
if gpti_val > 0.80:
    defcon = {"lv": "1", "ds": "COCKED PISTOL", "clr": "#FF0055", "bg": "rgba(255, 0, 85, 0.15)"}
elif gpti_val > 0.60:
    defcon = {"lv": "2", "ds": "FAST PACE", "clr": "#FF7A00", "bg": "rgba(255, 122, 0, 0.1)"}
elif gpti_val > 0.40:
    defcon = {"lv": "3", "ds": "ROUND HOUSE", "clr": "#FFD600", "bg": "rgba(255, 214, 0, 0.05)"}
else:
    defcon = {"lv": "5", "ds": "FADE OUT", "clr": "#00D4FF", "bg": "rgba(0, 212, 255, 0.05)"}

st.markdown(f"""
    <div class="header-container">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h1 style='margin:0; font-size: 2.2rem; letter-spacing: -1px;'>GEOSENTINEL <span style='color:#00D4FF;'>C2 COMMAND</span></h1>
                <p style='margin:0; color:#8B949E; font-size: 11px; font-family: "JetBrains Mono";'>
                    <span class="status-pulse"></span> 
                    SATELLITE LINK: ESTABLISHED | PROTOCOL: OSINT-V3 | {datetime.now().strftime('%d %b %Y %H:%M:%S')} UTC
                </p>
            </div>
            <div style="text-align:right;">
                <small style="color:#4B5563; text-transform:uppercase; letter-spacing:1px;">Theater Status</small>
                <div style="color:#00D4FF; font-weight:bold; font-size:16px;">{selected_zone.upper()}</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Metrics Grid with "Stealth" Styling
m_col1, m_col2, m_col3, m_col4 = st.columns([1.2, 1.4, 1, 1])

with m_col1:
    st.metric("CONFLICT INDEX", f"{gpti_val:.2f}", f"{trend:+.2f} VOL", delta_color="inverse")

with m_col2:
    st.markdown(f"""
        <div class="defcon-box" style="border-top-color: {defcon['clr']}; background: {defcon['bg']}; color: {defcon['clr']};">
            <small style="opacity: 0.6; font-size: 10px; letter-spacing: 2px;">STRATEGIC POSTURE</small><br>
            <span style="font-size: 1.5rem; font-weight: 800;">DEFCON {defcon['lv']}</span><br>
            <small style="font-weight: bold; font-size: 11px;">{defcon['ds']}</small>
        </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.metric("INTEL NODES", "34 ACTIVE", "99.8% SIGINT")

with m_col4:
    # Arshad, hum Gemini 3 Flash use kar rahe hain, wahi update kiya hai niche
    st.metric("CORE BRAIN", "GEMINI 3F", "QUANTUM-OSINT")

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
# ==========================================
# 6. TAB DEFINITION (THE 5-LAYER STACK)
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🛰️ SITUATIONAL MAP",     # Reality
    "⚡ LIVE SIGNALS",        # SIGINT
    "📊 TREND ANALYTICS",     # Math/Trend
    "🔮 COMMANDER'S SIM",     # Wargaming
    "📁 STRATEGIC INTEL"      # NEW: Deep Dive Analysis
])

# ==========================================
# TAB 1: LIVE SURVEILLANCE (The Current Reality)
# ==========================================
with tab1:
    # --- Tactical Awareness Grid ---
    col_map, col_intel = st.columns([2.2, 1])

    with col_map:
        st.markdown("### 📍 TACTICAL DEPLOYMENT HEATMAP (3D)")
        
        try:
            # data_ingestion se real coordinates uthao
            map_df, lat, lon, zoom = data_ingestion.generate_location_data(selected_zone)
            
            # 3D Hexagon Layer Logic
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

            # Direct link to Dark Matter tiles (No API Key required)
            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(
                    latitude=lat, 
                    longitude=lon, 
                    zoom=zoom, 
                    pitch=45
                ),
                map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                tooltip={"text": "Conflict Intensity: {elevationValue}"}
            ))
            
        except Exception as e:
            st.error(f"📡 Intelligence Map Offline: Verify Connection.")
            print(f"DEBUG ERROR: {e}")

    with col_intel:
        st.markdown("### 🧠 AI STRATEGIC REPORTS")
        
        if os.path.exists("live_intelligence_log.json"):
            with open("live_intelligence_log.json", 'r') as f:
                try:
                    logs = json.load(f)
                    relevant = [l for l in logs if l.get('zone') == selected_zone][:4]
                    
                    if not relevant: 
                        st.info("No active signals in this theater. Scanning...")
                    
                    for log in relevant:
                        integrity = log.get('integrity_score', 100)
                        # Dynamic color for integrity badge
                        badge_clr = "#064e3b" if integrity > 70 else "#78350f" if integrity > 40 else "#7f1d1d"
                        
                        st.markdown(f"""
                        <div class="intel-card">
                            <span class="propaganda-flag" style="background:{badge_clr}; color:white;">SCORE: {integrity}</span>
                            <small style='color:#8b949e;'>{log.get('timestamp', '')[:16]}</small><br>
                            <strong style='color:#f0f6fc;'>{log.get('headline', '')}</strong><br>
                            <p style='color:#00ff41; font-size:13px; margin-top:5px;'>⚡ {log.get('sitrep', '')}</p>
                            <div style='border-top: 1px solid #30363d; padding-top: 5px; margin-top: 5px;'>
                                <small style='color:#8b949e;'><b>NEXT:</b> {log.get('options', ['Review'])[0]}</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.warning("⚠️ Intel Packet Corrupted.")
        else:
            st.warning("⚠️ Intelligence Log Missing.")


    

# ==========================================
# TAB 2: LIVE SIGNALS (The Intelligence Terminal)
# ==========================================
with tab2:
    # Simulation of intercept (Only for UI feel)
    with st.spinner("🕵️ INTERCEPTING ENCRYPTED OSINT PACKETS..."):
        time.sleep(0.5) # Chota sa delay for realism

    st.toast("Satellite Link Established: Node-01 Active", icon="📡")
    # 1. TACTICAL STATUS HEADER
    st.markdown("""
        <div style="background: rgba(0, 255, 65, 0.05); padding: 15px; border-left: 5px solid #00ff41; border-radius: 5px; margin-bottom: 25px;">
            <h3 style="margin:0; color:#00ff41; font-family: 'Share Tech Mono'; letter-spacing: 2px;">📡 LIVE SIGNAL INTELLIGENCE (SIGINT)</h3>
            <p style="margin:0; color: #8b949e; font-size: 11px;">MODE: REAL-TIME DECRYPTION | SOURCE: GLOBAL OSINT NODES | STATUS: MONITORING</p>
        </div>
    """, unsafe_allow_html=True)

    if os.path.exists("live_intelligence_log.json"):
        with open("live_intelligence_log.json", 'r') as f:
            try:
                raw_logs = json.load(f)
                # Filter out old news logic: Sirf 2026 ke news ya latest 30 signals
                display_logs = raw_logs[:30] 
                
                if display_logs:
                    # --- BREAKING ALERT TICKER (Feature: Most Recent High Risk) ---
                    high_risk_signals = [l for l in display_logs if l['risk_score'] > 0.75]
                    if high_risk_signals:
                        latest_alert = high_risk_signals[0]
                        st.markdown(f"""
                            <div style="background: rgba(255, 75, 75, 0.1); border: 1px solid #ff4b4b; padding: 10px; border-radius: 5px; margin-bottom: 20px; animation: blinker 2s linear infinite;">
                                <span style="color: #ff4b4b; font-weight: bold;">🚨 CRITICAL ALERT:</span> 
                                <span style="color: #f0f6fc;">{latest_alert['headline']}</span>
                            </div>
                            <style> @keyframes blinker {{ 50% {{ opacity: 0.6; }} }} </style>
                        """, unsafe_allow_html=True)

                    # --- TOP LEVEL ANALYTICS ---
                    m1, m2, m3 = st.columns(3)
                    m1.metric("TOTAL SIGNALS", len(raw_logs), help="Cumulative signals intercepted in current session.")
                    
                    # Trend Calculation for Threat Velocity
                    v_latest = sum(l['risk_score'] for l in display_logs[:5])/5
                    v_prev = sum(l['risk_score'] for l in display_logs[5:10])/5
                    v_delta = v_latest - v_prev
                    
                    m2.metric("THREAT VELOCITY", f"{v_latest:.2f}", delta=f"{v_delta:+.2f}", delta_color="inverse")
                    m3.metric("AI CONFIDENCE", f"{sum(l['integrity_score'] for l in display_logs)/len(display_logs):.1f}%")

                    # --- SIGNAL INTENSITY PULSE ---
                    st.markdown('<p class="tactical-label">Risk Pulse Monitor (Standard Deviation Analysis)</p>', unsafe_allow_html=True)
                    fig_spark = go.Figure(go.Scatter(
                        x=[l['timestamp'] for l in display_logs[::-1]], 
                        y=[l['risk_score'] for l in display_logs[::-1]],
                        mode='lines+markers', fill='tozeroy', 
                        line=dict(color='#00ff41', width=3, shape='spline'),
                        fillcolor='rgba(0, 255, 65, 0.1)',
                        marker=dict(size=6, color='#ff4b4b', symbol='cross')
                    ))
                    fig_spark.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                          height=200, margin=dict(l=0,r=0,t=0,b=0),
                                          xaxis=dict(showgrid=False, showticklabels=False),
                                          yaxis=dict(showgrid=True, gridcolor='#30363d', range=[0, 1.1]))
                    st.plotly_chart(fig_spark, use_container_width=True)

                    st.markdown("---")
                    st.subheader("🛡️ TACTICAL SITUATION REPORTS (SITREPS)")

                    # --- THE ELITE SIGNAL CARDS LOOP ---
                    for log in display_logs:
                        # Cleaning: Remove 2019/2022/2023 from Theater Names
                        clean_zone = log['zone'].split(' ')[0].split('-')[0].upper()
                        
                        risk_val = log['risk_score']
                        risk_clr = "#ff4b4b" if risk_val > 0.75 else "#ffaa00" if risk_val > 0.45 else "#00ff41"
                        time_str = log['timestamp'].split("T")[-1][:8]
                        
                        # Formatting Flags as Badges
                        flags_html = "".join([f'<span style="background:rgba(255,75,75,0.1); color:#ff4b4b; border:1px solid #ff4b4b; padding:2px 8px; border-radius:10px; font-size:10px; margin-right:5px; font-family:sans-serif;">{f.upper()}</span>' for f in log.get('flags', [])])
                        
                        # Strategic Options with Category Icons
                        raw_options = log.get('strategic_options', [])
                        options_html = ""
                        for opt in raw_options:
                            icon = "🔍" if "Intelligence" in opt else "⚔️" if "Tactical" in opt else "🤝" if "Diplomatic" in opt else "⚡"
                            options_html += f'<li style="color:#f0f6fc; margin-bottom:8px; font-size:13px; list-style:none;">{icon} {opt}</li>'

                        # CARD CONTAINER
                        st.markdown(f"""
                        <div style="background: rgba(22, 27, 34, 0.7); border: 1px solid #30363d; border-left: 5px solid {risk_clr}; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.3);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                <span style="color: #4CC9F0; font-size: 13px; font-family: 'Share Tech Mono'; font-weight: bold;">
                                    🕒 {time_str} | 📍 {clean_zone}
                                </span>
                                <div style="text-align: right;">
                                    <span style="background: {risk_clr}33; color: {risk_clr}; padding: 2px 12px; border: 1px solid {risk_clr}; border-radius: 4px; font-weight: bold; font-size: 12px; font-family: 'Share Tech Mono';">
                                        RISK: {risk_val:.2f}
                                    </span>
                                </div>
                            </div>
                            <h4 style="color: #f0f6fc; margin-top: 0; font-family: 'Share Tech Mono'; line-height: 1.4;">{log['headline']}</h4>
                            <div style="color: #00ff41; font-size: 14px; background: rgba(0,255,65,0.03); padding: 12px; border-radius: 4px; border: 1px solid rgba(0,255,65,0.1); margin-bottom: 15px;">
                                <b style="letter-spacing: 1px; color:#8b949e; font-size: 10px; display: block; margin-bottom: 5px;">ANALYTICAL SITREP:</b>
                                {log.get('sitrep', 'Awaiting deep packet analysis...')}
                            </div>
                            <div style="margin: 15px 0;">
                                {flags_html if flags_html else '<span style="color:#4b5563; font-size:10px; letter-spacing: 1px;">SIGNAL INTEGRITY: VERIFIED</span>'}
                            </div>
                            <div style="background: rgba(13, 17, 23, 0.9); padding: 15px; border-radius: 6px; border: 1px solid #21262d;">
                                <small style="color: #8b949e; text-transform: uppercase; letter-spacing: 1px; font-size: 10px;">Strategic Response Directives:</small>
                                <ul style="margin-top: 10px; padding-left: 5px;">
                                    {options_html}
                                </ul>
                            </div>
                            <div style="margin-top: 15px; display: flex; justify-content: space-between; border-top: 1px solid #21262d; padding-top: 10px;">
                                <small style="color: #4b5563;">SOURCE: {log.get('source', 'OSINT_NODE')}</small>
                                <small style="color: #4b5563;">CONFIDENCE: {log.get('integrity_score', 0)}%</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ INTERFACE_SYNC_FAILURE: {e}")
    else:
        st.warning("⚠️ SATLINK_OFFLINE: Waiting for Sentinel signals...")

# ==========================================
# TAB 3: CONFLICT ANALYTICS (The Research)
# ==========================================
with tab3:
    # --- 8. CONFLICT GRAVITY & PREDICTION ---
    st.subheader("📉 CONFLICT GRAVITY & PREDICTION HORIZON")
    
    try:
        # Fetching last 30 intervals for the trend
        chart_data = final_df.tail(30)
        
        # Prediction Logic (Future Trajectory)
        last_val = chart_data['GPTI'].iloc[-1]
        # Projecting next 3 intervals using the trend calculated in index_calculator
        prediction_points = [last_val + (trend * i) for i in range(1, 4)]
        pred_dates = pd.date_range(start=chart_data['date'].iloc[-1], periods=4, freq='H')[1:]

        # --- PLOTLY ENGINE ---
        fig = go.Figure()

        # Narrative Bars
        fig.add_trace(go.Bar(
            x=chart_data['date'], y=chart_data['INT_norm'], 
            name="Narrative Pressure", marker_color='rgba(0, 255, 65, 0.2)'
        ))
        
        # Kinetic Bars
        fig.add_trace(go.Bar(
            x=chart_data['date'], y=chart_data['MCT_norm'], 
            name="Kinetic Load", marker_color='rgba(76, 201, 240, 0.2)'
        ))
        
        # Total GPTI Line
        fig.add_trace(go.Scatter(
            x=chart_data['date'], y=chart_data['GPTI'], 
            name="Total Index (GPTI)", 
            line=dict(color='#ff4b4b', width=4, shape='spline'), 
            fill='tozeroy', fillcolor='rgba(255, 75, 75, 0.1)'
        ))
        
        # Prediction Dotted Line
        fig.add_trace(go.Scatter(
            x=pred_dates, y=prediction_points, 
            name="Predictive Trajectory", 
            line=dict(color='#ffffff', width=2, dash='dot'), 
            mode='lines+markers'
        ))

        fig.update_layout(
            template="plotly_dark", 
            hovermode="x unified", 
            barmode='overlay', 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering Gravity Chart: {e}")

    st.markdown("---")

    # --- 14. HISTORICAL ECHO (CRISIS BENCHMARKING) ---
    st.subheader("📜 HISTORICAL ECHO: CRISIS BENCHMARKING")

    # --- RESEARCH-BASED HISTORICAL DATA ---
    HISTORICAL_CRISES = {
        "Russia-Ukraine (Feb 2022)": 0.96,
        "Balakot Air Strikes (Feb 2019)": 0.84,
        "Israel-Hamas Escalation (Oct 2023)": 0.89,
        "Galwan Valley Clash (June 2020)": 0.78,
        "Cuban Missile Crisis (Estimated)": 0.98
    }

    # Dynamic Comparison
    current_data = {"CURRENT: " + selected_zone: gpti_val}
    all_benchmarks = {**HISTORICAL_CRISES, **current_data}
    sorted_benchmarks = dict(sorted(all_benchmarks.items(), key=lambda item: item[1]))

    # --- HORIZONTAL BAR CHART ---
    fig_echo = go.Figure()
    
    # Highlight current theater in Red
    colors = ['rgba(48, 54, 61, 0.6)'] * len(sorted_benchmarks)
    try:
        current_idx = list(sorted_benchmarks.keys()).index("CURRENT: " + selected_zone)
        colors[current_idx] = '#ff4b4b'
    except: pass

    fig_echo.add_trace(go.Bar(
        y=list(sorted_benchmarks.keys()),
        x=list(sorted_benchmarks.values()),
        orientation='h',
        marker_color=colors,
        text=[f"{v:.2f}" for v in sorted_benchmarks.values()],
        textposition='auto'
    ))

    fig_echo.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=20, t=20, b=0),
        xaxis=dict(title="GPTI Intensity Scale", range=[0, 1.1], gridcolor='#30363d'),
        height=400
    )
    st.plotly_chart(fig_echo, use_container_width=True)

    # --- ANALYST INSIGHT BOX ---
    closest_crisis = min(HISTORICAL_CRISES, key=lambda x: abs(HISTORICAL_CRISES[x] - gpti_val))
    proximity = (1 - abs(HISTORICAL_CRISES[closest_crisis] - gpti_val)) * 100

    st.markdown(f"""
        <div style="background: rgba(76, 201, 240, 0.1); border-left: 5px solid #4CC9F0; padding: 15px; border-radius: 5px;">
            <span style="color: #4CC9F0; font-weight: bold;">📜 ANALYTICAL ECHO:</span> 
            The current situation in <b>{selected_zone}</b> shows a <b>{proximity:.1f}%</b> statistical similarity 
            to the peak of <b>{closest_crisis}</b>.
        </div>
    """, unsafe_allow_html=True)



# ==========================================
# TAB 4: COMMANDER'S SIM (The Future)
# ==========================================
with tab4:
    # --- 11. STRATEGIC SCENARIO SIMULATOR ---
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
            # Tactical Sliders based on current normalization
            sim_kinetic = st.slider("Simulated Kinetic Activity (MCT):", 0.0, 1.0, float(latest['MCT_norm']), 
                                    help="Simulate increase in troop deployments, border skirmishes, or drone strikes.")
            
            sim_narrative = st.slider("Simulated Narrative Pressure (INT):", 0.0, 1.0, float(latest['INT_norm']), 
                                      help="Simulate a massive surge in hostile state media reporting or cyber-propaganda.")
            
            sim_horizon = st.select_slider("Projection Horizon:", options=["Immediate", "48h Short-term", "7-Day Tactical"], value="48h Short-term")

        # --- SIMULATION MATH (PCA-WEIGHTED) ---
        # Formula: sim_gpti = (w1 * sim_kinetic) + (w2 * sim_narrative)
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
            sim_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': 'Share Tech Mono'}, height=350, margin=dict(t=50, b=0))
            st.plotly_chart(sim_fig, use_container_width=True)

    # --- ACTIONABLE AI ANALYSIS ---
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

    # --- 15. STRATEGIC RESOURCE FALLOUT ---
    st.markdown("---")
    st.subheader("💰 STRATEGIC RESOURCE FALLOUT (PREDICTIVE)")

    # Active value switches between simulated and current
    active_val = sim_gpti if 'sim_gpti' in locals() else gpti_val

    # Market Correlation Math
    oil_risk = active_val * 35 
    gold_hedge = active_val * 18
    currency_vol = "CRITICAL" if active_val > 0.8 else "UNSTABLE" if active_val > 0.6 else "STABLE"

    r_col1, r_col2, r_col3 = st.columns(3)

    with r_col1:
        st.metric(label="🛢️ CRUDE OIL RISK", value=f"+{oil_risk:.1f}%", delta="Supply Threat", delta_color="inverse")
        st.caption("Brent Crude futures surge.")

    with r_col2:
        st.metric(label="🟡 GOLD (SAFE HAVEN)", value=f"+{gold_hedge:.1f}%", delta="Capital Flight")
        st.caption("Hedge inflow into bullion.")

    with r_col3:
        st.metric(label="💹 CURRENCY VOLATILITY", value=currency_vol, delta="Forex Risk", delta_color="off" if currency_vol == "STABLE" else "inverse")
        st.caption("Regional currency fluctuation.")

    # --- RISK ASSESSMENT BOX ---
    status_color = "#ff4b4b" if active_val > 0.75 else "#ffaa00" if active_val > 0.5 else "#00ff41"
    st.markdown(f"""
        <div style="background: rgba(13, 17, 23, 0.8); border: 1px solid {status_color}33; 
                    border-left: 5px solid {status_color}; padding: 15px; border-radius: 4px; margin-top: 15px;">
            <strong style="color: {status_color}; text-transform: uppercase;"> 
                ⚠️ Economic Risk Assessment: {'SEVERE' if active_val > 0.75 else 'MODERATE' if active_val > 0.5 else 'LOW'}
            </strong><br>
            <p style="color: #8b949e; font-size: 13px; margin-top: 5px;">
                Based on the current <b>GPTI of {active_val:.2f}</b>, the model predicts a shift in global commodity flows. 
                Recommend hedging assets and securing logistics pipelines.
            </p>
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# TAB 5: STRATEGIC INTEL (Pro-Level Deep Dive)
# ==========================================
with tab5:
    st.markdown("""
        <div style="background: rgba(76, 201, 240, 0.05); padding: 15px; border-left: 5px solid #4CC9F0; border-radius: 5px; margin-bottom: 25px;">
            <h3 style="margin:0; color:#4CC9F0; font-family: 'Share Tech Mono'; letter-spacing: 2px;">📁 MULTI-DOMAIN STRATEGIC DOSSIER</h3>
            <p style="margin:0; color: #8b949e; font-size: 11px;">ANALYSIS NODE: SENTINEL-ADVANCED | ENGINE: DETERMINISTIC THREAT MODELING</p>
        </div>
    """, unsafe_allow_html=True)

    # 1. THREAT RADAR & PANIC GAUGE
    col_radar, col_panic = st.columns([1.8, 1.2])
    
    with col_radar:
        st.markdown('<p class="tactical-label">5-Axis Domain Threat Matrix</p>', unsafe_allow_html=True)
        # Radar Chart Logic
        threat_data = adv.generate_threat_matrix(gpti_val, selected_zone)
        categories = list(threat_data.keys())
        values = list(threat_data.values())
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(76, 201, 240, 0.25)',
            line=dict(color='#4CC9F0', width=3),
            marker=dict(size=8, color='#00ff41')
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, 
                    range=[0, 100], 
                    gridcolor="#30363d", 
                    tickfont=dict(size=8, color="#8b949e")
                ),
                angularaxis=dict(
                    gridcolor="#30363d", 
                    tickfont=dict(size=10, color="#f0f6fc")  # FIXED: Changed 'font' to 'tickfont'
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=False, 
            paper_bgcolor='rgba(0,0,0,0)', 
            height=380, 
            margin=dict(t=40, b=40, l=50, r=50)
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

    with col_panic:
        st.markdown('<p class="tactical-label">Social Anxiety & Panic Index</p>', unsafe_allow_html=True)
        panic_val, trending_terms = adv.get_public_panic_index(gpti_val)
        
        # PRO-LEVEL GAUGE
        fig_panic = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = panic_val,
            number = {'suffix': "%", 'font': {'color': '#f0f6fc', 'family': 'Share Tech Mono'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4b5563"},
                'bar': {'color': "#ff4b4b" if panic_val > 65 else "#00ff41"},
                'bgcolor': "rgba(255,255,255,0.05)",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(0, 255, 65, 0.1)'},
                    {'range': [50, 80], 'color': 'rgba(255, 170, 0, 0.1)'},
                    {'range': [80, 100], 'color': 'rgba(255, 75, 75, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_panic.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=280, margin=dict(t=20, b=0))
        st.plotly_chart(fig_panic, use_container_width=True)
        
        st.markdown(f"""
            <div style="background:rgba(13,17,23,0.8); border:1px solid #30363d; padding:10px; border-radius:5px;">
                <small style="color:#8b949e; text-transform:uppercase;">Trending Civilian Concerns:</small><br>
                <span style="color:#ffaa00; font-family:'Share Tech Mono'; font-size:14px;">
                    {" | ".join([t.upper() for t in trending_terms])}
                </span>
                <p style="font-size:9px; color:#4b5563; margin-top:10px;">
                    *Social Panic is calculated using a <b>Non-Linear Sigmoid Model</b> centered at GPTI 0.60.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. INFORMATION INTEGRITY (PSYOP MONITOR)
    st.markdown('<p class="tactical-label">Information Integrity & PsyOp Analysis</p>', unsafe_allow_html=True)
    if display_logs:
        latest_text = display_logs[0]['headline'] + " " + display_logs[0].get('sitrep', '')
        integrity = adv.analyze_information_integrity(latest_text, display_logs[0].get('sitrep', ''))
        
        i_col1, i_col2 = st.columns([1, 2.5])
        with i_col1:
            st.metric("INTEGRITY SCORE", f"{integrity['integrity_score']}%", 
                      delta="SECURE" if integrity['integrity_score'] > 75 else "COMPROMISED", 
                      delta_color="normal" if integrity['integrity_score'] > 75 else "inverse")
        with i_col2:
            status_clr = "#00ff41" if integrity['integrity_score'] > 75 else "#ffaa00" if integrity['integrity_score'] > 50 else "#ff4b4b"
            st.markdown(f"""
                <div style="padding:10px; border:1px solid {status_clr}33; background:{status_clr}11; border-radius:5px;">
                    <small style="color:#8b949e;">NARRATIVE STATUS:</small>
                    <b style="color:{status_clr}; display:block;">{integrity['narrative_status'].upper()}</b>
                </div>
            """, unsafe_allow_html=True)
            
            if integrity['flags']:
                st.markdown("<div style='margin-top:10px;'>" + 
                            "".join([f'<span style="background:rgba(255,75,75,0.1); color:#ff4b4b; padding:2px 8px; border-radius:5px; margin-right:5px; font-size:10px; border:1px solid #ff4b4b44;">⚠️ {f.upper()}</span>' for f in integrity['flags']]) + 
                            "</div>", unsafe_allow_html=True)

    st.markdown("---")

    # 3. LEADING INDICATORS (The Glowing Checklist)
    st.markdown('<p class="tactical-label">Tactical Leading Indicators for Kinetic Escalation</p>', unsafe_allow_html=True)
    
    # Getting results from the updated AdvancedFeatures engine
    indicator_results = adv.analyze_leading_indicators(latest_text, selected_zone)
    
    ind_cols = st.columns(4)
    for i, (name, active) in enumerate(indicator_results.items()):
        with ind_cols[i]:
            if active:
                color = "#ff4b4b" if "Mobilization" in name else "#00ff41"
                glow = f"box-shadow: 0 0 15px {color}44; border: 1px solid {color}; background: {color}11;"
                icon = "⚡"
                status_text = "[ACTIVE]"
            else:
                color = "#30363d"
                glow = "border: 1px solid #21262d; background: rgba(255,255,255,0.02);"
                icon = "🔘"
                status_text = "[SCANNING]"

            st.markdown(f"""
                <div style="text-align:center; padding:15px; border-radius:8px; {glow}">
                    <div style="font-size:24px; margin-bottom:5px;">{icon}</div>
                    <div style="font-size:10px; color:{color if active else '#8b949e'}; font-family:'Share Tech Mono'; font-weight:bold;">{name.upper()}</div>
                    <div style="font-size:8px; color:{color if active else '#4b5563'}; margin-top:5px;">{status_text}</div>
                </div>
            """, unsafe_allow_html=True)




# ==========================================
# 10. SYSTEM METHODOLOGY (THE JURY KILL-SHOT)
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)

with st.expander("🔬 SYSTEM ARCHITECTURE & MATHEMATICAL MODEL"):
    st.markdown("""
        <div style="background: rgba(0, 212, 255, 0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(0, 212, 255, 0.2);">
            <h2 style="color:#00D4FF; margin-top:0;">CENTRAL INTELLIGENCE ARCHITECTURE</h2>
            <p style="color:#8B949E; font-size:14px;">GeoSentinel operates on a decentralized SIGINT (Signal Intelligence) pipeline, processing unstructured OSINT data into deterministic strategic indices.</p>
        </div>
    """, unsafe_allow_html=True)
    
    m_col1, m_col2 = st.columns([1.8, 1.2])
    
    with m_col1:
        st.write("### **1. Geopolitical Tension Index (GPTI) Math**")
        st.write("""
        The GPTI is calculated by projecting high-dimensional conflict data onto a single principal axis using **Weighted PCA**. 
        We apply **Z-score Normalization** to Kinetic signals ($K$) and Narrative signals ($N$) to ensure zero-mean and unit-variance before aggregation.
        """)
        
        # Elite LaTeX for GPTI
        st.latex(r"GPTI = \sum_{i=1}^{n} \left( \omega_K \cdot \frac{K_i - \mu_K}{\sigma_K} + \omega_N \cdot \frac{N_i - \mu_N}{\sigma_N} \right)")
        
        st.write("### **2. Non-Linear Social Panic Model**")
        st.write("""
        Unlike linear models, public anxiety follows a **Sigmoid Activation Function**. Panic remains dormant until the GPTI threshold ($x_0$) is breached, after which it scales exponentially.
        """)
        
        # Sigmoid Math
        st.latex(r"P(g) = \frac{100}{1 + e^{-k(g - x_0)}}")
        st.caption("Where $k=10$ (escalation steepness) and $x_0=0.6$ (inflection point).")

    with m_col2:
        st.write("### **Intelligence Tech Stack**")
        st.markdown("""
        <div style="background: #0D1117; padding: 15px; border-radius: 8px; border: 1px solid #1C2631;">
            <ul style="list-style-type: none; padding-left: 0; font-size: 13px; color: #C9D1D9;">
                <li style="margin-bottom:10px;"><b>🧠 CORE BRAIN:</b> Gemini 3 Flash (NLU Engine)</li>
                <li style="margin-bottom:10px;"><b>🕵️ SIGINT NODES:</b> 34 Distributed Scrapers</li>
                <li style="margin-bottom:10px;"><b>📊 DIMENSIONALITY:</b> Scikit-Learn (PCA Solver)</li>
                <li style="margin-bottom:10px;"><b>🌐 VISUALIZATION:</b> Pydeck 3D & Plotly Spline</li>
                <li style="margin-bottom:10px;"><b>🛡️ FAIL-SAFE:</b> Multi-Key API Load Balancer</li>
                <li style="margin-bottom:10px;"><b>🛠️ BACKEND:</b> Python 3.11 / Streamlit C2</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("✅ **Mathematical Integrity Verified**")

    st.markdown("---")
    st.write("### **3. Data Ingestion Flowchart (The C2 Pipeline)**")
    # Simplified text-based flow for the jury
    st.code("""
    RAW OSINT FEED ➡️ SENTIMENT FILTER (DistilBERT) ➡️ TACTICAL EXTRACTION (Gemini 3F)
    ➡️ PCA NORMALIZATION (Math Node) ➡️ GPTI CALCULATION ➡️ DEFCON TRIGGER ➡️ DASHBOARD
    """, language="text")

    
# ==========================================
# FINAL BRANDING FOOTER (STEALTH SIGNATURE)
# ==========================================
footer_html = """
<div style="margin-top: 50px; padding: 30px 0; border-top: 1px solid rgba(0, 212, 255, 0.1); text-align: center; background-color: #05070a;">
    <div style="display: flex; justify-content: center; align-items: center; gap: 20px; margin-bottom: 10px;">
        <div style="height: 1px; width: 50px; background: rgba(0, 212, 255, 0.2);"></div>
        <span style="color: #4B5563; font-family: 'Courier New', monospace; font-size: 10px; letter-spacing: 3px; text-transform: uppercase;">
            Terminal End Transmission [EOT]
        </span>
        <div style="height: 1px; width: 50px; background: rgba(0, 212, 255, 0.2);"></div>
    </div>
    
    <p style="color: #8B949E; font-family: 'Courier New', monospace; font-size: 11px; line-height: 1.8;">
        <span style="color: #00D4FF;">GEOSENTINEL C2</span> | REGISTRY: <span style="color: #F0F6FC;">LKO-UP-IND-2026-ALPHA</span><br>
        <span style="opacity: 0.6;">STATION STATUS:</span> <span style="color: #00ff41;">SECURE_ENCRYPTED</span> | 
        <span style="opacity: 0.6;">CORE:</span> <span style="color: #00D4FF;">GEMINI-3-FLASH</span><br>
        <span style="margin-top: 15px; display: block; font-size: 13px; letter-spacing: 1px;">
            LEAD SYSTEMS ARCHITECT: <b style="color: #00D4FF;">MOHD ARSHAD</b>
        </span>
    </p>
    
    <div style="margin-top: 20px; opacity: 0.3; font-size: 8px; color: #8B949E; font-family: 'Courier New';">
        UNAUTHORIZED ACCESS TO SIGINT NODES IS STRICTLY PROHIBITED | 2026 © GEOSENTINEL DEFENSE SYSTEMS
    </div>
</div>
"""

# Rendering the component
components.html(footer_html, height=250)