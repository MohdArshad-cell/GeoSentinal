import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import index_calculator
import data_ingestion
import analysis_engine
import config

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    layout="wide", 
    page_title="GeoSentinel Commander",
    page_icon="🛡️"
)

# Custom CSS for "Command Center" Dark Mode Look
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #E63946; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; }
    .stMetric { background-color: #1E1E1E; border: 1px solid #333; padding: 15px; border-radius: 10px; }
    div[data-testid="stMetricValue"] { color: #F1FAEE; }
    .stSlider > div > div > div > div { background-color: #E63946; }
    .css-1d391kg { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INITIALIZE AI ENGINE (CACHED)
# ==========================================
@st.cache_resource
def load_ai():
    return analysis_engine.NarrativeAI()

try:
    ai_brain = load_ai()
except Exception as e:
    # Fail gracefully if AI libraries are missing so the dashboard doesn't crash
    ai_brain = None

# ==========================================
# 3. SIDEBAR CONTROL PANEL
# ==========================================
with st.sidebar:
    st.title("📡 Control Panel")
    st.markdown("---")
    
    # --- A. CONFLICT SELECTION ---
    st.subheader("📍 Target Zone")
    selected_dyad = st.selectbox(
        "Select Conflict to Monitor:",
        ["India-Pakistan", "Russia-Ukraine", "Israel-Palestine"],
        index=0
    )
    st.caption(f"Status: **Monitoring {selected_dyad}**")
    
    # --- DATA LOADING ---
    # This happens immediately so we can use the data for filters below
    with st.spinner("Establishing Secure Uplink..."):
        # 1. Generate Synthetic Data (The Eyes)
        df = data_ingestion.generate_synthetic_data(selected_dyad)
        # 2. Calculate Index (The Judge)
        calculator = index_calculator.IndexCalculator()
        final_df = calculator.process_index(df)
        final_df['date'] = pd.to_datetime(final_df['date'])

    st.markdown("---")
    
    # --- B. TIME RANGE FILTERS ---
    st.subheader("⏳ Time Window")
    enable_date_filter = st.checkbox("Enable Date Filter")
    
    if enable_date_filter:
        min_date = final_df['date'].min().date()
        max_date = final_df['date'].max().date()
        
        date_range = st.slider(
            "Select Analysis Period:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="MMM YYYY"
        )
        # Apply Filter to Dataframe
        mask = (final_df['date'].dt.date >= date_range[0]) & (final_df['date'].dt.date <= date_range[1])
        final_df = final_df.loc[mask]
    
    st.markdown("---")

    # --- C. VIEW SETTINGS ---
    st.subheader("⚙️ Analysis Options")
    smoothing = st.slider("Signal Smoothing (Days):", min_value=1, max_value=14, value=1, help="Reduce noise to see long-term trends.")
    show_markers = st.checkbox("Show Major Event Markers", value=True)

    # Apply Smoothing Logic
    if smoothing > 1:
        cols_to_smooth = ['GPTI', 'norm_kinetic', 'norm_narrative']
        for col in cols_to_smooth:
            final_df[col] = final_df[col].rolling(window=smoothing).mean()

    st.markdown("---")

    # --- D. WARGAME SIMULATOR (WHAT-IF MODE) ---
    st.subheader("🎲 Wargame Simulator")
    st.caption("Simulate hypothetical escalation scenarios.")
    
    sim_kinetic = st.slider("Simulated Kinetic Events:", 0, 100, 10)
    sim_narrative = st.slider("Simulated News Volume:", 0, 5000, 500)
    sim_sentiment = st.slider("Simulated Hostility (0=Peace, 1=War):", 0.0, 1.0, 0.8)
    
    # Live Calculation of Hypothetical GPTI
    sim_norm_k = sim_kinetic / 100.0  # Normalized against max
    sim_norm_n = (sim_narrative * sim_sentiment) / 5000.0 # Normalized against max
    # We assume high variance in a crisis, so weights split 50/50 for simulation
    sim_gpti = (0.5 * sim_norm_k + 0.5 * sim_norm_n) * 100
    
    # Display Result
    if sim_gpti > 70:
        st.error(f"Projected Threat Level: CRITICAL ({sim_gpti:.1f})")
    elif sim_gpti > 40:
        st.warning(f"Projected Threat Level: ELEVATED ({sim_gpti:.1f})")
    else:
        st.success(f"Projected Threat Level: STABLE ({sim_gpti:.1f})")

    st.markdown("---")

    # --- E. MISSION REPORT EXPORT ---
    st.subheader("💾 Mission Report")
    csv_data = final_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Dataset (CSV)",
        data=csv_data,
        file_name=f'geosentinel_{selected_dyad.lower()}_report.csv',
        mime='text/csv',
        help="Export all metrics for external analysis."
    )

# ==========================================
# 4. MAIN DASHBOARD TABS
# ==========================================
tab_main, tab_global, tab_deep, tab_ai, tab_about = st.tabs([
    "📈 Crisis Monitor", 
    "🌍 Global Overwatch", 
    "🔬 Statistical Deep Dive", 
    "🧠 Live AI Lab", 
    "ℹ️ System Intel"
])

# ------------------------------------------
# TAB 1: CRISIS MONITOR (MAIN VIEW)
# ------------------------------------------
with tab_main:
    st.title("🛡️ GeoSentinel: Conflict Monitor")
    
    if not final_df.empty:
        # -- 1. Heads-Up Display (Metrics) --
        latest = final_df.iloc[-1]
        prev = final_df.iloc[-2] if len(final_df) > 1 else latest
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Tension (GPTI)", f"{latest['GPTI']:.1f}", f"{latest['GPTI'] - prev['GPTI']:.1f}", delta_color="inverse")
        m2.metric("Kinetic Intensity", f"{latest['norm_kinetic']*100:.1f}%", f"{(latest['norm_kinetic'] - prev['norm_kinetic'])*100:.1f}%")
        m3.metric("Narrative Hostility", f"{latest['norm_narrative']*100:.1f}%", f"{(latest['norm_narrative'] - prev['norm_narrative'])*100:.1f}%")
        
        dom_pillar = "Kinetic" if latest['weight_kinetic'] > latest['weight_narrative'] else "Narrative"
        m4.metric("Dominant Driver", dom_pillar)

        # -- 2. Main Charts --
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Upper Chart: Tension & Pillars
        fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['GPTI'], mode='lines', name='Tension Index (GPTI)', line=dict(color='#EF233C', width=3), fill='tozeroy'), row=1, col=1)
        fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['norm_kinetic']*100, mode='lines', name='Kinetic Force', line=dict(color='#F4A261', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['norm_narrative']*100, mode='lines', name='Narrative Hype', line=dict(color='#4CC9F0', width=1, dash='dot')), row=1, col=1)
        
        # Lower Chart: PCA Weights
        fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['weight_kinetic'], stackgroup='one', name='Kinetic Weight', line=dict(width=0, color='rgba(244, 162, 97, 0.6)')), row=2, col=1)
        fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['weight_narrative'], stackgroup='one', name='Narrative Weight', line=dict(width=0, color='rgba(76, 201, 240, 0.6)')), row=2, col=1)
        
        # -- 3. Event Markers Logic --
        if show_markers:
            events = []
            if selected_dyad == "India-Pakistan":
                events = [(config.PULWAMA_ATTACK, "Pulwama"), (config.BALAKOT_STRIKE, "Balakot"), (config.ABHINANDAN_CAPTURE, "Dogfight")]
            elif selected_dyad == "Russia-Ukraine":
                events = [(config.INVASION_START, "Invasion Starts"), (config.TROOP_BUILDUP_START, "Buildup")]
            elif selected_dyad == "Israel-Palestine":
                events = [(config.OCT_7_ATTACK, "Oct 7 Attack"), (config.HOSPITAL_BLAST, "Hospital Blast"), (config.GROUND_INVASION, "Ground Ops")]

            for date_str, label in events:
                event_date = pd.to_datetime(date_str)
                # Ensure marker is within the currently filtered date range
                if event_date >= final_df['date'].min() and event_date <= final_df['date'].max():
                    # Use timestamp * 1000 to fix Pandas/Plotly bug
                    fig.add_vline(x=event_date.timestamp() * 1000, line_dash="dash", line_color="white", annotation_text=label, annotation_position="top right")

        fig.update_layout(height=600, template="plotly_dark", hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left"))
        st.plotly_chart(fig, use_container_width=True)
        
        # -- 4. Intelligence Drill-Down (Time Machine) --
        st.markdown("---")
        st.subheader("🕵️‍♂️ Intelligence Drill-Down")
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            # Set default date based on conflict
            default_date = config.PULWAMA_ATTACK
            if selected_dyad == "Russia-Ukraine": default_date = config.INVASION_START
            if selected_dyad == "Israel-Palestine": default_date = config.OCT_7_ATTACK
                
            investigate_date = st.date_input(
                "Select Date to Investigate:",
                value=pd.to_datetime(default_date),
                min_value=final_df['date'].min(),
                max_value=final_df['date'].max()
            )

        with c2:
            target_date = pd.to_datetime(investigate_date)
            day_data = final_df[final_df['date'].dt.date == target_date.date()]
            
            if not day_data.empty:
                gpti_val = day_data['GPTI'].values[0]
                kin_val = day_data['kinetic_score'].values[0]
                nar_val = day_data['narrative_volume'].values[0]
                
                # Synthetic Situation Report Generator
                if gpti_val > 80:
                    status = "🔴 CRITICAL THREAT"
                    desc = "Major escalation detected. Military forces engaging in direct combat. Global media saturation."
                elif gpti_val > 50:
                    status = "🟠 ELEVATED TENSION"
                    desc = "Diplomatic channels stalling. Sharp rise in hostile rhetoric and border skirmishes."
                else:
                    status = "🟢 STABLE / ROUTINE"
                    desc = "Standard border operations. No major incidents reported in the sector."

                st.info(f"**Status:** {status} | **Index Score:** {gpti_val:.1f}")
                st.markdown(f"> *\"{desc}\"*")
                
                d1, d2, d3 = st.columns(3)
                d1.metric("Kinetic Events", f"{int(kin_val)}")
                d2.metric("News Volume", f"{int(nar_val)}")
                d3.metric("AI Sentiment", f"{day_data['sentiment_score'].values[0]:.2f}")
            else:
                st.warning("No intelligence data available for this date.")
    else:
        st.warning("No data available for the selected date range.")

# ------------------------------------------
# TAB 2: GLOBAL OVERWATCH (COMPARISON)
# ------------------------------------------
with tab_global:
    st.header("🌍 Global Risk Overwatch")
    st.markdown("Comparative analysis of all monitored conflict zones on a unified timeline.")
    
    with st.spinner("Aggregating Global Intelligence Feeds..."):
        conflicts = ["India-Pakistan", "Russia-Ukraine", "Israel-Palestine"]
        fig_global = go.Figure()
        
        for c in conflicts:
            # Generate data for each conflict on the fly
            temp_df = data_ingestion.generate_synthetic_data(c)
            temp_calc = calculator.process_index(temp_df)
            temp_calc['date'] = pd.to_datetime(temp_calc['date'])
            
            fig_global.add_trace(go.Scatter(
                x=temp_calc['date'], 
                y=temp_calc['GPTI'], 
                mode='lines', 
                name=c
            ))
            
    fig_global.update_layout(height=500, template="plotly_dark", title="Global Tension Comparison (GPTI)", hovermode="x unified")
    st.plotly_chart(fig_global, use_container_width=True)
    st.info("💡 Insight: This view allows you to compare relative intensity. Notice how the Russia-Ukraine conflict creates a sustained 'Wall of War' compared to the sharp 'Spike' of India-Pakistan.")

# ------------------------------------------
# TAB 3: STATISTICAL DEEP DIVE
# ------------------------------------------
with tab_deep:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Correlation Matrix")
        if not final_df.empty:
            corr = final_df[['norm_kinetic', 'norm_narrative', 'GPTI']].corr()
            st.dataframe(corr.style.background_gradient(cmap='Reds'))
            st.caption("Strong correlation (Red) indicates pillars moving together.")
    
    with c2:
        st.subheader("Granger Causality (Lead-Lag)")
        if not final_df.empty:
            # Run the Statistical Test
            p_value = calculator.test_causality(final_df)
            st.metric("P-Value", f"{p_value:.4f}")
            
            if p_value < 0.05:
                st.success("✅ **VALID WARNING SIGNAL:** The Narrative pillar statistically predicts Kinetic violence 3 days in advance.")
            else:
                st.warning("⚠️ **NO SIGNAL DETECTED:** No clear lead-lag relationship found in this time window.")

# ------------------------------------------
# TAB 4: LIVE AI LAB
# ------------------------------------------
with tab_ai:
    st.header("🧠 Hybrid AI Analysis Lab")
    st.markdown("Test the **DistilBERT Engine** live. Type a hypothetical news headline to see how the system scores it.")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        user_text = st.text_area("Enter News Headline:", height=150, placeholder="E.g., Border forces exchange heavy fire overnight...")
        
        if st.button("Analyze Text"):
            if user_text and ai_brain:
                with st.spinner("AI Processing..."):
                    # 1. Check Relevance (Gatekeeper)
                    is_relevant = ai_brain.llm_relevance_filter(user_text)
                    # 2. Check Sentiment (Analyst)
                    sentiment_score = ai_brain.get_sentiment_score(user_text)
                    
                    st.session_state['ai_result'] = {"text": user_text, "relevant": is_relevant, "score": sentiment_score}
            elif not ai_brain:
                st.error("AI Engine is not loaded. Please check your dependencies.")
            else:
                st.warning("Please enter some text.")

    with col_result:
        if 'ai_result' in st.session_state:
            res = st.session_state['ai_result']
            
            # Relevance Badge
            if res['relevant']:
                st.success("✅ RELEVANT: Geopolitical Content Detected")
            else:
                st.error("❌ IRRELEVANT: Noise Detected (Sports/Entertainment)")
            
            # Sentiment Meter
            st.markdown("### Hostility Score")
            # Invert score because Low Sentiment = High Hostility in our logic
            hostility_pct = (1.0 - res['score']) 
            st.progress(hostility_pct)
            st.caption(f"Hostility Level: {hostility_pct*100:.1f}%")
            
            # AI Verdict
            if hostility_pct > 0.7:
                st.markdown("🚨 **Verdict:** High Tension Event")
            elif hostility_pct > 0.4:
                st.markdown("⚠️ **Verdict:** Moderate Tension")
            else:
                st.markdown("🕊️ **Verdict:** Low Tension / Peaceful")

# ------------------------------------------
# TAB 5: SYSTEM INTEL
# ------------------------------------------
with tab_about:
    st.markdown("### System Architecture")
    # st.image("architecture.png")  <-- Uncomment if you have the image
    st.markdown("""
    **GeoSentinel** fuses synthetic data streams with a real DistilBERT analysis engine.
    
    1.  **Ingestion:** Simulated ACLED/GDELT data streams.
    2.  **Processing:** Hybrid AI (LLM Filter + DistilBERT Sentiment).
    3.  **Aggregation:** PCA-based Dynamic Weighting.
    """)