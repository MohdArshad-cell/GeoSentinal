import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import index_calculator
import data_ingestion
import analysis_engine
import config

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    layout="wide", 
    page_title="GeoSentinel Commander",
    page_icon="🛡️"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #E63946; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; }
    .stMetric { background-color: #1E1E1E; border: 1px solid #333; padding: 15px; border-radius: 10px; }
    div[data-testid="stMetricValue"] { color: #F1FAEE; }
    .stSlider > div > div > div > div { background-color: #E63946; }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD AI ENGINE (Cached) ---
@st.cache_resource
def load_ai():
    return analysis_engine.NarrativeAI()

try:
    ai_brain = load_ai()
except Exception as e:
    # Fail gracefully if AI can't load (so dashboard still works)
    ai_brain = None

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("📡 Control Panel")
    st.markdown("---")
    
    # A. Conflict Selection
    st.subheader("📍 Target Zone")
    selected_dyad = st.selectbox(
        "Select Conflict:",
        ["India-Pakistan", "Russia-Ukraine", "Israel-Palestine"],
        index=0
    )
    st.caption(f"Monitoring: **{selected_dyad}**")
    
    # --- LOAD DATA (Before filtering) ---
    with st.spinner("Establishing Secure Uplink..."):
        df = data_ingestion.generate_synthetic_data(selected_dyad)
        calculator = index_calculator.IndexCalculator()
        final_df = calculator.process_index(df)
        final_df['date'] = pd.to_datetime(final_df['date'])

    st.markdown("---")
    
    # B. Time Filters
    st.subheader("⏳ Time Range")
    enable_date_filter = st.checkbox("Filter Date Range")
    
    if enable_date_filter:
        min_date = final_df['date'].min().date()
        max_date = final_df['date'].max().date()
        
        date_range = st.slider(
            "Select Range:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="MMM YYYY"
        )
        # Apply Filter
        mask = (final_df['date'].dt.date >= date_range[0]) & (final_df['date'].dt.date <= date_range[1])
        final_df = final_df.loc[mask]
    
    st.markdown("---")

    # C. Visualization Settings
    st.subheader("⚙️ View Options")
    smoothing = st.slider("Chart Smoothing (Days):", min_value=1, max_value=14, value=1, help="Higher values make lines smoother.")
    show_markers = st.checkbox("Show Event Markers", value=True)

    # Apply Smoothing if requested
    if smoothing > 1:
        cols_to_smooth = ['GPTI', 'norm_kinetic', 'norm_narrative']
        for col in cols_to_smooth:
            final_df[col] = final_df[col].rolling(window=smoothing).mean()

    # D. DOWNLOAD REPORT
    st.markdown("---")
    st.subheader("💾 Mission Report")
    
    # Convert the dataframe to CSV string
    csv_data = final_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Full Dataset (CSV)",
        data=csv_data,
        file_name=f'geosentinel_{selected_dyad.lower()}_report.csv',
        mime='text/csv',
        help="Export the kinetic, narrative, and AI-weighted scores for further analysis."
    )

# --- 5. DASHBOARD TABS ---
tab_main, tab_deep, tab_ai, tab_about = st.tabs(["📈 Crisis Monitor", "🔬 Statistical Deep Dive", "🧠 Live AI Lab", "ℹ️ System Intel"])

# --- TAB 1: CRISIS MONITOR ---
with tab_main:
    st.title("🛡️ GeoSentinel: Conflict Monitor")
    
    # Metrics Row (Get latest available data point after filter)
    if not final_df.empty:
        latest = final_df.iloc[-1]
        # Use previous day if available, else 0
        prev = final_df.iloc[-2] if len(final_df) > 1 else latest
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Tension (GPTI)", f"{latest['GPTI']:.1f}", f"{latest['GPTI'] - prev['GPTI']:.1f}", delta_color="inverse")
        m2.metric("Kinetic Intensity", f"{latest['norm_kinetic']*100:.1f}%", f"{(latest['norm_kinetic'] - prev['norm_kinetic'])*100:.1f}%")
        m3.metric("Narrative Hostility", f"{latest['norm_narrative']*100:.1f}%", f"{(latest['norm_narrative'] - prev['norm_narrative'])*100:.1f}%")
        dom_pillar = "Kinetic" if latest['weight_kinetic'] > latest['weight_narrative'] else "Narrative"
        m4.metric("Dominant Driver", dom_pillar)

        # Charts
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Top Chart: GPTI and Pillars
        fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['GPTI'], mode='lines', name='Tension Index (GPTI)', line=dict(color='#EF233C', width=3), fill='tozeroy'), row=1, col=1)
        fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['norm_kinetic']*100, mode='lines', name='Kinetic Force', line=dict(color='#F4A261', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['norm_narrative']*100, mode='lines', name='Narrative Hype', line=dict(color='#4CC9F0', width=1, dash='dot')), row=1, col=1)
        
        # Bottom Chart: Weights
        fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['weight_kinetic'], stackgroup='one', name='Kinetic Weight', line=dict(width=0, color='rgba(244, 162, 97, 0.6)')), row=2, col=1)
        fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['weight_narrative'], stackgroup='one', name='Narrative Weight', line=dict(width=0, color='rgba(76, 201, 240, 0.6)')), row=2, col=1)
        
        # Event Markers
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
                    fig.add_vline(x=event_date.timestamp() * 1000, line_dash="dash", line_color="white", annotation_text=label, annotation_position="top right")

        fig.update_layout(height=600, template="plotly_dark", hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left"))
        st.plotly_chart(fig, use_container_width=True)

        # --- INTELLIGENCE DRILL-DOWN (Added Feature) ---
        st.markdown("---")
        st.subheader("🕵️‍♂️ Intelligence Drill-Down")
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            # Default to the peak crisis date for the selected conflict
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
            # Find data for that specific date
            target_date = pd.to_datetime(investigate_date)
            day_data = final_df[final_df['date'].dt.date == target_date.date()]
            
            if not day_data.empty:
                gpti_val = day_data['GPTI'].values[0]
                kin_val = day_data['kinetic_score'].values[0]
                nar_val = day_data['narrative_volume'].values[0]
                
                # Generate a "Synthetic Situation Report" based on the score
                if gpti_val > 80:
                    status = "🔴 CRITICAL THREAT"
                    desc = "Major escalation detected. Military forces engaging in direct combat. Global media saturation."
                elif gpti_val > 50:
                    status = "🟠 ELEVATED TENSION"
                    desc = "Diplomatic channels stalling. Sharp rise in hostile rhetoric and border skirmishes."
                else:
                    status = "🟢 STABLE / ROUTINE"
                    desc = "Standard border operations. No major incidents reported in the sector."

                # Display the Report Card
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

# --- TAB 2: DEEP DIVE ---
with tab_deep:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Correlation Matrix")
        if not final_df.empty:
            corr = final_df[['norm_kinetic', 'norm_narrative', 'GPTI']].corr()
            st.dataframe(corr.style.background_gradient(cmap='Reds'))
    with c2:
        st.subheader("Granger Causality Stats")
        if not final_df.empty:
            p_value = calculator.test_causality(final_df)
            st.metric("P-Value (Lead-Lag)", f"{p_value:.4f}")
            if p_value < 0.05:
                st.success("Valid Warning Signal Detected")
            else:
                st.warning("No significant signal")

# --- TAB 3: LIVE AI LAB ---
with tab_ai:
    st.header("🧠 Hybrid AI Analysis Lab")
    st.markdown("Test the **DistilBERT Engine** live. Type a headline to see how the system scores it.")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        user_text = st.text_area("Enter News Headline:", height=150, placeholder="E.g., Border forces exchange heavy fire overnight...")
        
        if st.button("Analyze Text"):
            if user_text and ai_brain:
                with st.spinner("AI Processing..."):
                    # 1. Check Relevance
                    is_relevant = ai_brain.llm_relevance_filter(user_text)
                    # 2. Check Sentiment
                    sentiment_score = ai_brain.get_sentiment_score(user_text)
                    
                    st.session_state['ai_result'] = {"text": user_text, "relevant": is_relevant, "score": sentiment_score}
            elif not ai_brain:
                st.error("AI Engine is not loaded. Please check your dependencies.")
            else:
                st.warning("Please enter some text.")

    with col_result:
        if 'ai_result' in st.session_state:
            res = st.session_state['ai_result']
            if res['relevant']:
                st.success("✅ RELEVANT: Geopolitical Content Detected")
            else:
                st.error("❌ IRRELEVANT: Noise Detected (Sports/Entertainment)")
            
            st.markdown("### Hostility Score")
            hostility_pct = (1.0 - res['score']) 
            st.progress(hostility_pct)
            st.caption(f"Hostility Level: {hostility_pct*100:.1f}%")
            
            if hostility_pct > 0.7:
                st.markdown("🚨 **Verdict:** High Tension Event")
            elif hostility_pct > 0.4:
                st.markdown("⚠️ **Verdict:** Moderate Tension")
            else:
                st.markdown("🕊️ **Verdict:** Low Tension / Peaceful")

# --- TAB 4: ABOUT ---
with tab_about:
    st.markdown("### System Architecture")
    # st.image("architecture.png") 
    st.markdown("GeoSentinel fuses synthetic data streams with a real DistilBERT analysis engine.")