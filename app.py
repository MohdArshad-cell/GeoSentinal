import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import index_calculator
import data_ingestion
import config

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    layout="wide", 
    page_title="GeoSentinel Commander",
    page_icon="🛡️"
)

# --- 2. CUSTOM CSS (THE "POLISH") ---
# This hides the default Streamlit menu and adds a professional header style
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #E63946; /* Urgent Red */
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
    }
    .stMetric {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        color: #F1FAEE;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("📡 Control Panel")
    st.markdown("---")
    
    # Conflict Selection
    selected_dyad = st.selectbox(
        "Target Conflict Zone:",
        ["India-Pakistan", "Russia-Ukraine", "Israel-Palestine"],
        index=0
    )
    
    st.caption(f"Monitoring: **{selected_dyad}**")
    st.markdown("---")
    
    # Date Filter (Optional Zoom)
    st.subheader("📅 Time Window")
    use_custom_date = st.checkbox("Filter Date Range")
    
    # Load Data immediately to get min/max dates for the slider
    with st.spinner("Establishing Secure Uplink..."):
        df = data_ingestion.generate_synthetic_data(selected_dyad)
        calculator = index_calculator.IndexCalculator()
        final_df = calculator.process_index(df)
        final_df['date'] = pd.to_datetime(final_df['date'])

    if use_custom_date:
        min_date = final_df['date'].min().date()
        max_date = final_df['date'].max().date()
        date_range = st.slider(
            "Select Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date)
        )
        # Filter Data
        mask = (final_df['date'].dt.date >= date_range[0]) & (final_df['date'].dt.date <= date_range[1])
        final_df = final_df.loc[mask]

# --- 4. HEADER SECTION ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🛡️ GeoSentinel: Conflict Monitor")
    st.markdown("Real-time AI fusion of **Kinetic Events** and **Narrative Sentiment**.")
with col2:
    # Statistical Status Badge
    p_value = calculator.test_causality(final_df)
    if p_value < 0.05:
        st.success("⚠️ LINK VERIFIED: Narrative predicting Violence")
    else:
        st.info("ℹ️ LINK UNSTABLE: No predictive signal detected")

# --- 5. KEY METRICS ROW (HUD) ---
# Get the latest day's data for the "Current Status"
latest = final_df.iloc[-1]
prev = final_df.iloc[-2]

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Current Tension (GPTI)", 
        f"{latest['GPTI']:.1f}", 
        f"{latest['GPTI'] - prev['GPTI']:.1f}",
        delta_color="inverse"
    )

with m2:
    st.metric(
        "Kinetic Intensity", 
        f"{latest['norm_kinetic']*100:.1f}%", 
        f"{(latest['norm_kinetic'] - prev['norm_kinetic'])*100:.1f}%"
    )

with m3:
    st.metric(
        "Narrative Hostility", 
        f"{latest['norm_narrative']*100:.1f}%", 
        f"{(latest['norm_narrative'] - prev['norm_narrative'])*100:.1f}%"
    )

with m4:
    # Show which pillar is currently dominant
    dom_pillar = "Kinetic" if latest['weight_kinetic'] > latest['weight_narrative'] else "Narrative"
    st.metric("Dominant Driver", dom_pillar)

st.markdown("---")

# --- 6. MAIN DASHBOARD TABS ---
tab_main, tab_deep, tab_about = st.tabs(["📈 Crisis Monitor", "🔬 Statistical Deep Dive", "ℹ️ System Intel"])

with tab_main:
    # --- THE MASTER CHART ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Main GPTI Line
    fig.add_trace(go.Scatter(
        x=final_df['date'], y=final_df['GPTI'],
        mode='lines', name='Tension Index (GPTI)',
        line=dict(color='#EF233C', width=3),
        fill='tozeroy', fillcolor='rgba(239, 35, 60, 0.1)'
    ), row=1, col=1)
    
    # Add Pillar Lines
    fig.add_trace(go.Scatter(
        x=final_df['date'], y=final_df['norm_kinetic']*100,
        mode='lines', name='Kinetic Force',
        line=dict(color='#F4A261', width=1, dash='dot')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=final_df['date'], y=final_df['norm_narrative']*100,
        mode='lines', name='Narrative Hype',
        line=dict(color='#4CC9F0', width=1, dash='dot')
    ), row=1, col=1)
    
    # Weights Chart (Bottom)
    fig.add_trace(go.Scatter(
        x=final_df['date'], y=final_df['weight_kinetic'],
        stackgroup='one', name='Kinetic Weight',
        line=dict(width=0, color='rgba(244, 162, 97, 0.6)')
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=final_df['date'], y=final_df['weight_narrative'],
        stackgroup='one', name='Narrative Weight',
        line=dict(width=0, color='rgba(76, 201, 240, 0.6)')
    ), row=2, col=1)

    # --- EVENT MARKERS (CONTEXT) ---
    # Add vertical lines for known events based on selection
    events = []
    if selected_dyad == "India-Pakistan":
        events = [
            (config.PULWAMA_ATTACK, "Pulwama"),
            (config.BALAKOT_STRIKE, "Balakot"),
            (config.ABHINANDAN_CAPTURE, "Dogfight")
        ]
    elif selected_dyad == "Russia-Ukraine":
        events = [
            (config.INVASION_START, "Invasion Starts"),
            (config.TROOP_BUILDUP_START, "Buildup")
        ]
    elif selected_dyad == "Israel-Palestine":
        events = [
            (config.OCT_7_ATTACK, "Oct 7 Attack"),
            (config.HOSPITAL_BLAST, "Hospital Blast"),
            (config.GROUND_INVASION, "Ground Ops")
        ]

    for date_str, label in events:
        # Convert to object for comparison
        event_date = pd.to_datetime(date_str)
        
        # Only add if within current date range
        if event_date >= final_df['date'].min() and event_date <= final_df['date'].max():
            fig.add_vline(
                # FIX: Convert Timestamp to numeric milliseconds to bypass Pandas error
                x=event_date.timestamp() * 1000, 
                line_dash="dash", 
                line_color="white", 
                annotation_text=label, 
                annotation_position="top right"
            )

    # Chart Polish
    fig.update_layout(
        height=600,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left")
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab_deep:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Correlation Matrix")
        corr = final_df[['norm_kinetic', 'norm_narrative', 'GPTI']].corr()
        st.dataframe(corr.style.background_gradient(cmap='Reds'))
    
    with c2:
        st.subheader("Granger Causality Stats")
        st.metric("P-Value (Lead-Lag)", f"{p_value:.4f}")
        if p_value < 0.05:
            st.success("The Narrative pillar provides a statistically significant warning signal for Kinetic events.")
        else:
            st.warning("No significant lead-lag relationship detected in this window.")

with tab_about:
    st.markdown("""
    ### System Architecture
    **GeoSentinel** uses a hybrid approach to quantify risk:
    1.  **Ingestion**: Synthetic data streams simulating ACLED (Kinetic) and GDELT (Narrative).
    2.  **Processing**: 
        -   *Kinetic*: Weighted severity scoring.
        -   *Narrative*: LLM-based relevance filtering + DistilBERT sentiment scoring.
    3.  **Aggregation**: Principal Component Analysis (PCA) dynamically weights the pillars based on volatility.
    """)