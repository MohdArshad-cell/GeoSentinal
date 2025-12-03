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
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ==========================================
# 1. PAGE CONFIGURATION & PROFESSIONAL STYLING
# ==========================================
st.set_page_config(
    layout="wide", 
    page_title="GeoSentinel Commander",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Command Center" Aesthetic
st.markdown("""
<style>
    /* Main Container Padding */
    .main .block-container { padding-top: 1rem; padding-bottom: 3rem; }
    
    /* Header Styling */
    h1, h2, h3 { color: #E63946; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    
    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background-color: #1A1A1A;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: scale(1.02);
        border-color: #E63946;
    }
    div[data-testid="stMetricValue"] { color: #F1FAEE; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #A8A8A8; font-size: 0.9rem; }
    
    /* Slider Styling */
    .stSlider > div > div > div > div { background-color: #E63946; }
    
    /* Expander Styling */
    .streamlit-expanderHeader { background-color: #262626; color: #E63946; border-radius: 5px; }
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
except Exception:
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
    
    # Dynamic Status Indicator with more detail
    if selected_dyad == "Israel-Palestine":
        status_color = "🔴"
        status_text = "CRITICAL"
    elif selected_dyad == "Russia-Ukraine":
        status_color = "🟠" 
        status_text = "HIGH"
    else:
        status_color = "🟡"
        status_text = "ELEVATED"
        
    st.info(f"{status_color} Status: **{status_text}**\n\nMonitoring: **{selected_dyad}**")
    
    # --- LOAD DATA ---
    with st.spinner("Establishing Secure Uplink..."):
        df = data_ingestion.generate_synthetic_data(selected_dyad)
        calculator = index_calculator.IndexCalculator()
        final_df = calculator.process_index(df)
        final_df['date'] = pd.to_datetime(final_df['date'])

    st.markdown("---")
    
    # --- B. ANALYSIS SETTINGS (Collapsible) ---
    with st.expander("🛠️ Analysis Settings", expanded=True):
        # Time Window
        st.caption("⏳ **Time Window**")
        enable_date_filter = st.checkbox("Enable Date Filter")
        
        if enable_date_filter:
            min_d = final_df['date'].min().date()
            max_d = final_df['date'].max().date()
            
            date_range = st.slider(
                "Select Period:",
                min_value=min_d,
                max_value=max_d,
                value=(min_d, max_d),
                format="MMM YYYY"
            )
            mask = (final_df['date'].dt.date >= date_range[0]) & (final_df['date'].dt.date <= date_range[1])
            final_df = final_df.loc[mask]
        
        st.markdown("---")

        # View Options
        st.caption("⚙️ **View Options**")
        smoothing = st.slider("Signal Smoothing (Days):", 1, 14, 1, help="Increase to reduce noise and see long-term trends.")
        show_markers = st.checkbox("Show Event Markers", value=True)

        if smoothing > 1:
            cols = ['GPTI', 'norm_kinetic', 'norm_narrative']
            for c in cols: final_df[c] = final_df[c].rolling(window=smoothing).mean()

    st.markdown("---")

    # --- C. WARGAME SIMULATOR ---
    with st.expander("🎲 Wargame Simulator"):
        st.caption("Simulate hypothetical scenarios to test index response.")
        
        sim_kinetic = st.slider("Sim Kinetic Events:", 0, 100, 10)
        sim_narrative = st.slider("Sim News Volume:", 0, 5000, 500)
        sim_sentiment = st.slider("Sim Hostility:", 0.0, 1.0, 0.8)
        
        sim_norm_k = sim_kinetic / 100.0
        sim_norm_n = (sim_narrative * sim_sentiment) / 5000.0
        sim_gpti = (0.5 * sim_norm_k + 0.5 * sim_norm_n) * 100
        
        if sim_gpti > 70:
            st.error(f"Threat: CRITICAL ({sim_gpti:.1f})")
        elif sim_gpti > 40:
            st.warning(f"Threat: ELEVATED ({sim_gpti:.1f})")
        else:
            st.success(f"Threat: STABLE ({sim_gpti:.1f})")

    # --- D. EXPORT ---
    st.markdown("---")
    st.markdown("### 💾 Reports")
    csv_data = final_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Mission Report", 
        data=csv_data, 
        file_name=f'geosentinel_{selected_dyad.lower()}_report.csv', 
        mime='text/csv',
        use_container_width=True
    )

# ==========================================
# 4. MAIN DASHBOARD TABS
# ==========================================
tab_main, tab_map, tab_forecast, tab_global, tab_deep, tab_ai, tab_about = st.tabs([
    "📈 Crisis Monitor", 
    "🗺️ War Room", 
    "🔮 Forecast",
    "🌍 Global Overwatch", 
    "🔬 Deep Dive", 
    "🧠 AI Lab", 
    "ℹ️ Intel"
])

# ------------------------------------------
# TAB 1: CRISIS MONITOR (MAIN HUD)
# ------------------------------------------
# --- CSS FIX FOR METRIC CARDS ---
st.markdown("""
<style>
div[data-testid="metric-container"] {
    background-color: #1E1E1E;
    border: 1px solid #333;
    padding: 15px;
    border-radius: 8px;
    height: 140px; /* This forces all cards to be the same height */
    display: flex;
    flex-direction: column;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

with tab_main:
    st.title(f"🛡️ {selected_dyad} Monitor")
    
    if not final_df.empty:
        latest = final_df.iloc[-1]
        prev = final_df.iloc[-2] if len(final_df) > 1 else latest
        
        # Calculate 30-day avg for context
        avg_30d = final_df['GPTI'].iloc[-30:].mean() if len(final_df) > 30 else final_df['GPTI'].mean()
        delta_val = latest['GPTI'] - avg_30d
        
        # 1. METRICS ROW
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Tension (GPTI)", f"{latest['GPTI']:.1f}", f"{delta_val:.1f} (vs 30d Avg)", delta_color="inverse")
        m2.metric("Kinetic Force", f"{latest['norm_kinetic']*100:.1f}%", f"{(latest['norm_kinetic'] - prev['norm_kinetic'])*100:.1f}%")
        m3.metric("Narrative Hype", f"{latest['norm_narrative']*100:.1f}%", f"{(latest['norm_narrative'] - prev['norm_narrative'])*100:.1f}%")
        
        dom_pillar = "Kinetic" if latest['weight_kinetic'] > latest['weight_narrative'] else "Narrative"
        m4.metric("Dominant Driver", dom_pillar, help="Which pillar is currently driving the conflict score?")

        # 2. MAIN CHARTS
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08, 
            row_heights=[0.7, 0.3],
            subplot_titles=("Tension Index & Components", "AI-Driven Weight Distribution")
        )
        
        # Row 1: Index & Pillars
        fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['GPTI'], mode='lines', name='GPTI Score', line=dict(color='#EF233C', width=3), fill='tozeroy', fillcolor='rgba(239, 35, 60, 0.1)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['norm_kinetic']*100, mode='lines', name='Kinetic', line=dict(color='#F4A261', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['norm_narrative']*100, mode='lines', name='Narrative', line=dict(color='#4CC9F0', width=1, dash='dot')), row=1, col=1)
        
        # Row 2: Weights
        fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['weight_kinetic'], stackgroup='one', name='Kinetic Wt', line=dict(width=0, color='rgba(244, 162, 97, 0.5)')), row=2, col=1)
        fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['weight_narrative'], stackgroup='one', name='Narrative Wt', line=dict(width=0, color='rgba(76, 201, 240, 0.5)')), row=2, col=1)
        
        # 3. EVENT MARKERS (Staggered Labels)
        if show_markers:
            events = []
            if selected_dyad == "India-Pakistan": events = [(config.PULWAMA_ATTACK, "Pulwama"), (config.BALAKOT_STRIKE, "Balakot"), (config.ABHINANDAN_CAPTURE, "Dogfight")]
            elif selected_dyad == "Russia-Ukraine": events = [(config.INVASION_START, "Invasion Starts"), (config.TROOP_BUILDUP_START, "Buildup")]
            elif selected_dyad == "Israel-Palestine": events = [(config.OCT_7_ATTACK, "Oct 7 Attack"), (config.HOSPITAL_BLAST, "Hospital Blast"), (config.GROUND_INVASION, "Ground Ops")]

            for i, (d_str, lbl) in enumerate(events):
                ed = pd.to_datetime(d_str)
                if ed >= final_df['date'].min() and ed <= final_df['date'].max():
                    # Stagger label height to avoid crash: Low, High, Medium
                    y_pos = 115 if i % 3 == 0 else (130 if i % 3 == 1 else 145)
                    fig.add_vline(x=ed.timestamp()*1000, line_dash="dash", line_color="white", opacity=0.5)
                    fig.add_annotation(
                        x=ed.timestamp()*1000, y=y_pos, text=lbl, showarrow=False, 
                        font=dict(color="white", size=10), bgcolor="#1E1E1E", bordercolor="#333"
                    )

        fig.update_layout(height=700, template="plotly_dark", hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. DRILL DOWN
        with st.expander("🕵️‍♂️ Open Daily Intelligence Log (Drill-Down)"):
            c1, c2 = st.columns([1, 2])
            with c1:
                def_date = config.PULWAMA_ATTACK if selected_dyad == "India-Pakistan" else config.INVASION_START
                if selected_dyad == "Israel-Palestine": def_date = config.OCT_7_ATTACK
                inv_date = st.date_input("Investigate Date:", value=pd.to_datetime(def_date), min_value=final_df['date'].min(), max_value=final_df['date'].max())
            
            with c2:
                dd = final_df[final_df['date'].dt.date == pd.to_datetime(inv_date).date()]
                if not dd.empty:
                    val = dd['GPTI'].values[0]
                    if val > 80: status, color = "CRITICAL", "red"
                    elif val > 50: status, color = "ELEVATED", "orange"
                    else: status, color = "STABLE", "green"

                    st.markdown(f"### Status: :{color}[{status}] (Score: {val:.1f})")
                    st.markdown(f"**Intelligence Summary:** Analysis of {inv_date} indicates {status.lower()} threat levels driven by {int(dd['narrative_volume'].values[0])} news reports and {int(dd['kinetic_score'].values[0])} kinetic events.")
                    
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Kinetic Events", int(dd['kinetic_score'].values[0]))
                    d2.metric("News Volume", int(dd['narrative_volume'].values[0]))
                    d3.metric("Net Sentiment", f"{dd['sentiment_score'].values[0]:.2f}")
                else:
                    st.warning("No data for this date.")
    else:
        st.warning("No data available for the selected date range.")

# ------------------------------------------
# TAB 2: WAR ROOM MAP
# ------------------------------------------
with tab_map:
    st.header(f"🗺️ Geospatial Intel: {selected_dyad}")
    
    # 1. Layout Configuration: Map on Left (3), Controls on Right (1)
    col_map, col_intel = st.columns([3, 1])
    
    with col_intel:
        st.markdown("### 📡 Sector Controls")
        layer_type = st.radio("Visualization Layer:", ["Heatmap (Density)", "Tactical (Events)", "Combined"], index=2)
        map_pitch = st.slider("Map Pitch (3D):", 0, 60, 45)
        
        st.markdown("---")
        st.markdown("### 🛑 Threat Level")
        if not final_df.empty:
            # Dynamic Threat Gauge based on latest Kinetic Score
            k_score = final_df.iloc[-1]['norm_kinetic']
            if k_score > 0.7:
                st.error("STATUS: CRITICAL")
                st.caption("Heavy kinetic activity detected across multiple sectors.")
            elif k_score > 0.4:
                st.warning("STATUS: ACTIVE")
                st.caption("Sporadic skirmishes reported.")
            else:
                st.success("STATUS: MONITORING")
                st.caption("No major incursions.")

    with col_map:
        if not final_df.empty:
            # A. Data Prep (Enriching the raw lat/lon with simulated metadata)
            total_kinetic = int(final_df['kinetic_score'].sum())
            display_points = min(total_kinetic, 2000)
            
            # Generate base coordinates
            map_df, lat, lon, zoom = data_ingestion.generate_location_data(selected_dyad, display_points)
            
            # ENRICHMENT: Add fake "Event Types" and "Severity" for the Tooltip
            event_types = ['Artillery Fire', 'Airstrike', 'Drone Strike', 'Skirmish']
            map_df['type'] = np.random.choice(event_types, display_points)
            map_df['casualties'] = np.random.randint(0, 15, display_points)
            # Color logic: [R, G, B] -> Red for high casualty, Orange for low
            map_df['color'] = map_df['casualties'].apply(lambda x: [200, 30, 30, 200] if x > 5 else [255, 165, 0, 160])
            map_df['radius'] = map_df['casualties'] * 50 + 100  # Size based on severity

            # B. PyDeck Layers
            layers = []
            
            # Layer 1: Hexagon Heatmap (Density)
            if layer_type in ["Heatmap (Density)", "Combined"]:
                layers.append(pdk.Layer(
                    'HexagonLayer',
                    data=map_df,
                    get_position='[lon, lat]',
                    radius=8000 if selected_dyad == "Russia-Ukraine" else 2000,
                    elevation_scale=50,
                    elevation_range=[0, 1000],
                    pickable=True,
                    extruded=True,
                    opacity=0.4 if layer_type == "Combined" else 0.8
                ))

            # Layer 2: Scatterplot (Specific Events with Tooltips)
            if layer_type in ["Tactical (Events)", "Combined"]:
                layers.append(pdk.Layer(
                    'ScatterplotLayer',
                    data=map_df,
                    get_position='[lon, lat]',
                    get_fill_color='color',
                    get_radius='radius',
                    pickable=True,
                    auto_highlight=True
                ))

            # C. Render Map with Tooltip
            st.pydeck_chart(pdk.Deck(
                map_style=None, 
                initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=map_pitch),
                layers=layers,
                tooltip={
                    "html": "<b>Type:</b> {type}<br/><b>Casualties:</b> {casualties}<br/><b>Location:</b> {lon:.2f}, {lat:.2f}",
                    "style": {"backgroundColor": "#1E1E1E", "color": "white"}
                }
            ))
    
    # 3. Strategic Summary Row
    st.markdown("---")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Active Hotspots", f"{display_points}", "Sites confirmed")
    i2.metric("Est. Casualty Impact", f"{map_df['casualties'].sum()}", "Simulated count")
    i3.metric("Primary Tactic", map_df['type'].mode()[0])
    i4.metric("Region", "Line of Control" if selected_dyad=="India-Pakistan" else "Donbas Front" if selected_dyad=="Russia-Ukraine" else "Gaza Strip")

# ------------------------------------------
# TAB 3: FORECAST
# ------------------------------------------
with tab_forecast:
    st.header("🔮 Predictive Early Warning System")
    st.markdown("AI-driven projection of conflict intensity over the next 30 days using **Holt-Winters Exponential Smoothing**.")

    if not final_df.empty:
        try:
            # 1. MODELING
            # Use Holt-Winters for trend + seasonality (Damped to prevent unrealistic infinite growth)
            model = ExponentialSmoothing(
                final_df['GPTI'].fillna(0),
                trend='add',
                seasonal=None,
                damped_trend=True
            ).fit()

            forecast_days = 30
            forecast = model.forecast(forecast_days)

            # 2. METRICS CALCULATION
            current_score = final_df['GPTI'].iloc[-1]
            max_forecast = forecast.max()
            avg_forecast = forecast.mean()

            # Determine Trend Direction
            if avg_forecast > current_score * 1.05: trend_dir = "📈 ESCALATING"
            elif avg_forecast < current_score * 0.95: trend_dir = "📉 DE-ESCALATING"
            else: trend_dir = "➡️ STABLE"

            # Determine Risk Level
            if max_forecast > 80: risk_level = "CRITICAL"
            elif max_forecast > 50: risk_level = "HIGH"
            else: risk_level = "MODERATE"

            # 3. HUD METRICS
            c1, c2, c3 = st.columns(3)
            c1.metric("Projected Trend (30d)", trend_dir, f"{avg_forecast - current_score:.1f} change", delta_color="inverse")
            c2.metric("Peak Forecast Score", f"{max_forecast:.1f}", help="Highest predicted tension point in next 30 days")
            c3.metric("Threat Assessment", risk_level, f"Avg: {avg_forecast:.1f}")

            st.markdown("---")

            # 4. VISUALIZATION
            last_date = final_df['date'].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=forecast_days + 1)[1:]

            fig_p = go.Figure()

            # Historical Context (Last 60 days for clarity)
            history_subset = final_df.iloc[-60:]
            fig_p.add_trace(go.Scatter(
                x=history_subset['date'],
                y=history_subset['GPTI'],
                name="Historical Data",
                mode='lines',
                line=dict(color='#EF233C', width=2)
            ))

            # Forecast Line
            fig_p.add_trace(go.Scatter(
                x=future_dates,
                y=forecast,
                name="AI Forecast",
                mode='lines',
                line=dict(color='#4CC9F0', width=3, dash='dash')
            ))

            # Critical Threshold
            fig_p.add_hline(y=80, line_dash="dot", line_color="red", annotation_text="CRITICAL THRESHOLD", annotation_position="top right")

            # Layout Polish
            fig_p.update_layout(
                height=500,
                template="plotly_dark",
                title="30-Day Risk Trajectory",
                xaxis_title="Timeline",
                yaxis_title="Projected Tension (GPTI)",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=60, b=20),
                legend=dict(orientation="h", y=1.05)
            )
            st.plotly_chart(fig_p, use_container_width=True)

            # 5. AI ANALYSIS BOX
            st.info(f"**Strategic Insight:** The predictive model indicates a **{trend_dir.lower()}** trajectory. Policy intervention is recommended if the projected score exceeds **50.0** within the next 2 weeks.")

        except Exception as e:
            st.error(f"Forecasting Module Error: {e}")
            st.warning("Insufficient data points to generate a reliable forecast. Please adjust the time window.")
    else:
        st.warning("No data available for forecasting.")

# ------------------------------------------
# TAB 4: GLOBAL OVERWATCH
# ------------------------------------------
with tab_global:
    st.header("🌍 Global Risk Overwatch")
    st.markdown("Real-time comparative analysis of all monitored conflict zones.")

    with st.spinner("Integrating Global Intelligence Feeds..."):
        conflicts = ["India-Pakistan", "Russia-Ukraine", "Israel-Palestine"]
        
        # Store data for all conflicts to calculate global metrics
        all_data = {}
        current_scores = {}
        
        # Generate Data loop
        for c in conflicts:
            t_df = data_ingestion.generate_synthetic_data(c)
            t_calc = calculator.process_index(t_df)
            t_calc['date'] = pd.to_datetime(t_calc['date'])
            all_data[c] = t_calc
            # Capture the latest score
            if not t_calc.empty:
                current_scores[c] = t_calc['GPTI'].iloc[-1]
            else:
                current_scores[c] = 0

        # --- 1. GLOBAL STATUS HUD ---
        # Calculate Global Metrics
        avg_global_tension = sum(current_scores.values()) / len(current_scores) if current_scores else 0
        highest_risk_zone = max(current_scores, key=current_scores.get) if current_scores else "None"
        highest_score = current_scores[highest_risk_zone] if current_scores else 0
        
        # Determine Global Condition
        if avg_global_tension > 50: global_cond = "🔴 DEFCON 2 (HIGH)"
        elif avg_global_tension > 30: global_cond = "🟠 DEFCON 3 (ELEVATED)"
        else: global_cond = "🟢 DEFCON 4 (MODERATE)"

        m1, m2, m3 = st.columns(3)
        m1.metric("Global Avg Tension", f"{avg_global_tension:.1f}", help="Average GPTI across all monitored zones")
        m2.metric("Critical Hotspot", highest_risk_zone, f"{highest_score:.1f} GPTI", delta_color="inverse")
        m3.metric("Global Threat Condition", global_cond)

        st.markdown("---")

        # --- 2. COMPARATIVE TIMELINE (Line Chart) ---
        st.subheader("📉 Tension Timeline Comparison")
        fig_global = go.Figure()
        
        # Custom colors for each conflict
        colors = {"India-Pakistan": "#FFD700", "Russia-Ukraine": "#00CC96", "Israel-Palestine": "#EF553B"}
        
        for c, df in all_data.items():
            fig_global.add_trace(go.Scatter(
                x=df['date'], 
                y=df['GPTI'], 
                mode='lines', 
                name=c,
                line=dict(width=2, color=colors.get(c, "white"))
            ))
        
        # Add Critical Threshold Line
        fig_global.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="CRITICAL THRESHOLD", annotation_position="bottom right")
        
        fig_global.update_layout(
            height=450, 
            template="plotly_dark", 
            hovermode="x unified", 
            margin=dict(l=20, r=20, t=40, b=20), 
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig_global, use_container_width=True)

        # --- 3. INTENSITY HEATMAP (New Visualization) ---
        st.subheader("🔥 Conflict Intensity Heatmap")
        st.caption("Visualizing periods of simultaneous global escalation. Brighter colors indicate higher tension.")
        
        # Align dates for the heatmap
        all_dates = sorted(list(set().union(*[df['date'] for df in all_data.values()])))
        heatmap_df = pd.DataFrame({'date': all_dates})
        
        for c, df in all_data.items():
            # Merge each conflict's GPTI score into the master dataframe
            temp = df[['date', 'GPTI']].rename(columns={'GPTI': c})
            heatmap_df = pd.merge(heatmap_df, temp, on='date', how='left').fillna(0)
            
        # Create Heatmap Matrix (Z)
        z_data = [heatmap_df[c].tolist() for c in conflicts]
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=z_data,
            x=heatmap_df['date'],
            y=conflicts,
            colorscale='Magma', # Dark to Bright Red/Yellow
            colorbar=dict(title='Tension Score')
        ))
        
        fig_heat.update_layout(
            height=300, 
            template="plotly_dark", 
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ------------------------------------------
# TAB 5: STATISTICAL DEEP DIVE
# ------------------------------------------
with tab_deep:
    st.header("🔬 Advanced Statistical Diagnostics")
    st.markdown("Rigorous validation of the causal relationship between **Narrative (Words)** and **Kinetic (War)**.")

    # --- 1. LEAD-LAG ANALYSIS (CROSS-CORRELATION) ---
    st.subheader("1. Lead-Lag Analysis (Cross-Correlation)")
    st.caption("Does News appear *before* Violence? High red bars on the right confirm Narrative is a Leading Indicator.")
    
    if not final_df.empty:
        # Calculate Cross-Correlation for lags -10 to +10 days
        lags = list(range(-10, 11))
        # Calculate correlation: Kinetic(t) vs Narrative(t-lag)
        # Positive Lag means Narrative happened BEFORE Kinetic (Predictive)
        cross_corrs = [final_df['norm_kinetic'].corr(final_df['norm_narrative'].shift(lag)) for lag in lags]
        
        # Color logic: Red for "Predictive" (Positive Lags), Grey for "Reactive"
        colors = ['#EF233C' if lag > 0 else '#8D99AE' for lag in lags]
        
        fig_ccf = go.Figure(go.Bar(
            x=lags, 
            y=cross_corrs,
            marker_color=colors,
            hovertemplate='Lag: %{x} days<br>Correlation: %{y:.3f}<extra></extra>'
        ))
        
        fig_ccf.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="Simultaneous")
        
        fig_ccf.update_layout(
            title="Time-Shifted Correlation Strength",
            xaxis_title="Time Lag (Days) [Positive = Narrative Leads]",
            yaxis_title="Correlation Coefficient",
            template="plotly_dark",
            height=400,
            bargap=0.2
        )
        st.plotly_chart(fig_ccf, use_container_width=True)
        st.info("💡 **Interpretation:** If the red bars (right side) are higher than the grey bars (left side), it proves that **Information Warfare precedes Physical Warfare**.")

    st.markdown("---")

    # --- 2. DYNAMIC CORRELATION & METRICS ---
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("2. Rolling Correlation (30-Day Window)")
        st.caption("How synchronized are the two pillars right now?")
        
        # Calculate 30-day rolling correlation
        rolling_corr = final_df['norm_kinetic'].rolling(window=30).corr(final_df['norm_narrative'])
        
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=final_df['date'], 
            y=rolling_corr, 
            mode='lines', 
            name='Correlation Strength',
            line=dict(color='#4CC9F0', width=2),
            fill='tozeroy',
            fillcolor='rgba(76, 201, 240, 0.1)'
        ))
        fig_roll.update_layout(
            template="plotly_dark", 
            height=350, 
            yaxis_range=[-1, 1],
            yaxis_title="Correlation (-1 to +1)"
        )
        st.plotly_chart(fig_roll, use_container_width=True)
        
    with c2:
        st.subheader("3. Scientific Proof")
        
        # Correlation Matrix
        st.markdown("**Global Correlation Matrix**")
        corr_matrix = final_df[['norm_kinetic', 'norm_narrative', 'GPTI']].corr()
        st.dataframe(corr_matrix.style.background_gradient(cmap='Reds').format("{:.2f}"))
        
        st.divider()
        
        # Granger Causality
        st.markdown("**Granger Causality Test**")
        pval = calculator.test_causality(final_df)
        
        if pval < 0.01:
            st.success(f"🌟 **HIGHLY SIGNIFICANT**\n\nP-Value: {pval:.4f}")
            st.caption("Extremely strong predictive link.")
        elif pval < 0.05:
            st.success(f"✅ **SIGNIFICANT**\n\nP-Value: {pval:.4f}")
            st.caption("Predictive link verified.")
        else:
            st.warning(f"⚠️ **WEAK SIGNAL**\n\nP-Value: {pval:.4f}")
            st.caption("No statistical lead-lag found.")

# ------------------------------------------
# TAB 6: LIVE AI LAB
# ------------------------------------------
with tab_ai:
    st.header("🧠 Advanced Intelligence Lab")
    st.markdown("Direct uplink to the **Hybrid AI Engine**. Analyze raw intelligence feeds in real-time.")
    
    # Layout: Input on Left, Analysis on Right
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("📥 Signal Input")
        
        # 1. Quick-Fire Scenario Buttons (For fast demos)
        st.caption("Select a simulation scenario or type your own:")
        b1, b2, b3 = st.columns(3)
        
        sample_text = ""
        if b1.button("🔴 War Scenario"): 
            sample_text = "BREAKING: Heavy artillery fire reported at the border. Diplomatic channels suspended immediately."
        if b2.button("🟢 Peace Scenario"): 
            sample_text = "Prime Ministers meet for historic trade summit, promising to de-escalate tensions."
        if b3.button("❌ Noise (Sports)"): 
            sample_text = "Cricket captain scores century in thrilling match against rival team."

        # Text Area (Pre-filled if button clicked, else empty)
        # We use session state to handle the button click text population
        if 'lab_input' not in st.session_state: st.session_state['lab_input'] = ""
        if sample_text: st.session_state['lab_input'] = sample_text
        
        txt = st.text_area("Intelligence Brief / Headline:", value=st.session_state['lab_input'], height=150)
        
        analyze_btn = st.button("🚀 Analyze Signal", type="primary", use_container_width=True)

    with c2:
        st.subheader("📊 AI Diagnostics")
        # Placeholder container for results
        result_container = st.empty()
        
        if analyze_btn and txt and ai_brain:
            with st.spinner("Decrypting & Analyzing..."):
                # 1. Run AI Logic
                is_relevant = ai_brain.llm_relevance_filter(txt)
                sentiment_score = ai_brain.get_sentiment_score(txt)
                
                # Convert to Hostility Percentage (0-100)
                # Logic: sentiment_score 0.0 = Angry, 1.0 = Happy. 
                # We want Hostility, so we invert it: (1 - score)
                hostility_val = (1.0 - sentiment_score) * 100
                
                # 2. Visualization: Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = hostility_val,
                    title = {'text': "HOSTILITY INDEX"},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#EF233C"}, # Red needle
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': "rgba(0, 255, 0, 0.3)"},  # Green Zone
                            {'range': [40, 70], 'color': "rgba(255, 165, 0, 0.3)"}, # Orange Zone
                            {'range': [70, 100], 'color': "rgba(255, 0, 0, 0.3)"}], # Red Zone
                    }
                ))
                fig_gauge.update_layout(
                    height=250, 
                    margin=dict(l=20,r=20,t=30,b=20), 
                    paper_bgcolor="rgba(0,0,0,0)", 
                    font={'color': "white", 'family': "Arial"}
                )
                
                # 3. Display Results
                with result_container.container():
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    if is_relevant:
                        if hostility_val > 70:
                            st.error("🚨 **VERDICT: CRITICAL THREAT DETECTED**")
                        elif hostility_val > 40:
                            st.warning("⚠️ **VERDICT: ELEVATED TENSION**")
                        else:
                            st.success("🕊️ **VERDICT: STABLE / PEACEFUL**")
                    else:
                        st.info("ℹ️ **VERDICT: IRRELEVANT NOISE (FILTERED)**")
                
                # 4. Save to Session History (The "Log")
                if 'ai_history' not in st.session_state: st.session_state['ai_history'] = []
                st.session_state['ai_history'].insert(0, {
                    "Timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "Brief": txt[:50] + "...",
                    "Relevance": "✅" if is_relevant else "❌",
                    "Hostility": f"{hostility_val:.1f}%"
                })

        elif not ai_brain:
            st.error("AI Engine Offline. Check dependencies.")

    # --- SESSION HISTORY LOG ---
    st.markdown("---")
    st.subheader("🗂️ Mission Log (Session History)")
    if 'ai_history' in st.session_state and st.session_state['ai_history']:
        st.dataframe(
            pd.DataFrame(st.session_state['ai_history']), 
            use_container_width=True,
            hide_index=True
        )
    else:
        st.caption("No analysis performed this session. Awaiting input...")

# ------------------------------------------
# TAB 7: INTEL
# ------------------------------------------
with tab_about:
    st.header("ℹ️ System Architecture & Methodology")
    
    # Display the diagram (Ensure 'architecture.png' is in your folder)
    # If the image is missing, this line will be skipped safely
    try:
        st.image("architecture.png", caption="Fig 1. GeoSentinel Data Flow Architecture", use_container_width=True)
    except:
        st.warning("Architecture diagram not found. Please ensure 'architecture.png' is in the script directory.")

    st.markdown("""
    ### **Core Framework**
    **GeoSentinel** operates on a **Two-Pillar Architecture** designed to quantify geopolitical risk by fusing physical events with information warfare signals.

    #### **1. Data Ingestion Layer (The Eyes)**
    * **Kinetic Pillar (MCT):** Ingests event data simulating **ACLED** (Armed Conflict Location & Event Data). It tracks physical violence, assigning severity weights to incidents like shelling, airstrikes, and border skirmishes.
    * **Narrative Pillar (INT):** Ingests unstructured text data simulating **GDELT** and Global News APIs. It monitors the volume and tone of discourse surrounding the conflict.

    #### **2. Hybrid AI Engine (The Brain)**
    * **Stage 1: The Gatekeeper (LLM):** A Large Language Model filters raw data to remove noise (e.g., filtering out "cricket matches" or "movie reviews" from conflict news).
    * **Stage 2: The Analyst (DistilBERT):** A fine-tuned **DistilBERT** transformer model analyzes the sentiment of relevant text. It quantifies hostility on a scale of 0.0 (War) to 1.0 (Peace).

    #### **3. Dynamic Aggregation (The Judge)**
    * **Normalization:** Uses **Min-Max Scaling** to standardize disparate data units (e.g., bomb counts vs. tweet counts) onto a unified 0-1 scale.
    * **PCA Weighting:** Applies **Principal Component Analysis (PCA)** over a 30-day rolling window. The system automatically assigns higher weight to the pillar with higher variance, ensuring the index adapts to the dominant driver of tension (Kinetic vs. Narrative).

    #### **4. Statistical Validation**
    * **Granger Causality:** The system runs real-time econometric tests to verify if narrative spikes act as a "leading indicator" for physical violence ($p < 0.05$).
    """)
    
    st.markdown("---")
    st.caption("© 2025 GeoSentinel Project | Developed for Computational Political Science Research")