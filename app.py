import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import index_calculator
import data_ingestion

st.set_page_config(layout="wide", page_title="GeoSentinal: Multi-Dyad Index")

st.title("🛡️ GeoSentinal: Global Geopolitical Tension Index")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Configuration")
selected_dyad = st.sidebar.selectbox(
    "Select Conflict Zone:",
    ["India-Pakistan", "Russia-Ukraine", "Israel-Palestine"] # Added new option
)

# --- GENERATE & CALCULATE ---
with st.spinner(f"Simulating {selected_dyad} Data..."):
    df = data_ingestion.generate_synthetic_data(selected_dyad)
    calculator = index_calculator.IndexCalculator()
    final_df = calculator.process_index(df)
    final_df['date'] = pd.to_datetime(final_df['date'])

# --- STATS ---
p_value = calculator.test_causality(final_df)
is_significant = p_value < 0.05

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Statistical Validation")
st.sidebar.write("Does Narrative Predict Violence?")
if is_significant:
    st.sidebar.success(f"✅ YES (Significant)")
    st.sidebar.metric("P-Value", f"{p_value:.4f}", delta="Strong Link")
else:
    st.sidebar.error(f"❌ NO (Not Significant)")
    st.sidebar.metric("P-Value", f"{p_value:.4f}")

# --- DASHBOARD ---
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    subplot_titles=(f"{selected_dyad} Tension Index (GPTI)", "Pillars", "AI Weights"))

fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['GPTI'], mode='lines', name='GPTI', line=dict(color='red', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['norm_kinetic'], mode='lines', name='Kinetic', line=dict(color='orange')), row=2, col=1)
fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['norm_narrative'], mode='lines', name='Narrative', line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['weight_kinetic'], stackgroup='one', name='Kinetic Weight'), row=3, col=1)
fig.add_trace(go.Scatter(x=final_df['date'], y=final_df['weight_narrative'], stackgroup='one', name='Narrative Weight'), row=3, col=1)

fig.update_layout(height=800, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)