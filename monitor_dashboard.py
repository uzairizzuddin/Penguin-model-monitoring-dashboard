import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Model Dashboard", layout="wide")

st.title("📊 Model Performance Dashboard")

LOG_FILE = "monitoring_logs.csv"

# Check if the log file exists
if os.path.exists(LOG_FILE):
    # Load data
    df = pd.read_csv(LOG_FILE)
    
    # 1. Summary Metrics
    st.subheader("System Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", len(df))
    col2.metric("Avg Latency", f"{df['latency'].mean():.4f} s")
    col3.metric("Avg User Score", f"{df['user_feedback'].mean():.2f} / 5")

    st.divider()

    # 2. Visualizations
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Latency vs Model Version")
        # Box plot to compare speed of v1 vs v2
        fig_latency = px.box(df, x='model_version', y='latency', title="Latency Distribution")
        st.plotly_chart(fig_latency)

    with col_chart2:
        st.subheader("User Satisfaction")
        # Bar chart for feedback scores
        fig_feedback = px.bar(
            df.groupby('model_version')['user_feedback'].mean().reset_index(), 
            x='model_version', 
            y='user_feedback',
            title="Average Feedback Score by Model"
        )
        st.plotly_chart(fig_feedback)

    # 3. Raw Logs
    st.subheader("Recent Logs")
    st.dataframe(df.sort_values(by="timestamp", ascending=False))

else:
    st.info("No logs found yet. Go to the 'Predictive App' and make some predictions to generate data!")